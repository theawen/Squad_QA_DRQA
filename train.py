import re
import os
import math
import random
import string
from collections import OrderedDict
import logging
from args import get_train_args
from shutil import copyfile
from datetime import datetime
from collections import Counter
from ujson import load as json_load
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sched
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F
import msgpack
from models import DRQA
import util
from util import str2bool, AverageMeter


def main():
    args, log = get_train_args()
    log.info('[Program starts. Loading data...]')
    train, dev, embedding, opt = load_data(vars(args))
    tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids = util.get_available_devices()
    log.info('[Data loaded.]')
    
    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model, step = util.load_model(model, args.load_path, args.gpu_ids)
    else:
        model = DRQA(opt, embedding = embedding)
        step = 0
        updates = 0
        
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adamax(parameters, weight_decay = opt['weight_decay'])
    scheduler = sched.LambdaLR(optimizer, lambda s: 1.)
    train_loss = AverageMeter()
        
    model = nn.DataParallel(model, args.gpu_ids)
    model = model.to(device)
    model.train()
    
    # get saver
    saver = util.CheckpointSaver(args.save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log)

    # ema = util.EMA(model, args.ema_decay)
    
    log.info('Training...')
    steps_till_eval = args.eval_steps
    batch_size = args.batch_size
    epoch = step // len(train)
    while epoch != args.epochs:
        epoch+=1
        log.info(f'Starting epoch {epoch}...')
        with torch.enable_grad(), \
                tqdm(total=len(train)) as progress_bar:
            # train
            batches = BatchGen(train, batch_size=batch_size, gpu=args.cuda)
            for i, batch in enumerate(batches):
                # Transfer to GPU
                inputs = [e.to(device) for e in batch[:7]]
                target_s = batch[7].to(device)
                target_e = batch[8].to(device)
                optimizer.zero_grad()

                # Forward
                score_s,score_e = model(*inputs)
                loss = F.nll_loss(score_s, target_s) + F.nll_loss(score_e, target_e)
                loss_val = loss.item()

                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),opt['grad_clipping'])
                optimizer.step()
                scheduler.step(step // batch_size)
                step += batch_size
                
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         NLL=loss_val)
                tbx.add_scalar('train/NLL', loss_val, step)
#                 tbx.add_scalar('train/LR',
#                                optimizer.param_groups[0]['lr'],
#                                step)
                
                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps
                    # Evaluate and save checkpoint
                    log.info(f'Evaluating at updates {step}...')
                    results, pred_dict = evaluate(model, dev, args, device,
                                                  args.dev_eval_file,
                                                  args.max_ans_len,
                                                  args.use_squad_v2)
                    
                    saver.save(step,opt, model, results[args.metric_name], device)
                        
                    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
                    log.info(f'Dev {results_str}')
                
                    # Log to TensorBoard
                    log.info('Visualizing in TensorBoard...')
                    for k, v in results.items():
                        tbx.add_scalar(f'dev/{k}', v, step)
                    util.visualize(tbx,
                                   pred_dict=pred_dict,
                                   eval_path=args.dev_eval_file,
                                   step=step,
                                   split='dev',
                                   num_visuals=args.num_visuals)
            log.debug('\n')

            

def lr_decay(optimizer, lr_decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    return optimizer


def load_data(opt):
    with open('data/meta.msgpack', 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')
    embedding = torch.Tensor(meta['embedding'])
    opt['pretrained_words'] = True
    opt['vocab_size'] = embedding.size(0)
    opt['embedding_dim'] = embedding.size(1)
    opt['pos_size'] = len(meta['vocab_tag'])
    opt['ner_size'] = len(meta['vocab_ent'])
    BatchGen.pos_size = opt['pos_size']
    BatchGen.ner_size = opt['ner_size']
    with open(opt['data_file'], 'rb') as f:
        data = msgpack.load(f, encoding='utf8')
    train = data['train']
    dev = data['dev']
    return train, dev, embedding, opt

def evaluate(model, dev,args, device, eval_file, max_len, use_squad_v2):
    model.eval()
    pred_dict = {}
    batch_size = args.batch_size
    batches = BatchGen(dev, batch_size=args.batch_size, evaluation=True, gpu=args.cuda)
    nll_meter = util.AverageMeter()
    
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), \
            tqdm(total=len(dev)) as progress_bar:
        for i, batch in enumerate(batches):
            # Setup for forward
            inputs = [e.to(device) for e in batch[:7]]
            target_s = batch[7].to(device)
            target_e = batch[8].to(device)
            ids = batch[-1]
            
            # Run forward
            with torch.no_grad():
                score_s, score_e = model(*inputs)
            loss = F.nll_loss(score_s, target_s) + F.nll_loss(score_e, target_e)
            nll_meter.update(loss.item())
            
            # Get F1 and EM scores
            p1, p2 = score_s, score_e
            starts, ends = util.discretize(p1, p2, max_len, use_squad_v2)  
            
            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(NLL=nll_meter.avg)
            
            preds, _ = util.convert_tokens(gold_dict,
                                           ids.tolist(),
                                           starts.tolist(),
                                           ends.tolist(),
                                           use_squad_v2)
            pred_dict.update(preds)

    model.train()

    results = util.eval_dicts(gold_dict, pred_dict, use_squad_v2)
    results_list = [('NLL', nll_meter.avg),
                    ('F1', results['F1']),
                    ('EM', results['EM'])]
    if use_squad_v2:
        results_list.append(('AvNA', results['AvNA']))
    results = OrderedDict(results_list)

    return results, pred_dict

class BatchGen:
    pos_size = None
    ner_size = None

    def __init__(self, data, batch_size, gpu, evaluation=False, is_test = False):
        """
        input:
            data - list of lists
            batch_size - int
        """
        self.batch_size = batch_size
        self.eval = evaluation
        self.is_test = is_test
        self.gpu = gpu

        # sort by len
        data = sorted(data, key=lambda x: len(x[1]))
        # chunk into batches
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

        # shuffle
        if not evaluation:
            random.shuffle(data)

        self.data = data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for batch in self.data:
            batch_size = len(batch)
            batch = list(zip(*batch))

            context_len = max(len(x) for x in batch[1])
            context_id = torch.LongTensor(batch_size, context_len).fill_(0)
            for i, doc in enumerate(batch[1]):
                context_id[i, :len(doc)] = torch.LongTensor(doc)

            feature_len = len(batch[2][0][0])

            context_feature = torch.Tensor(batch_size, context_len, feature_len).fill_(0)
            for i, doc in enumerate(batch[2]):
                for j, feature in enumerate(doc):
                    context_feature[i, j, :] = torch.Tensor(feature)

            context_tag = torch.Tensor(batch_size, context_len, self.pos_size).fill_(0)
            for i, doc in enumerate(batch[3]):
                for j, tag in enumerate(doc):
                    context_tag[i, j, tag] = 1

            context_ent = torch.Tensor(batch_size, context_len, self.ner_size).fill_(0)
            for i, doc in enumerate(batch[4]):
                for j, ent in enumerate(doc):
                    context_ent[i, j, ent] = 1

            question_len = max(len(x) for x in batch[5])
            question_id = torch.LongTensor(batch_size, question_len).fill_(0)
            for i, doc in enumerate(batch[5]):
                question_id[i, :len(doc)] = torch.LongTensor(doc)

            context_mask = torch.eq(context_id, 0)
            question_mask = torch.eq(question_id, 0)
            text = list(batch[6])
            span = list(batch[7])
            ids = torch.LongTensor(batch[0])
            if not self.is_test:
                y_s = torch.LongTensor(batch[8])
                y_e = torch.LongTensor(batch[9])
            if self.gpu:
                context_id = context_id.pin_memory()
                context_feature = context_feature.pin_memory()
                context_tag = context_tag.pin_memory()
                context_ent = context_ent.pin_memory()
                context_mask = context_mask.pin_memory()
                question_id = question_id.pin_memory()
                question_mask = question_mask.pin_memory()
            if self.is_test:
                yield (context_id, context_feature, context_tag, context_ent, context_mask,
                       question_id, question_mask, text, span, ids)
            else:
                yield (context_id, context_feature, context_tag, context_ent, context_mask,
                       question_id, question_mask, y_s, y_e, text, span, ids)

def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _exact_match(pred, answers):
    if pred is None or answers is None:
        return False
    pred = _normalize_answer(pred)
    for a in answers:
        if pred == _normalize_answer(a):
            return True
    return False


def _f1_score(pred, answers):
    def _score(g_tokens, a_tokens):
        common = Counter(g_tokens) & Counter(a_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1. * num_same / len(g_tokens)
        recall = 1. * num_same / len(a_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    if pred is None or answers is None:
        return 0
    g_tokens = _normalize_answer(pred).split()
    scores = [_score(g_tokens, _normalize_answer(a).split()) for a in answers]
    return max(scores)


def score(pred, truth):
    assert len(pred) == len(truth)
    f1 = em = total = 0
    for p, t in zip(pred, truth):
        total += 1
        em += _exact_match(p, t)
        f1 += _f1_score(p, t)
    em = 100. * em / total
    f1 = 100. * f1 / total
    return em, f1


def load_statedict(model, state_dict):
    # Book-keeping.
    self.opt = opt
    self.device = torch.cuda.current_device() if opt['cuda'] else torch.device('cpu')
    self.updates = state_dict['updates']
    self.train_loss = AverageMeter()
    if state_dict:
        self.train_loss.load(state_dict['loss'])

    # Building network.
    self.network = RnnDocReader(opt, embedding=embedding)
    if state_dict:
        new_state = set(self.network.state_dict().keys())
        for k in list(state_dict['network'].keys()):
            if k not in new_state:
                del state_dict['network'][k]
        self.network.load_state_dict(state_dict['network'])
    self.network.to(self.device)

    # Building optimizer.
    self.opt_state_dict = state_dict['optimizer'] if state_dict else None
    self.build_optimizer()

if __name__ == '__main__':
    main()