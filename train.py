import re
import os
import math
import random
import string
import logging
from args import get_train_args
from shutil import copyfile
from datetime import datetime
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.nn.functional as F
import msgpack
from models import DRQA
import util
from util import str2bool, AverageMeter


def main():
    args, log = get_train_args()
    log.info('[Program starts. Loading data...]')
    train, dev, dev_y, embedding, opt = load_data(vars(args))
    device, args.gpu_ids = util.get_available_devices()
    log.info('[Data loaded.]')
    if args.save_dawn_logs:
        dawn_start = datetime.now()
        log.info('dawn_entry: epoch\tf1Score\thours')
    
    if args.resume:
        log.info('[loading previous model...]')
        checkpoint = torch.load(os.path.join(args.model_dir, args.resume))
        if args.resume_options:
            opt = checkpoint['config']
        state_dict = checkpoint['state_dict']
        model = DocReaderModel(opt, embedding, state_dict)
        epoch_0 = checkpoint['epoch'] + 1
        # synchronize random seed
        random.setstate(checkpoint['random_state'])
        torch.random.set_rng_state(checkpoint['torch_state'])
        if args.cuda:
            torch.cuda.set_rng_state(checkpoint['torch_cuda_state'])
        if args.reduce_lr:
            lr_decay(model.optimizer, lr_decay=args.reduce_lr)
            log.info('[learning rate reduced by {}]'.format(args.reduce_lr))

        batches = BatchGen(dev, batch_size=args.batch_size, evaluation=True, gpu=args.cuda)
        predictions = []
        for i, batch in enumerate(batches):
            text = batch[-2]
            spans = batch[-1]
            pred = []
            max_len = opt['max_len'] or score_s.size(1)
            for i in range(score_s.size(0)):
                scores = torch.ger(score_s[i], score_e[i])
                scores.triu_().tril_(max_len - 1)
                scores = scores.numpy()
                s_idx, e_idx = np.unravel_index(np.argmax(scores), scores.shape)
                s_offset, e_offset = spans[i][s_idx][0], spans[i][e_idx][1]
                pred.append(text[i][s_offset:e_offset])
            predictions.extend(pred)
            log.debug('> evaluating [{}/{}]'.format(i, len(batches)))
        em, f1 = score(predictions, dev_y)

        log.info("[dev EM: {} F1: {}]".format(em, f1))
        if math.fabs(em - checkpoint['em']) > 1e-3 or math.fabs(f1 - checkpoint['f1']) > 1e-3:
            log.info('Inconsistent: recorded EM: {} F1: {}'.format(checkpoint['em'], checkpoint['f1']))
            log.error('Error loading model: current code is inconsistent with code used to train the previous model.')
            exit(1)
        best_val_score = checkpoint['best_eval']
    else:
        model = DocReaderModel(opt, embedding)
        epoch_0 = 1
        best_val_score = 0.0
        updates = 0
        optimizer = optim.Adamax(parameters, weight_decay = opt['weight_decay'])
        scheduler = sched.LambdaLR(optimizer, lambda s: 1.)
        train_loss = AverageMeter()
        
    model = nn.DataParallel(model, args.gpu_ids)
    model = model.to(device)

    # ema = util.EMA(model, args.ema_decay)
    

    for epoch in range(epoch_0, epoch_0 + args.epochs):
        log.warning('Epoch {}'.format(epoch))
        # train
        batches = BatchGen(train, batch_size=args.batch_size, gpu=args.cuda)
        start = datetime.now()
        
        model.train()
        for i, batch in enumerate(batches):
            # Transfer to GPU
            inputs = [e.to(device) for e in batch[:7]]
            target_s = batch[7].to(device)
            target_e = batch[8].to(device)
            optimizer.zero_grad()
            
            # Forward
            score_s,score_e = model(*inputs)
            loss = F.nll_loss(score_s, target_s) + F.nll_loss(score_e, target_e)
            train_loss.update(loss.item())
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),opt['grad_clipping'])
            optimizer.step()
            updates +=1
            
            # Clip gradients

            if i % args.log_per_updates == 0:
                log.info('> epoch [{0:2}] updates[{1:6}] train loss[{2:.5f}] remaining[{3}]'.format(
                    epoch,updates, train_loss.value,
                    str((datetime.now() - start) / (i + 1) * (len(batches) - i - 1)).split('.')[0]))
        log.debug('\n')
        # eval
        batches = BatchGen(dev, batch_size=args.batch_size, evaluation=True, gpu=args.cuda)
        predictions = []
        for i, batch in enumerate(batches):
            model.eval()
            inputs = [e.to(device) for e in batch[:7]]
            
            # Run forward
            with torch.no_grad():
                score_s, score_e = model(*inputs)
                
            # Get argmax test spans
            text = batch[-2]
            spans = batch[-1]
            pred = []
            max_len = opt['max_len'] or score_s.size(1)
            for i in range(score_s.size(0)):
                scores = torch.ger(score_s[i], score_e[i])
                scores.triu_().tril_(max_len - 1)
                scores = scores.numpy()
                s_idx, e_idx = np.unravel_index(np.argmax(scores), scores.shape)
                s_offset, e_offset = spans[i][s_idx][0], spans[i][e_idx][1]
                pred.append(text[i][s_offset:e_offset])
            predictions.extend(pred)
            log.debug('> evaluating [{}/{}]'.format(i, len(batches)))
        em, f1 = score(predictions, dev_y)
        log.warning("dev EM: {} F1: {}".format(em, f1))
        if args.save_dawn_logs:
            time_diff = datetime.now() - dawn_start
            log.warning("dawn_entry: {}\t{}\t{}".format(epoch, f1/100.0, float(time_diff.total_seconds() / 3600.0)))
        # save
        if not args.save_last_only or epoch == epoch_0 + args.epochs - 1:
            model_file = os.path.join(args.model_dir, 'checkpoint_epoch_{}.pt'.format(epoch))
            params = {
                'state_dict': {
                    'network': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'updates': updates,
                    'loss': train_loss.state_dict()
                },
                'config': opt,
                'epoch': epoch,
                'em': em,
                'f1': f1,
                'best_eval': best_val_score,
                'random_state': random.getstate(),
                'torch_state': torch.random.get_rng_state(),
                #'torch_cuda_state': torch.cuda.get_rng_state()
            }
            torch.save(params, model_file)
            logger.info('model saved to {}'.format(model_file))
            if f1 > best_val_score:
                best_val_score = f1
                copyfile(
                    model_file,
                    os.path.join(args.model_dir, 'best_model.pt'))
                log.info('[new best model saved.]')


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
    data['dev'].sort(key=lambda x: len(x[1]))
    dev = [x[:-1] for x in data['dev']]
    dev_y = [x[-1] for x in data['dev']]
    return train, dev, dev_y, embedding, opt


class BatchGen:
    pos_size = None
    ner_size = None

    def __init__(self, data, batch_size, gpu, evaluation=False):
        """
        input:
            data - list of lists
            batch_size - int
        """
        self.batch_size = batch_size
        self.eval = evaluation
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
            log.off(batch)
            batch_size = len(batch)
            batch = list(zip(*batch))
            if self.eval:
                assert len(batch) == 8
            else:
                assert len(batch) == 10

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
            if not self.eval:
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
            if self.eval:
                yield (context_id, context_feature, context_tag, context_ent, context_mask,
                       question_id, question_mask, text, span)
            else:
                yield (context_id, context_feature, context_tag, context_ent, context_mask,
                       question_id, question_mask, y_s, y_e, text, span)


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