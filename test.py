"""Test a model and generate submission CSV.

Usage:
    > python test.py --split SPLIT --load_path PATH --name NAME
    where
    > SPLIT is either "dev" or "test"
    > PATH is a path to a checkpoint (e.g., save/train/model-01/best.pth.tar)
    > NAME is a name to identify the test run

Author:
    Chris Chute (chute@stanford.edu)
"""

import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import util
from args import get_test_args
from collections import OrderedDict
from json import dumps
from os.path import join
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load

import re
import os
import sys
import math
import random
import string
import logging
import argparse
from shutil import copyfile
from datetime import datetime
import msgpack
from models import DRQA
from util import str2bool, AverageMeter
from train import BatchGen, load_data


def main(args):
    # Set up
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=False)
    log = util.get_logger(args.save_dir, args.name)
    device, gpu_ids = util.get_available_devices()

    # Get embeddings
    log.info('Loading embeddings...')
    with open('data/meta.msgpack','rb') as f:
        meta = msgpack.load(f, encoding = 'utf8')
    embedding = torch.Tensor(meta['embedding'])
    opt = vars(args)
    opt['pretrained_words'] = True
    opt['vocab_size'] = embedding.size(0)
    opt['embedding_dim'] = embedding.size(1)
    opt['pos_size'] = len(meta['vocab_tag'])
    opt['ner_size'] = len(meta['vocab_ent'])
       
    # Get model
    log.info('Building model...')
    model = DRQA(opt, embedding = embedding)
    model = nn.DataParallel(model, gpu_ids)
    log.info(f'Loading checkpoint from {args.load_path}...')
    model = util.load_model(model, args.load_path, gpu_ids, return_step=False)
    model = model.to(device)
    model.eval()
    BatchGen.pos_size = opt['pos_size']
    BatchGen.ner_size = opt['ner_size']

    # Get data loader
    log.info('Building dataset...')
    with open(opt['data_file'], 'rb') as f:
        data = msgpack.load(f, encoding='utf8')
    test = data['test']

    # Evaluate
    log.info(f'Evaluating on test split...')
    batches = BatchGen(test, batch_size=args.batch_size, evaluation=True, is_test = True, gpu=args.cuda)
    
    nll_meter = util.AverageMeter()
    pred_dict = {}  # Predictions for TensorBoard
    sub_dict = {}   # Predictions for submission
    eval_file = './data/test_eval.json'
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), \
            tqdm(total=len(test)) as progress_bar:
        model.eval()
        for i, batch in enumerate(batches):
            # Setup for forward
            inputs = [e.to(device) for e in batch[:7]]
            ids = batch[-1]

            # Forward
            with torch.no_grad():
                score_s, score_e = model(*inputs)
                
            p1, p2 = score_s, score_e
            starts, ends = util.discretize(p1, p2, args.max_ans_len, args.use_squad_v2)

            # Log info
            progress_bar.update(args.batch_size)
            idx2pred, uuid2pred = util.convert_tokens(gold_dict,
                                                      ids.tolist(),
                                                      starts.tolist(),
                                                      ends.tolist(),
                                                      args.use_squad_v2)
            pred_dict.update(idx2pred)
            sub_dict.update(uuid2pred)

    # Log results (except for test set, since it does not come with labels)
    if args.split != 'test':
        results = util.eval_dicts(gold_dict, pred_dict, args.use_squad_v2)
        results_list = [('NLL', nll_meter.avg),
                        ('F1', results['F1']),
                        ('EM', results['EM'])]
        if args.use_squad_v2:
            results_list.append(('AvNA', results['AvNA']))
        results = OrderedDict(results_list)

        # Log to console
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
        log.info(f'{args.split.title()} {results_str}')

        # Log to TensorBoard
        tbx = SummaryWriter(args.save_dir)
        util.visualize(tbx,
                       pred_dict=pred_dict,
                       eval_path=eval_file,
                       step=0,
                       split=args.split,
                       num_visuals=args.num_visuals)

    # Write submission file
    sub_path = join(args.save_dir, args.split + '_' + args.sub_file)
    log.info(f'Writing submission file to {sub_path}...')
    with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
        csv_writer = csv.writer(csv_fh, delimiter=',')
        csv_writer.writerow(['Id', 'Predicted'])
        for uuid in sorted(sub_dict):
            csv_writer.writerow([uuid, sub_dict[uuid]])


if __name__ == '__main__':
    main(get_test_args())
