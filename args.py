"""Command-line arguments for setup.py, train.py, test.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import argparse
from util import str2bool
import torch
import sys
import os
import multiprocessing
from util import str2bool
import random
import logging

# setup logger
class ProgressHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        log_entry = self.format(record)
        if record.message.startswith('> '):
            sys.stdout.write('{}\r'.format(log_entry.rstrip()))
            sys.stdout.flush()
        else:
            sys.stdout.write('{}\n'.format(log_entry))


def add_setup_args(parser):
    parser.add_argument('--trn_file', default='data/train-v2.0.json',
                        help='path to train file.')
    parser.add_argument('--dev_file', default='data/dev-v2.0.json',
                        help='path to dev file.')
    parser.add_argument('--tst_file', default='data/test-v2.0.json',
                        help='path to test file.')
    parser.add_argument('--wv_file', default='data/glove.840B.300d/glove.840B.300d.txt',
                        help='path to word vector file.')
    parser.add_argument('--wv_dim', type=int, default=300,
                        help='word vector dimension.')
    parser.add_argument('--wv_cased', type=str2bool, nargs='?',
                        const=True, default=True,
                        help='treat the words as cased or not.')
    parser.add_argument('--sort_all', action='store_true',
                        help='sort the vocabulary by frequencies of all words. '
                             'Otherwise consider question words first.')
    parser.add_argument('--sample_size', type=int, default=0,
                        help='size of sample data (for debugging).')
    parser.add_argument('--threads', type=int, default=min(multiprocessing.cpu_count(), 16),
                        help='number of threads for preprocessing.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size for multiprocess tokenizing and tagging.')
    args = parser.parse_args()
    return args
    
def add_train_args(parser):
    # system
    parser.add_argument('--log_per_updates', type=int, default=3,
                        help='log model loss per x updates (mini-batches).')
    parser.add_argument('--data_file', default='data/data.msgpack',
                        help='path to preprocessed data file.')
    parser.add_argument('--model_dir', default='models',
                        help='path to store saved models.')
    parser.add_argument('--save_last_only', action='store_true',
                        help='only save the final models.')
    parser.add_argument('--save_dawn_logs', action='store_true',
                        help='append dawnbench log entries prefixed with dawn_entry:')
    parser.add_argument('--seed', type=int, default=1013,
                        help='random seed for data shuffling, dropout, etc.')
    parser.add_argument("--cuda", type=str2bool, nargs='?',
                        const=True,default=torch.cuda.is_available(),
                        help='whether to use GPU acceleration.')
    # training
    parser.add_argument('-e', '--epochs', type=int, default=40)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-rs', '--resume', default='best_model.pt',
                        help='previous model file name (in `model_dir`). '
                             'e.g. "checkpoint_epoch_11.pt"')
    parser.add_argument('-ro', '--resume_options', action='store_true',
                        help='use previous model options, ignore the cli and defaults.')
    parser.add_argument('-rlr', '--reduce_lr', type=float, default=0.,
                        help='reduce initial (resumed) learning rate by this factor.')
    parser.add_argument('-op', '--optimizer', default='adamax',
                        help='supported optimizer: adamax, sgd')
    parser.add_argument('-gc', '--grad_clipping', type=float, default=10)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1,
                        help='only applied to SGD.')
    parser.add_argument('-mm', '--momentum', type=float, default=0,
                        help='only applied to SGD.')
    parser.add_argument('-tp', '--tune_partial', type=int, default=1000,
                        help='finetune top-x embeddings.')
    parser.add_argument('--fix_embeddings', action='store_true',
                        help='if true, `tune_partial` will be ignored.')
    parser.add_argument('--rnn_padding', action='store_true',
                        help='perform rnn padding (much slower but more accurate).')
    # model
    parser.add_argument('--question_merge', default='self_attn')
    parser.add_argument('--doc_layers', type=int, default=3)
    parser.add_argument('--question_layers', type=int, default=3)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_features', type=int, default=4)
    parser.add_argument('--pos', type=str2bool, nargs='?', const=True, default=True,
                        help='use pos tags as a feature.')
    parser.add_argument('--ner', type=str2bool, nargs='?', const=True, default=True,
                        help='use named entity tags as a feature.')
    parser.add_argument('--use_qemb', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--concat_rnn_layers', type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument('--dropout_emb', type=float, default=0.4)
    parser.add_argument('--dropout_rnn', type=float, default=0.4)
    parser.add_argument('--dropout_rnn_output', type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument('--max_len', type=int, default=15)
    parser.add_argument('--rnn_type', default='lstm',
                        help='supported types: rnn, gru, lstm')

    args = parser.parse_args()
    return args




def get_setup_args():
    """Get arguments needed in setup.py."""
    parser = argparse.ArgumentParser(
        description='Preprocessing data files, about 10 minitues to run.'
    )
    
    add_setup_args(parser)
    args = parser.parse_args()
    
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG,
                        datefmt='%m/%d/%Y %I:%M:%S')
    log = logging.getLogger(__name__)
    log.info(vars(args))
    log.info('start data preparing...')
    
    return args, log

def get_train_args():
    """Get arguments needed in train.py"""
    parser = argparse.ArgumentParser(
        description='Train a Document Reader model.'
    )
    
    add_train_args(parser)
    args = parser.parse_args()
     # set model dir
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    args.model_dir = os.path.abspath(model_dir)

    if args.resume == 'best_model.pt' and not os.path.exists(os.path.join(args.model_dir, args.resume)):
        # means we're starting fresh
        args.resume = ''
        
    # set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(args.model_dir, 'log.txt'))
    fh.setLevel(logging.INFO)
    ch = ProgressHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    log.addHandler(fh)
    log.addHandler(ch)
    
    return args, log
    
    

def get_test_args():
    """Get arguments needed in test.py"""
    parser = argparse.ArgumentParser(description='Test a Document Reader model.')
    parser.add_argument('--model_file', default='models_check/best_model.pt', # models/best_model.pt
                    help='path to model file')
    
    parser.add_argument('--max_ans_len',
                        type=int,
                        default=15,
                        help='Maximum length of a predicted answer.')
    parser.add_argument('--save_dir',
                        type=str,
                        default='./save/',
                        help='Base directory for saving information.')
    parser.add_argument('--use_squad_v2',
                        type=lambda s: s.lower().startswith('t'),
                        default=True,
                        help='Whether to use SQuAD 2.0 (unanswerable) questions.')
    parser.add_argument('--test_eval_file',
                        type=str,
                        default='./data/test_eval.json')
    add_train_args(parser)
    args = parser.parse_args()
    
    
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG,
                        datefmt='%m/%d/%Y %I:%M:%S')
    log = logging.getLogger(__name__)
    return args, log 
