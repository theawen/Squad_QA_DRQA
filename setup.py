"""Download and pre-process SQuAD and GloVe.

Usage:
    > source activate squad
    > python setup.py

Pre-processing code adapted from:
    > https://github.com/HKUST-KnowComp/R-Net/blob/master/prepro.py

Author:
    Chris Chute (chute@stanford.edu)
"""

import numpy as np
import re
import os
import msgpack
import unicodedata
import collections
import spacy

import ujson as json
import urllib.request

from args import get_setup_args
from codecs import open
from functools import partial
from multiprocessing import Pool
from collections import Counter
from subprocess import run
from tqdm import tqdm
from util import str2bool
from zipfile import ZipFile

def download_url(url, output_path, show_progress=True):
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    if show_progress:
        # Download with a progress bar
        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url,
                                       filename=output_path,
                                       reporthook=t.update_to)
    else:
        # Simple download with no progress bar
        urllib.request.urlretrieve(url, output_path)

def url_to_data_path(url):
    return os.path.join('./data/', url.split('/')[-1])

def download(args):
    downloads = [
        # Can add other downloads here (e.g., other word vectors)
        ('GloVe word vectors', args.glove_url),
    ]

    for name, url in downloads:
        output_path = url_to_data_path(url)
        if not os.path.exists(output_path):
            print(f'Downloading {name}...')
            download_url(url, output_path)

        if os.path.exists(output_path) and output_path.endswith('.zip'):
            extracted_path = output_path.replace('.zip', '')
            if not os.path.exists(extracted_path):
                print(f'Unzipping {name}...')
                with ZipFile(output_path, 'r') as zip_fh:
                    zip_fh.extractall(extracted_path)

    print('Downloading spacy language model...')
    run(['python', '-m', 'spacy', 'download', 'en'])
    

def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print(f"Token {token} cannot be found")
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans

def normalize_text(text):
    return unicodedata.normalize('NFD', text)

def build_vocab(questions, contexts, wv_vocab):
    """
    Build vocabulary sorted by global word frequency, or consider frequencies in questions first,
    which is controlled by `args.sort_all`.
    """

    counter_q = collections.Counter(w for doc in questions for w in doc)
    counter_c = collections.Counter(w for doc in contexts for w in doc)
    counter = counter_c + counter_q
    vocab = sorted([t for t in counter_q if t in wv_vocab], key=counter_q.get, reverse=True)
    vocab += sorted([t for t in counter_c.keys() - counter_q.keys() if t in wv_vocab],
                    key=counter.get, reverse=True)
    vocab.insert(0, "<PAD>")
    vocab.insert(1, "<UNK>")
    return vocab, counter

def clean_spaces(text):
    """normalize spaces in a string."""
    text = re.sub(r'\s', ' ', text)
    return text

def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]

def process_file(filename, data_type):
    print(f"Pre-processing {data_type} examples...")
    rows = []
    eval_examples = {}
    total = 0
    with open(filename, "r") as fh:
        source = json.load(fh)
        for article in tqdm(source["data"]):
            for para in article["paragraphs"]:
                context = para["context"].replace("''", '" ').replace("``", '" ')
                context_tokens = word_tokenize(context)
                spans = convert_idx(context, context_tokens)
                
                for qa in para["qas"]:
                    total += 1
                    question = qa["question"].replace("''", '" ').replace("``", '" ')
                    answer_s, answer_e = [], []
                    answers = []
                    for answer in qa["answers"]:
                        answer_text = answer["text"]
                        answer_start = answer['answer_start']
                        answer_end = answer_start + len(answer_text)
                        answers.append(answer_text)
                        answer_s.append(answer_start)
                        answer_e.append(answer_end)
                    
                    rows.append((total, context, question, answers, answer_s, answer_e))
                    eval_examples[str(total)] = {"context": context,
                                                 "question": question,
                                                 "spans": spans,
                                                 "answers": answers,
                                                 "uuid": qa["id"]}
        print(f"{len(rows)} questions in total")
    return rows, eval_examples

def annotate(row, wv_cased):
    global nlp
    id_, context, question = row[:3]
    q_doc = nlp(clean_spaces(question))
    c_doc = nlp(clean_spaces(context))
    question_tokens = [normalize_text(w.text) for w in q_doc]
    context_tokens = [normalize_text(w.text) for w in c_doc]
    question_tokens_lower = [w.lower() for w in question_tokens]
    context_tokens_lower = [w.lower() for w in context_tokens]
    context_token_span = [(w.idx, w.idx + len(w.text)) for w in c_doc]
    context_tags = [w.tag_ for w in c_doc]
    context_ents = [w.ent_type_ for w in c_doc]
    question_lemma = {w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower() for w in q_doc}
    question_tokens_set = set(question_tokens)
    question_tokens_lower_set = set(question_tokens_lower)
    match_origin = [w in question_tokens_set for w in context_tokens]
    match_lower = [w in question_tokens_lower_set for w in context_tokens_lower]
    match_lemma = [(w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower()) in question_lemma for w in c_doc]
    # term frequency in document
    counter_ = collections.Counter(context_tokens_lower)
    total = len(context_tokens_lower)
    context_tf = [counter_[w] / total for w in context_tokens_lower]
    context_features = list(zip(match_origin, match_lower, match_lemma, context_tf))
    if not wv_cased:
        context_tokens = context_tokens_lower
        question_tokens = question_tokens_lower
    return (id_, context_tokens, context_features, context_tags, context_ents,
            question_tokens, context, context_token_span) + row[3:]


def index_answer(row):
    token_span = row[-4]
    starts, ends = zip(*token_span)
    answer_start = row[-2]
    answer_end = row[-1]
    if len(answer_start) >= 1:
        try:
            return row[:-3] + (starts.index(answer_start[-1]), ends.index(answer_end[-1]))
        except:
            return row[:-3] + (None, None)
    else:
        return row[:-3] + (None, None)

def save(filename, obj, message=None):
    if message is not None:
        print(f"Saving {message}...")
        with open(filename, "w") as fh:
            json.dump(obj, fh)         

def to_id(row, w2id, tag2id, ent2id, unk_id=1):
    context_tokens = row[1]
    context_features = row[2]
    context_tags = row[3]
    context_ents = row[4]
    question_tokens = row[5]
    question_ids = [w2id[w] if w in w2id else unk_id for w in question_tokens]
    context_ids = [w2id[w] if w in w2id else unk_id for w in context_tokens]
    tag_ids = [tag2id[w] for w in context_tags]
    ent_ids = [ent2id[w] for w in context_ents]
    return (row[0], context_ids, context_features, tag_ids, ent_ids, question_ids) + row[6:]            
            
def pre_process(args):
    # Process train, dev, test dataset
    train, train_eval = process_file(args.train_file, "train")
    dev, dev_eval = process_file(args.dev_file, "dev")
    test, test_eval = process_file(args.test_file, "test")
    
    with Pool(args.threads, initializer=init) as p:
        annotate_ = partial(annotate, wv_cased=args.wv_cased)
        train = list(tqdm(p.imap(annotate_, train, chunksize=args.batch_size), total=len(train), desc='train'))
        dev = list(tqdm(p.imap(annotate_, dev, chunksize=args.batch_size), total=len(dev), desc='dev'))
        test = list(tqdm(p.imap(annotate_, test, chunksize=args.batch_size), total=len(dev), desc='dev'))
    
    train = list(map(index_answer, train))
    dev = list(map(index_answer, dev))
    initial_len_train = len(train)
    initial_len_dev = len(dev)
    train = list(filter(lambda x: x[-1] is not None, train))
    dev = list(filter(lambda x: x[-1] is not None, dev))
    log.info('drop {} inconsistent train samples.'.format(initial_len_train - len(train)))
    log.info('drop {} inconsistent dev samples.'.format(initial_len_dev - len(dev)))
    log.info('tokens generated')
    
    # load vocabulary from word vector files
    wv_vocab = set()
    with open(args.wv_file) as f:
        for line in f:
            token = normalize_text(line.rstrip().split(' ')[0])
            wv_vocab.add(token)
    log.info('glove vocab loaded.')
    
    # build vocabulary
    full = train+test+dev
    vocab, counter = build_vocab([row[5] for row in full], [row[1] for row in full], wv_vocab)
    total = sum(counter.values())
    matched = sum(counter[t] for t in vocab)
    log.info('vocab coverage {1}/{0} | OOV occurrence {2}/{3} ({4:.4f}%)'.format(
        len(counter), len(vocab), (total - matched), total, (total - matched) / total * 100))
    counter_tag = collections.Counter(w for row in full for w in row[3])
    vocab_tag = sorted(counter_tag, key=counter_tag.get, reverse=True)
    counter_ent = collections.Counter(w for row in full for w in row[4])
    vocab_ent = sorted(counter_ent, key=counter_ent.get, reverse=True)
    w2id = {w: i for i, w in enumerate(vocab)}
    tag2id = {w: i for i, w in enumerate(vocab_tag)}
    ent2id = {w: i for i, w in enumerate(vocab_ent)}
    log.info('Vocabulary size: {}'.format(len(vocab)))
    log.info('Found {} POS tags.'.format(len(vocab_tag)))
    log.info('Found {} entity tags: {}'.format(len(vocab_ent), vocab_ent))
    
    to_id_ = partial(to_id, w2id=w2id, tag2id=tag2id, ent2id=ent2id)
    train = list(map(to_id_, train))
    dev = list(map(to_id_, dev))
    test = list(map(to_id_, test))
    log.info('converted to ids.')
    
    
    # get embeddings
    vocab_size = len(vocab)
    embeddings = np.zeros((vocab_size, args.wv_dim))
    embed_counts = np.zeros(vocab_size)
    embed_counts[:2] = 1  # PADDING & UNK
    with open(args.wv_file) as f:
        for line in f:
            elems = line.rstrip().split(' ')
            token = normalize_text(elems[0])
            if token in w2id:
                word_id = w2id[token]
                embed_counts[word_id] += 1
                embeddings[word_id] += [float(v) for v in elems[1:]]
    embeddings /= embed_counts.reshape((-1, 1))
    log.info('got embedding matrix.')
    
    # save meta data
    meta = {
        'vocab': vocab,
        'vocab_tag': vocab_tag,
        'vocab_ent': vocab_ent,
        'embedding': embeddings.tolist(),
        'wv_cased': args.wv_cased,
    }
    with open('data/meta.msgpack', 'wb') as f:
        msgpack.dump(meta, f)
    
    # data: id, context_id, context_features, tag_id, ent_id,
    #        question_id, context, context_token_span, answer_start, answer_end
    result = {
        'train': train,
        'dev': dev,
        'test':test
    }
    with open('data/data.msgpack', 'wb') as f:
        msgpack.dump(result, f)
    log.info('Data saved to disk')
    
    # save eval files
    save(args.train_eval_file, train_eval, message="train eval")
    save(args.test_eval_file, test_eval, message="test eval")
    save(args.dev_eval_file, dev_eval, message="dev eval")

def init():
    """initialize spacy in each process"""
    global nlp
    nlp = spacy.load('en', parser=False)    

if __name__ == '__main__':
    # Get command-line args
    args, log = get_setup_args()

    # Download resources
    # download(args)

    # Import spacy language model
    nlp = spacy.load("en", parser = False)

    # Preprocess dataset
    args.train_file = url_to_data_path(args.train_url)
    args.dev_file = url_to_data_path(args.dev_url)
    args.test_file = url_to_data_path(args.test_url)
    glove_dir = url_to_data_path(args.glove_url.replace('.zip', ''))
    glove_ext = f'.txt' if glove_dir.endswith('d') else f'.{args.glove_dim}d.txt'
    args.glove_file = os.path.join(glove_dir, os.path.basename(glove_dir) + glove_ext)
    pre_process(args)
