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
import os
import spacy
import ujson as json
import urllib.request

from args1 import get_setup_args
from codecs import open
from collections import Counter
from subprocess import run
from tqdm import tqdm
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

def process_file(filename, data_type):
    print(f"Pre-processing {data_type} examples...")
    examples = []
    eval_examples = {}
    total = 0
    with open(filename, "r") as fh:
        source = json.load(fh)
        for article in tqdm(source["data"]):
            for para in article["paragraphs"]:
                context = para["context"].replace("''", '" ').replace("``", '" ')
                c_doc = nlp(context)
                context_tokens = [normalize(w.text) for w in c_doc]
                context_token_span = [(w.idx, w.idx + len(w.text)) for w in c_doc]
                context_tokens_lower = [w.lower() for w in context_tokens]
                context_tags = [w.tag_ for w in c_doc]
                context_ents = [w.ent_type_ for w in c_doc]
                length = len(context_tokens_lower)
                counter_ = collections.Counter(context_tokens_lower)
                spans = convert_idx(context, context_tokens)
                
                for qa in para["qas"]:
                    total += 1
                    question = qa["question"].replace("''", '" ').replace("``", '" ')
                    q_doc = nlp(quesition)
                    question_tokens = [normalize_text(w.text) for w in q_doc]
                    question_tokens_set = set(question_tokens)
                    match_origin = [w in question_tokens_set for w in context_tokens]
                    question_lemma = {w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower() for w in q_doc}
                    match_lemma = [(w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower()) in question_lemma for w in c_doc]
                    context_features = list(zip(match_origin, match_lower, match_lemma, context_tf))
                    
                    answer_s, answer_e = [], []
                    answer = []
                    for answer in qa["answers"]:
                        answer_text = answer["text"]
                        answer_start = answer['answer_start']
                        answer_end = answer_start + len(answer_text)
                        answer.append(answer_text)
                        answer_s.append(answer_start)
                        answer_e.append(answer_end)
                        
                    example = {"context_tokens": context_tokens,
                               "context_features": context_features,
                               "context_tags":context_tags,
                               "context_ents":context_ents,
                               "question_tokens": question_tokens,
                               "context": context,
                               "question":question,
                               "context_token_span":context_token_span,
                               "answer":answer, 
                               "answer_start": answer_s,
                               "answer_end": answer_e}
                    examples.append(example)
                    eval_examples[str(total)] = {"context": context,
                                                 "question": quesition,
                                                 "spans": spans,
                                                 "answers": answer,
                                                 "uuid": qa["id"]}
        print(f"{len(examples)} questions in total")
    return examples, eval_examples


def save(filename, obj, message=None):
    if message is not None:
        print(f"Saving {message}...")
        with open(filename, "w") as fh:
            json.dump(obj, fh)
            
def pre_process(args):
    # Process train, dev, test dataset
    train_examples, train_eval = process_file(args.train_file, "train")
    dev_examples, dev_eval = process_file(args.dev_file, "dev")
    test_examples, test_eval = process_file(args.test_file, "test")
    
    # load vocabulary from word vector files
    wv_vocab = set()
    with open(args.wv_file) as f:
        for line in f:
            token = normalize_text(line.rstrip().split(' ')[0])
            wv_vocab.add(token)
    log.info('glove vocab loaded.')
    
    # build vocabulary
    full = train_examples+test_examples+dev_examples
    context = [rows['context'] for rows in full]
    question = [rows['question'] for rows in full]
    vocab, counter = build_vocab(question, context, wv_vocab)
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

    
    
    
    # Process dev and test sets
    
    build_features(args, train_examples, "train", args.train_record_file, word2idx_dict, char2idx_dict)
    dev_meta = build_features(args, dev_examples, "dev", args.dev_record_file, word2idx_dict, char2idx_dict)
    if args.include_test_examples:
        test_examples, test_eval = process_file(args.test_file, "test", word_counter, char_counter)
        save(args.test_eval_file, test_eval, message="test eval")
        test_meta = build_features(args, test_examples, "test",
                                   args.test_record_file, word2idx_dict, char2idx_dict, is_test=True)
        save(args.test_meta_file, test_meta, message="test meta")

    save(args.train_eval_file, train_eval, message="train eval")
    save(args.dev_eval_file, dev_eval, message="dev eval")
    save(args.dev_meta_file, dev_meta, message="dev meta")

if __name__ == '__main__':
    # Get command-line args
    args_ = get_setup_args()

    # Download resources
    download(args_)

    # Import spacy language model
    nlp = spacy.blank("en")

    # Preprocess dataset
    args_.train_file = url_to_data_path(args_.train_url)
    args_.dev_file = url_to_data_path(args_.dev_url)
    if args_.include_test_examples:
        args_.test_file = url_to_data_path(args_.test_url)
    glove_dir = url_to_data_path(args_.glove_url.replace('.zip', ''))
    glove_ext = f'.txt' if glove_dir.endswith('d') else f'.{args_.glove_dim}d.txt'
    args_.glove_file = os.path.join(glove_dir, os.path.basename(glove_dir) + glove_ext)
    pre_process(args_)
