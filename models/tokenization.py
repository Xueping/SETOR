from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import logging
import pickle
import numpy as np
from pytorch_pretrained_bert import BertTokenizer as origin_tokenizer

logger = logging.getLogger(__name__)


def load_vocab_seqs(vocab_file):
    """Loads a vocabulary file into a dictionary."""

    special_words = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '[SPAN1M]', '[SPAN3M]',
                     '[SPAN6M]', '[SPAN12M]', '[SPAN12M+]']
    vocab = collections.OrderedDict()
    for word in special_words:
        vocab[word] = len(vocab)
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = len(vocab)
    return vocab


def load_vocab_ent(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    vocab['[UNK]'] = 0

    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            line = token.strip().split(',')
            vocab[line[1]] = int(line[0])+1
    return vocab


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class DescTokenizer(object):
    def __init__(self, code2desc_file):
        self.code2desc = pickle.load(open(code2desc_file, 'rb'))
        # self.code2desc['[UNK]'] = '[UNK]'
        # self.code2desc['[MASK]'] = '[MASK]'
        self.code2indexed_tokens = {}
        self.decs_tokenize()

    def decs_tokenize(self):
        tokenizer = origin_tokenizer.from_pretrained('bert-base-uncased')
        # tokenize the description of each code
        max_len = 0
        for code, desc in self.code2desc.items():
            tokenized_text = tokenizer.tokenize(desc)
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            self.code2indexed_tokens[code] = indexed_tokens
            len_tokens = len(indexed_tokens)
            if len_tokens > max_len:
                max_len = len_tokens
        # padding the indexed tokens
        for id, inds in self.code2indexed_tokens.items():
            ln = len(inds)
            dif = max_len - ln
            if dif == 0:
                continue
            else:
                self.code2indexed_tokens[id] = np.array(inds + [0] * dif, dtype=np.long)

    def tokenize(self, codes):
        split_tokens = []
        split_vectors = []
        for code in codes:
            token_vec = self.code2indexed_tokens[code]
            split_tokens.append(code)
            split_vectors.append(token_vec)
        return split_tokens, np.stack(split_vectors)


class EntityTokenizer(object):
    def __init__(self, ent_vocab_file):
        self.vocab = load_vocab_ent(ent_vocab_file)

    def tokenize(self, text):
        split_tokens = []
        ids = []
        for token in whitespace_tokenize(text):
            split_tokens.append(token)
            ids.append(self.vocab[token])
        return split_tokens, np.stack(ids)


class SeqsTokenizer(object):
    def __init__(self, vocab_file):
        self.vocab = load_vocab_seqs(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])

    def tokenize(self, text):
        split_tokens = []
        ids = []
        for token in whitespace_tokenize(text):
            split_tokens.append(token)
            ids.append(self.vocab[token])
        return split_tokens, np.stack(ids)

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        return np.stack(ids)

    def convert_ids_to_tokens(self, ids):
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens


