# -*- coding: utf-8 -*-
# /usr/bin/python2

__author__ = "Luo Yifan"
__version__ = "0.1"

from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import json
import os

def create_vocab():
    count = 0
    action_set = list()
    with open(hp.logdesignid_freq_file, "r") as load_f:
        for line in load_f:
            action_set.append(line.split(':')[0])
            count += 1
            if count >= hp.vocab_size:
                break

    special_words = ['<PAD>', '<EOS>', '<GO>', '<UNK>']
    id2action = {word_i: word for word_i, word in
                                     enumerate(special_words + list(action_set))}
    action2id = {word: word_i for word_i, word in id2action.items()}
    return id2action,action2id

def pad_sentence_batch(sentence_batch, pad_int, pad_eos, pad_go, codertype='encoder'):
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    if max_sentence <= hp.maxlen:
        to_return = [sentence+[pad_eos]+[pad_int]*(hp.maxlen - len(sentence)) if codertype=='encoder'
                     else [pad_go]+sentence+[pad_int]*(hp.maxlen - len(sentence)) for sentence in sentence_batch]
    else:
        max_sentence = hp.maxlen
        to_return = [sentence[:hp.maxlen]+[pad_eos]+[pad_int]*(max_sentence-len(sentence)) if codertype=='encoder'
                     else [pad_go]+sentence[:hp.maxlen]+[pad_int]*(max_sentence - len(sentence)) for sentence in sentence_batch]
    return to_return
