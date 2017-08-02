#-*- coding: utf8 -*-
__author__ = "Danilo S. Carvalho <danilo@jaist.ac.jp>"

import numpy as np
from config import encoder as conf


def encode_char(c, char_size=conf.ENC_SIZE_CHAR):
    enc = np.zeros(char_size, dtype=np.float16)
    c_ord = ord(c)

    if (c_ord < char_size):
        enc[c_ord] = 1
    else:
        enc[1] = 1

    return enc


def encode_word(w, word_size=conf.ENC_SIZE_WORD, char_size=conf.ENC_SIZE_CHAR):
    enc = np.zeros((word_size, char_size), dtype=np.float16)
    delim_word = u"{" + w + u"}"

    for i in xrange(len(delim_word)):
        enc[i] = encode_char(delim_word[i])

    return enc


def encode_sample_weights(w, word_size=conf.ENC_SIZE_WORD, char_size=conf.ENC_SIZE_CHAR):
    enc = np.zeros(word_size, dtype=np.float16)
    delim_word = u"{" + w + u"}"

    for i in xrange(len(delim_word)):
        enc[i] = 1.

    return enc



def encode_morphodb(morphodb, word_size_input=conf.ENC_SIZE_WORD, char_size_input=conf.ENC_SIZE_CHAR, 
                    word_size_output=conf.ENC_SIZE_WORD, char_size_output=conf.ENC_SIZE_CHAR):
    input_seqs = []
    output_seqs = []
    sample_weights = []

    for (word, morphemes) in morphodb: 
        input_seqs.append(encode_word(word, word_size_input, char_size_input))
        output_seqs.append(encode_word(morphemes, word_size_output, char_size_output))
        sample_weights.append(encode_sample_weights(morphemes, word_size_output, char_size_output))

    return (np.array(input_seqs, dtype=np.float16), np.array(output_seqs, dtype=np.float16), np.array(sample_weights, dtype=np.float16))
