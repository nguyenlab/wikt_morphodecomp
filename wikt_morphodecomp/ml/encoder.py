#-*- coding: utf8 -*-
__author__ = "Danilo S. Carvalho <danilo@jaist.ac.jp>"

import string
import numpy as np

CHAR_CODES = dict([(unicode(c), idx) for (idx, c) in enumerate(list(string.ascii_lowercase) + ['\'', '-', '{', '}', ' ', chr(1)])])
ENC_SIZE_CHAR = len(CHAR_CODES)

def encode_char(c, char_size=ENC_SIZE_CHAR):
    enc = np.zeros(char_size, dtype=np.uint8) 

    if (c.lower() in CHAR_CODES):
        c_code = CHAR_CODES[c.lower()]
        enc[c_code] = 1
    else:
        enc[CHAR_CODES[unichr(1)]] = 1

    return enc


def encode_word(w, word_size, char_size, ctx_win_size, reverse=False):
    enc = np.zeros((word_size, char_size * ctx_win_size), dtype=np.uint8)

    assert (ctx_win_size % 2) == 1
    assert ctx_win_size >= 1

    w = w[0:word_size - 2]

    if (not reverse):
        charseq = list("{"+ w + "}")
    else:
        charseq = list(u"{"+ w[::-1] + u"}")

    lpadded = ctx_win_size // 2 * [u"{"] + charseq + ctx_win_size // 2 * [u"}"]
    context_windows = [lpadded[i:(i + ctx_win_size)] for i in range(len(charseq))]

    for i in xrange(len(context_windows)):
        enc[i] = np.concatenate([encode_char(c) for c in context_windows[i]])

    return enc


def encode_sample_weights(w, word_size):
    enc = np.zeros(word_size, dtype=np.uint8)
    delim_word = u"{" + w + u"}"

    for i in xrange(len(delim_word)):
        enc[i] = 1.

    return enc



def encode_morphodb(morphodb, word_size_input, char_size_input, word_size_output, char_size_output, ctx_win_size, reverse=False):
    input_seqs = []
    output_seqs = []
    sample_weights = []

    for (word, morphemes) in morphodb: 
        input_seqs.append(encode_word(word, word_size_input, char_size_input, ctx_win_size=ctx_win_size, reverse=reverse))
        output_seqs.append(encode_word(morphemes, word_size_output, char_size_output, ctx_win_size=1))
        sample_weights.append(encode_sample_weights(morphemes, word_size_output))

    return (np.array(input_seqs, dtype=np.uint8), np.array(output_seqs, dtype=np.uint8), np.array(sample_weights, dtype=np.uint8))
