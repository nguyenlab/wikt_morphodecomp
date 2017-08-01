__author__ = "Danilo S. Carvalho <danilo@jaist.ac.jp>"

import numpy as np
from config import encoder as conf


def encode_char(c):
    enc = np.zeros(conf.ENC_SIZE_CHAR)
    c_ord = ord(c)

    if (c_ord < conf.ENC_SIZE_CHAR):
        enc[c_ord] = 1
    else:
        enc[1] = 1

    return enc


def encode_word(w):
    enc = np.zeros((conf.ENC_SIZE_WORD, conf.ENC_SIZE_CHAR))

    for i in xrange(w):
        enc[i] = encode_char(w[i])

    return enc


def encode_morphodb(morphodb):
    input_seqs = []
    output_seqs = []
    for (word, decomp) in morphodb:
        input_seqs.append(encode_word(word))
        output_seqs.append(encode_word(decomp))

    return (np.array(input_seqs), np.array(output_seqs))
