__author__ = "Danilo S. Carvalho <danilo@jaist.ac.jp>"

import numpy as np

from ml import encoder as enc

CODE_CHARS = dict([(v,k) for (k,v) in enc.CHAR_CODES.items()])


def decode_char(enc_char):
    char_code = np.argmax(enc_char)
    return CODE_CHARS[char_code]


def decode_word(enc_word):
    return u"".join([decode_char(enc_char) for enc_char in enc_word])


def confidence(enc_word):
    return [float(np.nanmax(enc_char)) for enc_char in enc_word]

