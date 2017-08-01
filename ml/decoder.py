__author__ = "Danilo S. Carvalho <danilo@jaist.ac.jp>"

import numpy as np


def decode_char(enc_char):
    return unichr(np.argmax(enc_char))


def decode_word(enc_word):
    return u"".join([decode_char(enc_char) for enc_char in enc_word])
