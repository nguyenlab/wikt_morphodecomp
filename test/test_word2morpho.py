# -*- coding: utf-8 -*-
__author__ = "Danilo S. Carvalho <danilo@jaist.ac.jp>"

import unittest

from data_access.load_morphodb import morphodb_load
from ml.encoder import encode_morphodb
from ml.word2morpho import Word2Morpho

TEST_WORDS = []
MORPHODB_PATH = ""


class TestWord2Morpho(unittest.TestCase):
    def __init__(self):
        self.input_seqs, self.output_seqs = encode_morphodb(morphodb_load(MORPHODB_PATH))

    def test_accuracy(self):
        w2m = Word2Morpho()
        w2m.train(self.input_seqs, self.output_seqs, 10, batch_size=1, validation_split=0.1)