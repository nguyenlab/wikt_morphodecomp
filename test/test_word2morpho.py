# -*- coding: utf-8 -*-
__author__ = "Danilo S. Carvalho <danilo@jaist.ac.jp>"

import unittest
import numpy as np

from data_access.load_morphodb import morphodb_load
from ml.encoder import encode_morphodb, encode_word
from ml.word2morpho import Word2Morpho
from config import encoder as encconf

TEST_WORDS = ["unlockability", "busheater"]
MORPHODB_PATH = "/home/danilo/tdv_family/wikt_morphodecomp/data/enmorphodb.json"


class TestWord2Morpho(unittest.TestCase):
    def test_accuracy(self):
        morphodb = morphodb_load(MORPHODB_PATH)

        input_seqs, output_seqs = encode_morphodb(morphodb, word_size_output=encconf.ENC_SIZE_WORD + 10)

        w2m = Word2Morpho()
        w2m.train(input_seqs[0:1000], output_seqs[0:1000], 1, batch_size=1, validation_split=0.0)
        w2m.save("/home/danilo/tdv_family/w2m_model.hdf5")

        test_seqs = np.array([encode_word(word) for word in TEST_WORDS])

        print w2m.predict(test_seqs)
