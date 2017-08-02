# -*- coding: utf-8 -*-
__author__ = "Danilo S. Carvalho <danilo@jaist.ac.jp>"

import unittest
import numpy as np
import tensorflow as tf
from keras import backend as K

from data_access.load_morphodb import morphodb_load
from ml.encoder import encode_morphodb, encode_word
from ml.decoder import decode_word
from ml.word2morpho import Word2Morpho
from config import encoder as encconf

TEST_WORDS = ["unlockability", "busheater"]
MORPHODB_PATH = "../data/enmorphodb.json"

#config = tf.ConfigProto(intra_op_parallelism_threads=8, 
#                         inter_op_parallelism_threads=8, 
#                         allow_soft_placement=True, 
#                         device_count = {'CPU': 8})
#session = tf.Session(config=config)
#K.set_session(session)


class TestWord2Morpho(unittest.TestCase):
    def test_accuracy(self):
        morphodb = morphodb_load(MORPHODB_PATH)

        #for worddecomp in morphodb:
        #    print worddecomp

        #return

        input_seqs, output_seqs, sample_weights = encode_morphodb(morphodb, word_size_output=encconf.ENC_SIZE_WORD_OUT)

        w2m = Word2Morpho()
        w2m.train(input_seqs, output_seqs, 100, batch_size=100, validation_split=0.1, sample_weight=sample_weights)
        w2m.save("/home/danilo/tdv_family/w2m_model.hdf5")

        test_seqs = np.array([encode_word(word) for word in TEST_WORDS], dtype=np.float16)

        print [decode_word(enc_word) for enc_word in w2m.predict(test_seqs)]

