# -*- coding: utf-8 -*-
__author__ = "Danilo S. Carvalho <danilo@jaist.ac.jp>"

import unittest
import json
import re
import random
import numpy as np
import tensorflow as tf
from itertools import izip
from keras import backend as K

from data_access.load_morphodb import morphodb_load
from ml.encoder import encode_morphodb, encode_word
from ml.decoder import decode_word, confidence
from ml.word2morpho import Word2Morpho
from config import encoder as encconf
from config import word2morpho as w2mconf

TEST_WORDS = ["unlockability", "busheater", "drivethru", "cryomancy"]
MORPHODB_PATH = "../data/enmorphodb.json"
PNOUNDB_PATH = "../data/enpnoundb.json"
VOCAB_PATH = "../data/wikt_vocab.json"

#config = tf.ConfigProto(intra_op_parallelism_threads=8, 
#                         inter_op_parallelism_threads=8, 
#                         allow_soft_placement=True, 
#                         device_count = {'CPU': 8})
#session = tf.Session(config=config)
#K.set_session(session)


class TestWord2Morpho(unittest.TestCase):
    def test_accuracy(self):
        return
        morphodb = morphodb_load(MORPHODB_PATH)
        #pnoundb = morphodb_load(PNOUNDB_PATH)

        input_seqs, output_seqs, sample_weights = encode_morphodb(morphodb, word_size_output=encconf.ENC_SIZE_WORD_OUT, reverse=False) 
        
        #input_morpho_seqs, output_morpho_seqs, sample_morpho_weights = encode_morphodb(morphodb, word_size_output=encconf.ENC_SIZE_WORD_OUT) 
        #input_pnoun_seqs, output_pnoun_seqs, sample_pnoun_weights = encode_morphodb(pnoundb, word_size_output=encconf.ENC_SIZE_WORD_OUT)
        #input_seqs = np.append(input_pnoun_seqs, input_morpho_seqs, axis=0)
        #output_seqs = np.append(output_pnoun_seqs, output_morpho_seqs, axis=0)
        #sample_weights = np.append(sample_pnoun_weights, sample_morpho_weights, axis=0)

        w2m = Word2Morpho()

        w2m.train(input_seqs, output_seqs, 25, batch_size=200, validation_split=0.01, sample_weight=sample_weights)
        w2m.save("/home/danilo/tdv_family/w2m_model_%d-%d-%d_rms.hdf5" % (w2mconf.HIDDEN_DIM_ENC_1, w2mconf.HIDDEN_DIM_DEC_1, w2mconf.HIDDEN_DIM_DEC_2))
        
        test_vocab = TEST_WORDS
        test_seqs = np.array([encode_word(word, reverse=False) for word in test_vocab], dtype=np.uint8)

        results = []
        for (word, enc_word) in izip(test_vocab, w2m.predict(test_seqs)):
            unpadded_output = re.search(r"^{(?P<decomp>[^}]+)}.*", decode_word(enc_word))

            if (unpadded_output):
                decomp = unpadded_output.group("decomp")
                char_confidence = confidence(enc_word)[1:len(decomp) + 1]
                avg_confidence = float(np.mean(char_confidence))
                results.append({"word": word, "decomp": decomp, "char_confidence": char_confidence, "confidence": avg_confidence})

        print json.dumps(results, indent=2)


    def test_decomp(self):
        return
        w2m = Word2Morpho()
        w2m.load("/home/danilo/tdv_family/w2m_model_%d-%d-%d_rms.hdf5" % (w2mconf.HIDDEN_DIM_ENC_1, w2mconf.HIDDEN_DIM_DEC_1, w2mconf.HIDDEN_DIM_DEC_2))

        test_vocab = []
        with open(VOCAB_PATH) as vocab_file:
            vocab = json.load(vocab_file)
            test_vocab = [term for term in vocab if (" " not in term and re.match(r"^[a-zA-Z]+", term) and len(term) < 40 and len(term) > 3 and not term.isupper())]

        #test_vocab = random.sample(test_vocab, 10000)
        test_seqs = np.array([encode_word(word, reverse=False) for word in test_vocab], dtype=np.uint8)

        results = []
        for (word, enc_word) in izip(test_vocab, w2m.predict(test_seqs)):
            unpadded_output = re.search(r"^{(?P<decomp>[^}]+)}.*", decode_word(enc_word))

            if (unpadded_output):
                decomp = unpadded_output.group("decomp")
                char_confidence = confidence(enc_word)[1:len(decomp) + 1]
                avg_confidence = float(np.mean(char_confidence))

                if (avg_confidence < 0.5):
                    results.append({"word": word, "decomp": decomp}) #, "char_confidence": char_confidence, "confidence": avg_confidence})

        print "Lower 50%%: %d/%d" % (len(results), len(test_vocab))
        print json.dumps(results, indent=2)
        
        return

