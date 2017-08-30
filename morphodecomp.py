# -*- coding: utf-8 -*-
__author__ = "Danilo S. Carvalho <danilo@jaist.ac.jp>"

import sys
import json
import re
import random
import numpy as np
from itertools import izip
from keras import backend as K

from data_access.load_morphodb import morphodb_load
from ml.encoder import encode_morphodb, encode_word, ENC_SIZE_CHAR
from ml.decoder import decode_word, confidence
from ml.word2morpho import Word2Morpho
from config.loader import load_config

model_cache = None


def decompose(word_list, config_path, model=None, cache=True):
    global model_cache
    config = load_config(config_path)
    word_size = config["model"]["ENC_SIZE_WORD"]
    ctx_win_size = config["model"]["ENC_SIZE_CTX_WIN"]

    if (model is None):
        if (cache and model_cache is not None):
            model = model_cache
        else:
            model = Word2Morpho(config)
            model.load(config["data"]["model_path"] % config["id"])
            model_cache = model

    enc_words = np.array([encode_word(word, word_size, ENC_SIZE_CHAR, ctx_win_size, reverse=False) for word in word_list], dtype=np.uint8)

    results = []
    for (word, enc_decomp) in izip(word_list, model.predict(enc_words)):
        unpadded_output = re.search(r"^{(?P<decomp>[^}]+)}.*", decode_word(enc_decomp))

        if (unpadded_output):
            decomp = unpadded_output.group("decomp")
            char_confidence = confidence(enc_decomp)[1:len(decomp) + 1]
            avg_confidence = float(np.mean(char_confidence))

            if (avg_confidence > config["hyperparam"]["confidence_thresh"]):
                results.append({"word": word, "decomp": decomp.split(), "confidence": avg_confidence})
            else:
                results.append({"word": word, "decomp": [word], "confidence": avg_confidence})
        else:
            results.append({"word": word, "decomp": [word], "confidence": 0.})

    return results


def train_model(config, excluded_words=set()):
    morphodb = morphodb_load(config["data"]["morphodb_path"], excluded_words=excluded_words)

    input_seqs, output_seqs, sample_weights = encode_morphodb(morphodb, config["model"]["ENC_SIZE_WORD"], ENC_SIZE_CHAR, 
                                                              config["model"]["ENC_SIZE_WORD_OUT"], ENC_SIZE_CHAR, 
                                                              config["model"]["ENC_SIZE_CTX_WIN"], reverse=config["model"]["REVERSE"]) 
    

    w2m = Word2Morpho(config)

    w2m.train(input_seqs, output_seqs, 25, batch_size=200, validation_split=0.005, sample_weight=sample_weights)
    w2m.save(config["data"]["model_path"] % config["id"])

    return w2m


def test_accuracy(config, model):
    word_size = config["model"]["ENC_SIZE_WORD"]
    ctx_win_size = config["model"]["ENC_SIZE_CTX_WIN"]
    test_vocab = config["data"]["test_vocab"]
    test_seqs = np.array([encode_word(word, word_size, ENC_SIZE_CHAR, ctx_win_size, reverse=config["model"]["REVERSE"]) for word in test_vocab], dtype=np.uint8)

    results = []
    for (word, enc_word) in izip(test_vocab, model.predict(test_seqs)):
        unpadded_output = re.search(r"^{(?P<decomp>[^}]+)}.*", decode_word(enc_word))

        if (unpadded_output):
            decomp = unpadded_output.group("decomp")
            char_confidence = confidence(enc_word)[1:len(decomp) + 1]
            avg_confidence = float(np.mean(char_confidence))
            #results.append({"word": word, "decomp": decomp, "char_confidence": char_confidence, "confidence": avg_confidence})
            results.append({"word": word, "decomp": decomp, "confidence": avg_confidence})

    print json.dumps(results, indent=2)


def test_decomp(config, model):
    word_size = config["model"]["ENC_SIZE_WORD"]
    ctx_win_size = config["model"]["ENC_SIZE_CTX_WIN"]
    test_vocab = []
    with open(config["data"]["vocab_path"]) as vocab_file:
        vocab = json.load(vocab_file)
        test_vocab = [term for term in vocab if (" " not in term and re.match(r"^[a-zA-Z]+", term) and len(term) < 40 and len(term) > 3 and not term.isupper())]

    #test_vocab = random.sample(test_vocab, 10000)
    test_seqs = np.array([encode_word(word, word_size, ENC_SIZE_CHAR, ctx_win_size, reverse=config["model"]["REVERSE"]) for word in test_vocab], dtype=np.uint8)

    results = []
    for (word, enc_word) in izip(test_vocab, model.predict(test_seqs)):
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


def main(argv):
    config = load_config(argv[1])
    ops = argv[2].split(",")
    model = None

    if ("train" in ops):
        model = train_model(config)

    if (model is None):
        model = Word2Morpho(config)
        model.load(config["data"]["model_path"] % config["id"])

    if ("test" in ops):
        test_accuracy(config, model)
    if ("test_full" in ops):
        test_decomp(config, model)




if __name__ == "__main__":
    main(sys.argv)
