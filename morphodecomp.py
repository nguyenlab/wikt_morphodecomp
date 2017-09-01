# -*- coding: utf-8 -*-
__author__ = "Danilo S. Carvalho <danilo@jaist.ac.jp>"

import sys
import json
import re
import numpy as np
from itertools import izip
from collections import Counter
from joblib import Parallel, delayed

from data_access.load_morphodb import morphodb_load
from ml.encoder import encode_morphodb, encode_word, ENC_SIZE_CHAR
from ml.decoder import decode_word, confidence
from ml.word2morpho import Word2Morpho
from config.loader import load_config

model_cache = None


class EnsembleMode:
    MAJORITY_OVERALL = 1
    MAJORITY_CHAR = 2
    CONFIDENCE_OVERALL = 3
    CONFIDENCE_CHAR = 4


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
            model.load(config["data"]["model_path"] % (config["id"], 0))
            model_cache = model

    enc_words = np.array([encode_word(word, word_size, ENC_SIZE_CHAR, ctx_win_size, reverse=config["model"]["REVERSE"])
                          for word in word_list], dtype=np.uint8)

    results = []
    for (word, enc_decomp) in izip(word_list, model.predict(enc_words)):
        unpadded_output = re.search(r"^{(?P<decomp>[^}]+)}.*", decode_word(enc_decomp))

        if (unpadded_output):
            decomp = unpadded_output.group("decomp")
            char_confidence = confidence(enc_decomp)[1:len(decomp) + 1]
            avg_confidence = float(np.mean(char_confidence))

            if (avg_confidence > config["hyperparam"]["confidence_thresh"]):
                results.append({"word": word,
                                "decomp": decomp.split(),
                                "confidence": avg_confidence,
                                "char_confidence": char_confidence})
            else:
                results.append({"word": word,
                                "decomp": [word],
                                "confidence": avg_confidence,
                                "char_confidence": char_confidence})
        else:
            results.append({"word": word,
                            "decomp": [word],
                            "confidence": 0.,
                            "char_confidence": [0.] * len(word)})

    return results


def decompose_ensemble(word_list, config_paths, models=[], mode=EnsembleMode.MAJORITY_CHAR, num_procs=4):
    assert len(config_paths) == len(models)
    results = []

    #ensembled_results = Parallel(n_jobs=num_procs)(delayed(decompose)(word_list, config_path, model=model, cache=False)
    #                           for (config_path, model) in izip(config_paths, models))

    ensembled_results = []
    for (config_path, model) in izip(config_paths, models):
        ensembled_results.append(decompose(word_list, config_path, model=model, cache=False))
        

    for word_results in izip(*ensembled_results):
        if (mode == EnsembleMode.MAJORITY_OVERALL):
            votes = Counter([tuple(wres["decomp"]) for wres in word_results])
            selected = votes.most_common(1)[0][0]
            majority_confidence = [wres["confidence"] for wres in word_results if (tuple(wres["decomp"]) == selected)]

            results.append({"word": word_results[0]["word"],
                            "decomp": list(selected),
                            "confidence": float(sum(majority_confidence)) / len(majority_confidence),
                            "votes": votes})

        elif (mode == EnsembleMode.CONFIDENCE_OVERALL):
            results.append(max(word_results, key=lambda wr: wr["confidence"]))

        elif (mode == EnsembleMode.MAJORITY_CHAR):
            maxlen_wres = max(word_results, key=lambda wr: len(" ".join(wr["decomp"])))
            plain_decomp = " ".join(maxlen_wres["decomp"])
            result = {"word": word_results[0]["word"], "decomp": [], "confidence": 0., "char_confidence": [], "votes": []}

            for i in range(len(plain_decomp)):
                votes = Counter([" ".join(wres["decomp"])[i] if (i < len(" ".join(wres["decomp"]))) else " "
                                 for wres in word_results])
                selected = votes.most_common(1)[0][0]
                majority_confidence = [wres["char_confidence"][i] if (i < len(" ".join(wres["decomp"])) and " ".join(wres["decomp"])[i] == selected) else 0.
                                       for wres in word_results]
                result["decomp"].append(selected)
                
                if (sum(majority_confidence) > 0.0001):
                    result["char_confidence"].append(float(sum(majority_confidence)) / len([c for c in majority_confidence if (c > 0.)]))
                else:
                    result["char_confidence"].append(0.)

                result["votes"].append(votes.most_common(1)[0])

            result["decomp"] = "".join(result["decomp"]).strip().split()
            result["confidence"] = sum(result["char_confidence"]) / len(result["char_confidence"])

            results.append(result)

        elif (mode == EnsembleMode.CONFIDENCE_CHAR):
            maxlen_wres = max(word_results, key=lambda wr: len(" ".join(wr["decomp"])))
            plain_decomp = " ".join(maxlen_wres["decomp"])
            result = {"word": word_results[0]["word"], "decomp": [], "confidence": 0., "char_confidence": []}

            for i in range(len(plain_decomp)):
                max_confidence_wres = max([wres for wres in word_results if (i < len(" ".join(wres["decomp"])))],
                                        key=lambda wr: wr["char_confidence"][i])
                selected = " ".join(max_confidence_wres["decomp"])[i] if (i < len(" ".join(max_confidence_wres["decomp"]))) else " "
                result["decomp"].append(selected)
                result["char_confidence"].append(max_confidence_wres["char_confidence"][i] if (i < len(" ".join(max_confidence_wres["decomp"]))) else 1.)

            result["decomp"] = "".join(result["decomp"]).strip().split()
            result["confidence"] = sum(result["char_confidence"]) / len(result["char_confidence"])

            results.append(result)

    return results


def train_model(config, excluded_words=set(), model_seq=0):
    morphodb = morphodb_load(config["data"]["morphodb_path"], excluded_words=excluded_words)

    input_seqs, output_seqs, sample_weights = encode_morphodb(morphodb, config["model"]["ENC_SIZE_WORD"], ENC_SIZE_CHAR, 
                                                              config["model"]["ENC_SIZE_WORD_OUT"], ENC_SIZE_CHAR, 
                                                              config["model"]["ENC_SIZE_CTX_WIN"], reverse=config["model"]["REVERSE"])

    w2m = Word2Morpho(config)

    w2m.train(input_seqs, output_seqs, 25, batch_size=200, validation_split=0.001, sample_weight=sample_weights)
    w2m.save(config["data"]["model_path"] % (config["id"], model_seq))

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
        for i in xrange(5):
            model = train_model(config, model_seq=i+3)

    if (model is None):
        model = Word2Morpho(config)
        model.load(config["data"]["model_path"] % (config["id"], 0))

    if ("test" in ops):
        test_accuracy(config, model)
    if ("test_full" in ops):
        test_decomp(config, model)




if __name__ == "__main__":
    main(sys.argv)
