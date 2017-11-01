# -*- coding: utf-8 -*-
__author__ = "Danilo S. Carvalho <danilo@jaist.ac.jp>"


import unittest
import json
import re
import random
from collections import Counter
import Levenshtein

from morphodecomp import train_model, decompose, decompose_ensemble, load_models, EnsembleMode
from data_access.load_morphodb import morphodb_load
from data_access.load_morphochallenge import load_morphochallenge_data
from config.loader import load_config
from ml.word2morpho import Word2Morpho

from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors

CONFIG_PATH_LIST = ["./data/config/0010.config.json"] * 10
#CONFIG_PATH_LIST = ["./data/config/0011.config.json"] * 10
#CONFIG_PATH_LIST = ["./data/config/0006.config.json"] * 30
#CONFIG_PATH_LIST = ["./data/config/0007.config.json"] * 30
#CONFIG_PATH_LIST = ["./data/config/0008.config.json"] * 3
#CONFIG_PATH_LIST = ["./data/config/0009.config.json"] * 3
#CONFIG_PATH_LIST = ["./data/config/0006.config.json"] * 5 + ["./data/config/0007.config.json"] * 5 + ["./data/config/0008.config.json"] * 3

W2V_VEC_PATH = "/home/danilo/uv/GoogleNews-vectors-negative300.bin"
W2V_MODEL_PATH = "/home/danilo/tdv_family/wikt_morphodecomp/data/w2v_decomp_map.pickle"


def calc_performance(test_morphodb, morpho_analyses):
    pairs_result = []
    pairs_golden = []
    word_precision = dict()
    word_recall = dict()

    for morpho_anlz in morpho_analyses:
        for morpheme in morpho_anlz["decomp"]:
            try:
                pair = (morpho_anlz, 
                        random.choice([anlz for anlz in morpho_analyses if (morpheme in anlz["decomp"] and anlz["word"] != morpho_anlz["word"])]),
                        morpheme)
                pairs_result.append(pair)
            except (IndexError):
                pass

    for word in test_morphodb:
        for decomp in test_morphodb[word]:
            for morpheme in [morph for morph in decomp if ("~" not in morph)]:
                try:
                    pair = (word, 
                            random.choice([w for w in test_morphodb if (morpheme in test_morphodb[w][0])]), 
                            morpheme)
                    pairs_golden.append(pair)
                except (IndexError):
                    pass

    hits = Counter()
    misses = Counter()
    for pair in pairs_result:
        word1 = pair[0]["word"]
        word2 = pair[1]["word"]
        morphemes1 = []
        morphemes2 = []
        for decomp in test_morphodb[word1]:
            morphemes1.extend(decomp)
        for decomp in test_morphodb[word2]:
            morphemes2.extend(decomp)

        if (pair[2] in morphemes1 and pair[2] in morphemes2):
            hits[word1] += 1
        else:
            misses[word1] += 1

    for word in test_morphodb:
        if (word in hits):
            word_precision[word] = float(hits[word]) / (hits[word] + misses[word])

    analyses_dict = dict([(anlz["word"], anlz) for anlz in morpho_analyses])
    hits = Counter()
    misses = Counter()
    for pair in pairs_golden:
        word1 = pair[0]
        word2 = pair[1]

        #print pair
        #print (pair[2], analyses_dict[word1]["decomp"], analyses_dict[word2]["decomp"])

        if (pair[2] in analyses_dict[word1]["decomp"] and pair[2] in analyses_dict[word2]["decomp"]):
            hits[word1] += 1
        else:
            misses[word1] += 1

    for word in test_morphodb:
        if (word in hits):
            word_recall[word] = float(hits[word]) / (hits[word] + misses[word])

    precision = sum([word_precision[word] for word in word_precision]) / len(word_precision)
    recall = sum([word_recall[word] for word in word_recall]) / len(word_recall)

    return (precision, recall, 2 * (precision * recall) / (precision + recall))


class TestMorphoChallenge(unittest.TestCase):
    def test_accuracy(self):
        train_morphodb, test_morphodb = load_morphochallenge_data()

        for i in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 20]:
            models = load_models(CONFIG_PATH_LIST[0:i])

            word_list = []
            for word in test_morphodb:
                mo = re.search(r"^(?P<core>.+)(?P<gen>'s?)$", word)
                if (mo):
                    word_list.append([mo.group("core"), "-" + mo.group("gen")])
                else:
                    word_list.append([word])
    
            # morpho_analyses = decompose([w[0] for w in word_list], config_path, model=model, cache=False)
            morpho_analyses = decompose_ensemble([w[0] for w in word_list], CONFIG_PATH_LIST, models=models, mode=EnsembleMode.MAJORITY_OVERALL, num_procs=2)
    
            # for anlz in morpho_analyses:
            #     print anlz
            # 
            # return
    
            for i in xrange(len(word_list)):
                if (len(word_list[i]) > 1):
                    morpho_analyses[i]["word"] = word_list[i][0] + word_list[i][1][1:]
                    morpho_analyses[i]["decomp"].append(word_list[i][1])
    
            print calc_performance(test_morphodb, morpho_analyses)


class TestMorphoChallengeW2V(unittest.TestCase):
    def test_accuracy(self):
        train_morphodb, test_morphodb = load_morphochallenge_data()
        config = load_config(CONFIG_PATH_LIST[0])
        morphodb = morphodb_load(config["data"]["morphodb_path"], excluded_words=set(test_morphodb.keys()))

        word_list = []
        for word in test_morphodb:
            mo = re.search(r"^(?P<core>.+)(?P<gen>'s?)$", word)
            if (mo):
                word_list.append([mo.group("core"), "-" + mo.group("gen")])
            else:
                word_list.append([word])

        decomp_list = []
        for (word, morphemes) in morphodb:
            decomp_list.append({"word": word, "decomp": morphemes.split(), "confidence": 1.0})

        w2v = KeyedVectors.load_word2vec_format(W2V_VEC_PATH, binary=True)

        morpho_analyses = []
        for word in word_list:
            try:
                most_sim = sorted(decomp_list, key=lambda m: w2v.wv.similarity(m["word"], word[0]))[-1]
            except (KeyError):
                most_sim = sorted(decomp_list, key=lambda m: Levenshtein.ratio(m["word"], word[0]))[-1]

            morpho_analyses.append(most_sim)

        for i in xrange(len(word_list)):
            if (len(word_list[i]) > 1):
                morpho_analyses[i]["word"] = word_list[i][0] + word_list[i][1][1:]
                morpho_analyses[i]["decomp"].append(word_list[i][1])

        print calc_performance(test_morphodb, morpho_analyses)






