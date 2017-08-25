# -*- coding: utf-8 -*-
__author__ = "Danilo S. Carvalho <danilo@jaist.ac.jp>"


import unittest
import os
import json
import re
import random
from collections import Counter

from morphodecomp import train_model, decompose
from config.loader import load_config

MORPHOCHALLENGE_DATA_PATH = ""
CONFIG_PATH_LIST = ["./data/config/default.config.json"]


def load_morphochallenge_data(path):
    test_word_set = set()
    test_morphodb = dict()
    train_morphodb = dict()
    with open(os.path.join(path, "goldstd_develset.labels.eng")) as test_file:
        for line in test_file:
            test_word_set.add(line.split("\t")[0])

    with open(os.path.join(path, "goldstd_combined.segmentation.eng")) as combined_file:
        for line in combined_file:
            word, decomps = line.split("\t")
            morpheme_seqs = []
            for decomp in decomps.split(","):
                morphemes = []

                for part in decomp:
                    stem, morph_cls = part.split(":")

                    if ("_" in morph_cls):
                        morph, cls = morph_cls.split("_")
                        if (cls == "V" or cls == "N"):
                            morphemes.append(morph)
                        elif (cls == "p"):
                            morphemes.append(stem + "-")
                        elif (cls == "s"):
                            morphemes.append("-" + stem)
                    else:
                        if (morph_cls != "~"):
                            morphemes.append("-" + stem)

                morpheme_seqs.append(morphemes)

            if (word in test_word_set):
                test_morphodb[word] = morpheme_seqs
            else:
                train_morphodb[word] = morpheme_seqs

    return (train_morphodb, test_morphodb)


def calc_performance(test_morphodb, morpho_analyses):
    pairs_result = []
    pairs_golden = []
    word_precision = dict()
    word_recall = dict()

    for morpho_anlz in morpho_analyses:
        for morpheme in morpho_anlz["decomp"]:
            try:
                pair = (morpho_anlz, random.choice([anlz for anlz in morpho_analyses if (morpheme in anlz["decomp"])]), morpheme)
                pairs_result.append(pair)
            except (IndexError):
                pass

    for word in test_morphodb:
        for morpheme in test_morphodb[word]:
            try:
                pair = (word, random.choice([word for word in test_morphodb if (morpheme in test_morphodb[word])]), morpheme)
                pairs_golden.append(pair)
            except (IndexError):
                pass

    hits = Counter()
    misses = Counter()
    for pair in pairs_result:
        word1 = pair[0]["word"]
        word2 = pair[2]["word"]

        if (pair[2] in test_morphodb[word1] and pair[2] in test_morphodb[word2]):
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
        word1 = pair[0]["word"]
        word2 = pair[2]["word"]

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
        train_morphodb, test_morphodb = load_morphochallenge_data(MORPHOCHALLENGE_DATA_PATH)
        exp_decomps = dict()

        for config_path in CONFIG_PATH_LIST:
            config = load_config(config_path)
            model = train_model(config, set(test_morphodb.keys()))
            word_list = []

            for word in test_morphodb:
                mo = re.search(r"^(?P<core>.+)(?P<gen>'s?)$")
                if (mo):
                    word_list.append([mo.group("core"), "-" + mo.group("gen")])
                else:
                    word_list.append([word])

            morpho_analyses = decompose([w[0] for w in word_list], config_path, model=model, cache=False)

            for i in xrange(len(word_list)):
                if (len(word_list[i]) > 1):
                    morpho_analyses[i]["decomp"].extend(word_list[i][1:])

            exp_decomps[config_path] = morpho_analyses

        for config_path in exp_decomps:
            print calc_performance(test_morphodb, exp_decomps[config_path])






