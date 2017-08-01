__author__ = "Danilo S. Carvalho <danilo@jaist.ac.jp>"

import json


def morphodb_load(path):
    morphodb = None
    with open(path) as morphodbfile:
        morphodb = json.load(morphodbfile)

    for word in morphodb:
        yield (word, " ".join(morphodb[word]["morphemes"]["seq"]))
