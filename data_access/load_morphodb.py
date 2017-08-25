__author__ = "Danilo S. Carvalho <danilo@jaist.ac.jp>"

import json

DECOMP_PASS_LIMIT = 4


def morphodb_load(path, excluded_words=set()):
    morphodb = None
    with open(path) as morphodbfile:
        morphodb = json.load(morphodbfile)

    for word in morphodb:
        if (word not in excluded_words):
            total_decomp = False
            decomp = None
            tmp_morphemes = list(morphodb[word]["morphemes"]["seq"])
            passes = 0

            while (not total_decomp):
                total_decomp = True
                decomp = []
                for morpheme in tmp_morphemes:
                    if (morpheme in morphodb and passes < DECOMP_PASS_LIMIT):
                        decomp.extend(morphodb[morpheme]["morphemes"]["seq"])
                        total_decomp = False
                    else:
                        decomp.append(morpheme)
                    passes += 1

                tmp_morphemes = list(decomp)

            yield (word, " ".join(decomp))
