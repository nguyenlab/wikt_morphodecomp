__author__ = "Danilo S. Carvalho <danilo@jaist.ac.jp>"

import os

MORPHOCHALLENGE_DATA_PATH = "/home/danilo/tdv_family/wikt_morphodecomp/data/morphochallenge-2010/"


def load_morphochallenge_data(path=MORPHOCHALLENGE_DATA_PATH):
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

                for part in decomp.split():
                    stem, morph_cls = part.split(":")

                    if ("_" in morph_cls):
                        morphcls_sep = morph_cls.split("_")
                        morph = "_".join(morphcls_sep[0:len(morphcls_sep) - 1])
                        cls = morphcls_sep[-1]
                        if (cls == "V" or cls == "N"):
                            morphemes.append(morph)
                        elif (cls == "p"):
                            morphemes.append(stem + "-")
                        elif (cls == "s"):
                            morphemes.append("-" + stem)
                    else:
                        if (morph_cls != "~"):
                            morphemes.append("-" + stem)

                morpheme_seqs.append([m.lower() for m in morphemes])

            if (word in test_word_set):
                test_morphodb[word] = morpheme_seqs
            else:
                train_morphodb[word] = morpheme_seqs

    return (train_morphodb, test_morphodb)