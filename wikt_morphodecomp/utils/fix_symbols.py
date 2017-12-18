#-*- coding: utf8 -*-
import json
import re
import sys


morphodb = json.load(open(sys.argv[1]))


def fix_confix():
    for key in morphodb:
        term = morphodb[key]
        if (term["morphemes"]["type"] == "confix"):
            for i in xrange(len(term["morphemes"]["seq"])):
                if (i == 0 and not re.match(r".+-$", term["morphemes"]["seq"][i])):
                    term["morphemes"]["seq"][i] += "-"

                elif (i == len(term["morphemes"]["seq"]) - 1 and not re.match(r"^-.+", term["morphemes"]["seq"][i])):
                    term["morphemes"]["seq"][i] = "-" + term["morphemes"]["seq"][i]


def fix_ensuffix():
    for key in morphodb:
        term = morphodb[key]
        if (term["morphemes"]["seq"][0] == "n" and not re.match(r"^n.+", key)):
            term["morphemes"]["seq"][0] = term["morphemes"]["seq"][-1].replace("-", "")
            term["morphemes"]["seq"][-1] = "-en"


#fix_confix()
fix_ensuffix()

json.dump(morphodb, open(sys.argv[2], "w"), indent=2)


