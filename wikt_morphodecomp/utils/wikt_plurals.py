#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys
import json
import re
import difflib

LANG = "English"

def get_singular(meanings):
    for pos in meanings:
        for meaning in meanings[pos]:
            if ("attrs" in meaning):
                for attr in meaning["attrs"]:
                    if (attr[0] == "inflec" and attr[1] == "plural"):
                        return attr[2]

    return None


def main(argv):
    wiktdb = None
    with open(argv[1]) as wiktfile:
        wiktdb = json.load(wiktfile)

    wikt_morphdb = dict()
    count = 0

    for term_entry in wiktdb:
        title = term_entry["title"]

        if (" " in title or not re.match(r".+(s|a|n)$", title)):
            continue

        meanings = term_entry["langs"][LANG]["meanings"]
        
        singular = get_singular(meanings)
        if (singular):
            morphemes = {"type": "plural", "seq": [singular, "-" + "".join([c[-1] for c in difflib.ndiff(singular, title) if (c[0] == "+")])]}
            wikt_morphdb[term_entry["title"]] = {"morphemes": morphemes, "pos_order": term_entry["langs"][LANG]["pos_order"]}

        count += 1
        if (count % 10000 == 0):
            print float(count) / len(wiktdb)

    print "Total words: ", len(wikt_morphdb)

    with open(argv[2], "w") as morphdbfile:
        json.dump(wikt_morphdb, morphdbfile, indent=2)





if __name__ == "__main__":
    main(sys.argv)
