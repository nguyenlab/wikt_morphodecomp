#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys
import json

LANG = "English"
MORPH_KEYS = set(["prefix", "suffix", "confix", "affix", "stem", "af", "compound", "blend"])

def main(argv):
    wiktdb = None
    with open(argv[1]) as wiktfile:
        wiktdb = json.load(wiktfile)

    wikt_morphdb = dict()
    count = 0

    for term_entry in wiktdb:
        morphemes = {"type": "", "seq": []}
        if (LANG in term_entry["langs"] and "etymology" in term_entry["langs"][LANG] and " " not in term_entry["title"]):
            etym = term_entry["langs"][LANG]["etymology"]

            if ("attrs" in etym):
                for etym_attr in etym["attrs"]:
                    if (etym_attr[0] in MORPH_KEYS):
                        morphemes["type"] = mtype = etym_attr[0]
                        if (mtype == "af"):
                            morphemes["type"] = mtype = "affix"

                        for morph in etym_attr[1:]:
                            if (("=" in morph) or (morph == "en" and not term_entry["title"].startswith("en")) or morph == ""):
                                continue
                            else:
                                morphemes["seq"].append(morph)
                        
                        if (len(morphemes["seq"]) > 0):
                            if (mtype == "prefix"):
                                morphemes["seq"][0] += "-"
                            elif (mtype == "suffix"):
                                morphemes["seq"][-1] = "-" + morphemes["seq"][-1]

                        if (len(morphemes["seq"]) == 1):
                            single_morph = morphemes["seq"][0].replace("-", "")
                            remainder = term_entry["title"].replace(single_morph, "")
                            
                            if (mtype == "prefix"):
                                morphemes["seq"] = [single_morph + "-", remainder]
                            elif (mtype == "suffix"):
                                morphemes["seq"] = [remainder, "-" + single_morph]
                            else:
                                if (remainder + single_morph == term_entry["title"]):
                                    morphemes["seq"] = [remainder, single_morph]
                                else:
                                    morphemes["seq"] = [single_morph, remainder]

                if (morphemes["seq"]):
                    wikt_morphdb[term_entry["title"]] = {"morphemes": morphemes, "pos_order": term_entry["langs"][LANG]["pos_order"]}

        count += 1
        if (count % 10000 == 0):
            print float(count) / len(wiktdb)

    print "Total words: ", len(wikt_morphdb)

    with open(argv[2], "w") as morphdbfile:
        json.dump(wikt_morphdb, morphdbfile, indent=2)





if __name__ == "__main__":
    main(sys.argv)
