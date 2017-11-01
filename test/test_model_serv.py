#!/usr/bin/env python
#-*- coding: utf8 -*-

import sys
import web
import json
import re
import numpy as np

from morphodecomp import train_model, decompose, decompose_ensemble, load_models, EnsembleMode

CONFIG_PATH_LIST = ["./data/config/0007.config.json"] * 20

try:
    assert len(models) > 0
except:
    models = load_models(CONFIG_PATH_LIST)

urls = (
    '/w2morph/(.+)', 'W2Morph',
)
app = web.application(urls, globals())

class W2Morph:
    def GET(self, words):
        global CONFIG_PATH_LIST
        global models
        word_list = words.split(",")
        morpho_analyses = decompose_ensemble(word_list, CONFIG_PATH_LIST, models=models, mode=EnsembleMode.MAJORITY_CHAR)

        for anlz in morpho_analyses:
            del anlz["char_confidence"]
            del anlz["votes"]

        return json.dumps(morpho_analyses, indent=2)



if __name__ == "__main__":
    app.run()
 
