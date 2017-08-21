#!/usr/bin/env python
#-*- coding: utf8 -*-

import sys
import web
import json
import re
import numpy as np

from data_access.load_morphodb import morphodb_load
from ml.encoder import encode_morphodb, encode_word
from ml.decoder import decode_word
from ml.word2morpho import Word2Morpho
from config import encoder as encconf

urls = (
    '/w2morph/(.+)', 'W2Morph',
)
app = web.application(urls, globals())

TEST_WORDS = ["unlockability", "busheater"]

w2m = Word2Morpho()
w2m.load("/home/danilo/tdv_family/w2m_model_4.hdf5")


class W2Morph:
    def GET(self, word):
        test_word = np.array([encode_word(word)], dtype=np.float16)

        return json.dumps(re.search(r"^{(?P<decomp>[^}]+)}.*", decode_word(w2m.predict(test_word)[0])).group("decomp").split())



if __name__ == "__main__":
    app.run()
 
