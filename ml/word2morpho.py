__author__ = "Danilo S. Carvalho <danilo@jaist.ac.jp>"

import seq2seq
from seq2seq.models import AttentionSeq2Seq
from keras.models import load_model

from config import encoder as encconf
from config import word2morpho as w2mconf


class Word2Morpho(object):
    def __init__(self):
        self.model = AttentionSeq2Seq(input_dim=encconf.ENC_SIZE_CHAR,
                                      input_length=encconf.ENC_SIZE_WORD,
                                      hidden_dim=w2mconf.HIDDEN_DIM,
                                      output_length=encconf.ENC_SIZE_WORD + 10,
                                      output_dim=encconf.ENC_SIZE_CHAR,
                                      depth=1, unroll=True)
        self.model.compile(loss='mse', optimizer='rmsprop')

    def train(self, input_seqs, output_seqs, num_epochs, batch_size=1, validation_split=0.):
        self.model.fit(input_seqs, output_seqs, batch_size, num_epochs, verbose=1, validation_split=validation_split)

    def train_from_generator(self, generator, steps_per_epoch, num_epochs):
        self.model.fit_generator(generator, steps_per_epoch, epochs=num_epochs)
    
    def predict(self, insts):
        return self.model.predict(insts)

    def load(self, path):
        try:
            self.model = load_model(path)
            print "Loaded model."
            return True
        except(IOError):
            return False

    def save(self, path):
        self.model.save(path)

