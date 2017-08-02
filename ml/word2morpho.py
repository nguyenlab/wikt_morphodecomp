__author__ = "Danilo S. Carvalho <danilo@jaist.ac.jp>"

import seq2seq
from seq2seq.models import AttentionSeq2Seq, Seq2Seq
from keras.models import Sequential, load_model
from keras.layers import Masking, TimeDistributed, Activation, LSTM, RepeatVector, Dense, Bidirectional, Dropout

from config import encoder as encconf
from config import word2morpho as w2mconf


class Word2Morpho(object):
    def __init__(self):
        self.model = Sequential()
        self.model.add(Masking(mask_value=0., input_shape=(encconf.ENC_SIZE_WORD, encconf.ENC_SIZE_CHAR)))
        self.model.add(Dropout(0.25))
        self.model.add(Bidirectional(LSTM(w2mconf.HIDDEN_DIM_ENC, return_sequences=False)))
        self.model.add(RepeatVector(encconf.ENC_SIZE_WORD_OUT))
        self.model.add(LSTM(w2mconf.HIDDEN_DIM_DEC, return_sequences=True))
        self.model.add(TimeDistributed(Dense(encconf.ENC_SIZE_CHAR, activation="softmax"), input_shape=(encconf.ENC_SIZE_WORD_OUT, w2mconf.HIDDEN_DIM_DEC)))

        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', sample_weight_mode="temporal")

    def train(self, input_seqs, output_seqs, num_epochs, batch_size=1, validation_split=0., sample_weight=None):
        self.model.fit(input_seqs, output_seqs, batch_size, num_epochs, verbose=1, validation_split=validation_split, sample_weight=sample_weight)

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

