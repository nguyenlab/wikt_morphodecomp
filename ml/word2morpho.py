__author__ = "Danilo S. Carvalho <danilo@jaist.ac.jp>"

from keras.models import Sequential, Model, load_model
from keras.layers import Masking, TimeDistributed, Activation, LSTM, RepeatVector, Dense, Bidirectional, Dropout, Concatenate, Input, Lambda
from keras import backend as K

from ml import encoder as enc
from config import encoder as encconf
from config import word2morpho as w2mconf


def concat_lstm(x):
    input_shape = K.shape(x)
    return K.reshape(x, (input_shape[0], input_shape[1] * input_shape[2]))


class Word2Morpho(object):
    def __init__(self):
        self.model = Sequential()
        self.model.add(Masking(mask_value=0., input_shape=(encconf.ENC_SIZE_WORD, (enc.ENC_SIZE_CHAR) * encconf.ENC_SIZE_CTX_WIN)))
        self.model.add(Bidirectional(LSTM(w2mconf.HIDDEN_DIM_ENC_1, return_sequences=False, implementation=2, activation="tanh")))
        self.model.add(Dropout(0.25))
        #self.model.add(Bidirectional(LSTM(w2mconf.HIDDEN_DIM_ENC_2, return_sequences=False, implementation=2, activation="tanh")))
        self.model.add(RepeatVector(encconf.ENC_SIZE_WORD_OUT))
        self.model.add(Bidirectional(LSTM(w2mconf.HIDDEN_DIM_DEC_1, return_sequences=True, implementation=2, activation="tanh")))
        self.model.add(Dropout(0.25))
        self.model.add(TimeDistributed(Dense(w2mconf.HIDDEN_DIM_DEC_2, activation="tanh"), input_shape=(encconf.ENC_SIZE_WORD_OUT, w2mconf.HIDDEN_DIM_DEC_1)))
        self.model.add(TimeDistributed(Dense(enc.ENC_SIZE_CHAR, activation="softmax"), input_shape=(encconf.ENC_SIZE_WORD_OUT, w2mconf.HIDDEN_DIM_DEC_2)))

        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', sample_weight_mode="temporal")
        self.model.summary()

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

