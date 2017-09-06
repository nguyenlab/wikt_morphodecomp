__author__ = "Danilo S. Carvalho <danilo@jaist.ac.jp>"

from keras.models import Sequential, Model, load_model
from keras.layers import Masking, TimeDistributed, Activation, LSTM, RepeatVector, Dense, Bidirectional, Dropout
from keras import backend as K

from ml import encoder as enc


class Word2Morpho(object):
    def __init__(self, config):
        init_dist = config["model"]["INIT_DIST"]
        enc_size_word = config["model"]["ENC_SIZE_WORD"]
        enc_size_ctx_win = config["model"]["ENC_SIZE_CTX_WIN"]
        hidden_dim_enc = (config["model"]["HIDDEN_DIM_ENC_1"],)
        enc_size_word_out = config["model"]["ENC_SIZE_WORD_OUT"]
        hidden_dim_dec = (config["model"]["HIDDEN_DIM_DEC_1"], config["model"]["HIDDEN_DIM_DEC_2"])

        self.model = Sequential()
        self.model.add(Masking(mask_value=0., input_shape=(enc_size_word, enc.ENC_SIZE_CHAR * enc_size_ctx_win)))
        self.model.add(Bidirectional(LSTM(hidden_dim_enc[0], return_sequences=False, implementation=2, activation="tanh", kernel_initializer=init_dist)))
        self.model.add(Dropout(0.25))
        self.model.add(RepeatVector(enc_size_word_out))
        self.model.add(Bidirectional(LSTM(hidden_dim_dec[0], return_sequences=True, implementation=2, activation="tanh", kernel_initializer=init_dist)))
        self.model.add(Dropout(0.25))
        self.model.add(TimeDistributed(Dense(hidden_dim_dec[1], activation="tanh", kernel_initializer=init_dist), input_shape=(enc_size_word_out, hidden_dim_dec[0])))
        self.model.add(TimeDistributed(Dense(enc.ENC_SIZE_CHAR, activation="softmax", kernel_initializer=init_dist), input_shape=(enc_size_word_out, hidden_dim_dec[1])))

        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', sample_weight_mode="temporal")
        #self.model.summary()

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

