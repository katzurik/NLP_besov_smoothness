from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import nltk
import gensim
import numpy as np


nltk.download('stopwords')

# WORD2VEC
W2V_SIZE = 300
W2V_WINDOW = 7
W2V_EPOCH = 32
W2V_MIN_COUNT = 10

# KERAS
SEQUENCE_LENGTH = 300
EPOCHS = 8
BATCH_SIZE = 1024
filters = 600
kernel_size = 3

def create_w2v_model(df_train):
    documents = [_text.split() for _text in df_train.text]

    w2v_model = gensim.models.word2vec.Word2Vec(size=W2V_SIZE,
                                                window=W2V_WINDOW,
                                                min_count=W2V_MIN_COUNT,
                                                workers=8)

    w2v_model.build_vocab(documents)
    return w2v_model


def create_embedding_matrix_w2v(w2v_model, df_train):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df_train.text)

    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((vocab_size, W2V_SIZE))
    for word, i in tokenizer.word_index.items():
      if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]
    print(embedding_matrix.shape)

    return embedding_matrix, vocab_size


def build_model_w2v(embedding_matrix, vocab_size):

    embedding_layer = Embedding(vocab_size, W2V_SIZE, weights=[embedding_matrix], input_length=SEQUENCE_LENGTH, trainable=False)

    model = Sequential()
    model.add(embedding_layer)
    model.add(Dropout(0.4))
    model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))
    model.add(Conv1D(300, kernel_size, padding='valid', activation='relu', strides=1))
    model.add(Conv1D(150, kernel_size, padding='valid', activation='relu', strides=1))
    model.add(Conv1D(75, kernel_size, padding='valid', activation='relu', strides=1))
    model.add(Flatten())
    model.add(Dense(600))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def train_w2v_model(word_seq_train, y_train, model):

    callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
              EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)]

    model.fit(word_seq_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_split=0.1,
                    verbose=1,
                    callbacks=callbacks)
    return model
