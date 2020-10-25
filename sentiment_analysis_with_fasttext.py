from tqdm import tqdm
import codecs
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Embedding, Conv1D
from keras.callbacks import EarlyStopping

MAX_NB_WORDS = 100000

#training params
batch_size = 1024
num_epochs = 15

#model parameters
num_filters = 64
embed_dim = 300
weight_decay = 1e-4
filters = 600
kernel_size = 3


def get_embeddings_index():
    # load embeddings
    print('loading word embeddings...')
    embeddings_index = {}
    f = codecs.open(r"C:\Users\danielleuser\Documents\wiki.simple.vec", encoding='utf-8')
    for line in tqdm(f):
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('found %s word vectors' % len(embeddings_index))
    return embeddings_index


# embedding matrix
def create_embedding_matrix(word_index, embeddings_index):
    print('preparing embedding matrix...')
    words_not_found = []
    nb_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((nb_words, embed_dim))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            words_not_found.append(word)
    print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    print("sample words not found: ", np.random.choice(words_not_found, 10))
    return nb_words, embedding_matrix


def build_model(nb_words, embedding_matrix, max_seq_len):
    print("training CNN ...")
    model = Sequential()
    model.add(Embedding(nb_words, embed_dim, weights=[embedding_matrix], input_length=max_seq_len, trainable=True))
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


# model training
def train_model(model, word_seq_train, y_train):
    # define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
    callbacks_list = [early_stopping]

    model.fit(
        word_seq_train,
        y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        callbacks=callbacks_list,
        validation_split=0.1,
        shuffle=True, verbose=2)
    return model

