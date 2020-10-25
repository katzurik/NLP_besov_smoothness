
from keras.layers import Lambda
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv1D
from keras.callbacks import EarlyStopping

# training params
batch_size = 1024
num_epochs = 15

# model parameters
num_filters = 64
embed_dim = 300
weight_decay = 1e-4
filters = 600
kernel_size = 3

MAX_NB_WORDS = 100000


def OneHot(input_dim=None, input_length=None):
    # Check if inputs were supplied correctly
    if input_dim is None or input_length is None:
        raise TypeError("input_dim or input_length is not set")

    # Helper method (not inlined for clarity)
    def _one_hot(x, num_classes):
        return K.one_hot(K.cast(x, 'uint8'),
                          num_classes=num_classes)

    # Final layer representation as a Lambda layer
    return Lambda(_one_hot,
                  arguments={'num_classes': input_dim},
                  input_shape=(input_length,))


def build_one_hot_model(nb_words, max_seq_len):
    print("training CNN ...")
    model = Sequential()
    model.add(OneHot(input_dim=nb_words, input_length=max_seq_len))
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


def train_one_hot_model(model, word_seq_train, y_train):
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
        shuffle=True,
        verbose=2)
    return model



