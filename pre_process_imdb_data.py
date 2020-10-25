import numpy as np
import pandas as pd
import os
from nltk.tokenize import RegexpTokenizer
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import re
from tqdm import tqdm
import random
from tensorflow import set_random_seed
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from path_config import imdb_path

import nltk
nltk.download('punkt')
nltk.download('wordnet')

#set random seed for the session and also for tensorflow that runs in background for keras
set_random_seed(123)
random.seed(123)

encoding = "ISO-8859-1"


TestSize = 0.3


def read_data():
    df = pd.read_csv(imdb_path)
    return df


def clean_sentences(df):
    reviews = []

    for sent in tqdm(df['review']):
        # remove html content
        review_text = BeautifulSoup(sent).get_text()

        # remove non-alphabetic characters
        review_text = re.sub("[^a-zA-Z]", " ", review_text)

        # tokenize the sentences
        words = word_tokenize(review_text.lower())

        # lemmatize each word to its lemma
        lemma_words = [lemmatizer.lemmatize(i) for i in words]

        reviews.append(lemma_words)

    return (reviews)


def get_y_test_y_train(df_train, df_test):
    encoder = LabelEncoder()
    encoder.fit(df_train.sentiment.tolist())

    y_train = encoder.transform(df_train.sentiment.tolist())
    y_test = encoder.transform(df_test.sentiment.tolist())

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    print("y_train", y_train.shape)
    print("y_test", y_test.shape)

    return y_train, y_test


def split_train_set(df, sample):
    df_train, df_test = train_test_split(df, test_size=TestSize, random_state=42)

    if sample:
        df_train = df_train.sample(1000)
        df_test = df_test.sample(500)

    print("TRAIN size:", len(df_train))
    print("TEST size:", len(df_test))

    return df_train, df_test


def toknize_data(df_train, df_test):
# It is needed for initializing tokenizer of keras and subsequent padding

    unique_words = set()
    len_max = 0

    for sent in tqdm(df_train['sentences']):

        unique_words.update(sent)

        if (len_max < len(sent)):
            len_max = len(sent)

    # length of the list of unique_words gives the no of unique words
    print(len(list(unique_words)))
    print(len_max)

    tokenizer = Tokenizer(num_words=len(list(unique_words)))
    tokenizer.fit_on_texts(list(df_train['sentences']))

    #texts_to_sequences(texts)

    # Arguments- texts: list of texts to turn to sequences.
    #Return: list of sequences (one per text input).
    train = tokenizer.texts_to_sequences(df_train['sentences'])
    test = tokenizer.texts_to_sequences(df_test['sentences'])
    word_index = tokenizer.word_index

    #padding done to equalize the lengths of all input reviews. LSTM networks needs all inputs to be same length.
    #Therefore reviews lesser than max length will be made equal using extra zeros at end. This is padding.

    train_seq = sequence.pad_sequences(train, maxlen=300)
    test_seq = sequence.pad_sequences(test, maxlen=300)

    return train_seq, test_seq, word_index


def main_idb_process(sample):
    df = read_data()

    df['text'] = df['review']

    df_train, df_test, = split_train_set(df, sample)

    df_train['sentences'] = clean_sentences(df_train)
    df_test['sentences'] = clean_sentences(df_test)

    y_train, y_test = get_y_test_y_train(df_train, df_test)
    print(len(df_train['sentences']))

    word_seq_train, word_seq_test, word_index = toknize_data(df_train, df_test)

    return y_train, y_test, word_seq_train, word_seq_test, word_index, 300, df_train


if __name__ == '__main__':
    main_idb_process(sample=True)







