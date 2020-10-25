import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
from path_config import fpath

encoding = "ISO-8859-1"
col = ["target", "ids", "date", "flag", "user", "text"]

stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")
stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
regstring = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

TestSize = 0.3

MAX_NB_WORDS = 100000


def read_data():
    df = pd.read_csv(fpath, encoding=encoding, names=col)
    return df


def pre_process(text, stem=False):
    # Remove link,user and special characters
    text = re.sub(regstring, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
                print(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)


def split_train_set(df, sample):
    df_train, df_test = train_test_split(df, test_size=TestSize, random_state=42)

    if sample:
        df_train = df_train.sample(1000)
        df_test = df_test.sample(500)

    print("TRAIN size:", len(df_train))
    print("TEST size:", len(df_test))

    return df_train, df_test


def get_y_test_y_train(df_train, df_test):
    encoder = LabelEncoder()
    encoder.fit(df_train.target.tolist())

    y_train = encoder.transform(df_train.target.tolist())
    y_test = encoder.transform(df_test.target.tolist())

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    print("y_train", y_train.shape)
    print("y_test", y_test.shape)

    return y_train, y_test

def tokenize_input_data(processed_docs_train, processed_docs_test):
    tokenizer = RegexpTokenizer(r'\w+')
    print("tokenizing input data...")
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=False)
    tokenizer.fit_on_texts(processed_docs_train + processed_docs_test)  #leaky
    word_seq_train = tokenizer.texts_to_sequences(processed_docs_train)
    word_seq_test = tokenizer.texts_to_sequences(processed_docs_test)
    word_index = tokenizer.word_index
    print("dictionary size: ", len(word_index))

    return word_index, word_seq_train, word_seq_test


def pre_process_data(sample):
    # load data :
    df = read_data()

    # pre process data :
    df['clean_text'] = df.text.apply(lambda x: pre_process(x))
    df_train, df_test = split_train_set(df, sample)
    processed_docs_train = df_train['clean_text'].tolist()
    processed_docs_test = df_test['clean_text'].tolist()
    y_train, y_test  = get_y_test_y_train(df_train, df_test)
    df_train['doc_len'] = df_train['text'].apply(lambda words: len(words.split(" ")))
    # max_seq_len = np.round(df_train['doc_len'].mean() + df_train['doc_len'].std()).astype(int)
    max_seq_len = 300
    print(max_seq_len)

    # tokenize data:
    word_index, word_seq_train, word_seq_test = tokenize_input_data(processed_docs_train, processed_docs_test)

    # pad sequences
    word_seq_train = sequence.pad_sequences(word_seq_train, maxlen=max_seq_len)
    word_seq_test = sequence.pad_sequences(word_seq_test, maxlen=max_seq_len)

    return y_train, y_test, word_seq_train, word_seq_test, word_index, max_seq_len, df_train


if __name__ == '__main__':
    pre_process_data(sample=True)