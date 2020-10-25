import pandas as pd
from pathlib import Path
from path_config import SAVE_PATH
from WaveletsForestRegressor import WaveletsForestRegressor
from keras import backend as K
from sentiment_analysis_with_fasttext import get_embeddings_index, create_embedding_matrix, build_model, train_model
from sentiment_analysis_one_hot import build_one_hot_model, train_one_hot_model
from sentiment_analysis_word2vec import build_model_w2v, create_embedding_matrix_w2v, create_w2v_model, train_w2v_model
from pre_process_imdb_data import main_idb_process
from pre_process_data import pre_process_data

SAMPLE = True
TWITTER = False
IMDB = True


def smoothness_test(model, model_name):
    smoothness_df = pd.DataFrame(columns=['alpha'], index=[layer.name for layer in model.layers])
    layer_output = word_seq_train
    for i, layer in enumerate(model.layers):
        get_layer_output = K.function([model.layers[i].input], [model.layers[i].output])
        layer_output = get_layer_output([layer_output])[0]

        regressor = WaveletsForestRegressor()
        rf = regressor.fit(layer_output.reshape(-1, layer_output.shape[0]).transpose(), y_train)

        alpha, n_wavelets, errors = rf.evaluate_smoothness()
        smoothness_df.loc[layer.name, 'alpha'] = alpha

    smoothness_df.to_csv(Path(SAVE_PATH, f'smoothness_{model_name}.csv'))
    return smoothness_df


if __name__ == '__main__':

    # pre_process data:
    if TWITTER:
        y_train, y_test, word_seq_train, word_seq_test, word_index, max_seq_len, df_train = pre_process_data(sample=SAMPLE)

    elif IMDB:
        y_train, y_test, word_seq_train, word_seq_test, word_index, max_seq_len, df_train = main_idb_process(sample=SAMPLE)

    # test word2v smoothness :
    w2v_model = create_w2v_model(df_train)
    embed_matrix, vocab_size = create_embedding_matrix_w2v(w2v_model, df_train)
    model_word2vec = build_model_w2v(embed_matrix, vocab_size)
    train_w2v = train_w2v_model(word_seq_train, y_train, model_word2vec)
    smoothness_test(train_w2v, 'w2v')

    # test fasttext smoothness :
    embeddings_index = get_embeddings_index()
    nb_words, embedding_matrix = create_embedding_matrix(word_index, embeddings_index)
    model = build_model(nb_words, embedding_matrix, max_seq_len)
    train_model = train_model(model, word_seq_train, y_train)
    smoothness_test(train_model, 'fasttext')

    # test one hot smoothness
    one_hot_model = build_one_hot_model(nb_words, max_seq_len)
    train_one_hot_model = train_one_hot_model(one_hot_model, word_seq_train, y_train)
    smoothness_test(train_one_hot_model, 'one_hot')