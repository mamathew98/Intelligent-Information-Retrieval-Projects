import numpy as np
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tools.make_dir import make_dir
from tools.pickle_manager import save_object
from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
# from keras.initializers import Constant
# import keras.initializers
# import keras.optimizers
import tensorflow as tf


# Function returning vector reperesentation of a document
def get_embedding_w2v(doc_tokens, w2v_model):
    embeddings = []
    if len(doc_tokens) < 1:
        return np.zeros(100)
    else:
        for tok in doc_tokens:
            if tok in w2v_model.wv.key_to_index:
                embeddings.append(w2v_model.wv.get_vector(tok))
            else:
                embeddings.append(np.random.rand(100))
        # mean the vectors of individual words to get the vector of the document
        return np.mean(embeddings, axis=0)


def word2vector(train_q, train_a, test_q, test_a, target, f_type):
    df1_q = pd.DataFrame(train_q)
    train_q_ids = df1_q["Id"]
    train_count = len(list(df1_q["Id"]))
    df1_a = pd.DataFrame(train_a)
    train_a_ids = df1_a["Id"]
    train_judges = df1_a["Judge"]
    train_sentences = []
    for i, question in df1_q.iterrows():
        # corpus.append(str(question["Question"]))
        qt = str(question["Question"])
        rel_answers = df1_a[(i * 10):((i * 10) + 10)]
        for j, answer in rel_answers.iterrows():
            # corpus.append(str(answer["Answer"]))
            at = str(answer["Answer"])
            train_sentences.append(qt + " " + at)

    df2_q = pd.DataFrame(test_q)
    test_q_ids = df2_q["Id"]
    test_count = len(list(df2_q["Id"]))
    df2_a = pd.DataFrame(test_a)
    test_a_ids = df2_a["Id"]
    test_judges = df2_a["Judge"]
    test_sentences = []
    for a, question in df2_q.iterrows():
        # corpus.append(str(question["Question"]))
        # print(question)
        qt = str(question["Question"])
        rel_answers = df2_a[(a * 10):((a * 10) + 10)]
        for b, answer in rel_answers.iterrows():
            # corpus.append(str(answer["Answer"]))
            at = str(answer["Answer"])
            test_sentences.append(qt + " " + at)

    train_corpus = [x.split(' ') for x in train_sentences]
    combined_training = train_sentences + test_sentences

    # Creating data for the model training
    train_data = []
    for i in combined_training:
        train_data.append(i.split())

    # Training a word2vec model from the given data set
    w2v_model = Word2Vec(sentences=train_data, vector_size=100, min_count=1, window=50, sg=1, workers=4)

    num_words = len(train_corpus)
    print(num_words)
    max_len = 50

    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(train_sentences)
    train_sequences = tokenizer.texts_to_sequences(train_sentences)

    # print(train_sequences)

    train_padded = pad_sequences(
        train_sequences, maxlen=max_len, truncating="post", padding="post"
    )

    test_sequences = tokenizer.texts_to_sequences(test_sentences)
    test_padded = pad_sequences(
        test_sequences, maxlen=max_len, padding="post", truncating="post"
    )

    word_index = tokenizer.word_index
    print("Number of unique words:", len(word_index))

    embedding_dict = {}
    for tok in tqdm(w2v_model.wv.key_to_index):
        embedding_dict[tok] = w2v_model.wv.get_vector(tok)

    # print(embedding_dict)

    num_words = len(word_index) + 1
    embedding_matrix = np.zeros((num_words, 100))

    for word, i in tqdm(word_index.items()):
        if i < num_words:
            emb_vec = embedding_dict.get(word)
            if emb_vec is not None:
                embedding_matrix[i] = emb_vec

    print((embedding_matrix[270] == embedding_dict.get("reason")).all())

    model = Sequential()

    model.add(
        Embedding(
            num_words,
            100,
            embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
            input_length=max_len,
            trainable=False,
        )
    )
    model.add(LSTM(100, dropout=0.1))
    model.add(Dense(1, activation="sigmoid"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    history = model.fit(
        train_padded,
        train_judges,
        epochs=8,
        # validation_data=(test_padded, test_judges),
        verbose=1,
    )

    sequences = tokenizer.texts_to_sequences(test_sentences)
    padded = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

    pred = model.predict(padded)
    pred_int = pred.round().astype("int")
    print(pred)
    print(pred_int)

    pred_labels = []
    for tag in pred_int:
        if tag[0] == 0:
            pred_labels.append(False)
        if tag[0] == 1:
            pred_labels.append(True)

    pred_prob = []
    for tag in pred:
        pred_prob.append(tag[0])

    out = pd.DataFrame(zip(test_q_ids, test_a_ids, pred_labels, pred_prob))
    out.to_csv('w2v_out.csv')


if __name__ == '__main__':
    questions_train = pd.read_csv('./processed_data/train/lemmatized_question_train')
    answers_train = pd.read_csv('./processed_data/train/lemmatized_answer_train')
    # word2vector(questions_train, answers_train, 'w2v_train', 'train')

    questions_test = pd.read_csv('./processed_data/test/lemmatized_question_test')
    answers_test = pd.read_csv('./processed_data/test/lemmatized_answer_test')
    # word2vector(questions_test, answers_test, 'w2v_test', 'test')

    questions_dev = pd.read_csv('./processed_data/dev/lemmatized_question_dev')
    answers_dev = pd.read_csv('./processed_data/dev/lemmatized_answer_dev')
    word2vector(questions_train, answers_train,questions_test, answers_test, 'w2v_dev', 'dev')
