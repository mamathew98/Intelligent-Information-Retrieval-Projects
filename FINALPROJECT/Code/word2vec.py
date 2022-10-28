import numpy as np
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from tqdm import tqdm

from tools.make_dir import make_dir
from tools.pickle_manager import save_object
from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm


# Function returning vector reperesentation of a document
def get_embedding_w2v(doc_tokens, w2v_model):
    embeddings = []
    if len(doc_tokens) < 1:
        return np.zeros(300)
    else:
        for tok in doc_tokens:
            if tok in w2v_model.wv.key_to_index:
                embeddings.append(w2v_model.wv.get_vector(tok))
            else:
                embeddings.append(np.random.rand(300))
        # mean the vectors of individual words to get the vector of the document
        return np.mean(embeddings, axis=0)


def word2vector(questions, answers, target, f_type):
    vectors = []
    df_q = pd.DataFrame(questions)
    q_ids = df_q["Id"]
    count = len(list(df_q["Id"]))
    df_a = pd.DataFrame(answers)
    a_ids = df_a["Id"]
    a_judges = df_a["Judge"]
    corpus = []
    c_ =[]
    for i, question in df_q.iterrows():
        corpus.append(str(question["Question"]))
        qt = str(question["Question"])
        rel_answers = df_a[(i * 10):((i * 10) + 10)]
        for j, answer in rel_answers.iterrows():
            corpus.append(str(answer["Answer"]))
            at = str(answer["Answer"])
            c_.append(qt+" "+at)

    # Creating data for the model training
    train_data = []
    for i in c_:
        train_data.append(i.split())

    # Training a word2vec model from the given data set
    w2v_model = Word2Vec(sentences=train_data, vector_size=300, min_count=1, window=100, sg=1, workers=4)
    # print(w2v_model.wv.most_similar('place', topn=10))

    X = [get_embedding_w2v(x.split(), w2v_model) for x in corpus]

    points = []
    for i in tqdm(range(count)):
        sel = X[(i * 11):((i + 1) * 11)]
        # print(sel)
        a = sel[0]
        for b in sel[1:]:
            cos_sim = dot(a, b) / (norm(a) * norm(b))
            points.append(cos_sim)

    out = pd.DataFrame(zip(q_ids, a_ids, points, a_judges), columns=["QuestionID", "AnswerID", "Points", "Judges"])

    output_dir = make_dir('w2v_vectors', f_type)

    save_object(out, output_dir / '{}.pickle'.format(target))

    out.to_csv(output_dir / '{}.csv'.format(target), sep=',', index=False, header=["QuestionID", "AnswerID", "vector", "Judges"])


if __name__ == '__main__':
    questions_train = pd.read_csv('./processed_data/train/lemmatized_question_train')
    answers_train = pd.read_csv('./processed_data/train/lemmatized_answer_train')
    word2vector(questions_train, answers_train, 'w2v_train', 'train')

    questions_test = pd.read_csv('./processed_data/test/lemmatized_question_test')
    answers_test = pd.read_csv('./processed_data/test/lemmatized_answer_test')
    word2vector(questions_test, answers_test, 'w2v_test', 'test')

    questions_dev = pd.read_csv('./processed_data/dev/lemmatized_question_dev')
    answers_dev = pd.read_csv('./processed_data/dev/lemmatized_answer_dev')
    word2vector(questions_dev, answers_dev, 'w2v_dev', 'dev')
