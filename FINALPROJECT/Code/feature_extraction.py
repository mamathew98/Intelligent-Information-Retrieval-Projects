import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from tqdm import tqdm

from measures.bm25 import BM25
from measures.overlap_ratio import OverlappingRatio
from measures.answer_span import AnswerSpan
from measures.same_word_seq import SameWordSeq
from measures.overall_match import OverallMatch
from measures.informativeness import Informativeness
from measures.tf_idf_sums import QueryIDFSum
from measures.tf_idf_sums import AnswerIDFSum
from measures.tf_idf_sums import QueryTFSum
from measures.tf_idf_sums import AnswerTFSum
from tools.make_dir import make_dir
from tools.pickle_manager import save_object


def feature_extraction(questions, answers, target, f_type):
    vectors = []
    df_q = pd.DataFrame(questions)
    q_ids = df_q["Id"]
    count = len(list(df_q["Id"]))
    df_a = pd.DataFrame(answers)
    a_ids = df_a["Id"]
    a_judges = df_a["Judge"]
    corpus = []
    for i, question in df_q.iterrows():
        corpus.append(str(question["Question"]))
        rel_answers = df_a[(i * 10):((i * 10) + 10)]
        for j, answer in rel_answers.iterrows():
            corpus.append(str(answer["Answer"]))

    # print(corpus)
    tfidf_vec = TfidfVectorizer(use_idf=True, ngram_range=(1, 3))
    count_vec = CountVectorizer(ngram_range=(1, 3))

    X = tfidf_vec.fit_transform(corpus)
    Y = count_vec.fit_transform(corpus)

    for i in tqdm(range(count)):
        sel_tfidf = X[(i * 11):((i + 1) * 11)]
        sel_corpus = corpus[(i * 11):((i + 1) * 11)]
        sel_count = Y[(i * 11):((i + 1) * 11)]
        sel_idf = tfidf_vec.idf_

        cos_sim = linear_kernel(sel_tfidf[0:1], sel_tfidf[1:]).flatten()

        bm25 = BM25()
        bm25.fit(sel_corpus[1:])
        bm25_data = bm25.transform(sel_corpus[0], sel_corpus[1:])

        overlap_rat = OverlappingRatio(sel_tfidf.toarray())

        answer_span = AnswerSpan(sel_corpus)

        same_word_seq = SameWordSeq(sel_corpus)

        overall_match = OverallMatch(sel_tfidf.toarray())

        informativeness = Informativeness(sel_tfidf.toarray())

        query_length = 10 * [len(sel_corpus[0])]

        answer_length = [len(x) for x in sel_corpus[1:]]

        question_idf_sum = QueryIDFSum(sel_tfidf.toarray(), sel_idf)

        answer_idf_sum = AnswerIDFSum(sel_tfidf.toarray(), sel_idf)

        question_tf_sum = QueryTFSum(sel_count)

        answer_tf_sum = AnswerTFSum(sel_count)

        query_tfidf_max = 10*[np.amax(sel_tfidf[0])]

        answer_tfidf_max = [np.amax(x) for x in sel_tfidf[1:]]

        query_tf_max = 10*[np.amax(sel_count[0])]

        answer_tf_max = [np.amax(x) for x in sel_count[1:]]

        query_tfidf_mean = 10*[np.mean(sel_tfidf[0])]

        answer_tfidf_mean = [np.mean(x) for x in sel_tfidf[1:]]

        query_tf_mean = 10*[np.mean(sel_count[0])]

        answer_tf_mean = [np.mean(x) for x in sel_count[1:]]

        vectors.extend(list(zip(
            10 * [str(q_ids[i])], a_ids[(i * 10):((i + 1) * 10)],
            bm25_data, cos_sim,
            overlap_rat,
            answer_span, same_word_seq,
            overall_match, informativeness,
            query_length, answer_length,
            question_idf_sum, answer_idf_sum,
            question_tf_sum, answer_tf_sum,
            query_tfidf_max, answer_tfidf_max,
            query_tf_max, answer_tf_max,
            query_tfidf_mean, answer_tfidf_mean,
            query_tf_mean, answer_tf_mean,
            a_judges[(i * 11):((i + 1) * 11)]
        )))

    out = pd.DataFrame(vectors, columns=["QuestionID", "AnswerID",
                                         "BM25", 'CosSim',
                                         "OverlapRat",
                                         "AnswerSpan", "SameWordSeq",
                                         "OverallMatch", "Informative",
                                         "query_length", "answer_length",
                                         "question_idf_sum", "answer_idf_sum",
                                         "question_tf_sum", "answer_tf_sum",
                                         "query_tfidf_max", "answer_tfidf_max",
                                         "query_tf_max", "answer_tf_max",
                                         "query_tfidf_mean", "answer_tfidf_mean",
                                         "query_tf_mean", "answer_tf_mean",
                                         "Judges"])

    output_dir = make_dir('feature_vectors', f_type)

    save_object(out, output_dir / '{}.pickle'.format(target))

    out.to_csv(output_dir / '{}.csv'.format(target), sep=',',
               index=False, header=["QuestionID", "AnswerID",
                                    "BM25", 'CosSim',
                                    "OverlapRat",
                                    "AnswerSpan", "SameWordSeq",
                                    "OverallMatch", "Informative",
                                    "query_length", "answer_length",
                                    "question_idf_sum", "answer_idf_sum",
                                    "question_tf_sum", "answer_tf_sum",
                                    "query_tfidf_max", "answer_tfidf_max",
                                    "query_tf_max", "answer_tf_max",
                                    "query_tfidf_mean", "answer_tfidf_mean",
                                    "query_tf_mean", "answer_tf_mean",
                                    "Judges"])


if __name__ == '__main__':
    questions_train = pd.read_csv('./processed_data/train/stemmed_question_train')
    answers_train = pd.read_csv('./processed_data/train/stemmed_answer_train')
    feature_extraction(questions_train, answers_train, 'features_train', 'train')

    questions_test = pd.read_csv('./processed_data/test/stemmed_question_test')
    answers_test = pd.read_csv('./processed_data/test/stemmed_answer_test')
    feature_extraction(questions_test, answers_test, 'features_test', 'test')

    questions_dev = pd.read_csv('./processed_data/dev/stemmed_question_dev')
    answers_dev = pd.read_csv('./processed_data/dev/stemmed_answer_dev')
    feature_extraction(questions_dev, answers_dev, 'features_dev', 'dev')
