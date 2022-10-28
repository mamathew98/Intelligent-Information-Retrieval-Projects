from gensim.test.utils import common_texts
import pandas as pd

from tools.make_dir import make_dir


def make_corpus(questions, answers, target, f_type):
    corpus = []
    df_q = pd.DataFrame(questions)
    df_a = pd.DataFrame(answers)
    for i, question in df_q.iterrows():
        corpus.append(str(question["Question"]))
        rel_answers = df_a[(i * 10):((i * 10) + 10)]
        for j, answer in rel_answers.iterrows():
            corpus.append(str(answer["Answer"]))

    output_dir = make_dir('corpus', f_type)

    with open(output_dir / '{}.txt'.format(target), 'w', encoding="utf-8") as f:
        for item in corpus:
            f.write("%s\n" % item)


if __name__ == '__main__':
    questions_train = pd.read_csv('./processed_data/train/lemmatized_question_train')
    answers_train = pd.read_csv('./processed_data/train/lemmatized_answer_train')
    make_corpus(questions_train, answers_train, 'corpus_train', 'train')

    questions_test = pd.read_csv('./processed_data/test/lemmatized_question_test')
    answers_test = pd.read_csv('./processed_data/test/lemmatized_answer_test')
    make_corpus(questions_test, answers_test, 'corpus_test', 'test')

    questions_dev = pd.read_csv('./processed_data/dev/lemmatized_question_dev')
    answers_dev = pd.read_csv('./processed_data/dev/lemmatized_answer_dev')
    make_corpus(questions_dev, answers_dev, 'corpus_dev', 'dev')
