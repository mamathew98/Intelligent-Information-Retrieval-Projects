import xml.etree.ElementTree as ET
import pandas as pd
import string
import os
from pathlib import Path
from tools.make_dir import make_dir


def parse_questions(file, target, fType):
    tree = ET.parse(file)
    root = tree.getroot()

    ids = []
    questions = []

    for item in root.findall('./Thread/RelQuestion'):

        ids.append(str(item.attrib['RELQ_ID']))

        text = ''
        for q in item:
            if q.text is not None:
                text = text + ' ' + str(q.text)
        text = text.translate(str.maketrans('', '', string.punctuation)).lower()
        questions.append(text)

    results = pd.DataFrame(list(zip(ids, questions)))

    output_dir = make_dir('raw_data', fType)

    results.to_csv(output_dir / target, sep=',', index=None, header=["Id", "Question"])


def parse_answers(file, target, fType):
    tree = ET.parse(file)
    root = tree.getroot()

    ids = []
    labels = []
    answers = []

    for item in root.findall('./Thread/RelComment'):

        ids.append(str(item.attrib['RELC_ID']))
        judge = str(item.attrib['RELC_RELEVANCE2RELQ'])
        if judge == "Good":
            labels.append(True)
        elif judge == "Bad" or judge == "PotentiallyUseful":
            labels.append(False)
        else:
            labels.append(judge)

        text = ''
        for q in item:
            if q.text is not None:
                text = text + ' ' + str(q.text)
        text = text.translate(str.maketrans('', '', string.punctuation)).lower()
        answers.append(text)

    results = pd.DataFrame(list(zip(ids, answers, labels)))

    output_dir = make_dir('raw_data', fType)

    results.to_csv(output_dir / target, sep=',', index=False, header=["Id", "Answer", "Judge"])


if __name__ == '__main__':
    parse_questions("./datasets/train_data.xml", 'question-train.csv', 'train')
    parse_answers("./datasets/train_data.xml", 'answers-train.csv', 'train')
    parse_questions("./datasets/test_data.xml", 'question-test.csv', 'test')
    parse_answers("./datasets/test_data.xml", 'answers-test.csv', 'test')
    parse_questions("./datasets/dev_data.xml", 'question-dev.csv', 'dev')
    parse_answers("./datasets/dev_data.xml", 'answers-dev.csv', 'dev')

