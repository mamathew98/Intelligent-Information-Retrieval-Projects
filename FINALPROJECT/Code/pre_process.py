import pandas as pd
from tqdm import tqdm
import re
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from tools.make_dir import make_dir


stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()


def pre_process_question(file, f_type):
    df = pd.read_csv(file)
    ids = df["Id"]
    stemmed_documents = []
    lemmatized_documents = []
    no_stem_documents = []
    for idx, line in tqdm(df.iterrows(), total=df.shape[0]):
        document = line["Question"]

        document = re.sub(r'\W', ' ', str(document))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Remove words with len(w) < 4
        document = re.sub(r'\b\w\b', '', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        document = document.lower()

        nltk_tokenizer = RegexpTokenizer(r'\w+')
        tokens = nltk_tokenizer.tokenize(re.sub(r'\d+', '', document))

        no_stem_documents.append(' '.join(tokens))

        document = [stemmer.stem(word) for word in tokens
                    if word not in set(stopwords.words('english')) and word.isalpha()]

        stemmed_documents.append(' '.join(document))

        document = [lemmatizer.lemmatize(word) for word in tokens
                    if word not in set(stopwords.words('english')) and word.isalpha()]

        lemmatized_documents.append(' '.join(document))

    output_dir = make_dir('processed_data', f_type)

    not_stemmed_results = pd.DataFrame(list(zip(ids, no_stem_documents)))\
        .to_csv(output_dir / 'not_stemmed_question_{}'.format(f_type), sep=',', index=False, header=["Id", "Question"])
    stemmed_results = pd.DataFrame(list(zip(ids, stemmed_documents)))\
        .to_csv(output_dir / 'stemmed_question_{}'.format(f_type), sep=',', index=False, header=["Id", "Question"])
    lemmatized_results = pd.DataFrame(list(zip(ids, lemmatized_documents)))\
        .to_csv(output_dir / 'lemmatized_question_{}'.format(f_type), sep=',', index=False, header=["Id", "Question"])


def pre_process_answer(file, f_type):
    df = pd.read_csv(file)
    ids = df["Id"]
    labels = df["Judge"]
    stemmed_documents = []
    lemmatized_documents = []
    no_stem_documents = []
    for idx, line in tqdm(df.iterrows(), total=df.shape[0]):
        document = line["Answer"]

        document = re.sub(r'\W', ' ', str(document))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Remove words with len(w) < 4
        document = re.sub(r'\b\w\b', '', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        document = document.lower()

        nltk_tokenizer = RegexpTokenizer(r'\w+')
        tokens = nltk_tokenizer.tokenize(re.sub(r'\d+', '', document))

        no_stem_documents.append(' '.join(tokens))

        document = [stemmer.stem(word) for word in tokens
                    if word not in set(stopwords.words('english')) and word.isalpha()]

        stemmed_documents.append(' '.join(document))

        document = [lemmatizer.lemmatize(word) for word in tokens
                    if word not in set(stopwords.words('english')) and word.isalpha()]

        lemmatized_documents.append(' '.join(document))

    output_dir = make_dir('processed_data', f_type)

    not_stemmed_results = pd.DataFrame(list(zip(ids, no_stem_documents, labels)))\
        .to_csv(output_dir / 'not_stemmed_answer_{}'.format(f_type), sep=','
                , index=False, header=["Id", "Answer", "Judge"])

    stemmed_results = pd.DataFrame(list(zip(ids, stemmed_documents, labels)))\
        .to_csv(output_dir / 'stemmed_answer_{}'.format(f_type), sep=','
                , index=False, header=["Id", "Answer", "Judge"])

    lemmatized_results = pd.DataFrame(list(zip(ids, lemmatized_documents, labels)))\
        .to_csv(output_dir / 'lemmatized_answer_{}'.format(f_type), sep=','
                , index=False, header=["Id", "Answer", "Judge"])


if __name__ == '__main__':
    pre_process_question('./raw_data/train/question-train.csv', 'train')
    pre_process_question('./raw_data/test/question-test.csv', 'test')
    pre_process_question('./raw_data/dev/question-dev.csv', 'dev')
    pre_process_answer('./raw_data/train/answers-train.csv', 'train')
    pre_process_answer('./raw_data/test/answers-test.csv', 'test')
    pre_process_answer('./raw_data/dev/answers-dev.csv', 'dev')
