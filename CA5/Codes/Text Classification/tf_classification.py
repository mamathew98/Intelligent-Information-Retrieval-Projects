import pickle

import numpy as np
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from file_manager import save_object, load_object
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import LinearSVC

nltk.download('stopwords')

movie_data = pd.read_csv("IMDB_Movie_Reviews.csv")
rev, sent = list(movie_data["review"]), list(movie_data["sentiment"])

sent = [int(x) for x in sent]

stemmer = WordNetLemmatizer()
documents = []

for sen in range(0, len(rev)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(rev[sen]))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()

    # Lemmatization
    nltk_tokenizer = RegexpTokenizer(r'\w+')
    tokens = nltk_tokenizer.tokenize(re.sub(r'\d+', '', document))

    document = [stemmer.lemmatize(word) for word in tokens]
    document = ' '.join(document)

    documents.append(document)

print("Vectorizing")
vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
cv = vectorizer.fit_transform(documents).toarray()
vocabulary = vectorizer.get_feature_names()

X_train, X_test, y_train, y_test = train_test_split(cv, sent, test_size=0.2, random_state=0)


mnb_clf = MultinomialNB().fit(X_train, y_train)
y_pred = mnb_clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))


lgr_clf = LogisticRegression().fit(X_train, y_train)
y_pred = lgr_clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

svm_clf = LinearSVC(random_state=0, tol=1e-5).fit(X_train, y_train)
y_pred = svm_clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

