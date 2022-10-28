import pandas as pd
from tools.pickle_manager import load_object
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, SelectFdr, f_regression
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold
from sklearn.neural_network import MLPRegressor


def re_ranking(train, test):
    X_train = train[[x for x in train.columns if x not in ["QuestionID", "AnswerID", "Judges"]]].to_numpy()
    y_train = list(train["Judges"])

    # X_test = dev[[x for x in dev.columns if x not in ["QuestionID", "AnswerID", "Judges"]]].to_numpy()
    # y_truth = list(dev["Judges"])

    sel1 = VarianceThreshold(threshold=0.01)
    X_train = sel1.fit_transform(X_train)
    print(X_train.shape)
    mask1 = sel1.get_support()

    # sel2 = SelectKBest()
    # X_train = sel2.fit_transform(X_train, y_train)
    # mask2 = sel2.get_support()

    df = pd.DataFrame(test[[x for x in test.columns if x not in ["QuestionID", "AnswerID", "Judges"]]])
    sub_df = df.loc[:, mask1]
    # sub_df = sub_df.loc[:, mask2]

    X_test = sub_df.to_numpy()
    # y_truth = list(test["Judges"])

    clf = MLPClassifier(random_state=1).fit(X_train, y_train)
    print(clf.predict_proba(X_test))

    pred_prob = clf.predict_proba(X_test)

    y_test = clf.predict(X_test)

    # print(list(y_test))

    pred_ = [(x[1] - x[0]) for x in pred_prob]
    # for i in range(len(y_test)):
    #     if y_test[i] == True or y_test[i] == "True":
    #         pred_.append(pred_prob[i][1])
    #     else:
    #         pred_.append(pred_prob[i][0])

    # print(clf.score(X_test, y_truth))
    q_ids = list(test["QuestionID"])
    a_ids = list(test["AnswerID"])

    out = pd.DataFrame(list(zip(q_ids, a_ids, y_test, pred_)))
    out.to_csv('mlp_test.csv')


def test_re_ranking(train, dev):
    X_train = train[[x for x in train.columns if x not in ["QuestionID", "AnswerID", "Judges"]]].to_numpy()
    y_train = list(train["Judges"])

    # X_test = dev[[x for x in dev.columns if x not in ["QuestionID", "AnswerID", "Judges"]]].to_numpy()
    # y_truth = list(dev["Judges"])

    sel1 = VarianceThreshold(threshold=0.01)
    X_train = sel1.fit_transform(X_train)
    print(X_train.shape)
    mask1 = sel1.get_support()

    # sel2 = SelectKBest()
    # X_train = sel2.fit_transform(X_train, y_train)
    # mask2 = sel2.get_support()

    df = pd.DataFrame(dev[[x for x in dev.columns if x not in ["QuestionID", "AnswerID", "Judges"]]])
    sub_df = df.loc[:, mask1]
    # sub_df = sub_df.loc[:, mask2]

    X_test = sub_df.to_numpy()
    y_truth = list(dev["Judges"])

    clf = MLPClassifier(random_state=1).fit(X_train, y_train)
    print(clf.predict_proba(X_test))

    pred_prob = clf.predict_proba(X_test)

    y_test = clf.predict(X_test)

    # print(list(y_test))

    pred_ = [(x[1] - x[0]) for x in pred_prob]
    # for i in range(len(y_test)):
    #     if y_test[i] == True or y_test[i] == "True":
    #         pred_.append(pred_prob[i][1])
    #     else:
    #         pred_.append(pred_prob[i][0])

    print(clf.score(X_test, y_truth))
    q_ids = list(dev["QuestionID"])
    a_ids = list(dev["AnswerID"])

    out = pd.DataFrame(list(zip(q_ids, a_ids, y_test, pred_)))
    out.to_csv('mlp_dev.csv')

    from sklearn.metrics import classification_report
    print('Results on the test set:')
    print(classification_report(y_truth, y_test))

    # mlp_gs = MLPClassifier(max_iter=20, random_state=1)
    # parameter_space = {
    #     'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
    #     'activation': ['tanh', 'relu'],
    #     'solver': ['sgd', 'adam'],
    #     'alpha': [0.0001, 0.05],
    #     'learning_rate': ['constant', 'adaptive'],
    # }
    # from sklearn.model_selection import GridSearchCV
    # clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=5)
    # clf.fit(X_train, y_train)  # X is train samples and y is the corresponding labels
    # print('Best parameters found:\n', clf.best_params_)
    #
    # y_true, y_pred = y_truth, clf.predict(X_test)
    #
    # from sklearn.metrics import classification_report
    # print('Results on the test set:')
    # print(classification_report(y_true, y_pred))


# train_features = load_object('./w2v_vectors/train/w2v_train.pickle')
# test_features = load_object('./w2v_vectors/test/w2v_test.pickle')
# dev_features = load_object('./w2v_vectors/dev/w2v_dev.pickle')

train_features = load_object('./feature_vectors/train/features_train.pickle')
test_features = load_object('./feature_vectors/test/features_test.pickle')
dev_features = load_object('./feature_vectors/dev/features_dev.pickle')
re_ranking(train_features, test_features)
# test_re_ranking(train_features, dev_features)
