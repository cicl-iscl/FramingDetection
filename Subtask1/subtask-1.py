import pandas as pd
from tqdm import tqdm
import os
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.metrics import classification_report as report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import argparse


def make_dataframe(input_folder, labels_folder=None):
    # MAKE TXT DATAFRAME
    text = []

    for fil in tqdm(filter(lambda x: x.endswith('.txt'), os.listdir(input_folder))):
        iD, txt = fil[7:].split('.')[0], open(input_folder + fil, 'r', encoding='utf-8').read()
        text.append((iD, txt))

    df_text = pd.DataFrame(text, columns=['id', 'text']).set_index('id')

    df = df_text

    # MAKE LABEL DATAFRAME
    if labels_folder:
        labels = pd.read_csv(labels_folder, sep='\t', header=None)
        labels = labels.rename(columns={0: 'id', 1: 'type'})
        labels.id = labels.id.apply(str)
        labels = labels.set_index('id')

        # JOIN
        df = labels.join(df_text)[['text', 'type']]

    return df

def main():
    path = os.getcwd()
    print(path)
    language = "en"
    folder_train = "../Data/data/" + language + "/train-articles-subtask-1/"
    folder_dev = "../Data/data/" + language + "/dev-articles-subtask-1/"
    labels_train_fn = "../Data/data/" + language + "/train-labels-subtask-1.txt"
    out_fn = "results/output-subtask-1-dev-" + language + ".txt"

    # Read Data
    print('Loading training...')
    train = make_dataframe(folder_train, labels_train_fn)
    print('Loading dev...')
    test = make_dataframe(folder_dev)

    X_train = train['text'].values
    X_test = test['text'].values
    Y_train = train['type'].values

    # bigger wordbag: F1 = 0.15704
    #pipe = Pipeline([('vectorizer', CountVectorizer(ngram_range=(10, 10),
    #                                                  analyzer='char')),
    #                   ('RandomForestClassifier', svm.SVC(class_weight='balanced', C=0.1, kernel='linear'))])

    # bigger wordbag: F1 = 0.12418
    #pipe = Pipeline([('vectorizer', CountVectorizer(ngram_range=(20, 20),
    #                                                 analyzer='char')),
    #                  ('RandomForestClassifier', svm.SVC(class_weight='balanced', C=0.1, kernel='linear'))])

    # tried: RandomForestClassifier -> bad F1= 0.12945
    # pipe = Pipeline([('vectorizer', CountVectorizer(ngram_range=(5, 5),
    #                                                 analyzer='char')),
    #                  ('RandomForestClassifier', RandomForestClassifier(class_weight='balanced', n_estimators=100, max_depth=None,
    #                             min_samples_split=2, random_state=0))])

    # bad: training accuracy just 0.8314
    # pipe = Pipeline([('vectorizer', CountVectorizer(ngram_range=(10, 10),
    #                                                 analyzer='char')),
    #                  ('RandomForestClassifier', AdaBoostClassifier(n_estimators=100, random_state=0))])

    # F1 of decision tree: 0.27652
    pipe = Pipeline([('vectorizer', CountVectorizer(ngram_range=(5, 5),
                                                    analyzer='char')),
                     ('RandomForestClassifier', DecisionTreeClassifier(class_weight='balanced', max_depth=None,
                                                                       min_samples_split=2, random_state=0))])
    # F1 of decision tree: 0.27652
    # F1 of decision tree: 0.38702 (it)
    pipe = Pipeline([('vectorizer', CountVectorizer(ngram_range=(10, 10),
                                                     analyzer='char')),
                      ('RandomForestClassifier', DecisionTreeClassifier(class_weight='balanced', max_depth=None,
                                 min_samples_split=2, random_state=0))])

    # F1 = 0.22499
    # pipe = Pipeline([('vectorizer', CountVectorizer(ngram_range=(20, 20),
    #                                                  analyzer='char')),
    #                   ('RandomForestClassifier', DecisionTreeClassifier(class_weight='balanced', max_depth=None,
    #                              min_samples_split=2, random_state=0))])

    # eliminate balanced == None
    # tried: DecisionTree
    # tried: KNeighborsClassifier

    pipe.fit(X_train, Y_train)

    print('In-sample Acc: \t\t', pipe.score(X_train, Y_train))

    Y_pred = pipe.predict(X_test)

    out = pd.DataFrame(Y_pred, test.index)
    out.to_csv(out_fn, sep='\t', header=None)
    print('Results on: ', out_fn)

#if __name__ == "__main__":
#    main()