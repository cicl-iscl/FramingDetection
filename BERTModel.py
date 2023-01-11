#try:
import pandas as pd
from tqdm import tqdm
import numpy as np
import os

import torch
import transformers as ppb  # pytorch transformers

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.metrics import classification_report as report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

import warnings

import swifter

tqdm.pandas()

warnings.filterwarnings('ignore')
#except Exception as e:
#    pass

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

class BertTokenizer(object):

    def __init__(self, text=[]):
        self.text = text

        # For DistilBERT:
        self.model_class, self.tokenizer_class, self.pretrained_weights = (
        ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

        # Load pretrained model/tokenizer
        self.tokenizer = self.tokenizer_class.from_pretrained(self.pretrained_weights)

        self.model = self.model_class.from_pretrained(self.pretrained_weights)

    def get(self):

        df = pd.DataFrame(data={"text": self.text})
        tokenized = df["text"].swifter.apply((lambda x: self.tokenizer.encode(x, add_special_tokens=True)))

        max_len = 0
        for i in tokenized.values:
            if len(i) > max_len:
                max_len = len(i)

        padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])

        attention_mask = np.where(padded != 0, 1, 0)
        input_ids = torch.tensor(padded)
        attention_mask = torch.tensor(attention_mask)

        with torch.no_grad():
            last_hidden_states = self.model(input_ids, attention_mask=attention_mask)

        features = last_hidden_states[0][:, 0, :].numpy()

        return features

def main():
    print("Read Data from disk:")
    #loaddata.load_trainingdata()

    language = "en"
    folder_train = "../Data/data/" + language + "/train-articles-subtask-1/"
    folder_dev = "../Data/data/" + language + "/dev-articles-subtask-1/"
    labels_train_fn = "../Data/data/" + language + "/train-labels-subtask-1.txt"
    out_fn = "resultsBERT/output-subtask-1-dev-" + language + ".txt"

    # Read Data
    print('Loading training...')
    train = make_dataframe(folder_train, labels_train_fn)
    print('Loading dev...')
    test = make_dataframe(folder_dev)

    X_train = train['text'].values
    X_test = test['text'].values
    Y_train = train['type'].values


    encoder = LabelEncoder()
    Y_train = encoder.fit_transform(Y_train)
    #x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3)

    _instance = BertTokenizer(text=X_train)
    tokens = _instance.get()

    #lr_clf = LogisticRegression()
    #lr_clf.fit(tokens, Y_train)

    pipe = Pipeline([('vectorizer', CountVectorizer(ngram_range=(10, 10),
                                                     analyzer='char')),
                      ('RandomForestClassifier', DecisionTreeClassifier(class_weight='balanced', max_depth=None,
                                 min_samples_split=2, random_state=0))])

    pipe.fit(tokens, Y_train)

    print('In-sample Acc: \t\t', pipe.score(X_train, Y_train))

    Y_pred = pipe.predict(X_test)

    out = pd.DataFrame(Y_pred, test.index)
    out.to_csv(out_fn, sep='\t', header=None)
    print('Results on: ', out_fn)

    #_instance = BertTokenizer(text=x_test)
    #tokensTest = _instance.get()

    #predicted = lr_clf.predict(tokensTest)

    #np.mean(predicted == y_test)


if __name__ == "__main__":
    main()

