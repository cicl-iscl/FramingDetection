"""Subtask 1 SemEval Calllange
Autors: Rosina Baumann & Sabrina ..."""

import sys
import os
import loaddata
import pandas as pd
from tqdm import tqdm
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.metrics import classification_report as report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import TransformerMixin, BaseEstimator
import torch


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

Optional = []
Callable = []
List = []
class BertTransformer(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            bert_tokenizer,
            bert_model,
            #embedding_func: None,  # embedding_func: #Optional[Callable[[torch.tensor], torch.tensor]] = None,
            max_length: int = 60,
            embedding_func: Optional[Callable[[torch.tensor], torch.tensor]] = None,
    ):
        self.tokenizer = bert_tokenizer
        self.model = bert_model
        self.model.eval()
        self.max_length = max_length
        self.embedding_func = embedding_func

        if self.embedding_func is None:
            self.embedding_func = lambda x: x[0][:, 0, :]

    def tokenize(self, text: str) -> Tuple[torch.tensor, torch.tensor]:
        # Tokenize the text with the provided tokenizer
        tokenized_text = self.tokenizer.encode_plus(text,
                                                    add_special_tokens=True,
                                                    max_length=self.max_length
                                                    )["input_ids"]

        # Create an attention mask telling BERT to use all words
        attention_mask = [1] * len(tokenized_text)

        # bert takes in a batch so we need to unsqueeze the rows
        return (
            torch.tensor(tokenized_text).unsqueeze(0),
            torch.tensor(attention_mask).unsqueeze(0),
        )

    def tokenize_and_predict(self, text: str) -> torch.tensor:
        tokenized, attention_mask = self.tokenize(text)

        embeddings = self.model(tokenized, attention_mask)
        return self.embedding_func(embeddings)

    def transform(self, text: List[str]):
        if isinstance(text, pd.Series):
            text = text.tolist()

        with torch.no_grad():
            return torch.stack([self.tokenize_and_predict(string) for string in text])

    def fit(self, X, y=None):
        """No fitting necessary so we just return ourselves"""
        return self

# Defining main function
def main():
    print("Read Data from disk:")
    loaddata.load_trainingdata()

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

    # BERT model:
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenized_dict = tokenizer.encode_plus(
        "hi my name is nicolas",
        add_special_tokens=True,
        max_length=5
    )

    bert_model = BertModel.from_pretrained("bert-base-uncased")
    tokenized_text = torch.tensor(tokenized_dict["input_ids"])
    with torch.no_grad():
        embeddings = bert_model(torch.tensor(tokenized_text.unsqueeze(0)))

    bert_transformer = BertTransformer(tokenizer, bert_model)

    classifier = svm.LinearSVC(C=1.0, class_weight="balanced")

    pipe = Pipeline(
        [
            ("vectorizer", bert_transformer),
            ("classifier", classifier),
        ]
    )

    # pipe = Pipeline([('vectorizer', CountVectorizer(ngram_range=(10, 10),
    #                                                  analyzer='char')),
    #                   ('RandomForestClassifier', DecisionTreeClassifier(class_weight='balanced', max_depth=None,
    #                              min_samples_split=2, random_state=0))])
    pipe.fit(X_train, Y_train)

    print('In-sample Acc: \t\t', pipe.score(X_train, Y_train))

    Y_pred = pipe.predict(X_test)

    out = pd.DataFrame(Y_pred, test.index)
    out.to_csv(out_fn, sep='\t', header=None)
    print('Results on: ', out_fn)

# Execute main:
if __name__ == "__main__":
    main()