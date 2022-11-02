import pandas as pd
from tqdm import tqdm
import os
import sys
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.metrics import classification_report as report
from sklearn.feature_extraction.text import CountVectorizer
import argparse
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer


def make_dataframe(input_folder, labels_fn=None):
    #MAKE TXT DATAFRAME
    text = []
    for fil in tqdm(filter(lambda x: x.endswith('.txt'), os.listdir(input_folder))):
        iD = fil[7:].split('.')[0]
        lines = list(enumerate(open(input_folder+fil,'r',encoding='utf-8').read().splitlines(),1))
        text.extend([(iD,) + line for line in lines])

    df_text = pd.DataFrame(text, columns=['id','line','text'])
    df_text.id = df_text.id.apply(int)
    df_text.line = df_text.line.apply(int)
    df_text = df_text[df_text.text.str.strip().str.len() > 0].copy()
    df_text = df_text.set_index(['id','line'])
    
    df = df_text

    if labels_fn:
        #MAKE LABEL DATAFRAME
        labels = pd.read_csv(labels_fn,sep='\t',encoding='utf-8',header=None)
        labels = labels.rename(columns={0:'id',1:'line',2:'labels'})
        labels = labels.set_index(['id','line'])
        labels = labels[labels.labels.notna()].copy()

        #JOIN
        df = labels.join(df_text)[['text','labels']]

    return df

def main():
    
    
    parser = argparse.ArgumentParser(description='Subtask-2')
    parser.add_argument('train_folder',  type=str, nargs=1,
                        help='Path to training articles')
    parser.add_argument('dev_folder',  type=str, nargs=1,
                    help='Path to dev articles')
    parser.add_argument('train_labels',  type=str, nargs=1,
                    help='Path to training labels')
    parser.add_argument('-o', "--output",  type=str, nargs=1,
                help='Path to output predictions on dev (mandatory)')
    
    args = parser.parse_args()
    if not args.output:
        print("argument -o is mandatory")
        sys.exit(1)
    
    folder_train = args.train_folder[0]
    folder_dev = args.dev_folder[0]
    labels_train_fn = args.train_labels[0]
    out_fn = args.output[0]
    
    #Read Data
    print('Loading dataset...')
    train = make_dataframe(folder_train, labels_train_fn)
    test = make_dataframe(folder_dev)

    X_train = train['text'].values
    Y_train = train['labels'].fillna('').str.split(',').values
    
    X_test = test['text'].values

    multibin= MultiLabelBinarizer() #use sklearn binarizer
    
    Y_train = multibin.fit_transform(Y_train)
    #Create train-test split
    
    pipe = Pipeline([('vectorizer',CountVectorizer(ngram_range = (1, 2), 
                                               analyzer='word')),
                ('SVM_multiclass', MultiOutputClassifier(svm.SVC(class_weight= None,C=1, kernel='linear'),n_jobs=1))])

    print('Fitting SVM...')
    pipe.fit(X_train,Y_train)

    print('In-sample Acc: \t\t', pipe.score(X_train,Y_train))
    
    Y_pred = pipe.predict(X_test)
    out = multibin.inverse_transform(Y_pred)
    out = list(map(lambda x: ','.join(x), out))
    out = pd.DataFrame(out, test.index)
    out.to_csv(out_fn, sep='\t', header=None)
    print('Results on: ', out_fn)

if __name__ == "__main__":
    main()
