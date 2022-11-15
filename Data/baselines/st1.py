import pandas as pd
from tqdm import tqdm
import os
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.metrics import classification_report as report
from sklearn.feature_extraction.text import CountVectorizer
import argparse



def make_dataframe(input_folder, labels_folder=None):
    #MAKE TXT DATAFRAME
    text = []
    
    for fil in tqdm(filter(lambda x: x.endswith('.txt'), os.listdir(input_folder))):

        iD, txt = fil[7:].split('.')[0], open(input_folder +fil, 'r', encoding='utf-8').read() 
        text.append((iD, txt))

    df_text = pd.DataFrame(text, columns=['id','text']).set_index('id')

    df = df_text

    #MAKE LABEL DATAFRAME
    if labels_folder:
        labels = pd.read_csv(labels_folder, sep='\t', header=None)
        labels = labels.rename(columns={0:'id',1:'type'})
        labels.id = labels.id.apply(str)
        labels = labels.set_index('id')

        #JOIN
        df = labels.join(df_text)[['text','type']]

    return df

def main():
    
    
    parser = argparse.ArgumentParser(description='Subtask-1')
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
    

    #if len(sys.argv) != 4:
    #    parser.print_help()
    #    return
    
    print(args)

    folder_train = args.train_folder[0]
    folder_dev = args.dev_folder[0]
    labels_train_fn = args.train_labels[0]
    out_fn = args.output[0]

    #Read Data
    print('Loading training...')
    train = make_dataframe(folder_train, labels_train_fn)
    print('Loading dev...')
    test = make_dataframe(folder_dev)

    X_train = train['text'].values
    X_test = test['text'].values
    Y_train = train['type'].values

    pipe = Pipeline([('vectorizer',CountVectorizer(ngram_range = (5, 5), 
                                                   analyzer='char')),
                    ('SVM_multiclass', svm.SVC(class_weight= 'balanced', C=0.1, kernel='linear'))])

    pipe.fit(X_train,Y_train)

    print('In-sample Acc: \t\t', pipe.score(X_train,Y_train))
    
    Y_pred = pipe.predict(X_test)
    
    out = pd.DataFrame(Y_pred, test.index)
    out.to_csv(out_fn, sep='\t', header=None)
    print('Results on: ', out_fn)

if __name__ == "__main__":
    main()
