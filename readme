# Data for SemEval 2023 task 3

This Readme is distributed with the data for participating in SemEval 2023 task 3. 
The website of the shared task (https://propaganda.math.unipd.it/semeval2023task3/), includes the submission instructions, updates on the competition and a live leaderboard.


# List of Versions
* __v4.0 [2023/1/14]__ test articles released, including 3 suprise languages: Spanish, Greek, Georgian
* __v3.0 [2023/1/14]__ gold labels for dev set released
* __v2.0 [2022/11/4]__ all languages released
* __v1.0 [2022/9/23]__ data for subtasks 1-3 in English, Italian and Russian released.


## Task Description

**Subtask 1:** Given a news article, determine whether it is an opinion piece, aims at objective news reporting, or if it is a satire piece. This is a multi-class task at article-level.

**Subtask 2:** Given a news article, identify the frames used in the article. This is a multi-label task at article-level.

**Subtask 3:** Given a news article, identify the persuasion techniques in each paragraph. This is a multi-label task at paragraph level.


## Data Format

The input documents are news articles. Each article is contained in its own UTF8-encoded .txt file. 
For the English subtasks, the title is on the first row, followed by an empty row. The content of the article starts from the third row.
The training and dev articles are in the folders ```data/{en,it,fr,ge,ru,pl}/{train,dev}-articles-subtask{1,2,3}```.
For example, the training articles for subtask 3 in English are in the folder ```data/en/train-articles-subtask-3```. 
The name of the files of the input articles have the following structure: article{ID}.txt, where {ID} is a unique numerical identifier (the starting digits of the ID identify the language). For example article111111112.txt

### Input data format

#### Subtask 1 and 2:

As said above, the input for the subtasks are in the folders ```data/{en,it,fr,ge,ru,pl}/{train,dev}-articles-subtask{1,2,3}```, each article is in its own text file. 

#### Subtask 3:

In principle the input for subtask 3 could again be the articles in the folders ```data/{en,it,fr,ge,ru,pl}/{train,dev}-articles-subtask3```.
However, participants are supposed to identify the techniques in each paragraph. In order to make sure the identification of the paragraphs by the participants is consistent with ours, we provide ```.template``` files in the folders ```{train,dev}-labels-subtask-3/*.template``` (and also a single file ```{train,dev}-labels-subtask-3.template```merging the .template files of all articles). There is one such file per article. Each row has three tab-separated columns: the first column is the ID of the article, the second column is the index of the paragraph (starting from 1), the third column is the content of the paragraph. 
Note that, empty lines in the input files are not reported in the .template files and therefore predictions for those lines is not expected. 
For example the text in the file article1234.txt
```
The book is on the table

The table is under the book
```
would result in the following .template file
```
1234	1	The book is on the table
1234	3	The table is under the book
```
In addition, the full list of techniques is in the file ```scorers/techniques_subtask3.txt```, one technique per line (these are the exact strings you are supposed to use in your predictions). If for a language we considered a shorter list, then a corresponding file  ```scorers/techniques_subtask3_en.txt``` (here ```en``` stands for the language) is provided.  


### Prediction Files Format

For all subtasks, a prediction file, for example for the development set, must be one single txt file.

#### Subtask 1

The format of a tab-separated line of the gold label and of a submission files for subtask 1 is:
```
 article_id     label
```	    
where article_id is the numeric id in the name of the input article file (e.g. the id of file article123456.txt is 123456), label is one the strings representing the three categories: reporting, opinion, satire. This is an example of a section of the gold file for the articles with ids 123456 - 123460:
```
123456    opinion
123457    opinion
123458    satire
123459    reporting
123460    satire
```						

#### Subtask 2

The format of a tab-separated line of the gold label and of a submission files for subtask 2 is:
```
 article_id     frame_1,frame_2,...,frame_N
```	    
where article_id is the numeric id in the name of the input article file (e.g. the id of file article123456.txt is 123456), frame_x is one the strings in the txt file ```scorers/frames_subtask2.txt```, one frame per line (these are the exact strings you are supposed to use in your predictions). This is the full list of frames, if for a language we considered a shorter list, then a corresponding file  ```scorers/frames_subtask2_en.txt``` (here ```en``` stands for the language) is provided. 
This is an example of a section of the gold file for the articles with ids 123456 - 123460:
```
  123456    Crime_and_punishment,Policy_prescription_and_evaluation    
  123457    Legality,Constitutionality_and_jurisprudence,Security_and_defense
  123458    Health_and_safety,Quality_of_life,Cultural_identity
  123469    Public_opinion
```		

#### Subtask 3

The format of a tab-separated line of the gold label and of a submission files for subtask 3 is:
```
 article_id		paragraph_id	technique_1,technique_2,...,technique_N
```	    
where article_id is the identifier of the article, paragraph_id is the identifier of the paragraph, technique_1,technique_2,...,technique_N is a comma-separated list of techniques that are present in the paragraph, i.e. the strings in ```scorers/techniques_subtask3_en.txt```, where en specify the language (if there is no such file, the default list is ```scorers/techniques_subtask3.txt```). 
This is an example of a gold file for the article 111111112: 
```
111111112	1	
111111112	3	
111111112	5	Slogans
111111112	7	
111111112	9	
111111112	11	False_Dilemma-No_Choice
111111112	13	
111111112	14	Slogans
111111112	15	Loaded_Language
```		
We provide one single file with the gold labels: ```data/{en,it,ru,po,fr,ge}/{train,dev}-labels-subtask-3.txt```. In addition, we release, for each article, the corresponding gold file in ```data/{en,it,ru,po,fr,ge}/{train,dev}-labels-subtask-3/```: article{ID}-labels-subtask-{N}.txt, where {ID} is the numerical identifier of the corresponding article, and {N} is the index of the subtask. For example data/it/train-labels-subtask-3/article111111112-labels-subtask-3.txt 


## Baselines

You can install all prerequisites with
```
pip3 install -r requirements.txt
```
The command above installs all prerequisites for the scorers as well.

### Subtask 1
By typing
```
cd baselines; 
python3 st1.py -o baseline-output-subtask1-dev-it.txt ../data/it/train-articles-subtask-1/ ../data/it/dev-articles-subtask-1/ ../data/it/train-labels-subtask-1.txt 
```
(notice that the arguments referring to folders, i.e. the second and the third, end in /) you should get the following output
```
Namespace(train_folder=['../data/it/train-articles-subtask-1/'], dev_folder=['../data/it/dev-articles-subtask-1/'], train_labels=['../data/it/train-labels-subtask-1.txt'], output=['baseline-output-subtask1-dev-it.txt'])
Loading training...
226it [00:00, 7694.66it/s]
Loading dev...
77it [00:00, 15127.00it/s]
In-sample Acc: 		 1.0
Results on:  baseline-output-subtask1-dev-it.txt
```

If you submit the file with the predictions of the baseline on the development set, i.e. ```baseline-output-subtask1-dev-it.txt```, to the shared task website, you would get a Macro-F1 score of 0.44609. The Macro-F1 of the baseline for some of the other languages are: 0.25162 (English), 0.34339 (Russian). 

### Subtask 2
By typing
```
cd baselines; python3 st2.py -o baseline-output-subtask2-dev-it.txt ../data/it/train-articles-subtask-2/ ../data/it/dev-articles-subtask-2/ ../data/it/train-labels-subtask-2.txt
```
you should get the following output
```
Namespace(train_folder=['../data/it/train-articles-subtask-2/'], dev_folder=['../data/it/dev-articles-subtask-2/'], train_labels=['../data/it/train-labels-subtask-2.txt'], output=['baseline-output-subtask2-dev-it.txt'])
Loading training...
227it [00:00, 13081.27it/s]
Loading dev...
76it [00:00, 14232.58it/s]
In-sample Acc: 		 1.0
Results on:  baseline-output-subtask2-dev-it.txt
```
If you submit the file with the predictions of the baseline on the development set, i.e. ```baseline-output-subtask2-dev-it.txt```, to the shared task website, you would get a Macro-F1 score of 0.43084. The Macro-F1 of the baseline for some of the other languages are: 0.60545 (English), 0.21583 (Russian). 

### Subtask 3
By typing
```
cd baselines; python3 st3.py -o baseline-output-subtask3-dev-it.txt ../data/it/train-articles-subtask-3/ ../data/it/dev-articles-subtask-3/ ../data/it/train-labels-subtask-3.txt
```
you should get the following output
```
Loading dataset...
227it [00:00, 11514.03it/s]
76it [00:00, 12017.61it/s]
Fitting SVM...
In-sample Acc: 		 0.997134670487106
Results on:  baseline-output-subtask3-dev-it.txt
```
If you submit the file with the predictions of the baseline on the development set, i.e. ```baseline-output-subtask3-dev-it.txt```, to the shared task website, you would get a Macro-F1 score of 0.38918. The Macro-F1 of the baseline for some of the other languages are: 0.16125 (English), 0.25316 (Russian). 


## Scorers and Official Evaluation Metrics

The scorer for the subtasks is located in the ```scorers``` folder.
The scorer will report the official evaluation metric and in some cases additional metrics.

If you have not done so already, you can install all prerequisites with
```
pip install -r requirements.txt
```

### Subtask 1
The official evaluation metric for the task is **macro-F1**. The scorer also reports micro-F1. 

To launch the scorer, run the following commands:
```
cd scorers;
python3 scorer-subtask-1.py --gold_file_path <path_to_gold_labels> --pred_file_path <path_to_your_results_file> --classes_file_path=<path_to_techniques_categories_for_task>
```
For example (here we are assuming we have done training and predict on the training set):
```
cd scorers;
python3 scorer-subtask-1.py -g ../data/it/train-labels-subtask-1.txt -p ../baselines/baseline-output-subtask1-train-it.txt
```

### Subtask 2
The official evaluation metric for the task is **micro-F1**. The scorer also reports macro-F1. 
Run it as
```
cd scorers;
python3 scorer-subtask-2.py --gold_file_path <path_to_gold_labels> --pred_file_path <path_to_your_results_file> --frame_file_path <path_to_frame_list_file>
```
For example:
```
cd scorers;
python3 scorer-subtask-2.py -g ../data/it/train-labels-subtask-2.txt -p ../baselines/baseline-output-subtask2-train-it.txt --frame_file_path frames_subtask2.txt
```

### Subtask 3
The official evaluation metric for the task is **micro-F1**. The scorer also reports macro-F1. 
Run it as
```
cd scorers;
python3 scorer-subtask-2.py --gold_file_path <path_to_gold_labels> --pred_file_path <path_to_your_results_file> --frame_file_path <path_to_frame_list_file>
```
For example (here we are assuming we have done training and predict on the training set):
```
cd scorers;
python3 scorer-subtask-3.py -g ../data/it/train-labels-subtask-3.txt -p ../baselines/baseline-output-subtask3-train-it.txt -f techniques_subtask3.txt
```


## Licensing

The dataset may include content which is protected by copyright of third parties. It may only be used in the context of this shared task, and only for scientific research purposes. The dataset may not be redistributed or shared in part or full with any third party.


## Citation

Besides the SemEval publication that describes the shared task (please cite it if you use the data in your work)

```bibtex
@InProceedings{SemEval2023:task3,
  author    = {TBA},
  title     = {TBA},
  booktitle = {TBA},
  series    = {TBA},
  year      = {TBA},
  url = {TBA},
}
```

the data for the task include the following previous dataset:

* PTC Dataset

```bibtex
@InProceedings{EMNLP19DaSanMartino,
	author = {Da San Martino, Giovanni and
	Yu, Seunghak and
	Barr\'{o}n-Cede\~no, Alberto and
	Petrov, Rostislav and
	Nakov, Preslav},
	title = {Fine-Grained Analysis of Propaganda in News Articles},
	booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP 2019, Hong Kong, China, November 3-7, 2019},
	series = {EMNLP-IJCNLP 2019},
	year = {2019},
	address = {Hong Kong, China},
	month = {November},
}
```



