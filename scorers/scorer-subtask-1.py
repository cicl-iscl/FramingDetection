import sys
import logging.handlers
import argparse
import os
from sklearn.metrics import f1_score

"""
Scorer for SemEval 2023 task 3 subtask-1. 
The script checks the format of the prediction file and score it if a gold file is provided. 
This is a 3-class classification problem, we compute the macro F1 over the three classes, as indicated by the list CLASSES below. 
One row of the prediction file has the following format:

article_id|TAB|class

where article_id is the numerical code in the name of the file with the input article, |TAB| is the tab character and class is one the output classes
For example:

111111111    opinion
111111112    opinion
111111113    satire

In addition to Macro-F1, the scorer computes also Micro-F1
"""
CLASSES = ["reporting", "opinion", "satire"]


logger = logging.getLogger("task1_scorer")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)


def _read_csv_input_file(file_full_name):
  """ 
  Read a csv file with two columns TAB separated:
   - first column is the id of the example
   - second column is the label of the example
  """
  with open(file_full_name, encoding='utf-8') as f:
    return dict([ line.rstrip().split("\t") for line in f.readlines()])
  

def _labels_correct(labels, CLASSES, debug=False):
  """
  Make sure all the labels correspond to strings in the CLASSES array
  :param labels: a dictionary with strings as values
  :param CLASSES: a list of allowed labels
  """
  if debug:
    return set(labels.values())-set(CLASSES)
  else:
    return len(set(labels.values())-set(CLASSES))==0
  

def _correct_number_of_examples(pred_labels, gold_labels):
  """
  Make sure that the number of predictions is exactly the same as the gold labels
  """
  return len(pred_labels.keys())==len(gold_labels.keys())


def _correct_id_list(pred_labels, gold_labels, debug=False):
  """
  Check that the list of keys of pred_labels is the same as the gold file
  """
  if debug:
    return ", ".join(set(pred_labels.keys()).symmetric_difference(set(gold_labels.keys())))
  return len(set(pred_labels.keys()).symmetric_difference(set(gold_labels.keys())))==0


def _extract_matching_lists(pred_labels, gold_labels):
  """
  Extract the list of values from the two dictionaries ensuring that elements with the same key are in the same position.
  """
  pred_values, gold_values = ([],[])
  for k in gold_labels.keys():
    pred_values.append(pred_labels[k])
    gold_values.append(gold_labels[k])
  return pred_values, gold_values


def correct_format(pred_labels, gold_labels, CLASSES):
  """
  Check whether the format of the prediction file is correct. 
  The number of checks that can be performed depends on the availability of the gold labels
  """
  if not _labels_correct(pred_labels, CLASSES):
    logger.error('The following labels in the prediction file are not valid: {}.'
                 .format(_labels_correct(pred_labels, CLASSES, True)))
    return False
  if gold_labels: # we can do further checks if the gold_labels are available
    if not _correct_number_of_examples(pred_labels, gold_labels):
      logger.error('The number of predictions (%d) is not the expected one (%d)'
                   %(len(pred_labels.keys()), len(gold_labels.keys())))
      return False
    if not _correct_id_list(pred_labels, gold_labels):
      logger.error('The list of articles ids is not correct. The following ids are not in the gold file: %s'
                   %(_correct_id_list(pred_labels, gold_labels, True)))
      return False
  return True


if __name__ == '__main__':
  
  parser = argparse.ArgumentParser()
  parser.add_argument("--gold_file_path", '-g', type=str, required=False, help="Paths to the file with gold annotations.")
  parser.add_argument("--pred_file_path", '-p', type=str, required=True, help="Path to the file with predictions")
  parser.add_argument("--log_to_file", "-l", action='store_true', default=False,
                      help="Set flag if you want to log the execution file. The log will be appended to <pred_file>.log")
  parser.add_argument('--output-for-script', "-o", dest='output_for_script', required=False, action='store_true',
                      default=False, help="Prints the output in a format easy to parse for a script")
  args = parser.parse_args()

  output_for_script = bool(args.output_for_script)
  if not output_for_script:
    logger.addHandler(ch)
        
  pred_file = args.pred_file_path
  if args.gold_file_path:
    gold_file = args.gold_file_path
  else:
    gold_file = None
  if args.log_to_file:
    output_log_file = pred_file + ".log"
    logger.info("Logging execution to file " + output_log_file)
    fileLogger = logging.FileHandler(output_log_file)
    fileLogger.setLevel(logging.DEBUG)
    fileLogger.setFormatter(formatter)
    logger.addHandler(fileLogger)
    logger.setLevel(logging.DEBUG) #

  if args.log_to_file:
    logger.info('Reading predictions file') 
  else:
    logger.info('Reading predictions file {}'.format(args.pred_file_path))
  if gold_file:
    if args.log_to_file:
      logger.info('Reading gold file')
    else:
      logger.info("Reading gold predictions from file {}".format(args.gold_file_path))
  else:
    logger.info('No gold file provided')

  pred_labels = _read_csv_input_file(pred_file)    
  gold_labels = _read_csv_input_file(gold_file) if gold_file else None
    
  if correct_format(pred_labels, gold_labels, CLASSES):
    logger.info('Prediction file format is correct')
    if gold_labels:
      pred_values, gold_values = _extract_matching_lists(pred_labels, gold_labels)
      macro_f1 = f1_score(gold_values, pred_values, average="macro", zero_division=0)
      micro_f1 = f1_score(gold_values, pred_values, average="micro", zero_division=0)
      logger.info("macro-F1={:.5f}\tmicro-F1={:.5f}".format(macro_f1, micro_f1))
      if output_for_script:
        print("{}\t{}".format(macro_f1, micro_f1))
