# Loading and preprocessing of the data:
import sys
import os
import numpy as np

def load_trainingdata(language = "en"):
    # change directory to the current language:
    dir = "../data/" + language
    os.chdir(dir)

    # load labels:
    file = "train-labels-subtask-1.txt"
    np.loadtxt(file)