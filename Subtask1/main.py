"""Subtask 1 SemEval Calllange
Autors: Rosina Baumann & Sabrina ..."""

import sys
import os
import loaddata

# Defining main function
def main():
    print("Read Data from disk:")
    loaddata.load_trainingdata()
    # evaluate score with scorer-subtask-1.py

# Execute main:
if __name__ == "__main__":
    main()