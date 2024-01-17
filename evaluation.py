#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import os

from common import calculate_f1_max
from sklearn.metrics import roc_auc_score

import csv



def evaluate():
    
    # search the csv files for all classes
    files_in_folder = os.listdir(os.getcwd())
    predictions_csv_files = [file for file in files_in_folder if file.startswith("predictions") and file.endswith(".csv")]

    y_true = list()
    y_score = list()
    print('prediction files on all classes:')
    for file in predictions_csv_files:
        print(file)
        with open(file, 'r') as f:
            csv_reader = csv.reader(f)
            next(csv_reader)
            for row in csv_reader:
                y_true.append(float(row[-2]))
                y_score.append(float(row[-1]))

    # auc and F1 score
    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    f1, threshold = calculate_f1_max(np.array(y_true), np.array(y_score))

    print('Evaluation on all classes of test set, global image classification auc: {:.4f}, F1: {:.4f}, threshold: {:.4f}'.format(auc*100, f1*100, threshold))


if __name__ == '__main__':
    evaluate()
