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

    y_trues = list()
    y_scores = list()
    f1_scores = list()
    print('prediction files on all classes:')
    for file in predictions_csv_files:
        print(file)
        with open(file, 'r') as f:
            csv_reader = csv.reader(f)
            next(csv_reader)
            y_true = list()
            y_score = list()
            for row in csv_reader:
                y_trues.append(float(row[-2]))
                y_scores.append(float(row[-1]))

                y_true.append(float(row[-2]))
                y_score.append(float(row[-1]))
            f1, threshold = calculate_f1_max(np.array(y_true), np.array(y_score))
            print('image classification F1: {:.4f}, threshold: {:.4f}'.format(f1*100, threshold))
            f1_scores.append(f1)

    # auc and F1 score
    f1, threshold = calculate_f1_max(np.array(y_true), np.array(y_score))

    print('Evaluation on all classes of test set, average image classification F1: {:.4f}'.format(np.mean(f1_scores)*100))
    print('Evaluation on all classes of test set, global image classification F1: {:.4f}, threshold: {:.4f}'.format(f1*100, threshold))


if __name__ == '__main__':
    evaluate()
