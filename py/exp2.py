# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, StratifiedKFold
from knn_classifier import KnnClassifier
from exp import *


# Include path to binaries on PATH variable
bin_path = os.path.join(os.path.dirname('../'))
os.environ['PATH'] = "{}:{}".format(bin_path, os.environ['PATH'])
DATA_PATH = '../data/train.csv'

# Global variables
X, y = load_data(DATA_PATH)

def train_sizes():
    rango_1000 = range(100, 1000, 100)
    rango_10000 = range(1000, 10000, 1000)
    rango_30000 = range(10000, 30000, 2000)
    rango_41990 = range(30000, 41990, 3000)
    rango = list(rango_1000) + list(rango_10000) + list(rango_30000) + list(rango_41990)
    rango.append(41990)
    return rango


# Sets X_train and y_train.
def set_X_y_with_train_size(train_size, test_size=10, seed=0):
    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_size, test_size=test_size, random_state=seed)
    for train_index, _ in sss.split(X, y):
        X_train = X[train_index]
        y_train = y[train_index]
        return X_train, y_train

# Runs a single time knn with data in X_train and y_train.
def run_knn_with_k_alpha(k, alpha, X_train, y_train, cv_k=5, seed=0, n_iters=3):
    scoring = {'acc': 'accuracy', 'f1': 'f1_macro'}
    results = {'acc': [], 'f1': [], 'times': []}
    clf = KnnClassifier(k=k, alpha=alpha)
    cv = StratifiedKFold(n_splits=cv_k, random_state=seed)
    cv = list(cv.split(X_train, y_train))[:n_iters]
    res = cross_validate(clf, X_train, y_train, cv=cv, scoring=scoring, return_train_score=False, n_jobs=-1)
    results['acc'].append(np.mean(res['test_acc']))
    results['f1'].append(np.mean(res['test_f1']))
    results['times'].append(np.mean(res['score_time']))
    return results

# Runs a single time knn with data in X_train and y_train.
def run_knn_with_k_alpha_kfold(k, alpha, X_train, y_train, cv, seed=0, n_iters=3):
    scoring = {'acc': 'accuracy', 'f1': 'f1_macro'}
    results = {'acc': [], 'f1': [], 'times': []}
    clf = KnnClassifier(k=k, alpha=alpha)
    res = cross_validate(clf, X_train, y_train, cv=cv, scoring=scoring, return_train_score=False, n_jobs=-1)
    results['acc'].append(np.mean(res['test_acc']))
    results['f1'].append(np.mean(res['test_f1']))
    results['times'].append(np.mean(res['score_time']))
    return results

def save_array(fname, arr):
    full_fname = '{}.npy'.format(fname)
    np.save(fname, arr)

def run_knn_with_k_alpha_incrementing_train_size(k, alpha, cv=5):
    sizes = train_sizes()
    file_name = "../doc/exp2_k{}_alpha{}_K{}".format(k, alpha, cv)
    results = []
    for size in sizes:
        X_train, y_train = set_X_y_with_train_size(size)
        assert(len(X_train) > 0)
        results.append(run_knn_with_k_alpha(k, alpha, X_train, y_train, cv))
        save_array(file_name, results)

def variate_kFOLD():
    results = []
    fname = "../doc/exp4_variate_KFold_k3_alpha37"
    X_train, y_train = set_X_y_with_train_size(10000)
    for K in range(10, 10000, 500):
        print(K)
        results.append(run_knn_with_k_alpha_kfold(3, 37, X_train, y_train, K))
        save_array(fname, results)
        print("Done with: {}".format(K))

# The experiment per se
def run_experiment_2():
    # run_knn_with_k_alpha_incrementing_train_size(3, 40)
    # run_knn_with_k_alpha_incrementing_train_size(3, 36)
    # run_knn_with_k_alpha_incrementing_train_size(10, 10)
    run_knn_with_k_alpha_incrementing_train_size(3, 40, cv=10)
    run_knn_with_k_alpha_incrementing_train_size(3, 40, cv=15)

# Run it all!
#run_experiment_2()
variate_kFOLD()
