# -*- coding: utf-8 -*-
from knn_classifier import KnnClassifier
from sklearn import metrics
from sklearn.model_selection import cross_validate, StratifiedKFold
import numpy as np
import time
import os

def load_data(path):
    data = np.loadtxt(path, delimiter=',', dtype=np.uint8, skiprows=1)
    X = data[:,1:]
    y = data[:,0]
    return X, y

def run_experiment(fname, method, *args, **kwargs):
    full_fname = '{}.npy'.format(fname)
    if os.path.exists(full_fname):
        return np.load(full_fname).item()
    res = method(*args, **kwargs)
    np.save(fname, res)
    return res

def load_experiment(fname):
    full_fname = '{}.npy'.format(fname)
    return np.load(full_fname).item()

def crossval_incrementing_k(X, y, k_range, alpha, cv_k, seed=0, n_iters=3, with_pca=True, quiet=True, times=1):
    # n_iters es la cantidad de iteraciones que se realizan de kfold.
    scoring = {'acc': 'accuracy', 'f1': 'f1_macro'}
    results = {'acc': [], 'f1': [], 'times': []}

    for k in k_range:
        clf = KnnClassifier(k=k, alpha=alpha, with_pca=with_pca, quiet=quiet)
        cv = StratifiedKFold(n_splits=cv_k, random_state=seed)
        cv = list(cv.split(X, y))[:n_iters]

        res_times = []
        for _ in range(times):
            res = cross_validate(clf, X, y, cv=cv, scoring=scoring,
                    return_train_score=False, n_jobs=-1)
            res_times.append(np.mean(res['score_time']))

        results['acc'].append(np.mean(res['test_acc']))
        results['f1'].append(np.mean(res['test_f1']))
        results['times'].append(np.mean(res_times))
        print(k)

    return results
