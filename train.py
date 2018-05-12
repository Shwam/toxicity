#!/usr/bin/env python

import pickle
import sklearn
import numpy as np
import scipy
from sklearn import feature_extraction, ensemble, tree, neighbors, model_selection, svm, naive_bayes
import sys
import util

def main():
    feature_extract("train", "test")

def feature_extract(train_path, test_path):
    NUM_FEATURES = 1000
    print("Loading preprocessed data")
    corpus = util.preprocessed(train_path)
    labels = util.labels(train_path)
    corp2 = util.preprocessed(test_path)
    labels2 = util.labels(test_path)
    print("Training")
    
    used = pickle.load(open(".features", "rb"))
    vectorizer = pickle.load(open("vectorizer", "rb"))
    deleted = 0
    for i in range(len(corpus)):
        s = ""
        for w in corpus[i].split(" "):
            if w in used:
                if s:
                    s += " "
                s += w
        corpus[i] = s
        if labels[i - deleted] == "-1":
            del labels[i - deleted]
            corpus[i] = ""
            deleted += 1
        elif not s:
            del labels[i - deleted]
            deleted += 1
    corpus = [c for c in corpus if c != ""]
    X = vectorizer.fit_transform(corpus)
    print("Training")
    
    deleted = 0
    for i in range(len(corp2)):
        s = ""
        for w in corp2[i].split(" "):
            if w in used:
                if s:
                    s += " "
                s += w
        corp2[i] = s
        if labels2[i - deleted] == "-1":
            del labels2[i - deleted]
            corp2[i] = ""
            deleted += 1
        elif not s:
            del labels2[i - deleted]
            deleted += 1
    corp2 = [c for c in corpus if c != ""]
    X2 = vectorizer.transform(corpus)
    print("Training")
    
    clfs = [sklearn.ensemble.RandomForestClassifier(n_estimators=20), svm.SVC(),naive_bayes.GaussianNB()] 
    for clf in clfs:
        clf.fit(X, labels)
        print(clf.score(X2, labels2))
    

if __name__ == "__main__":
    main()
