#!/usr/bin/env python

import pickle
import sklearn
import numpy as np
import scipy
from sklearn import feature_extraction, ensemble, tree, neighbors, model_selection
import sys
import util

def main():
    test("test")

def test(test_path):
    print("Loading preprocessed data")
    #vectorizer = sklearn.feature_extraction.text.TfidfVectorizer()
    corpus = util.preprocessed(test_path)
    labels = util.labels(test_path)
    print("loading featues")
    used = pickle.load(open(".features", "rb"))
    vectorizer = pickle.load(open("vectorizer", "rb"))
    print("processing corpus")
    for i in range(len(corpus)):
        s = ""
        for w in corpus:
            if w in used:
                if s:
                    s += " "
                s += w
        corpus[i] = s
        if not s:
            del labels[i]
    corpus = [c for c in corpus if c]
    X = vectorizer.transform(corpus)
    del corpus
    print("Testing")
    clf = pickle.load(open("model", "rb"))
    print(clf.score(X, labels))
if __name__ == "__main__":
    main()
