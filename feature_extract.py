#!/usr/bin/env python

import pickle
import sklearn
import numpy as np
import scipy
from sklearn import feature_extraction, ensemble, tree, neighbors, model_selection
import sys
import util

def main():
    feature_extract("train", "test")

def feature_extract(train_path, test_path):
    print("Loading preprocessed data")
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer()
    corpus = util.preprocessed(train_path)
    labels = util.labels(train_path)
    X = vectorizer.fit_transform(corpus)
    del corpus
    print("Training")
    clf = sklearn.ensemble.RandomForestClassifier(n_estimators=20)
    clf = clf.fit(X, labels)
    print("Analyzing")
    names = vectorizer.get_feature_names()
    
    # Random forest importances
    importances = dict(zip(names, np.multiply(clf.feature_importances_, 1.0/np.max(clf.feature_importances_))))

    # Frequency of word
    frequencies = (X.sum(axis=0))
    frequencies = np.squeeze(np.asarray(frequencies))
    maxfreq = np.max(frequencies)
    frequencies = np.multiply(frequencies, 1.0/maxfreq) 
    frequencies = {names[i]:frequencies[i] for i in range(len(names))} 

    # Select 100 most relevant features 
    names.sort(key=lambda name: -0.2*frequencies[name] +  -0.8*importances[name])
    for i in range(100):
        print(names[i], importances[names[i]], frequencies[names[i]])
    feature_names = names[:100]
    
    names = vectorizer.get_feature_names()
    mat = X.toarray()
    feature_vector = np.array([[] for _ in range(len(labels))])
    used_indices = []
    name_indices = [names.index(name) for name in feature_names]

    print("Converting to term frequency matrix for 100 best terms")    
    # Remove samples which have no features and grab top 100 terms
    import time
    for i in range(len(mat)):
        sample = []
        for j in range(100):
            sample.append(mat[i][name_indices[j]])
        if sum(1 for k in sample if k > 0) < 10:
            continue
        used_indices.append(i)
        feature_vector = np.append(feature_vector, sample)
    feature_vector.shape = (len(used_indices), 100)
    tf_matrix = scipy.sparse.csr_matrix(feature_vector, shape=(len(used_indices), 100))
    tf_matrix = sklearn.preprocessing.normalize(tf_matrix, norm='l1', axis=1)
    pickle.dump(tf_matrix, open("tfm.csr_matrix", "wb"))
    pickle.dump(feature_names, open(".features", "wb"))
    pickle.dump(used_indices, open(".indices", "wb"))

if __name__ == "__main__":
    main()

