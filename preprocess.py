#!/usr/bin/env python3

""" Preprocesses the specified dataset and saves the result """

import sys
import csv
import nltk

stopwords = nltk.corpus.stopwords.words('english')
tokenizer = nltk.tokenize.RegexpTokenizer(r"[a-zA-Z]+")
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
stemmer  = nltk.stem.snowball.SnowballStemmer("english")
tagger = nltk.tag.perceptron.PerceptronTagger()
universal = lambda t: nltk.tag.mapping.map_tag('en-ptb', 'universal', t)

def preprocess(comment):
    processed = ""
    tokens = tokenizer.tokenize(comment)
    tags = tagger.tag(tokens)
    for pair in tags:
        token = pair[0].lower()
        if universal(pair[1]) not in ("ADV", "ADP", "CONJ", "PRT") and token not in stopwords and len(pair[0]) > 2:
            token = stemmer.stem(token)
            if processed:
                processed += " "
            processed += token
    return processed

def main():
    fname = "dataset/train.csv"
    if len(sys.argv) > 1:
        fname = sys.argv[1]

    corpus = []
    with open(fname, "r") as f:
        reader = list(csv.reader(f))
        corpus.append(("id", "preprocessed_text"))
        for row in reader[1:]:
            corpus.append((row[0], preprocess(row[1])))
    
    with open(".".join(fname.split(".")[:-1]) + "_preprocessed" + ".csv", "w") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        for item in corpus:
            writer.writerow(item)

if __name__ == "__main__":
    main()
