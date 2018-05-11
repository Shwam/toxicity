#!/usr/bin/env python3
""" Combines the test data with its labels """

import sys
import csv

def main():
    fname = "dataset/test.csv"
    fname2 = "dataset/test_labels.csv"
    if len(sys.argv) > 1:
        fname = sys.argv[1]

    c1 = None
    with open(fname, "r") as f:
        c1 = list(csv.reader(f))
    c2 = None
    with open(fname2, "r") as f:
        c2 = list(csv.reader(f))

    with open("dataset/combined.csv", "w") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        for i in range(len(c1)):
            writer.writerow(c1[i] + c2[i][1:])
            

if __name__ == "__main__":
    main()
