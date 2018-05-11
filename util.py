import csv

def load_metadata(fname):
    reader = []
    with open(fname, "r") as f:
        reader = list(csv.reader(f))
    return reader

def preprocessed(fname):
    if fname[-4:].lower() != ".csv":
        fname = "dataset/" + fname + "_preprocessed.csv"
    return [m[1] for m in load_metadata(fname)]

def labels(fname):
    if fname[-4:].lower() != ".csv":
        fname = "dataset/" + fname + ".csv"
    return [m[1] for m in load_metadata(fname)]
