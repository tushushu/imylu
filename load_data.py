import os


def load_breast_cancer():
    path = os.path.join(os.getcwd(), "dataset", "breast_cancer.csv")
    f = open(path)
    X = []
    y = []
    for line in f:
        line = line[:-1].split(",")
        xi = [float(s) for s in line[:-1]]
        yi = int(line[-1])
        X.append(xi)
        y.append(yi)
    return X, y
