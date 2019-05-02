# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2019-01-09 10:19:31
"""
import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])

import sys
sys.path.append(os.path.abspath(".."))

from imylu.decomposition.pca import PCA
from imylu.utils.model_selection import train_test_split
from numpy.random import randint, seed
from sklearn.decomposition import PCA as _PCA
import numpy as np


def main():
    print("Tesing the performance of PCA...")
    # Generate data
    seed(100)
    X = randint(0, 10, (6, 3))
    X_train, X_test = train_test_split(X)
    # Fit and transform data.
    pca = PCA()
    pca.fit(X, n_components=2)
    print("Imylu PCA:\n", pca.transform(X))
    # Compare result with scikit-learn.
    _pca = _PCA(n_components=2)
    print("Sklearn PCA:\n", _pca.fit(X).transform(X))
    # TODO 为什么会有正负号的问题？


if __name__ == "__main__":
    main()
