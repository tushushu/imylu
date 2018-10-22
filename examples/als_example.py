# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-08-21 17:32:24
@Last Modified by:   tushushu
@Last Modified time: 2018-08-21 17:32:24
"""
import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])

import sys
sys.path.append(os.path.abspath(".."))

from imylu.utils import load_movie_ratings
from imylu.recommend.als import ALS


def main():
    print("Tesing the accuracy of ALS...")
    # Load data
    X = load_movie_ratings()
    # Train model
    model = ALS()
    model.fit(X, k=3, max_iter=4)
    # Predictions
    user_ids = range(1, 5)
    predictions = model.predict(user_ids, n_items=2)
    for user_id, prediction in zip(user_ids, predictions):
        print("User id:%d\nrecommedation: %s" % (user_id, prediction))


if __name__ == "__main__":
    main()
