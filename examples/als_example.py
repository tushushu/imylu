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

from imylu.recommend.als import ALS
from imylu.utils.load_data import load_movie_ratings
from imylu.utils.utils import run_time


def format_prediction(item_id, score):
    return "item_id:%d score:%.2f" % (item_id, score)


@run_time
def main():
    print("Tesing the performance of ALS...")
    # Load data
    X = load_movie_ratings()
    # Train model
    model = ALS()
    model.fit(X, k=3, max_iter=10)
    print()

    print("Showing the predictions of users...")
    # Predictions
    user_ids = range(1, 5)
    predictions = model.predict(user_ids, n_items=2)
    for user_id, prediction in zip(user_ids, predictions):
        _prediction = [format_prediction(item_id, score)
                       for item_id, score in prediction]
        print("User id:%d recommedation: %s" % (user_id, _prediction))


if __name__ == "__main__":
    main()
