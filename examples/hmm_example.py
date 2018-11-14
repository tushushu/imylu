# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-10-11 17:58:18
@Last Modified by:   tushushu
@Last Modified time: 2018-10-11 17:58:18
"""

import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])

import sys
sys.path.append(os.path.abspath(".."))

from imylu.probability_model.hmm import HMM
from imylu.utils.load_data import load_tagged_speech
from imylu.utils.model_selection import train_test_split
from imylu.utils.utils import run_time
from itertools import chain


@run_time
def main():
    print("Tesing the performance of HMM...")
    # Load data
    X, y = load_tagged_speech()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=40)
    # Train model
    model = HMM()
    model.fit(X_train, y_train)
    # Model infomation.
    print("There are %d unique observations!" % len(model.observations))
    print("There are %d unique states!" % len(model.states))
    # Correctness test.
    tolerance = 1e-6
    start_probs_check = abs(sum(model.start_probs.values()) - 1) < tolerance
    assert start_probs_check, "Start probs incorrect!"
    trans_probs_check = all(abs(sum(dic.values()) - 1) < tolerance
                            for dic in model.trans_probs.values())
    assert trans_probs_check, "Trans probs incorrect!"
    emit_probs_check = all(abs(sum(dic.values()) - 1) < tolerance
                           for dic in model.emit_probs.values())
    assert emit_probs_check, "Emit probs incorrect!"
    print("Correctness test passed!")
    # Model evaluation.
    predictions = model.predict(X_test)
    numerator = sum(prediction == yi for prediction, yi
                    in zip(chain(*predictions), chain(*y_test)))
    denominator = sum(map(len, y_test))
    acc = numerator / denominator
    print("Test accuracy is %.3f%%!" % (acc * 100))


if __name__ == "__main__":
    main()
