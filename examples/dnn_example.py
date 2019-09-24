"""
@Author: tushushu
@Date: 2019-09-09 14:34:58
"""

import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])

import sys
sys.path.append(os.path.abspath(".."))


from imylu.utils.utils import run_time
from imylu.utils.preprocessing import min_max_scale
from imylu.utils.model_selection import get_r2, train_test_split
from imylu.utils.load_data import load_boston_house_prices
from imylu.neural_network.dnn import DNN


@run_time
def main():
    """Tesing the performance of DNN.
    """
    print("Tesing the performance of DNN....")
    # Load data
    data, label = load_boston_house_prices()
    data = min_max_scale(data)
    # Split data randomly, train set rate 70%
    data_train, data_test, label_train, label_test = train_test_split(
        data, label, random_state=20)
    # Train model
    reg = DNN()
    reg.fit(data=data_train, label=label_train, n_hidden=8,
            epochs=1000, batch_size=8, learning_rate=0.0008)
    # Model evaluation
    get_r2(reg, data_test, label_test)
    print(reg)


if __name__ == "__main__":
    main()
