# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-11-06 11:15:25
@Last Modified by:   tushushu
@Last Modified time: 2018-11-06 11:15:25
"""
import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])

import sys
sys.path.append(os.path.abspath(".."))

from imylu.cluster.kmeans import KMeans
from imylu.utils.load_data import load_breast_cancer
from imylu.utils.preprocessing import min_max_scale
from imylu.utils.utils import run_time


@run_time
def main():
    print("Tesing the performance of Kmeans...")
    # Load data
    X, y = load_breast_cancer()
    X = min_max_scale(X)
    # Train model
    est = KMeans()
    k = 2
    est.fit(X, k)
    print()
    # Model performance
    prob_pos = sum(y) / len(y)
    print("Positive probability of X is:%.1f%%.\n" % (prob_pos * 100))
    y_hat = est.predict(X)
    cluster_pos_tot_cnt = {i: [0, 0] for i in range(k)}
    for yi_hat, yi in zip(y_hat, y):
        cluster_pos_tot_cnt[yi_hat][0] += yi
        cluster_pos_tot_cnt[yi_hat][1] += 1
    cluster_prob_pos = {k: v[0] / v[1] for k, v in cluster_pos_tot_cnt.items()}
    for i in range(k):
        tot_cnt = cluster_pos_tot_cnt[i][1]
        prob_pos = cluster_prob_pos[i]
        print("Count of elements in cluster %d is:%d." %
              (i, tot_cnt))
        print("Positive probability of cluster %d is:%.1f%%.\n" %
              (i, prob_pos * 100))


if __name__ == "__main__":
    main()
