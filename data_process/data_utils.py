#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
"""
@author: your name eg: Yangst
@file: main.py
@time: 2022/11/26 21:00
@contact:  your email
@desc: "this is a template for pycharm, please setting for python script."
"""
import numpy as np
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data', help='input data directory')
parser.add_argument('--output_dir', type=str, default='./data/10folds', help='output data directory')
parser.add_argument('--folds', type=int, default=10, help='# of split')
args = parser.parse_args()


if __name__ == '__main__':
    import sklearn.model_selection

    if __name__ == '__main__':
        pass
        import pandas as pd

        data1 = pd.read_csv(args.data_dir + "/blp23_sentiment_train.tsv", delimiter="\t")

        all_data = data1

        print("train:",len(data1))

        x = all_data["text"].to_list()
        x = np.array(x)
        y = all_data["label"].to_list()
        y = np.array(y)

        # print(all_data)

        from sklearn.model_selection import StratifiedKFold

        k_folder = StratifiedKFold(n_splits=args.folds)

        fold_index = 0

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        for train_index, test_index in tqdm(k_folder.split(x, y)):
            d = dict()
            d["text"] = x[train_index]
            d["label"] = y[train_index]
            df = pd.DataFrame(data=d)
            df.to_csv(args.output_dir+"/fold_{}_train.csv".format(fold_index), index=False, sep="\t")

            d = dict()
            d["text"] = x[test_index]
            d["label"] = y[test_index]
            df = pd.DataFrame(data=d)
            df.to_csv(args.output_dir+"/fold_{}_test.csv".format(fold_index), index=False, sep="\t")

            fold_index += 1
