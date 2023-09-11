#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
"""
@author: your name eg: Yangst
@file: main.py
@time: 2022/11/26 21:00
@contact:  your email
@desc: "this is a template for pycharm, please setting for python script."
"""
import pandas as pd
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--predict_file_dir', type=str, default='./output', help='path to test file')
parser.add_argument('--test_path', type=str, default='./data/blp23_sentiment_test.tsv', help='path to test file')
parser.add_argument('--output_dir', type=str, default='./output', help='output data directory')
parser.add_argument('--folds', type=int, default=10, help='total number of splits')
parser.add_argument('--threshold', type=float, defualt=8.5)
args.parser.parse_args()




if __name__ == '__main__':
    pass
    all_data = []
    id_list = []
    for i in range(args.folds):
        data = pd.read_csv(args.predict_file_dir + "/fold_{}_predict_results.tsv".format(i), sep="\t")
        all_data.append(data)
        id_list = data["id"]

    text_data = pd.read_csv(args.test_path, sep="\t")["text"]

    labels = []
    s = 0

    with open(args.output_dir+"/pseudo_labelling.csv", "w", encoding="utf8") as f:
        f.write("text\tlabel\n")
        for i in range(len(all_data[0])):
            d = []
            for j in range(10):
                label = all_data[j]["label"][i]
                label = eval(label)
                d.append(label)
            label_freq = dict()
            for _ in d:
                label_freq.setdefault(_[0], 0)
                label_freq[_[0]] += _[1]
            items = list(label_freq.items())
            items.sort(key=lambda x: x[1], reverse=True)
            labels.append(items[0][0])
            if items[0][1] > args.threshold:
                print(items, text_data[i])
                s += 1
                f.write(text_data[i]+"\t"+items[0][0]+"\n")
