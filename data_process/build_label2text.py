#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
"""
@author: your name eg: Yangst
@file: main.py
@time: 2022/11/26 21:00
@contact:  your email
@desc: "this is a template for pycharm, please setting for python script."
"""
import json

import numpy as np
import pandas as pd
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data', help='input data directory')
parser.add_argument('--output_dir', type=str, default='./data', help='output data directory')
args = parser.parse_args()


def build_label_to_text():
    data1 = pd.read_csv(agrs.data_dir + "/blp23_sentiment_train.tsv", delimiter="\t")
    data2 = pd.read_csv(agrs.data_dir + "/blp23_sentiment_dev.tsv", delimiter="\t")

    all_data = pd.concat([data1, data2])

    print(len(data1), len(data2), len(all_data))

    x = all_data["text"].to_list()
    x = np.array(x)
    y = all_data["label"].to_list()
    y = np.array(y)

    

    label_to_text = dict()
    for _label, _text in zip(y, x):
        _text = re.sub(r"(http|https)://[a-z./0-9A-Z]{1,}", " ", _text)
        _text = re.sub(r"\s+", " ", _text)
        label_to_text.setdefault(_label, set())
        label_to_text[_label].add(_text)
    for k in label_to_text.keys():
        arr = list(label_to_text[k])
        arr.sort()
        label_to_text[k] = arr

    with open(args.output_dir + "/label_to_text.json", "w", encoding="utf8") as f:
        f.write(json.dumps(label_to_text, ensure_ascii=False, indent=4))


if __name__ == '__main__':
    build_label_to_text()
