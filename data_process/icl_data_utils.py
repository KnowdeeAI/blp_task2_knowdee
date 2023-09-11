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
import random
import re

import numpy as np
import requests
import sklearn.model_selection
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from text2vec import SentenceModel, semantic_search, EncoderType
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data/', help='input data directory')
parser.add_argument('--output_dir', type=str, default='./data/icl_10folds', help='output data directory')
parser.add_argument('--folds', type=int, default=10, help='# of split')
parser.add_argument('--model_path', type=str, default='./ptm/google-muril-large-cased', help='embedding model path')
parser.add_argument('--label_to_text_path', type=str, default='./data/label_to_text.json', help='path to label to text')
args = parser.parse_args()


label_map = dict()
label_map["Positive"] = "ইতিবাচক"
label_map["Neutral"] = "নিরপেক্ষ"
label_map["Negative"] = "নেতিবাচক"

prompt = "{}==>{};"


embedder = SentenceModel(model_name_or_path=args.model_path,
                         encoder_type=EncoderType.FIRST_LAST_AVG)

label_to_embeddings = dict()
with open(args.label_to_text_path, encoding="utf8") as f:
    label_texts = json.loads(f.read())
    print('encoding')
    for item in tqdm(qlabel_texts.items()):
        label, texts = item
        text_embeddings = embedder.encode(texts)
        label_to_embeddings[label] = text_embeddings

def get_entity_relation(query, k):
    query_embedding = embedder.encode(query)

    result = dict()

    for _item in label_to_embeddings.items():
        _label, embeddings = _item
        _text = label_texts[_label]
        all_hits = semantic_search(query_embedding, embeddings, top_k=k)
        for hits in all_hits:
            text = [_text[hit['corpus_id']] for hit in hits]
            result[_label] = text
    return result


def get_similar_cases(_query):
    _new_text = ""
    res = get_entity_relation(_query, 10)
    for _item in res:
        _label, _cases = _item
        _cases = [_ for _ in _cases if _ != _query]
        _label = label_map[_label]
        _cases = [_ for _ in _cases if len(_.split(" ")) < 128]
        _cases = _cases[:1]
        for _case in _cases:
            _new_text += prompt.format(_case, _label)
    return _new_text


if __name__ == '__main__':

    data1 = pd.read_csv(args.data_dir + "/blp23_sentiment_train.tsv", delimiter="\t")
    all_data = data1

    x = all_data["text"].to_list()
    y = all_data["label"].to_list()
    y = [label_map[_] for _ in y]
    y = np.array(y)

    prompt = "{}=>{};"

    label_to_cases = dict()
    for case, label in zip(x, y):
        label_to_cases.setdefault(label, set())
        label_to_cases[label].add(case)

    new_x = []
    i = 0
    for text in x:
        text = re.sub(r"(http|https)://[a-z./0-9A-Z]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        new_text = get_similar_cases(text)
        new_text += "{}=>".format(text)
        new_x.append(new_text)
        i += 1
        if i % 100 == 0:
            print(i)

    x = new_x
    x = np.array(x)
    k_folder = StratifiedKFold(n_splits=args.folds)

    fold_index = 0
    for train_index, test_index in k_folder.split(x, y):
        d = dict()

        d["text"] = x[train_index]
        d["label"] = y[train_index]
        df = pd.DataFrame(data=d)
        df.to_csv(args.output_dir + "/icl_fold_{}_train.csv".format(fold_index), index=False, sep="\t")

        d = dict()
        d["text"] = x[test_index]
        d["label"] = y[test_index]
        df = pd.DataFrame(data=d)
        df.to_csv(args.output_dir + "/icl_fold_{}_test.csv".format(fold_index), index=False, sep="\t")

        fold_index += 1

    df = pd.read_csv(args.data_dir + "/blp23_sentiment_dev.tsv", sep="\t")
    new_x = []
    i = 0
    for text in df["text"]:
        text = re.sub(r"(http|https)://[a-z./0-9A-Z]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        new_text = get_similar_cases(text)
        new_text += "{}=>".format(text)
        new_x.append(new_text)
        i += 1
        if i % 100 == 0:
            print(i)
    df["text"]=new_x
    df.to_csv(args.output_dir + "/icl_blp23_sentiment_dev.csv", index=False, sep="\t")


    df = pd.read_csv(args.data_dir + "/blp23_sentiment_test.tsv", sep="\t")
    new_x = []
    i = 0
    for text in df["text"]:
        text = re.sub(r"(http|https)://[a-z./0-9A-Z]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        new_text = get_similar_cases(text)
        new_text += "{}=>".format(text)
        new_x.append(new_text)
        i += 1
        if i % 100 == 0:
            print(i)
    df["text"]=new_x
    df.to_csv(args.output_dir + "/icl_blp23_sentiment_test.csv", index=False, sep="\t")

