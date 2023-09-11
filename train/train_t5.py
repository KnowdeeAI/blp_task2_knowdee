#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
"""
@author: your name eg: Yangst
@file: main.py
@time: 2022/11/26 21:00
@contact:  your email
@desc: "this is a template for pycharm, please setting for python script."
"""

import os

os.environ["WANDB_DISABLED"] = "true"

import logging
import random
import sys
import pandas as pd
import datasets
import transformers
from sklearn.metrics import f1_score
import numpy as np
from datasets import Dataset, DatasetDict
from normalizer import normalize
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from torch.utils.data import DataLoader
import torch
from transformers import T5ForConditionalGeneration, AdamW, set_seed

from tqdm import tqdm
import datasets
import transformers

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data/10folds/', help='input data directory')
parser.add_argument('--test_path', type=str, default='./data/blp23_sentiment_dev.tsv', help='path to test file')
parser.add_argument('--output_dir', type=str, default='./output', help='output data directory')
parser.add_argument('--model_path', type=str, default='./ptm/google-mt5-large/', help='path to pretrained model')
parser.add_argument('--folds', type=int, default=10, help='total number of splits')

parser.add_argument('--learning_rate',type=float, default=2e-5)
parser.add_argument('--epoch',type=int, default=15)
parser.add_argument('--train_batch_size',type=int, default=64)
parser.add_argument('--eval_batch_size',type=int, default=64)
parser.add_argument('--max_seq_length',type=int, default=128)
parser.add_argument('--max_target_length', type=int, default=2)
parser.add_argument('--icl', action='store_true', default=False)
args.parser.parse_args()

if os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)


def train(fold_index):
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    icl_file_prefix=''
    if args.icl:
        icl_file_prefix = 'icl_'

    train_file = args.data_dir + '/{}fold_{}_train.csv'.format(icl_file_prefix, fold_index)
    validation_file = args.data_dir + '/{}fold_{}_test.csv'.format(icl_file_prefix,fold_index)
    test_file = args.test_file

    print("fold_index: ", fold_index)
    print("train_file: ", train_file)
    print("validation_file: ", validation_file)
    print("test_file: ", test_file)

    max_train_samples = None
    max_eval_samples = None

    model_name = args.model_path

    if args.icl:
        train_df = pd.read_csv(train_file, sep='\t')
        train_df = Dataset.from_pandas(train_df)

        validation_df = pd.read_csv(validation_file, sep='\t')
        validation_df = Dataset.from_pandas(validation_df)
    else:
        label_map = dict()
        label_map["Positive"] = "ইতিবাচক"
        label_map["Neutral"] = "নিরপেক্ষ"
        label_map["Negative"] = "নেতিবাচক"

        train_df = pd.read_csv(train_file, sep='\t')
        train_df["label"] = [label_map[_] for _ in train_df["label"]]
        train_df = Dataset.from_pandas(train_df)

        validation_df = pd.read_csv(validation_file, sep='\t')
        validation_df["label"] = [label_map[_] for _ in validation_df["label"]]
        validation_df = Dataset.from_pandas(validation_df)

    data_files = {"train": train_df, "validation": validation_df}
    for key in data_files.keys():
        logger.info(f"loading a local file for {key}")
    raw_datasets = DatasetDict(
        {"train": train_df, "validation": validation_df}
    )

    from transformers import AutoTokenizer
    import numpy as np

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prefix = "অনুভূতির বিশ্লেষণ"
    max_input_length = args.max_seq_length
    max_target_length = args.max_target_length

    def preprocess_examples(examples):
        # encode the documents
        articles = examples['text']
        summaries = examples['label']

        inputs = [prefix + article for article in articles]
        inputs = [normalize(_) for _ in inputs]

        model_inputs = tokenizer(inputs, max_length=max_input_length, padding="max_length", truncation=True)

        # encode the summaries
        summaries = [normalize(_) for _ in summaries]
        labels = tokenizer(summaries, max_length=max_target_length, padding="max_length", truncation=True).input_ids

        # important: we need to replace the index of the padding tokens by -100
        # such that they are not taken into account by the CrossEntropyLoss
        labels_with_ignore_index = []
        for labels_example in labels:
            labels_example = [label if label != 0 else -100 for label in labels_example]
            labels_with_ignore_index.append(labels_example)

        model_inputs["labels"] = labels_with_ignore_index

        return model_inputs

    raw_datasets = raw_datasets.map(
        preprocess_examples,
        batched=True,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    print(raw_datasets)

    train_dataset = raw_datasets["train"]
    train_dataset = train_dataset.remove_columns(["text", "label"])
    print(train_dataset)

    eval_dataset = raw_datasets["validation"]
    eval_dataset = eval_dataset.remove_columns(["text", "label"])

    print(eval_dataset)

    train_dataset.set_format(type="torch")
    eval_dataset.set_format(type="torch")

    training_args = TrainingArguments(
        learning_rate=args.learning_rate,
        num_train_epochs=args.epoch,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        output_dir=args.output_dir + "/{}fold_{}/".format(icl_file_prefix,fold_index),
        overwrite_output_dir=True,
        remove_unused_columns=False,
        local_rank=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=3,
        greater_is_better=True,
        metric_for_best_model="f1"
    )

    transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Instantiate the model, let Accelerate handle the device placement.
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    data_collator = default_data_collator

    def compute_metrics(p: EvalPrediction):
        val_pred = []
        pred_results = np.argmax(p.predictions[0], -1)
        for row in pred_results:
            pred = tokenizer.decode(row, skip_special_tokens=True)
            val_pred.append(pred)
        val_true = []
        for _ in p.label_ids:
            tag = tokenizer.decode(_, skip_special_tokens=True)
            val_true.append(tag)
        print(val_pred[:10])
        print(val_true[:10])
        f1 = f1_score(val_true, val_pred, average="micro")
        print("f1: ", f1)
        return {"f1": f1}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    train_result = trainer.train()
    metrics = train_result.metrics
    max_train_samples = (
        max_train_samples if max_train_samples is not None else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.save_model()
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Evaluate ***")

    metrics = trainer.evaluate(eval_dataset=eval_dataset)

    max_eval_samples = (
        max_eval_samples if max_eval_samples is not None else len(eval_dataset)
    )
    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_name, "tasks": "text-classification"}
    trainer.create_model_card(**kwargs)


if __name__ == '__main__':

    for i in range(args.folds):
        train(i)
