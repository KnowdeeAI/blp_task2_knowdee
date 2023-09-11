import os
os.environ["WANDB_DISABLED"] = "true"
import logging
import random
import sys
import pandas as pd
import datasets
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
from adv_trainer import AdvTrainer

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data/10folds/', help='input data directory')
parser.add_argument('--test_path', type=str, default='./data/blp23_sentiment_dev.tsv', help='path to test file')
parser.add_argument('--output_dir', type=str, default='./output', help='output data directory')
parser.add_argument('--model_path', type=str, default='./ptm/csebuetnlp-banglabert_large/', help='path to pretrained model')
parser.add_argument('--folds', type=int, default=10, help='total number of splits')

parser.add_argument('--learning_rate',type=float, default=2e-5)
parser.add_argument('--epoch',type=int, default=15)
parser.add_argument('--train_batch_size',type=int, default=64)
parser.add_argument('--eval_batch_size',type=int, default=64)
parser.add_argument('--max_seq_length',type=int, default=196)
parser.add_argument('--icl', action='store_true', defualt=False)
parser.add_argument('--pseudo_labeling', type=str, defualt='')
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

    train_file = args.data_dir + '/{}fold_{}_train.csv'.format(icl_file_prefix,fold_index)
    validation_file = args.data_dir + '/{}fold_{}_test.csv'.format(icl_file_prefix,fold_index)
    test_file = args.test_file

    print("fold_index: ", fold_index)
    print("train_file: ", train_file)
    print("validation_file: ", validation_file)
    print("test_file: ", test_file)

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

    max_train_samples = None
    max_eval_samples = None
    max_predict_samples = None
    max_seq_length = args.max_seq_length

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

    model_name = args.model_path

    set_seed(training_args.seed)

    l2id = {'Positive': 2, 'Neutral': 1, 'Negative': 0}
    train_df = pd.read_csv(train_file, sep='\t')
    if args.pseudo_labeling:
        ps_data = pd.read_csv(args.pseudo_labeling, sep="\t")
        all_data = dict()
        all_data["text"] = train_df["text"].tolist() + ps_data["text"].tolist()
        all_data["label"] = train_df["label"].tolist() + ps_data["label"].tolist()
        
        train_df = pd.DataFrame(data=all_data, index=None)
       
    train_df['label'] = train_df['label'].map(l2id)
    train_df = Dataset.from_pandas(train_df)
    validation_df = pd.read_csv(validation_file, sep='\t')
    validation_df['label'] = validation_df['label'].map(l2id)
    validation_df = Dataset.from_pandas(validation_df)
    test_df = pd.read_csv(test_file, sep='\t')
    # test_df['label'] = test_df['label'].map(l2id)
    test_df = Dataset.from_pandas(test_df)

    data_files = {"train": train_df, "validation": validation_df, "test": test_df}
    for key in data_files.keys():
        logger.info(f"loading a local file for {key}")
    raw_datasets = DatasetDict(
        {"train": train_df, "validation": validation_df, "test": test_df}
    )

    # Labels
    label_list = raw_datasets["train"].unique("label")
    label_list.sort()  # sort the labels for determine
    num_labels = len(label_list)

    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        finetuning_task=None,
        cache_dir=None,
        revision="main",
        use_auth_token=None,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=None,
        use_fast=True,
        revision="main",
        use_auth_token=None,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        from_tf=bool(".ckpt" in model_name),
        config=config,
        cache_dir=None,
        revision="main",
        use_auth_token=None,
        ignore_mismatched_sizes=False,
    )

    non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
    sentence1_key = non_label_column_names[0]

    # Padding strategy
    padding = "max_length"

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id:
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.", )

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({128}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}.")
    max_seq_length = min(max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        cases = [normalize(_) for _ in examples[sentence1_key]]
        args = (
            (cases,))
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    if "train" not in raw_datasets:
        raise ValueError("requires a train dataset")
    train_dataset = raw_datasets["train"]
    if max_train_samples is not None:
        max_train_samples_n = min(len(train_dataset), max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples_n))

    if "validation" not in raw_datasets:
        raise ValueError("requires a validation dataset")
    eval_dataset = raw_datasets["validation"]
    if max_eval_samples is not None:
        max_eval_samples_n = min(len(eval_dataset), max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples_n))

    if "test" not in raw_datasets and "test_matched" not in raw_datasets:
        raise ValueError("requires a test dataset")
    predict_dataset = raw_datasets["test"]
    if max_predict_samples is not None:
        max_predict_samples_n = min(len(predict_dataset), max_predict_samples)
        predict_dataset = predict_dataset.select(range(max_predict_samples_n))

    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        f1 = f1_score(p.label_ids, preds, average="micro")
        print("f1: ", f1)
        return {"f1": f1}

    data_collator = default_data_collator
    from transformers import EarlyStoppingCallback
    es = EarlyStoppingCallback(early_stopping_patience=5)
    trainer = AdvTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        do_adv=True,
        callbacks=[es]
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

    id2l = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    logger.info("*** Predict ***")
    # predict_dataset = predict_dataset.remove_columns("label")
    ids = predict_dataset['id']
    predict_dataset = predict_dataset.remove_columns("id")
    predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
    predictions = np.argmax(predictions, axis=1)
    output_predict_file = os.path.join(args.output_dir+'/predict/', "fold_{}_predict_results.tsv".format(fold_index))
    if trainer.is_world_process_zero():
        with open(output_predict_file, "w") as writer:
            logger.info(f"***** Predict results *****")
            writer.write("id\tlabel\n")
            for index, item in enumerate(predictions):
                item = label_list[item]
                item = id2l[item]
                writer.write(f"{ids[index]}\t{item}\n")

    kwargs = {"finetuned_from": model_name, "tasks": "text-classification"}
    trainer.create_model_card(**kwargs)

if __name__ == '__main__':
    for i in tqdm(range(args.folds)):
        train(i)