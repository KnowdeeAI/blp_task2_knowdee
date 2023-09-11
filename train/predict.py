import os

os.environ["WANDB_DISABLED"] = "true"

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

import torch
import logging
import sys
import pandas as pd
from datasets import Dataset
from normalizer import normalize
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    set_seed,
)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--test_path', type=str, default='./data/blp23_sentiment_dev.tsv', help='path to test file')
parser.add_argument('--model_dir', type=str, default='./output', help='output data directory')
parser.add_argument('--folds', type=int, default=10, help='total number of splits')

parser.add_argument('--model_type', type=str, default='class')
parser.add_argument('--save_path', type=str, default='./output/banglabert_large_pl/final_predict')
parser.add_argument('--batch_size',type=int, default=32)
parser.add_argument('--max_seq_length',type=int, default=128)
parser.add_argument('--icl', action='store_true', default=False)

args.parser.parse_args()

if os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict(fold_index, model_path, save_path, max_seq_length=args.max_seq_length, test_file=args.test_path):
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    icl_file_prefix=''
    if args.icl:
        icl_file_prefix='icl_'


    model_name = '{}/{}fold_{}/'.format(model_name, icl_file_prefix, fold_index)
    print("model path: ", model_name)

    test_df = pd.read_csv(test_file, sep='\t')
    test_df = Dataset.from_pandas(test_df)

    id2l = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    label_list = list(id2l.values())
    label_list.sort()

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
    model = model.to(device)

    sentence1_key = "text"
    
    padding = "max_length"

    def preprocess_function(examples):
        # Tokenize the texts
        # print(examples)
#         for row in examples['text']:
#             print(row)
        cases = [normalize(_) for _ in examples['text']]
        args = (
            (cases,))
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
        return result

    test_df = test_df.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    from torch import nn

    def get_answer(text):
        text = [normalize(x) for x in text]
        inputs = tokenizer(text, return_tensors="pt", max_length=max_seq_length, padding='max_length', truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        final_result = []
        with torch.no_grad():
            outputs = model(**inputs).logits
            outputs = nn.functional.softmax(outputs, -1)
#             print(outputs)
            scores = outputs.tolist()
            indexes = outputs.argmax(-1).tolist()
            for _case_score, _index in zip(scores, indexes):
                _case_score = _case_score[_index]
                final_result.append((_index, _case_score))
#         print(final_result)
        return final_result

    predictions = []
    index, batch_size = 0, 32

    while index < len(test_df['text']):
        predictions.extend(get_answer([x for x in test_df['text'][index:index + batch_size]]))
        index += batch_size

    predictions = [(id2l[_[0]], _[1]) for _ in predictions]

    ids = test_df['id']

    output_predict_file = os.path.join("{}/fold_{}_predict_results.tsv".format(save_path, fold_index))
    with open(output_predict_file, "w") as writer:
        logger.info(f"***** Predict results *****")
        writer.write("id\tlabel\n")
        for index, item in enumerate(predictions):
            writer.write(f"{ids[index]}\t{item}\n")


def predictT5(fold_index, model_path, save_path, max_seq_length=args.max_seq_length, test_file=args.test_path):
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    

    icl_file_prefix=''
    if args.icl:
        icl_file_prefix='icl_'

    model_name = '{}/{}fold_{}/'.format(model_name, icl_file_prefix, fold_index)
    print("model path: ", model_name)

    test_df = pd.read_csv(test_file, sep='\t')
    test_df = Dataset.from_pandas(test_df)

    id2l = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    label_list = list(id2l.values())
    label_list.sort()

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

    model = T5ForConditionalGeneration.from_pretrained(
        model_name,
        from_tf=bool(".ckpt" in model_name),
        config=config,
        cache_dir=None,
        revision="main",
        use_auth_token=None,
        ignore_mismatched_sizes=False,
    )
    model = model.to(device)

    sentence1_key = "text"

    padding = "max_length"
    prefix = "অনুভূতির বিশ্লেষণ"

    def preprocess_function(examples):

        inputs = [prefix + article for article in examples['text']]
        cases = [normalize(_) for _ in inputs]
        
        args = (
            (cases,))
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
        return result
    
    
    

    test_df = test_df.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )
    
    label_map = dict()
    label_map["Positive"] = "ইতিবাচক"
    label_map["Neutral"] = "নিরপেক্ষ"
    label_map["Negative"] = "নেতিবাচক"
    
    label_map = {value:key for key,value in label_map.items()}
    

    from torch import nn

    def get_answer(text):
        text = [normalize(x) for x in text]
        inputs = tokenizer(text, return_tensors="pt", max_length=max_seq_length, padding='max_length', truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        final_result = []
        with torch.no_grad():
            output = model.generate(inputs["input_ids"],
                                    attention_mask=inputs["attention_mask"],
                            decoder_start_token_id=tokenizer.cls_token_id,
                            eos_token_id=tokenizer.sep_token_id,
                            max_length=100).cpu().numpy()
        
            
            for result in output:
                result = tokenizer.decode(result[1:-1])
                result = label_map.get(result, "Neutral")

                
                final_result.append((result, 0.9))

        return final_result

    predictions = []
    index, batch_size = 0, args.batch_size

    while index < len(test_df['text']):
        predictions.extend(get_answer([x for x in test_df['text'][index:index + batch_size]]))
        index += batch_size

    ids = test_df['id']

    output_predict_file = os.path.join(training_args.output_dir, "{}/fold_{}_predict_results.tsv".format(save_path, fold_index))
    with open(output_predict_file, "w") as writer:
        logger.info(f"***** Predict results *****")
        writer.write("id\tlabel\n")
        for index, item in enumerate(predictions):
            writer.write(f"{ids[index]}\t{item}\n")


def vote(save_dir,num=10):
    all_data = []
    id_list = []
    for i in range(num):
        data = pd.read_csv("{}/fold_{}_predict_results.tsv".format(save_dir, i), sep="\t")
        all_data.append(data)
        id_list = data["id"]

    labels = []
    for i in range(len(all_data[0])):
        d = []
        for j in range(num):
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
#         if len(items) == 1 or items[0][1] > 7:
#             print(items)

    d = dict()
    d["id"] = id_list
    d["label"] = labels

    df = pd.DataFrame(data=d, index=None)
    df.to_csv("{}/task.tsv".format(save_dir), sep="\t", index=False)
    
if __name__ == '__main__':
    model_path = args.model_dir
    save_dir = args.output_dir
    k = args.folds
    modelType = "class"
    
    if args.model_type=="class":
        testfile = args.test_path
        for i in range(k):
            predict(i,model_path,save_dir,max_seq_length=args.max_seq_length, test_file=testfile )
    else:
        testfile = args.test_path
        for i in range(k):
            predictT5(i,model_path,save_dir,max_seq_length=args.max_seq_length, test_file=testfile)

    vote(save_dir, num=k)
    evaluate("{}/task.tsv".format(save_dir), "blp23_sentiment_test_with_label.tsv")