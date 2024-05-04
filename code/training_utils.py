import os
import torch
import random
import numpy as np
import pandas as pd
import datasets
from types import SimpleNamespace
import yaml
from seqeval.metrics import recall_score, precision_score


def seed_everything(seed):
    print(f'Setting seed {seed}')
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_config(config_path):
    cfg = yaml.safe_load(open(config_path).read())
    for k, v in cfg.items():
        if type(v) == dict:
            cfg[k] = SimpleNamespace(**v)
    cfg = SimpleNamespace(**cfg)
    print(cfg)
    return cfg


def compute_metrics(p, all_labels):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [all_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [all_labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    recall = recall_score(true_labels, true_predictions)
    precision = precision_score(true_labels, true_predictions)
    f1_score = (1 + 5*5) * recall * precision / (5*5*precision + recall)
    
    results = {
        'recall': recall,
        'precision': precision,
        'f1': f1_score
    }
    return results


def tokenize_infer(example, tokenizer, max_length):
    text = []; token_map = []
    
    idx = 0
    for t, ws in zip(example["tokens"], example["trailing_whitespace"]):
        text.append(t)
        token_map.extend([idx]*len(t))
        if ws:
            text.append(" ")
            token_map.append(-1)
        idx += 1
            
    tokenized = tokenizer("".join(text), return_offsets_mapping=True, max_length=max_length)
        
    return {
        **tokenized,
        "token_map": token_map,
    }


# https://www.kaggle.com/code/takanashihumbert/piidd-deberta-model-starter-training
def tokenize(example, tokenizer, label2id, max_length, target):
   
    text = []
    targets = []; labels = []   # at character level

    for t, l, ws in zip(example["tokens"], example["provided_labels"], example["trailing_whitespace"]):

        text.append(t)
        labels.extend([l]*len(t))
        
        if l in target:
            targets.append(1)
        else:
            targets.append(0)
        
        if ws:  # if there is trailing whitespace
            text.append(" ")
            labels.append("O")

    tokenized = tokenizer("".join(text), return_offsets_mapping=True, truncation=True, max_length=max_length)
    labels = np.array(labels)
    text = "".join(text)

    token_labels = []
    for start_idx, end_idx in tokenized.offset_mapping:

        if start_idx == 0 and end_idx == 0: # CLS token
            token_labels.append(label2id["O"])
            continue

        if text[start_idx].isspace(): # token starts with a whitespace
            start_idx += 1

        while start_idx >= len(labels):
            start_idx -= 1
        
        token_labels.append(label2id[labels[start_idx]])

    length = len(tokenized.input_ids)

    return {
        **tokenized,
        "labels": token_labels,
        "length": length
    }


def generate_oof(valid_df, preds, tokenizer, model_dir, oof_version, cfg, id2label):
    infer_ds = datasets.Dataset.from_pandas(valid_df)
    infer_ds = infer_ds.map(tokenize_infer, fn_kwargs={"tokenizer": tokenizer, "max_length": cfg.train_params.infer_max_len})
    
    document, token, label, token_str = [], [], [], []
    for p, token_map, offsets, tokens, doc in zip(preds, infer_ds["token_map"], infer_ds["offset_mapping"], infer_ds["tokens"], infer_ds["document"]):

        for token_pred, (start_idx, end_idx) in zip(p, offsets):
            label_pred = id2label[token_pred]

            if start_idx + end_idx == 0: continue

            if token_map[start_idx] == -1: 
                start_idx += 1

            # ignore "\n\n"
            while start_idx < len(token_map) and tokens[token_map[start_idx]].isspace():
                start_idx += 1

            if start_idx >= len(token_map): break

            token_id = token_map[start_idx]

            # ignore "O" predictions and whitespace preds
            if label_pred != "O" and token_id != -1:
                document.append(doc)
                token.append(token_id)
                label.append(label_pred)
                token_str.append(tokens[token_id])

    df = pd.DataFrame({
        "document": document,
        "token": token,
        "label": label,
        "token_str": token_str
    })

    df = df.drop_duplicates()
    df["row_id"] = list(range(len(df)))
    df[["row_id", "document", "token", "label", "token_str"]].to_csv(f"{model_dir}/oof_f{cfg.fold}_{oof_version}.csv", index=False)


