import torch
import os
import torch
import numpy as np
import pandas as pd
from scipy.special import softmax
from functools import partial
import argparse

import transformers
import datasets
import tokenizers
from transformers import AutoTokenizer, DataCollatorForTokenClassification, Trainer, TrainingArguments, AutoModelForTokenClassification

from pii_metrics import compute_oof_metrics
from training_utils import seed_everything, compute_metrics, tokenize, get_config, generate_oof
import warnings
warnings.filterwarnings("ignore")

print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(f"Tokenizers version: {tokenizers.__version__}")
print(f"Transformers version: {transformers.__version__}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument('--config_path', type=str)
    arguments = parser.parse_args()
    return arguments

    
def train(fold):

    print('*' * 10)
    seed = cfg.seed + fold
    seed_everything(seed)

    #----- Create Model Directories & Log -----#
    model_dir = os.path.join(cfg.out_dir, f'exp_{cfg.run_name}')
    os.makedirs(model_dir, exist_ok=True)

    print(f'Training fold: {fold} with seed {seed}')
    fold_dir = os.path.join(model_dir, f'fold{fold}')
    os.makedirs(fold_dir, exist_ok=True)

    
    #----- Data Prep -----#
    train = pd.read_json('../datasets/train.json')
    train_folds = pd.read_csv('../datasets/train_folds_strat_5folds.csv')
    train = train.merge(train_folds, on='document', how='left')

    if cfg.train_params.add_newline_token:
        print('Adding new line token to train data')
        train['tokens'] = train.tokens.apply(lambda l:list(map(lambda x: x.replace('\t\r', 'TBBRK'), l)))
        train['tokens'] = train.tokens.apply(lambda l:list(map(lambda x: x.replace('\n\n', 'LBBRK'), l)))
        train['tokens'] = train.tokens.apply(lambda l:list(map(lambda x: x.replace('\n', 'LBBRK2'), l)))


    ext_data = pd.read_json('../datasets/mpware_mixtral8x7b_v1.1.json')
    print(f'Ext data: {ext_data.shape}')

    if cfg.train_params.add_newline_token:
        print('Adding new line token to external data')
        ext_data['tokens'] = ext_data.tokens.apply(lambda l:list(map(lambda x: x.replace('\t\r', 'TBBRK'), l)))
        ext_data['tokens'] = ext_data.tokens.apply(lambda l:list(map(lambda x: x.replace('\n\n', 'LBBRK'), l)))
        ext_data['tokens'] = ext_data.tokens.apply(lambda l:list(map(lambda x: x.replace('\n', 'LBBRK2'), l)))


    train_df = ext_data.copy()
    valid_df = train.copy()
    print(f"Train data: {train_df.shape} Valid data: {valid_df.shape}")
    
    train_df['document'] = train_df['document'].astype(str)
    valid_df['document'] = valid_df['document'].astype(str)


    # ----- Tokenize ----- #
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if cfg.train_params.add_newline_token:
      print('Adding new line token to tokenizer')
      tokenizer.add_tokens([f"TBBRK"], special_tokens=True)
      tokenizer.add_tokens([f"LBBRK"], special_tokens=True)
      tokenizer.add_tokens([f"LBBRK2"], special_tokens=True)
    cfg.train_params.tokenizer_len = len(tokenizer)

    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)

    train_df.rename(columns={'labels':'provided_labels'}, inplace=True)
    valid_df.rename(columns={'labels':'provided_labels'}, inplace=True)
    use_cols = ['document', 'full_text', 'tokens', 'trailing_whitespace', 'provided_labels']
    
    train_ds = datasets.Dataset.from_pandas(train_df[use_cols])
    valid_ds = datasets.Dataset.from_pandas(valid_df[use_cols])

    target = [
        'B-EMAIL', 'B-ID_NUM', 'B-NAME_STUDENT', 'B-PHONE_NUM', 
        'B-STREET_ADDRESS', 'B-URL_PERSONAL', 'B-USERNAME', 'I-ID_NUM', 
        'I-NAME_STUDENT', 'I-PHONE_NUM', 'I-STREET_ADDRESS', 'I-URL_PERSONAL', 'I-USERNAME'
    ]

    all_labels = sorted(target + ['O'])
    label2id = {l: i for i,l in enumerate(all_labels)}
    id2label = {v:k for k,v in label2id.items()}
    print('id2label', id2label)
    
    train_ds = train_ds.map(tokenize, fn_kwargs={"tokenizer": tokenizer, "label2id": label2id, "max_length": cfg.train_params.train_max_len, "target": target})
    valid_ds = valid_ds.map(tokenize, fn_kwargs={"tokenizer": tokenizer, "label2id": label2id, "max_length": cfg.train_params.valid_max_len, "target": target})

    # ----- Check Dataset -----#
    x = train_ds[0]
    for t, l in zip(x["tokens"], x["provided_labels"]):
        if l != "O": print((t,l))

    print("*"*100)
    for t, l in zip(tokenizer.convert_ids_to_tokens(x["input_ids"]), x["labels"]):
        if id2label[l] != "O": print((t, id2label[l]))
    print(tokenizer.convert_ids_to_tokens(x["input_ids"]))

    
    # ----- Model & Training -----#
    if cfg.ext_data_pretrain:
        print(f'Loading pretrained model {cfg.ext_data_model_dir}')
        model = AutoModelForTokenClassification.from_pretrained(
            cfg.ext_data_model_dir,
            num_labels=len(all_labels),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            cfg.base_model,
            num_labels=len(all_labels),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )

    if cfg.train_params.freeze_layers>0:
        print(f'Freezing {cfg.train_params.freeze_layers} layers.')
        for layer in model.deberta.encoder.layer[:cfg.train_params.freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    if cfg.train_params.add_newline_token:
        model.resize_token_embeddings(len(tokenizer))

    args = TrainingArguments(
        output_dir=fold_dir, 
        fp16=True,
        dataloader_num_workers = cfg.train_params.num_workers,
        learning_rate=cfg.train_params.lr,
        num_train_epochs=cfg.train_params.num_epochs,
        per_device_train_batch_size=cfg.train_params.train_batch_size,
        per_device_eval_batch_size=cfg.train_params.valid_batch_size,
        gradient_accumulation_steps = cfg.train_params.grad_accum,
        report_to=None,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        lr_scheduler_type=cfg.train_params.scheduler,
        greater_is_better=True,
        warmup_ratio=cfg.train_params.lr_warmup_pct,
        weight_decay=cfg.train_params.weight_decay,
        logging_steps = cfg.train_params.logging_steps,
        eval_accumulation_steps = 100
    )
    
    trainer = Trainer(
        model=model, 
        args=args, 
        train_dataset=train_ds, 
        eval_dataset=valid_ds, 
        data_collator=data_collator, 
        tokenizer=tokenizer,
        compute_metrics=partial(compute_metrics, all_labels=all_labels),
    )

    trainer.train()
    torch.cuda.empty_cache()

    # ----- Predict and Base OOF -----#
    raw_preds = trainer.predict(valid_ds).predictions
    raw_preds = softmax(raw_preds, axis = -1)
    preds = raw_preds.argmax(-1)
    generate_oof(valid_df[use_cols], preds, tokenizer, model_dir, 1, cfg, id2label)
    

    # ----- OOF with Pre Proc -----#
    preds_without_O = raw_preds[:,:,:13].argmax(-1)
    O_preds = raw_preds[:,:,13]
    threshold = cfg.train_params.pred_thresh
    preds2 = np.where(O_preds < threshold, preds_without_O , preds)
    generate_oof(valid_df[use_cols], preds2, tokenizer, model_dir, 2, cfg, id2label)


    # ----- Ground Truth -----#
    valid_df.rename(columns={'provided_labels':'labels'}, inplace=True)

    gt_docs = []; gt_token_nums = []; gt_labels = []
    for i, row in valid_df.iterrows():
        for j, label in enumerate(row['labels']):
            if label!='O':
                gt_docs.append(row.document)
                gt_token_nums.append(j)
                gt_labels.append(label)
    
    gt_df = pd.DataFrame(gt_docs, columns=['document'])
    gt_df['token'] = gt_token_nums
    gt_df['label'] = gt_labels

    oof = pd.read_csv(f"{model_dir}/oof_f{cfg.fold}_1.csv")
    oof2 = pd.read_csv(f"{model_dir}/oof_f{cfg.fold}_2.csv")

    oof['document'] = oof['document'].astype(str)
    oof2['document'] = oof2['document'].astype(str)
    gt_df['document'] = gt_df['document'].astype(str)
    
    score = compute_oof_metrics(oof, gt_df)['ents_f5']
    score2 = compute_oof_metrics(oof2, gt_df)['ents_f5']

    print(f'Fold {fold}: {score:.3f}')
    print(f'Fold {fold}: {score2:.3f} (with preproc {threshold})')


# ----- Run Training -----#
if __name__ == '__main__':
    
    args = parse_arguments()
    cfg = get_config(args.config_path)
    cfg.fold = args.fold
    train(cfg.fold)
    

    

