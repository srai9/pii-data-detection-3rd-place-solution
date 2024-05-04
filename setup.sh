#!/usr/bin/env bash
mkdir datasets
mkdir code/models

kaggle competitions download -c pii-detection-removal-from-educational-data
unzip pii-detection-removal-from-educational-data.zip -d ./datasets/
rm pii-detection-removal-from-educational-data.zip

kaggle datasets download -d nbroad/pii-dd-mistral-generated
unzip pii-dd-mistral-generated.zip -d ./datasets/
rm pii-dd-mistral-generated.zip

kaggle datasets download -d mpware/pii-mixtral8x7b-generated-essays
unzip pii-mixtral8x7b-generated-essays.zip -d ./datasets/
rm pii-mixtral8x7b-generated-essays.zip

kaggle datasets download -d rai555/pii-data-detection-3rd-place-add-data
unzip pii-data-detection-3rd-place-add-data.zip
mv train_folds_strat_5folds.csv ./datasets/
mv exp_pretrain_v1 ./code/models/
rm pii-data-detection-3rd-place-add-data.zip






