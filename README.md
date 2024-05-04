# The Learning Agency Lab - PII Data Detection
This repo consists of the 3rd place solution for the PII Data Detection competition hosted by The Learning Agency Lab on Kaggle. The goal of the competition was to develop models that could detect personally identifiable information in student essays. You can read more about the competition [here](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/overview)

## Setup
All the models were trained locally on a single GPU. The specification for the system are below:

- OS: Ubuntu 22.04.2 LTS
- CPU: Intel Core i9 13900K
- GPU: 1 x NVIDIA RTX 3090 
- Python: 3.11.5

The Python packages and the versions used are listed in `requirements.txt`.

## Training

### Datasets

The training requires the competition dataset as well as some additional student essays generated using Mixtral-8x7B by Kaggle users during the competition:
- 2355 student essays with PII generated with Mixtral-8x7B-Instruct-v0.1. Details of this dataset can be found [here.](https://www.kaggle.com/datasets/nbroad/pii-dd-mistral-generated)
- 2692 student essays with PII generated with Mixtral8x7B-Instruct. Details of this dataset can be found [here.](https://www.kaggle.com/datasets/mpware/pii-mixtral8x7b-generated-essays)

### Training

To start the training, please install the Kaggle API and then run `./setup.sh` 

This will create the `./datasets` folder with the required datasets as well as the `./code/models` folder where the final checkpoints will be saved.

To train the final models run the following commands:
 ```
 cd code
 python train.py --fold=3 --config_path='../config/final_v1.yaml
 python train.py --fold=4 --config_path='../config/final_v1.yaml

 ```


The final checkpoints will be saved to the `./code/models` folder. The training uses the pretrained model located in the same folder. If you would like to run the pretraining yourself then use the same training script with the `./datasets/mpware_mixtral8x7b_v1.1.json` dataset and `pretrain_config.yaml`

### Inference
The final trained models are published on Kaggle Datasets and can be found [here](https://www.kaggle.com/datasets/rai555/pii-model-with-mp-pretrain-fulldata). The inference code is also published on Kaggle and can be found here: [https://www.kaggle.com/code/rai555/pii-detection-inference/notebook](https://www.kaggle.com/code/rai555/pii-detection-inference/notebook)

