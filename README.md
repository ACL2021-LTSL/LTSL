# Reinforced Learning to Select for Weakly-supervised Paraphrase Generation (RLTS)

This is the code and data for ACL2021 submission.

## Requirements
transformers >= 3.0

pytorch >= 1.4

python >= 3.6


## Illustation
data.py -- data processing

generate.py -- generate paraphrases using fine-tuned generator

pretrain_selector.py -- pretrain the selector using different ways

run.py -- the main entry of training RLTS

score.py -- evaluate the model performance

select_data.py -- select data using fine-tuned selector


## Usage example

python pretrain_selector_pariwise.py

python run.py -epochs 5 -batch_size 512

python generate.py
