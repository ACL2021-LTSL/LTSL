import pandas as pd
import torch
import os
import pickle
import random
import csv
import numpy as np
import torch.nn.functional as F


def get_batches(iterable, size):
    source = iter(iterable)
    # for _, val in zip(range(size), source):
    #     print(val)
    while True:
        chunk = [val for _, val in zip(range(size), source) if val is not None]
        if not chunk:
            # raise StopIteration
            break
        yield chunk


def load_data_train(args, bert_tokenizer, bart_tokenizer):
    data = pd.read_csv(args.data_path_train, sep='\t', quoting=csv.QUOTE_NONE)
    # label = data['is_perturbed'].values
    data_size = len(data)
    print("Data Size: {}".format(data_size))

    idx_list = list(range(data_size))
    random.shuffle(idx_list)
    idx_batches = list(get_batches(idx_list, int(args.batch_size)))

    if os.path.exists(args.pre_bert) and os.path.exists(args.pre_bart):
        tokenized_text_pair = pickle.load(open(args.pre_bert, 'rb'))
        tokenized_ids_bart = pickle.load(open(args.pre_bart, 'rb'))
        print("Load Pre-computed Training Data!")
    else:
        source = data['question1'].str.strip('"').values
        target = data['question2'].str.strip('"').values
        text_pair = zip(source, target)

        tokenized_text_pair = bert_tokenizer.batch_encode_plus(text_pair, add_special_tokens=True,
                                                               max_length=args.max_seq_len * 2,
                                                               is_pretokenized=False, pad_to_max_length=True,
                                                               return_tensors='pt')
        print("Finish Bert Tokenization!")

        tokenized_source_bart = bart_tokenizer.batch_encode_plus(source, add_special_tokens=True,
                                                                 truncation=True, max_length=args.max_seq_len,
                                                                 is_pretokenized=False, pad_to_max_length=True,
                                                                 return_tensors='pt')
        tokenized_target_bart = bart_tokenizer.batch_encode_plus(target, add_special_tokens=True,
                                                                 truncation=True, max_length=args.max_seq_len,
                                                                 is_pretokenized=False, pad_to_max_length=True,
                                                                 return_tensors='pt')
        print("Finish Bart Tokenization!")

        tokenized_ids_bart = (
        tokenized_source_bart['input_ids'], tokenized_source_bart['attention_mask'], tokenized_target_bart['input_ids'])
        pickle.dump(tokenized_text_pair, open(args.pre_bert, 'wb'))
        pickle.dump(tokenized_ids_bart, open(args.pre_bart, 'wb'))

    return data_size, idx_batches, tokenized_text_pair, tokenized_ids_bart


def load_data_dev(args, bart_tokenizer):
    data = pd.read_csv(args.data_path_dev, sep='\t')
    source = data['question1'].values
    target = data['question2'].values
    inputs = list(get_batches(zip(source, target), int(args.batch_size)))

    batches = []
    for source_target in inputs:
        source, target = zip(*source_target)
        batch_tokenized_source = bart_tokenizer.batch_encode_plus(source, add_special_tokens=True,
                                                                  truncation=True, max_length=args.max_seq_len,
                                                                  is_pretokenized=False, pad_to_max_length=True,
                                                                  return_tensors='pt')
        batch_tokenized_target = bart_tokenizer.batch_encode_plus(target, add_special_tokens=True,
                                                                  truncation=True, max_length=args.max_seq_len,
                                                                  is_pretokenized=False, pad_to_max_length=True,
                                                                  return_tensors='pt')

        batches.append((batch_tokenized_source['input_ids'], batch_tokenized_source['attention_mask'],
                        batch_tokenized_target['input_ids']))

    return batches



