import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import transformers as tfs
import argparse
import csv
import os
import pickle


def load_and_generate(device):
    bart_model_class, bart_tokenizer_class, bart_pretrained_weights = (
        tfs.BartForConditionalGeneration, tfs.BartTokenizer, 'facebook/bart-base')
    bart_tokenizer = bart_tokenizer_class.from_pretrained(bart_pretrained_weights)
    bart_model = bart_model_class.from_pretrained(args.save_best_bart)

    source = []
    with open(args.data_path, 'r') as f_in:
        for line in f_in.readlines():
            source.append(line.strip())
    source = np.array(source)
    data_size = len(source)

    print("Data Size: {}".format(data_size))
    print("Finish Data Loading")

    if torch.cuda.device_count() > 1:
        print("Multi-GPU")
        bart_model = nn.DataParallel(bart_model)
    bart_model.to(device)

    if os.path.exists(args.tokenized_bart):
        tokenized_source_bart = pickle.load(open(args.tokenized_bart, 'rb'))
    else:
        tokenized_source_bart = bart_tokenizer.batch_encode_plus(source, add_special_tokens=True,
                                                                 truncation=True, max_length=args.max_seq_len,
                                                                 is_pretokenized=False, pad_to_max_length=True,
                                                                 return_tensors='pt')
        pickle.dump(tokenized_source_bart, open(args.tokenized_bart, 'wb'))

        # tokenized_source_bart['input_ids'], tokenized_source_bart['attention_mask']
        print("Finish Data Tokenization")

    source_ids = tokenized_source_bart['input_ids']
    source_mask = tokenized_source_bart['attention_mask']

    idx_list = list(range(data_size))
    batch_size = 250
    idx_batches = [idx_list[i:i + batch_size] for i in range(0, len(idx_list), batch_size)]

    hypos = []
    for idx in idx_batches:
        source_ids_bactch = source_ids[idx].to(device)
        source_mask_bcatch = source_mask[idx].to(device)

        generated_ids_batch = bart_model.module.generate(input_ids=source_ids_bactch, attention_mask=source_mask_bcatch, num_beams=4,
                                                   max_length=33, early_stopping=True, no_repeat_ngram_size=3)
        hypos_batch = bart_tokenizer.batch_decode(generated_ids_batch, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=True)
        hypos.extend(hypos_batch)

    with open(args.output_path, 'w') as f_out:
        for line in hypos:
            f_out.write(line + "\n")


if __name__ == '__main__':
    #model = 'bm25_100k_top5'
    model = 'bm25_100k_top5_pretrain_clf'
    #model = 'wiki_domain_adapted_rlts' 
    dataset = 'bm25_100k_top5_pretrain_clf'

    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, default=dataset)
    parser.add_argument('-checkpoint', type=str, default=None)
    parser.add_argument('-save_best_bart', type=str, default='checkpoints/{}/bart'.format(model))
    parser.add_argument('-data_path', type=str, default='data/test_20k.source')
    parser.add_argument('-tokenized_bart', type=str, default='data/{}/tokenized_bart_20k.pkl'.format(dataset))
    parser.add_argument('-output_path', type=str, default='output/test_20k_{}.hypo'.format(model))
    parser.add_argument('-pretrain', type=str, default='bert-base-uncased')
    parser.add_argument('-max_seq_len', type=int, default=33)
    args = parser.parse_args()

    my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_and_generate(my_device)
