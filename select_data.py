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


def remove_low_value_data(device):
    # load saved model
    model_class, tokenizer_class, pretrained_weights = (
        tfs.AutoModelForSequenceClassification, tfs.AutoTokenizer, 'textattack/bert-base-uncased-QQP')
    # model_class, tokenizer_class, pretrained_weights = (
    #     tfs.BertForSequenceClassification, tfs.BertTokenizer, args.pretrain)
    bert_tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    # bert_model = model_class.from_pretrained(args.save_best_bert)
    bert_model = model_class.from_pretrained(pretrained_weights)
    print("Finish Model Loading")

    if torch.cuda.device_count() > 1:
        print("Multi-GPU")
        bert_model = nn.DataParallel(bert_model)
    bert_model.to(device)

    # read data
    data = pd.read_csv(args.data_path, sep='\t', quoting=csv.QUOTE_NONE)
    source = data['question1'].str.strip('"').values
    target = data['question2'].str.strip('"').values
    ground_truth = data['ground_truth'].str.strip('"').values

    print("Data Size: {}".format(data.shape))
    print("Finish Data Loading")

    if os.path.exists(args.pre_bert):
        data_values_all = pickle.load(open(args.pre_bert, 'rb'))
    else:
        text_pair = zip(source, target)
        tokenized_text_pair = bert_tokenizer.batch_encode_plus(text_pair, add_special_tokens=True, truncation=True,
                                                               max_length=args.max_seq_len * 2,
                                                               is_pretokenized=False, pad_to_max_length=True,
                                                               return_tensors='pt')
        pickle.dump(tokenized_text_pair, open(args.pre_bert, 'wb'))

        text_pair_ids = tokenized_text_pair['input_ids']
        mask_bert = tokenized_text_pair['attention_mask']
        token_type_ids = tokenized_text_pair['token_type_ids']
        print("Finish Data Tokenization")

        # compute final data value
        data_size = len(source)
        idx_list = range(data_size)
        idx_batches = [idx_list[i:i + 1000] for i in range(0, len(idx_list), 1000)]
        data_values_all = []
        with torch.no_grad():
            for batch_idx in idx_batches:
                text_pair_ids_batch = text_pair_ids[batch_idx].to(device)
                mask_bert_batch = mask_bert[batch_idx].to(device)
                token_type_ids_batch = token_type_ids[batch_idx].to(device)
                outputs_batch = bert_model(text_pair_ids_batch, attention_mask=mask_bert_batch,
                                           token_type_ids=token_type_ids_batch)
                data_values = F.softmax(outputs_batch[0], dim=-1)
                data_values = data_values[:, 1].detach().cpu().numpy().tolist()
                data_values_all.extend(data_values)

        data_values_all = np.array(data_values_all)
        pickle.dump(data_values_all, open(args.pre_bert, 'wb'))

    data_values_all = np.reshape(data_values_all, (-1, ))
    idx_sorted = np.argsort(data_values_all)
    divide = 10

    for itt in range(divide):
        source_new = source[idx_sorted[int(itt * len(source) / divide):]]
        target_new = target[idx_sorted[int(itt * len(target) / divide):]]
        ground_truth_new = ground_truth[idx_sorted[int(itt * len(target) / divide):]]
        data_value = data_values_all[idx_sorted[int(itt * len(source) / divide):]]

        with open('{}/remove_{:.2f}.txt'.format(args.output_path, itt/divide), 'w') as f:
            for i in range(len(source_new)):
                f.write(source_new[i] + "\t" + target_new[i] + "\t" + ground_truth_new[i] + "\t" + str(data_value[i]) + "\n")
    print("Finish")


def remove_low_value_data_2(device):
    """select topK for each query"""
    step = 5

    # load saved model
    #model_class, tokenizer_class, pretrained_weights = (
    #    tfs.AutoModelForSequenceClassification, tfs.AutoTokenizer, 'textattack/bert-base-uncased-QQP')
    model_class, tokenizer_class, pretrained_weights = (
        tfs.BertForSequenceClassification, tfs.BertTokenizer, args.pretrain)
    bert_tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    #bert_model = model_class.from_pretrained(pretrained_weights)
    bert_model = model_class.from_pretrained(args.save_best_bert)
    print("Finish Model Loading")

    if torch.cuda.device_count() > 1:
        print("Multi-GPU")
        bert_model = nn.DataParallel(bert_model)
    bert_model.to(device)

    # read data
    data = pd.read_csv(args.data_path, sep='\t', quoting=csv.QUOTE_NONE)
    source = data['question1'].str.strip('"').values
    target = data['question2'].str.strip('"').values
    ground_truth = data['ground_truth'].str.strip('"').values
    data_size = len(source)
    print("Data Size: {}".format(data_size))
    print("Finish Data Loading")

    if os.path.exists(args.pre_bert):
        data_values_all = pickle.load(open(args.pre_bert, 'rb'))
    else:
        text_pair = zip(source, target)
        tokenized_text_pair = bert_tokenizer.batch_encode_plus(text_pair, add_special_tokens=True, truncation=True,
                                                               max_length=args.max_seq_len * 2,
                                                               is_pretokenized=False, pad_to_max_length=True,
                                                               return_tensors='pt')

        text_pair_ids = tokenized_text_pair['input_ids']
        mask_bert = tokenized_text_pair['attention_mask']
        token_type_ids = tokenized_text_pair['token_type_ids']
        print("Finish Data Tokenization")

        # compute final data value
        idx_list = range(data_size)
        idx_batches = [idx_list[i:i + 1000] for i in range(0, len(idx_list), 1000)]
        data_values_all = []
        with torch.no_grad():
            for batch_idx in idx_batches:
                text_pair_ids_batch = text_pair_ids[batch_idx].to(device)
                mask_bert_batch = mask_bert[batch_idx].to(device)
                token_type_ids_batch = token_type_ids[batch_idx].to(device)
                outputs_batch = bert_model(text_pair_ids_batch, attention_mask=mask_bert_batch,
                                           token_type_ids=token_type_ids_batch)
                data_values = F.softmax(outputs_batch[0], dim=-1)
                data_values = data_values[:, 1].detach().cpu().numpy().tolist()
                data_values_all.extend(data_values)

        data_values_all = np.array(data_values_all)
        pickle.dump(data_values_all, open(args.pre_bert, 'wb'))

    data_values_all = np.reshape(data_values_all, (int(data_size/step), step))
    idx_sorted = np.argsort(data_values_all, axis=1)
    source = np.reshape(source, (int(data_size/step), step))
    target = np.reshape(target, (int(data_size/step), step))
    ground_truth = np.reshape(ground_truth, (int(data_size/step), step))

    divide = step
    for itt in range(divide):
        source_remain = []
        target_remain = []
        ground_truth_remain = []
        data_value_remain = []
        for id, sorted_list in enumerate(idx_sorted):
            source_remain.extend(source[id][sorted_list[itt:]])
            target_remain.extend(target[id][sorted_list[itt:]])
            ground_truth_remain.extend(ground_truth[id][sorted_list[itt:]])
            data_value_remain.extend(data_values_all[id][sorted_list[itt:]])

        with open('{}/{}_remove_{}.txt'.format(args.output_path, args.dataset, int(itt)), 'w') as f:
            for i in range(len(source_remain)):
                f.write(source_remain[i] + "\t" + target_remain[i] + "\t" + ground_truth_remain[i] + "\t" + str(data_value_remain[i]) + "\n")
    print("Finish")


def select_top1(device):
    K = 50

    # load saved model
    #model_class, tokenizer_class, pretrained_weights = (
    #    tfs.AutoModelForSequenceClassification, tfs.AutoTokenizer, 'textattack/bert-base-uncased-QQP')
    model_class, tokenizer_class, pretrained_weights = (
        tfs.BertForSequenceClassification, tfs.BertTokenizer, args.pretrain)
    bert_tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    #bert_model = model_class.from_pretrained(pretrained_weights)
    bert_model = model_class.from_pretrained(args.save_best_bert)
    print("Finish Model Loading")

    if torch.cuda.device_count() > 1:
        print("Multi-GPU")
        bert_model = nn.DataParallel(bert_model)
    bert_model.to(device)

    # read data
    data = pd.read_csv(args.data_path, sep='\t', quoting=csv.QUOTE_NONE)
    source = data['question1'].str.strip('"').values
    target = data['question2'].str.strip('"').values
    ground_truth = data['ground_truth'].str.strip('"').values
    data_size = len(source)
    print("Data Size: {}".format(data_size))
    print("Finish Data Loading")

    if os.path.exists(args.pre_bert):
        data_values_all = pickle.load(open(args.pre_bert, 'rb'))
    else:
        text_pair = zip(source, target)
        tokenized_text_pair = bert_tokenizer.batch_encode_plus(text_pair, add_special_tokens=True, truncation=True,
                                                               max_length=args.max_seq_len * 2,
                                                               is_pretokenized=False, pad_to_max_length=True,
                                                               return_tensors='pt')

        text_pair_ids = tokenized_text_pair['input_ids']
        mask_bert = tokenized_text_pair['attention_mask']
        token_type_ids = tokenized_text_pair['token_type_ids']
        print("Finish Data Tokenization")

        # compute final data value
        idx_list = range(data_size)
        idx_batches = [idx_list[i:i + 1000] for i in range(0, len(idx_list), 1000)]
        data_values_all = []
        with torch.no_grad():
            for batch_idx in idx_batches:
                text_pair_ids_batch = text_pair_ids[batch_idx].to(device)
                mask_bert_batch = mask_bert[batch_idx].to(device)
                token_type_ids_batch = token_type_ids[batch_idx].to(device)
                outputs_batch = bert_model(text_pair_ids_batch, attention_mask=mask_bert_batch,
                                           token_type_ids=token_type_ids_batch)
                data_values = F.softmax(outputs_batch[0], dim=-1)
                data_values = data_values[:, 1].detach().cpu().numpy().tolist()
                data_values_all.extend(data_values)

        data_values_all = np.array(data_values_all)
        pickle.dump(data_values_all, open(args.pre_bert, 'wb'))

    data_values_all = np.reshape(data_values_all, (int(data_size/K), K))
    # idx_sorted = np.argsort(data_values_all, axis=1)
    source = np.reshape(source, (int(data_size/K), K))
    target = np.reshape(target, (int(data_size/K), K))
    ground_truth = np.reshape(ground_truth, (int(data_size/K), K))

    selection_range = [5, 10, 15, 25, 50]

    for k in selection_range:
        data_values_k = data_values_all[:][:k]
        idx_sorted = np.argsort(data_values_k, axis=1)

        with open('{}/select_top_{}_from_{}.txt'.format(args.output_path, int(k), args.dataset), 'w') as f:
            for id, sorted_list in enumerate(idx_sorted):
                f.write(source[id][sorted_list[-1]] + "\t" + target[id][sorted_list[-1]] + "\t" +
                        ground_truth[id][sorted_list[-1]] + "\t" + str(data_values_k[id][sorted_list[-1]]) + "\n")
    print("Finish")


def select_topk(device):
    K = 50

    # load saved model
    #model_class, tokenizer_class, pretrained_weights = (
    #    tfs.AutoModelForSequenceClassification, tfs.AutoTokenizer, 'textattack/bert-base-uncased-QQP')
    model_class, tokenizer_class, pretrained_weights = (
        tfs.BertForSequenceClassification, tfs.BertTokenizer, args.pretrain)
    bert_tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    #bert_model = model_class.from_pretrained(pretrained_weights)
    bert_model = model_class.from_pretrained(args.save_best_bert)
    print("Finish Model Loading")

    if torch.cuda.device_count() > 1:
        print("Multi-GPU")
        bert_model = nn.DataParallel(bert_model)
    bert_model.to(device)

    # read data
    data = pd.read_csv(args.data_path, sep='\t', quoting=csv.QUOTE_NONE)
    source = data['question1'].str.strip('"').values
    target = data['question2'].str.strip('"').values
    ground_truth = data['ground_truth'].str.strip('"').values
    data_size = len(source)
    print("Data Size: {}".format(data_size))
    print("Finish Data Loading")

    if os.path.exists(args.pre_bert):
        data_values_all = pickle.load(open(args.pre_bert, 'rb'))
    else:
        text_pair = zip(source, target)
        tokenized_text_pair = bert_tokenizer.batch_encode_plus(text_pair, add_special_tokens=True, truncation=True,
                                                               max_length=args.max_seq_len * 2,
                                                               is_pretokenized=False, pad_to_max_length=True,
                                                               return_tensors='pt')

        text_pair_ids = tokenized_text_pair['input_ids']
        mask_bert = tokenized_text_pair['attention_mask']
        token_type_ids = tokenized_text_pair['token_type_ids']
        print("Finish Data Tokenization")

        # compute final data value
        idx_list = range(data_size)
        idx_batches = [idx_list[i:i + 1000] for i in range(0, len(idx_list), 1000)]
        data_values_all = []
        with torch.no_grad():
            for batch_idx in idx_batches:
                text_pair_ids_batch = text_pair_ids[batch_idx].to(device)
                mask_bert_batch = mask_bert[batch_idx].to(device)
                token_type_ids_batch = token_type_ids[batch_idx].to(device)
                outputs_batch = bert_model(text_pair_ids_batch, attention_mask=mask_bert_batch,
                                           token_type_ids=token_type_ids_batch)
                data_values = F.softmax(outputs_batch[0], dim=-1)
                data_values = data_values[:, 1].detach().cpu().numpy().tolist()
                data_values_all.extend(data_values)

        data_values_all = np.array(data_values_all)
        pickle.dump(data_values_all, open(args.pre_bert, 'wb'))

    data_values_all = np.reshape(data_values_all, (int(data_size/K), K))
    idx_sorted = np.argsort(-data_values_all, axis=1)
    source = np.reshape(source, (int(data_size/K), K))
    target = np.reshape(target, (int(data_size/K), K))
    ground_truth = np.reshape(ground_truth, (int(data_size/K), K))

    #selection_range = [5, 10, 15, 25, 35, 50]
    selection_range = [1, 3]
    for k in selection_range:

        source_remain = []
        target_remain = []
        ground_truth_remain = []
        data_value_remain = []
        for id, sorted_list in enumerate(idx_sorted):
            source_remain.extend(source[id][sorted_list[:k]])
            target_remain.extend(target[id][sorted_list[:k]])
            ground_truth_remain.extend(ground_truth[id][sorted_list[:k]])
            data_value_remain.extend(data_values_all[id][sorted_list[:k]])

        with open('{}/select_top_{}_{}_recall.txt'.format(args.output_path, int(k), args.model), 'w') as f:
            for i in range(len(source_remain)):
                f.write(source_remain[i] + "\t" + target_remain[i] + "\t" + ground_truth_remain[i] + "\t" + str(
                    data_value_remain[i]) + "\n")
    print("Finish")


if __name__ == '__main__':
    #model = 'bm25_100k_top5'
    # model = 'bm25_100k_top5'
    #model = 'bm25_100k_top5_pretrain_clf'
    # dataset = 'bm25_100k_top5_pretrain'
    model = 'selector_3k'
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default=model)
    parser.add_argument('-checkpoint', type=str, default=None)
    # parser.add_argument('-save_best_bert', type=str, default='checkpoints/{}/bert'.format(model))
    parser.add_argument('-save_best_bert', type=str, default='pretrained_model/selector_3k')
    # parser.add_argument('-data_path', type=str, default='data/{}/train.csv'.format(dataset))
    parser.add_argument('-data_path', type=str, default='preprocess/qqp_bm25_top50.csv')
    # parser.add_argument('-pre_bert', type=str, default='data/{}/train_data_values.pkl'.format(dataset))
    parser.add_argument('-pre_bert', type=str, default='preprocess/data_values_{}.pkl'.format(model))
    parser.add_argument('-output_path', type=str, default='select_output')
    parser.add_argument('-pretrain', type=str, default='bert-base-uncased')
    parser.add_argument('-max_seq_len', type=int, default=33)
    args = parser.parse_args()

    my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    select_topk(my_device)
