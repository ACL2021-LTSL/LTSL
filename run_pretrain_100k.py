from data import *
from torch import optim
import torch.nn as nn
from torch.distributions import Categorical
import transformers as tfs
import numpy as np
import argparse
from tqdm import trange
import sys
import torch.nn.functional as F


def compute_score_loss(bart_tokenizer, bart_model, dev_data, device):
    scores = []
    pad_token_id = bart_tokenizer.pad_token_id
    # sampled_dev_data = random.sample(dev_data, 5)
    for source_ids, source_mask, target_ids in dev_data:
        source_ids = source_ids.to(device)
        source_mask = source_mask.to(device)
        target_ids = target_ids.to(device)

        y_ids = target_ids[:, :-1].contiguous()
        lm_labels = target_ids[:, 1:].clone()
        lm_labels[target_ids[:, 1:] == pad_token_id] = -100

        valid_loss, _, _ = bart_model(source_ids, attention_mask=source_mask, decoder_input_ids=y_ids, labels=lm_labels)
        scores.append(valid_loss.mean(-1).detach().cpu().numpy())
    return np.mean(scores)


def compute_selection_accuracy(data_size, data_values):
    idx_sorted = np.argsort(-np.array(data_values))
    test_num = int(args.noise * data_size)
    count = 0
    for id in idx_sorted[:test_num]:
        if int(id) < data_size - test_num:
            count += 1
    return float(count) / test_num


def model_train(device):
    # define pre-trained encoder
    # model_class, tokenizer_class, pretrained_weights = (
    #     tfs.AutoModelForSequenceClassification, tfs.AutoTokenizer, 'textattack/bert-base-uncased-QQP')
    model_class, tokenizer_class, pretrained_weights = (
        tfs.BertForSequenceClassification, tfs.BertTokenizer, 'bert-base-uncased')
    bert_tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    bert_model = model_class.from_pretrained(pretrained_weights)
    # bert_model = model_class.from_pretrained('pretrained_model/pre-train-bert')
    # define data valuator
    # valuator = DataValuator()

    # define text generator
    bart_model_class, bart_tokenizer_class, bart_pretrained_weights = (
        tfs.BartForConditionalGeneration, tfs.BartTokenizer, 'facebook/bart-base')
    bart_tokenizer = bart_tokenizer_class.from_pretrained(bart_pretrained_weights)
    bart_model = bart_model_class.from_pretrained(bart_pretrained_weights)

    print(device)

    if torch.cuda.device_count() > 1:
        print("Multi-GPU")
        bert_model = nn.DataParallel(bert_model)
        # valuator = nn.DataParallel(valuator)
        bart_model = nn.DataParallel(bart_model)

    bert_model.to(device)
    # valuator.to(device)
    bart_model.to(device)

    valuator_opt = tfs.AdamW(bert_model.parameters(), lr=3e-5)
    generator_opt = tfs.AdamW(bart_model.parameters(), lr=1e-5)
    valuator_scheduler = tfs.get_linear_schedule_with_warmup(valuator_opt, num_warmup_steps=0, num_training_steps=30000)
    generator_scheduler = tfs.get_linear_schedule_with_warmup(generator_opt, num_warmup_steps=0,
                                                              num_training_steps=50000)

    bert_model.train()
    # valuator.train()
    bart_model.train()

    # load validation data
    dev_data = load_data_dev(args, bart_tokenizer)

    # load training data
    data_size, train_idx, tokenized_text_pair, (source_ids_bart, source_mask_bart, target_ids_bart) \
        = load_data_train(args, bert_tokenizer, bart_tokenizer)

    text_pair_ids = tokenized_text_pair['input_ids']
    mask_bert = tokenized_text_pair['attention_mask']
    token_type_ids = tokenized_text_pair['token_type_ids']

    print("Finish Loading Data, Start Warm-up")
    # post-train generator
    for epoch in range(args.warm_up):
        for step, batch_idx in enumerate(train_idx):
            selected_source_ids_bart = source_ids_bart[batch_idx].to(device)
            selected_source_mask_bart = source_mask_bart[batch_idx].to(device)
            selected_target_ids_bart = target_ids_bart[batch_idx].to(device)

            pad_token_id = bart_tokenizer.pad_token_id
            y_ids = selected_target_ids_bart[:, :-1].contiguous()
            lm_labels = selected_target_ids_bart[:, 1:].clone()
            lm_labels[selected_target_ids_bart[:, 1:] == pad_token_id] = -100

            batch_loss, _, _ = bart_model(input_ids=selected_source_ids_bart,
                                          attention_mask=selected_source_mask_bart,
                                          decoder_input_ids=y_ids, labels=lm_labels)
            generator_opt.zero_grad()
            batch_loss.sum().backward()
            generator_opt.step()
            generator_scheduler.step()

        with torch.no_grad():
            score = compute_score_loss(bart_tokenizer, bart_model, dev_data, device)
        print("Warm-up Epoch: {}, Current Score: {}".format(epoch, score))

    # run training epochs
    best_score = -sys.maxsize - 1
    for epoch in trange(args.epochs):
        with torch.no_grad():
            baseline_score = compute_score_loss(bart_tokenizer, bart_model, dev_data, device)
            best_score = baseline_score if baseline_score > best_score else best_score

        # iterate each training batch
        for step, batch_idx in enumerate(train_idx):

            # compute data values and select actions
            text_pair_ids_batch = text_pair_ids[batch_idx].to(device)
            mask_bert_batch = mask_bert[batch_idx].to(device)
            token_type_ids_batch = token_type_ids[batch_idx].to(device)
            data_values = bert_model(text_pair_ids_batch, attention_mask=mask_bert_batch,
                                     token_type_ids=token_type_ids_batch)  # batch_size * 2

            dist = Categorical(F.softmax(data_values[0], dim=-1))
            actions = dist.sample()
            log_prob = dist.log_prob(actions).sum().unsqueeze(-1)

            # fine-tune generator and get performance on dev data
            selected_idx = [batch_idx[i] for i, action in enumerate(actions) if action == 1]
            selected_source_ids_bart = source_ids_bart[selected_idx].to(device)
            selected_source_mask_bart = source_mask_bart[selected_idx].to(device)
            selected_target_ids_bart = target_ids_bart[selected_idx].to(device)

            pad_token_id = bart_tokenizer.pad_token_id
            y_ids = selected_target_ids_bart[:, :-1].contiguous()
            lm_labels = selected_target_ids_bart[:, 1:].clone()
            lm_labels[selected_target_ids_bart[:, 1:] == pad_token_id] = -100

            for i in range(args.inner_iters):
                batch_loss, _, _ = bart_model(input_ids=selected_source_ids_bart,
                                              attention_mask=selected_source_mask_bart,
                                              decoder_input_ids=y_ids, labels=lm_labels)
                generator_opt.zero_grad()
                batch_loss.sum().backward()
                generator_opt.step()
                generator_scheduler.step()

            if (step + 1) % args.T == 0:
                with torch.no_grad():
                    score = compute_score_loss(bart_tokenizer, bart_model, dev_data, device)
                print("Current Score: {}, Baseline Score: {}".format(score, baseline_score))
                # best_score = score if score > best_score else best_score
                reward = args.alpha * (np.exp(score) - np.exp(baseline_score)) + (1 - args.alpha) * (np.exp(score) - np.exp(best_score))
                loss = -log_prob * -reward #* 100
                valuator_opt.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(bert_model.parameters(), 40)
                valuator_opt.step()
                valuator_scheduler.step()

            if (step + 1) % 100 == 0:
                print("Process Batch {}!".format(step + 1))
                with torch.no_grad():
                    baseline_score = compute_score_loss(bart_tokenizer, bart_model, dev_data, device)
                print("{} Instances Selected!".format(len(selected_idx)))

                # save best model
                if baseline_score > best_score:
                    best_score = baseline_score
                    if torch.cuda.device_count() > 1:
                        bert_model.module.save_pretrained(args.save_best_bert)
                        bart_model.module.save_pretrained(args.save_best_bart)
                    else:
                        bert_model.save_pretrained(args.save_best_bert)
                        bart_model.save_pretrained(args.save_best_bart)

        print("Best score of epoch {}: {}".format(epoch, best_score))

    # return selector
    return bert_model, bert_tokenizer


if __name__ == '__main__':
    dataset = 'bm25_100k_top5_pretrain_clf'
    model = 'quora_exp_reward'

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, default='train')
    parser.add_argument('-checkpoint', type=str, default=None)
    parser.add_argument('-save_best_bert', type=str, default='checkpoints/{}/bert'.format(model))
    parser.add_argument('-save_best_bart', type=str, default='checkpoints/{}/bart'.format(model))
    parser.add_argument('-data_path_train', type=str, default='data/{}/train.csv'.format(dataset))
    parser.add_argument('-data_path_dev', type=str, default='data/{}/valid.csv'.format(dataset))
    parser.add_argument('-pretrain', type=str, default='bert-base-uncased')
    parser.add_argument('-pre_bert', type=str, default='data/{}/pre/embeddings_bert.pkl'.format(dataset))
    parser.add_argument('-pre_bart', type=str, default='data/{}/pre/tokenized_ids_bart.pkl'.format(dataset))
    parser.add_argument('-warm_up', type=int, default=8)
    parser.add_argument('-inner_iters', type=int, default=1)
    parser.add_argument('-alpha', type=float, default=1.0)
    parser.add_argument('-gamma', type=float, default=0.99)
    parser.add_argument('-T', type=int, default=1)
    parser.add_argument('-max_seq_len', type=int, default=33)
    parser.add_argument('-epochs', type=int, default=5)
    parser.add_argument('-batch_size', type=int, default=500)
    parser.add_argument('-window_size', type=int, default=10)
    parser.add_argument('-noise', type=float, default=0.5)
    args = parser.parse_args()

    my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bert_model, bert_tokenizer = model_train(my_device)


