import numpy as np
import pandas as pd
from data import *
from torch import nn
from torch import optim
import torch
import transformers as tfs
import warnings


warnings.filterwarnings('ignore')


def pre_train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_class, tokenizer_class, pretrained_weights = (
        tfs.BertForSequenceClassification, tfs.BertTokenizer, 'bert-base-uncased')
    bert_tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    bert_model = model_class.from_pretrained(pretrained_weights)

    if torch.cuda.device_count() > 1:
        bert_model = nn.DataParallel(bert_model)
        bert_model.to(device)

    data = pd.read_csv("data/selector_pretrain_data.csv", sep='\t', quoting=csv.QUOTE_NONE)

    data_size = len(data)
    print("Data Size: {}".format(data_size))
    sources = data['question1'].str.strip('"').values
    targets = data['question2'].str.strip('"').values
    labels = data['is_perturbed'].values
    labels = 1 - labels

    text_pair = zip(sources, targets)
    tokenized_text_pair = bert_tokenizer.batch_encode_plus(text_pair, add_special_tokens=True,
                                                           max_length=66, truncation=True,
                                                           is_pretokenized=False, pad_to_max_length=True,
                                                           return_tensors='pt')

    print("Finish Tokenization")

    text_pair_ids = tokenized_text_pair['input_ids']
    mask_bert = tokenized_text_pair['attention_mask']
    token_type_ids = tokenized_text_pair['token_type_ids']

    idx_list = list(range(data_size))
    random.shuffle(idx_list)
    batch_size = 256
    idx_batches = list(get_batches(idx_list, batch_size))

    optimizer = optim.SGD(bert_model.parameters(), lr=1e-5, momentum=0.9)

    print("Start Training...")
    bert_model.train()
    epochs = 10
    for epoch in range(epochs):
        epoch_loss = 0.0
        for step, idx in enumerate(idx_batches):
            if (step + 1) % 100 == 0:
                print("Process Batch: {}, Loss: {}".format(step + 1, epoch_loss/(step+1)))
            text_pair_ids_batch = text_pair_ids[idx].to(device)
            mask_bert_batch = mask_bert[idx].to(device)
            token_type_ids_batch = token_type_ids[idx].to(device)

            # logits_batch = bert_model(text_pair_ids_batch, attention_mask=mask_bert_batch, token_type_ids=token_type_ids_batch)[0]
            # logits_batch = torch.reshape(logits_batch[:, 1], (-1, n))
            # scores_batch = torch.from_numpy(scores[idx]).to(device)
            labels_batch = torch.from_numpy(labels[idx]).to(device)
            # logits_batch = F.softmax(logits_batch, dim=1)
            # scores_batch = F.softmax(scores_batch, dim=1)
            # batch_loss = -torch.sum(scores_batch * torch.log(logits_batch)) / batch_size
            outputs = bert_model(text_pair_ids_batch, attention_mask=mask_bert_batch,
                                 token_type_ids=token_type_ids_batch, labels=labels_batch)
            loss, logits = outputs[:2]
            batch_loss = loss.mean()

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss
        print("Epoch {} -- Loss: {}".format(epoch, epoch_loss/len(idx_batches)))
    if torch.cuda.device_count() > 1:
        bert_model.module.save_pretrained('pretrained_model/selector_3k')
    print("Training Finished!")

    return


if __name__ == '__main__':

    pre_train()
