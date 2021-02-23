import os
import argparse
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForTokenClassification
from sklearn.model_selection import train_test_split
from seqeval.metrics import f1_score, recall_score, precision_score, classification_report

from preprocessing import loadInputFile, tokenize_to_ids, padding_sequences, split_sentences, split_by_regex, split_by_period, create_tag_list

def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-no', type=int, default=0, help='no of gpu')
    parser.add_argument('--batchsize', type=int, default=32, help='batch size')
    parser.add_argument('--max-period-num', type=int, default=4, help='the max number of periods for segmentation')
    parser.add_argument('--max-token-size', type=int, default=128, help='max size of tokens')
    parser.add_argument('--segment', type=int, default=0, choices=[0, 1, 2], help='way to segment sentences')
    parser.add_argument('--pre-model', type=str, default='bert-base-chinese', choices=['bert-base-chinese', 'hfl/chinese-bert-wwm'], help='bert pretrained model for tokenizer and classification')
    parser.add_argument('--tokenizer', type=str, default='bert-base-chinese', choices=['bert-base-chinese', 'hfl/chinese-bert-wwm'], help='bert pretrained model for tokenizer and classification')
    parser.add_argument('--load-path', type=str, default='./output/', help='path to load model')
    parser.add_argument('--data-path', type=str, default='../data/train_4_new.txt', help='path to test data')
    parser.add_argument('--best', action='store_true', help='choose best model')

    return parser

def evaluate(args, tag_values, model, val_loader):
    model.eval()

    validation_loss_values = []
    eval_loss, eval_accuracy = 0, 0
    predictions , true_labels = [], []

    with torch.no_grad():
        val_iterator = tqdm(val_loader, desc='Validating', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        for batch in val_iterator:
            batch = tuple(t.cuda(non_blocking=True) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            
            # This will return the logits rather than the loss because we have not provided labels.
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

            # Move logits and labels to CPU
            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.cpu().numpy()

            # Calculate the accuracy for this batch of test sentences.
            eval_loss += outputs[0].mean().item()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.extend(label_ids)

    eval_loss = eval_loss / len(val_loader)
    validation_loss_values.append(eval_loss)

    pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                                     for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
    valid_tags = [tag_values[l_i] for l in true_labels
                                      for l_i in l if tag_values[l_i] != "PAD"]
    
    eval_prec = precision_score(valid_tags, pred_tags)
    eval_recall = recall_score(valid_tags, pred_tags)
    eval_f1 = f1_score(valid_tags, pred_tags)
    print(classification_report(valid_tags, pred_tags))
    return validation_loss_values, eval_loss, eval_prec, eval_recall, eval_f1

if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_no)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('--Inferencing mode--')
    
    print('Loading hyperparameters...')
    with open(os.path.join(args.load_path, 'params.txt'), 'r') as f:
        lines = f.readlines()
    params = {}
    for line in lines:
        name, val = line.strip().split(':')
        name, val = name.strip(), val.strip()
        params[name] = args.load_path if name == 'load_path' else val

    max_token_size = int(params['max_token_size'])
    max_period_num = int(params['max_period_num'])

    if args.best:
        checkpoint = torch.load(os.path.join(args.load_path, 'best.pth'))
        print('Loading checkpoint: {}...'.format(os.path.join(args.load_path, 'best.pth')))
    else:
        checkpoint = torch.load(os.path.join(args.load_path, 'checkpoint.pth'))
        print('Loading checkpoint: {}...'.format(os.path.join(args.load_path, 'checkpoint.pth')))
    tag2idx = checkpoint['tag2idx']
    idx2tag = {v: k for k, v in tag2idx.items()}
    print('idx -> tag:\n', idx2tag)

    # read file
    train_path = args.data_path
    print('Creating data...')
    articles, position = loadInputFile(train_path, is_test=False)
    if params['segment'] == '0':
        train_sentences_list = split_sentences(articles, max_token_size, max_period_num)   # training data
    elif params['segment'] == '1':
        train_sentences_list = split_by_period(articles, max_token_size, max_period_num)
    elif params['segment'] == '2':
        train_sentences_list = split_by_regex(articles, args.max_token_size)   # training data
    train_tag_list = create_tag_list(articles, train_sentences_list, position)    # labels
    # train_article_id_list = [[idx for _ in range(len(article))] for idx, article in enumerate(train_sentences_list)]    # article id record
    train_num_sent = sum([len(article) for article in train_sentences_list])    # num of sentences for training
    print('num of training sentences:', train_num_sent) 

    # make a dict that convert tag to index
    tag_values = [k for k in tag2idx.keys()]
    print(tag_values)

    # tokenizing
    print('Tokenizing...')
    tokenizer = BertTokenizer.from_pretrained(params['tokenizer'], do_lower_case=True)
    tokenized_training_ids, tokenized_training_labels = tokenize_to_ids(tokenizer, tag2idx, train_sentences_list, train_tag_list)

    # Padding
    print('Padding...')
    input_ids = padding_sequences(args.max_token_size, tokenized_training_ids, pad_val=0.0)    # sentence ids
    tags = padding_sequences(args.max_token_size, tokenized_training_labels, pad_val=tag2idx['PAD'])   # tags
    attention_mask = [[float(word != 0.0) for word in sent] for sent in input_ids]
    assert input_ids.size() == tags.size(), 'not matched'

    # Dataset
    print('Making datasets...')
    # split training data to training and validation sets
    tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags, random_state=2018, test_size=0.1)
    tr_masks, val_masks, _, _ = train_test_split(attention_mask, input_ids, random_state=2018, test_size=0.1)
    
    # convert to tensor
    tr_masks, val_masks = torch.tensor(tr_masks), torch.tensor(val_masks)
    
    # make pytorch dataset, each iteration will output a batch of tuples(inputs, masks, tags)
    # train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
    # train_loader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=args.batchsize, num_workers=8, pin_memory=True)

    val_data = TensorDataset(val_inputs, val_masks, val_tags)
    val_loader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=args.batchsize, num_workers=8, pin_memory=True)

    # model
    print('Loading model...')
    model = BertForTokenClassification.from_pretrained(
        params['pre_model'],
        num_labels=len(tag2idx),
        output_attentions=False,
        output_hidden_states=False
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()

    _, eval_loss, eval_prec, eval_recall, eval_f1 = evaluate(args, tag_values, model, val_loader)

    print("Validation loss: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1-Score: {:.4f}".format(eval_loss, eval_prec, eval_recall, eval_f1))