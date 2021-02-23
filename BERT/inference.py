import os
import argparse
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForTokenClassification
from keras.preprocessing.sequence import pad_sequences

from utils import create_test_dataset, output_file

def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-no', type=int, default=0, help='no of gpu')
    parser.add_argument('--batchsize', type=int, default=32, help='batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers to load data')
    parser.add_argument('--max-period-num', type=int, default=4, help='the max number of periods for segmentation')
    parser.add_argument('--max-token-size', type=int, default=128, help='max size of tokens')
    parser.add_argument('--segment', type=int, default=0, choices=[0, 1, 2], help='way to segment sentences')
    parser.add_argument('--pre-model', type=str, default='bert-base-chinese', choices=['bert-base-chinese', 'hfl/chinese-bert-wwm'], help='bert pretrained model for tokenizer and classification')
    parser.add_argument('--tokenizer', type=str, default='bert-base-chinese', choices=['bert-base-chinese', 'hfl/chinese-bert-wwm'], help='bert pretrained model for tokenizer and classification')
    parser.add_argument('--load-path', type=str, default='./output/', help='path to load model')
    parser.add_argument('--data-path', type=str, default='../data/test.txt', help='path to test data')
    parser.add_argument('--best', action='store_true', help='choose best model')

    return parser

if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_no)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dev_path = args.data_path    
    print('Loading hyperparameters...')
    with open(os.path.join(args.load_path, 'params.txt'), 'r') as f:
        lines = f.readlines()
    params = {}
    for line in lines:
        name, val = line.strip().split(':')
        name, val = name.strip(), val.strip()
        params[name] = args.load_path if name == 'load_path' else val
        if name in ['segment', 'gpu_no', 'batchsize', 'num_workers', 'max_token_size', 'max_period_num']:
            params[name] = int(params[name])

    if args.best:
        checkpoint = torch.load(os.path.join(args.load_path, 'best.pth'))
        print('Loading checkpoint: {}...'.format(os.path.join(args.load_path, 'best.pth')))
    else:
        checkpoint = torch.load(os.path.join(args.load_path, 'checkpoint.pth'))
        print('Loading checkpoint: {}...'.format(os.path.join(args.load_path, 'checkpoint.pth')))
    tag2idx = checkpoint['tag2idx']
    idx2tag = {v: k for k, v in tag2idx.items()}
    print('idx -> tag:\n', idx2tag)

    tokenized_test_texts, test_article_id_list, test_loader = create_test_dataset(dev_path, params, tag2idx)

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

    predictions = []
    print('Start Inferencing...')
    with torch.no_grad():
        test_iterator = tqdm(test_loader, desc='Inferencing', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        for batch in test_iterator:
            batch = tuple(t.cuda(non_blocking=True) for t in batch)
            b_input_ids, b_input_mask = batch

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs[0].detach().cpu().numpy()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])

    # convert the result and output
    print("Generating output file...")
    folder = args.load_path[9:-1]
    path = os.path.join(args.load_path, folder)
    if args.best:
        output_file(idx2tag, tokenized_test_texts, predictions, test_article_id_list, path + '_best.tsv')
    else:        
        output_file(idx2tag, tokenized_test_texts, predictions, test_article_id_list, path + '.tsv')
    print('Done.')