import os
import argparse
import torch

def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-no', type=int, default=0, help='no. of gpu')
    parser.add_argument('--batchsize', type=int, default=32, help='batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers to load data')
    parser.add_argument('--epoch', type=int, default=10, help='training epochs')
    parser.add_argument('--lr', type=float, default=3e-5, help='initial lr')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--warmup', type=float, default=0.01, help='portion of warmup steps')
    parser.add_argument('--max-grad-norm', type=float, default=1.0, help='max grad norm')
    parser.add_argument('--max-period-num', type=int, default=4, help='the max number of periods for segmentation')
    parser.add_argument('--max-token-size', type=int, default=128, help='max size of tokens')
    parser.add_argument('--segment', type=int, default=0, choices=[0, 1, 2, 3], help='way to segment sentences')
    parser.add_argument('--pre-model', type=str, default='bert-base-chinese', choices=['bert-base-chinese', 'hfl/chinese-bert-wwm'], help='bert pretrained model for classification')
    parser.add_argument('--tokenizer', type=str, default='bert-base-chinese', choices=['bert-base-chinese', 'hfl/chinese-bert-wwm'], help='bert pretrained model for tokenizer')
    parser.add_argument('--k', type=int, default=5, help='k-fold')
    parser.add_argument('--data-path', type=str, default='../data/train_3.txt', help='path to data')
    parser.add_argument('--save-path', type=str, default='./output/', help='path to save model')

    return parser

def get_args():
    parser = build_arg_parser()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_no)

    return args    