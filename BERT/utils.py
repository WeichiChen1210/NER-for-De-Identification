import os

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

from preprocessing import split_sentences, split_by_period, split_by_regex

# loading input
def loadInputFile(path, is_test=False):
    """Load input data and split to articles, word positions and keyword types.

    Parameters:
        path (str): path to the file
        is_test (bool): if true, the file has no ground truth
    
    Return:
        dataset (list): list of articles
        position (dict): dict of annotations, key is the article id, value is the list of its annotations(as tuple)
    """

    with open(path, 'r', encoding='utf8') as f:
        file_text = f.read().encode('utf-8').decode('utf-8-sig')
    
    dataset = list()  # store dataset [content, content,...]
    if not is_test:
        position = dict()  # store position [article_id, start_pos, end_pos, entity_text, entity_type, ...]
        # mentions = dict()  # store mentions[mention] = Type
    
    # split every article
    datas = file_text.split('\n\n--------------------\n\n')[:-1]
    for data in datas:
        data = data.split('\n')

        if is_test:
            # article_id = int(data[0].strip().split()[-1])
            content = data[1] # articles
            dataset.append(content)
        else:
            dataset.append(data[0]) # articles
            annotations = data[1:] # annotations
            article_id = None
            annots = [] # store the annotations of the article

            for annot in annotations[1:]: # for each line of annotation
                annot = annot.split('\t') # annot = article_id, start_pos, end_pos, entity_text, entity_type
                article_id = annot[0]
                annot = tuple(annot[1:]) # make annotations tuple
                annots.append(annot)
            
            position[article_id] = annots   # {article_id: annotation list of this article}    
        
    if is_test:
        return dataset
    else:
        return dataset, position

def output_file(idx2tag, test_sentence_list, predictions, article_id_list, output_path):    
    # create a dict to store combined sentneces of each article
    output = {i: ['', []] for i in range(max(article_id_list)+1)}
    # output_tags = []
    # combine sentences and cut the tag lists to correct length
    for sentence, prediction, article_id in zip(test_sentence_list, predictions, article_id_list):
        sent = ''.join(sentence)
        tags = prediction[:len(sent)]
        tags = [idx2tag[tag] for tag in tags]   # convert from idx to tag name
        # output_tags.extend(tags)
        output[article_id][0] += sent
        output[article_id][1].extend(tags)
    """
    output_tags = np.array(output_tags)
    print(output_tags.shape)
    np.save('bert3.npy', output_tags)
    """
    output = {k: output[k] for k in sorted(output.keys())}  # sort by article id    
    
    out_str = "article_id\tstart_position\tend_position\tentity_text\tentity_type\n"
    for key, val in output.items():
        sentence, tags = val
        start_pos, end_pos = None, None
        entity, entity_type = '', ''
        article_len = len(sentence)

        for idx, tag in enumerate(tags):  # traversal every tag
            if tag[0] == 'B':
                if start_pos is not None:
                    '''先輸出上一個B'''
                    end_pos = idx
                    entity = sentence[start_pos:end_pos]
                    line = '{}\t{}\t{}\t{}\t{}\n'.format(key, start_pos, end_pos, entity, entity_type)
                    out_str += line
                    '''再設這個B為新的start_pos'''
                    start_pos = idx
                    if (idx + 1) >= article_len:
                        end_pos = idx + 1
                        entity = sentence[start_pos:end_pos]
                        line = '{}\t{}\t{}\t{}\t{}\n'.format(key, start_pos, end_pos, entity, entity_type)
                        out_str += line
                        start_pos, end_pos = None, None
                    else:
                        end_pos = None
                else:
                    start_pos = idx  # record idx
                    entity_type = tag[2:]  # record type
                    if (idx + 1) >= article_len:
                        end_pos = idx + 1
                        entity = sentence[start_pos:end_pos]
                        line = '{}\t{}\t{}\t{}\t{}\n'.format(key, start_pos, end_pos, entity, entity_type)
                        out_str += line
                        start_pos, end_pos = None, None
                    else:
                        continue
            elif start_pos is not None and tag[0] == 'I':  # if encounter 'I'
                if (idx + 1) >= article_len:
                    end_pos = idx + 1
                    entity = sentence[start_pos:end_pos]
                    line = '{}\t{}\t{}\t{}\t{}\n'.format(key, start_pos, end_pos, entity, entity_type)
                    out_str += line
                    start_pos, end_pos = None, None
                else:
                    end_pos = idx

            elif tag[0] == 'O':
                if start_pos is not None:
                    end_pos = idx
                    entity = sentence[start_pos:end_pos]  # entity name
                    line = '{}\t{}\t{}\t{}\t{}\n'.format(key, start_pos, end_pos, entity, entity_type)
                    out_str += line
                    start_pos, end_pos = None, None
                else:
                    continue
    with open(output_path,'w',encoding='utf-8') as f:
        f.write(out_str)

# create tag list
def create_tag_list(articles, sentences_list, position):
    """Create tag list according to the segmented sentences and positions.
    
    First create a tag list only containing 'O', whose structure is like the article list ([['O', 'O', ...], ['O', ...], ...])
    Then according to the tags in position dict, change the tag of the corresponding idx.
    Finally according to the list of segmented sentences, split the tags to the same structure.

    Parameters:
        articles (list): list of original articles
        sentences_list (list): list of segmented sentences
        position (dict): dict that contains annotations

    Returns:
        tag_list (list): list of segmented tag sequences
    """
    # create same structure of list which contains only 'O'
    tags = []
    for article in articles:
        tags.append(['O' for _ in range(len(article))])

    # for every tag tuple, change the tag
    for key, val in position.items():
        tag_tuple = val
        article_id = int(key)
        article = articles[article_id]
        
        for (start_pos, end_pos, entity, entity_type) in tag_tuple:
            start_pos, end_pos = int(start_pos), int(end_pos)
            if article[start_pos:end_pos] == entity:    # check if the slicing result is correct
                for idx in range(start_pos, end_pos):
                    if idx == start_pos:
                        tags[article_id][idx] = 'B-' + entity_type  # the beggining of the tag
                    else:
                        tags[article_id][idx] = 'I-' + entity_type
    
    # create tag list that is segmented according to segmented sentence list
    tag_list = []
    for idx, article in enumerate(sentences_list):
        tag = tags[idx] # the tags of this article
        sentence_len = [len(sentence) for sentence in article]  # compute every length of sentences
        start_pos = 0
        article_tag = []    # storing the tags of the article
        for sent_len in sentence_len:
            cur_tag = tag[start_pos:start_pos+sent_len]
            article_tag.append(cur_tag)
            start_pos += sent_len
        tag_list.append(article_tag)

    # check if the segmentations match
    for article, tag in zip(sentences_list, tag_list):
        assert len(article) == len(tag), 'In \'create_tag_list\': number of sentences not match'
        
        for sentence, t in zip(article, tag):
            assert len(sentence) == len(t), 'In \'create_tag_list\': sentence lengths not match'
    
    return tag_list

def create_tag_list_overlap(articles, sentences_list, overlap_len, position):
    """Create tag list according to the segmented sentences and positions.
    
    First create a tag list only containing 'O', whose structure is like the article list ([['O', 'O', ...], ['O', ...], ...])
    Then according to the tags in position dict, change the tag of the corresponding idx.
    Finally according to the list of segmented sentences, split the tags to the same structure.

    Parameters:
        articles (list): list of original articles
        sentences_list (list): list of segmented sentences
        overlap_len (list): list of the overlapped subsentence's len in each sentence
        position (dict): dict that contains annotations

    Returns:
        tag_list (list): list of segmented tag sequences
    """
    # create same structure of list which contains only 'O'
    tags = []
    for article in articles:
        tags.append(['O' for _ in range(len(article))])

    # for every tag tuple, change the tag
    for key, val in position.items():
        tag_tuple = val
        article_id = int(key)
        article = articles[article_id]
        
        for (start_pos, end_pos, entity, entity_type) in tag_tuple:
            start_pos, end_pos = int(start_pos), int(end_pos)
            if article[start_pos:end_pos] == entity:    # check if the slicing result is correct
                for idx in range(start_pos, end_pos):
                    if idx == start_pos:
                        tags[article_id][idx] = 'B-' + entity_type  # the beggining of the tag
                    else:
                        tags[article_id][idx] = 'I-' + entity_type
    
    # create tag list that is segmented according to segmented sentence list
    tag_list = []
    for idx, (article, ov_len) in enumerate(zip(sentences_list, overlap_len)):
        tag = tags[idx] # the tags of this article
        sentence_len = [len(sentence) for sentence in article]  # compute every length of sentences
        start_pos = 0
        article_tag = []    # storing the tags of the article
        for i, sent_len in enumerate(sentence_len):
            cur_tag = tag[start_pos:start_pos+sent_len]
            article_tag.append(cur_tag)
            start_pos = start_pos + sent_len - ov_len[i] # overlapped
        tag_list.append(article_tag)

    # check if the segmentations match
    for article, tag in zip(sentences_list, tag_list):
        assert len(article) == len(tag), 'In \'create_tag_list\': number of sentences not match'
        
        for sentence, t in zip(article, tag):
            assert len(sentence) == len(t), 'In \'create_tag_list\': sentence lengths not match'
    
    return tag_list

# tokenization
def tokenize_to_ids(tokenizer, tag2idx, sentences_list, tag_list=None):
    """Tokenize the sentences and convert to token ids.
    
    For this task, we only use list() to tokenize to single words.
    If tag_list is None, only operate on sentences list.

    Parameters:
        tokenizer (BertTokenizer): to do tokenizing(not used in this case)
        tag2idx (dict): dict to convert tag to idx
        sentences_list (list): list of sentences to be tokenized
        tag_list (list): the corresponding tag list

    Returns:
        output_list (list): tokenized sentence as token ids
        output_tag_list (list): tag list(only for training data)
        tokenized_texts_list (list): tokenzied sentences(str)(only for test data)
    """

    output_list = []
    if tag_list is None:
        tokenized_texts_list = []
        for _, article in enumerate(sentences_list):
            for sentence in article:
                # tokenized_sent = tokenizer.tokenize(sentence)
                # tokenizing and adding '[CLS]' and '[SEP]' tokens
                tokenized_sent = list(sentence.lower()) # convert English words to lower case
                
                # convert to token ids and tag idx
                tokenized_ids = tokenizer.convert_tokens_to_ids(tokenized_sent)

                output_list.append(tokenized_ids)
                tokenized_texts_list.append(tokenized_sent)
        
        return output_list, tokenized_texts_list
    else:
        output_tag_list = []
        for article, tags in zip(sentences_list, tag_list):
            for sentence, tag in zip(article, tags):
                # tokenized_sent = tokenizer.tokenize(sentence)
                # tokenizing and adding '[CLS]' and '[SEP]' tokens
                tokenized_sent = list(sentence.lower()) # convert English words to lower case

                # convert to token ids and tag idx
                tokenized_ids = tokenizer.convert_tokens_to_ids(tokenized_sent)
                tag_ids = [tag2idx.get(t) for t in tag]

                output_list.append(tokenized_ids)
                output_tag_list.append(tag_ids)

        return output_list, output_tag_list

def padding_sequences(max_token_size, sequences, pad_val=0.0):
    """Pad sequences.

    Parameters:
        max_token_size (int): max senquence length
        sequences (list): list of sequences to be padded
        pad_val (float or others): value used for padding

    Return:
        padded_seq (torch tensor): tensor of padded sequences in shape [batchsize, max senquence length]
    """
    seq_list = [torch.tensor(seq) for seq in sequences] # convert each sequence to tensor to make list of tensors
    padded_seq = pad_sequence(seq_list, batch_first=True, padding_value=pad_val)    # padding

    assert padded_seq.size(1) == max_token_size, 'In function \'padding sequences\': padded sequence length not matched to max token size'
    
    return padded_seq

def compute_tag_weights(tag2idx, train_tag_list):
    """Count tag frequency and compute the weights for each tag.

    We use exponential to produce weights according to frequency.

    Parameters:
        tag2idx (dict): the dict to store the tag and its idx.
        train_tag_list (list): list that stores the tags of the articles.
    
    Return:
        weights (torch.tensor): weights of each tag.
    """
    # count the frequency of each tag
    count = {k: 0 for k in tag2idx.keys()}
    for article in train_tag_list:
        for sentence in article:
            for tag in sentence:
                count[tag] += 1
    
    # only count the 'O' tags and 'B-' tags
    b_count = {k: v for k, v in count.items() if 'O' in k or 'B' in k}

    # use exponential to produce weights
    w = np.array(list(b_count.values()))    # convert to ndarray
    w = np.exp(-np.log10(w))

    weights = {k: 0.0 for k in tag2idx.keys()}
    for val, k in zip(w, b_count.keys()):
        if k == 'PAD':
            continue
        if k == 'O':
            weights[k] = val
            continue
        else:
            i_item = 'I' + k[1:]
        
        weights[k] = val    # the B- tags
        weights[i_item] = val   # corresponding I- tags
    
    weights['PAD'] = weights['O']   # take weight of 'O' as PAD's weight
    weights = torch.tensor([v for v in weights.values()], dtype=torch.float32).cuda()   # convert ot tensor

    return weights

def create_dataset(args, output_dataset=True):
    """Load data, preprocess data to dataset

    Steps:
        1. Load and segment
        2. Tokenize and pad
        3. Make dataset and dataloader
    
    Parameters:
        args (Namespace): arguments
        output_dataset (bool): make the sentences dataset to output or not
    
    Returns:
        tag_values (list): the list of tags
        tag2idx (dict): the dict to store the tag and its idx.
        train_loader (Dataloader): dataloader for training. Output when output_dataset is True.
        val_loader (Dataloader): dataloader for validation. Output when output_dataset is True.
        input_ids (tensor): training data. Output when output_dataset is False.
        tags (tensor): training labels. Output when output_dataset is False.
        attention_mask (list): attention mask. Output when output_dataset is False.
        weights (torch tensor): weights of tags
    """
    ########## 1. Load and Segment ##########
    # load texts from file and process to lists of contents and labels
    print('Loading data...')
    articles, position = loadInputFile(args.data_path, is_test=False)
    if args.segment == 0:
        train_sentences_list = split_sentences(articles, args.max_token_size, args.max_period_num)   # training data
    elif args.segment == 1:
        train_sentences_list = split_by_period(articles, args.max_token_size, args.max_period_num)
    elif args.segment == 2:
        train_sentences_list = split_by_regex(articles, args.max_token_size)   # training data
    
    train_tag_list = create_tag_list(articles, train_sentences_list, position)    # labels
    # train_article_id_list = [[idx for _ in range(len(article))] for idx, article in enumerate(train_sentences_list)]    # article id record
    train_num_sent = sum([len(article) for article in train_sentences_list])    # num of sentences for training
    print('num of training sentences:', train_num_sent)    

    # make a dict that convert tag to index
    tag_values = [tag for article in train_tag_list for sentence in article for tag in sentence]
    tag_values = list(set(tag_values))
    tag_values.sort()
    tag_values.append('PAD')
    tag2idx = {t: i for i, t in enumerate(tag_values)}
    print('tag -> idx:\n', tag2idx)  

    weights = compute_tag_weights(tag2idx, train_tag_list)

    ########## 2. Tokenzie and pad ##########
    # tokenizing
    print('Tokenizing...')
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer, do_lower_case=True)
    tokenized_training_ids, tokenized_training_labels = tokenize_to_ids(tokenizer, tag2idx, train_sentences_list, train_tag_list)
   
    # Padding
    print('Padding...')
    input_ids = padding_sequences(args.max_token_size, tokenized_training_ids, pad_val=0.0)    # sentence ids
    tags = padding_sequences(args.max_token_size, tokenized_training_labels, pad_val=tag2idx['PAD'])   # tags
    attention_mask = [[float(word != 0.0) for word in sent] for sent in input_ids]
    assert input_ids.size() == tags.size(), 'not matched'
    
    if output_dataset:
        ########## 3. Make dataset and dataloader ##########
        # Dataset
        print('Making datasets...')
        # split training data to training and validation sets
        tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags, random_state=2018, test_size=0.1)
        tr_masks, val_masks, _, _ = train_test_split(attention_mask, input_ids, random_state=2018, test_size=0.1)
        
        # convert to tensor
        tr_masks, val_masks = torch.tensor(tr_masks), torch.tensor(val_masks)
        
        # make pytorch dataset, each iteration will output a batch of tuples(inputs, masks, tags)
        train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
        train_loader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=args.batchsize, num_workers=args.num_workers, pin_memory=True)

        val_data = TensorDataset(val_inputs, val_masks, val_tags)
        val_loader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=args.batchsize, num_workers=args.num_workers, pin_memory=True)

        return (tag_values, tag2idx), (train_loader, val_loader), weights
    else:
        return (tag_values, tag2idx), (input_ids, tags, attention_mask), weights

def create_test_dataset(path, params, tag2idx):
    """Create test dataset.

    Read the parameters in the load path.
    Process the input data to test dataset

    Parameters:
        path (str): path to load test file
        params (dict): dict that stores the arguments
        tag2idx (dict): stores the tags and their idx
    
    Returns:
        tokenized_test_texts (list): list of tokenized setences
        test_article_id_list (list): list of article id of sentences
        test_loader (DataLoader): dataloader for testing
    """
    max_token_size = params['max_token_size']
    max_period_num = params['max_period_num']
    # read file
    print('Creating data...')
    articles = loadInputFile(path, is_test=True)
    if params['segment'] == 0:
        test_sentences_list = split_sentences(articles, max_token_size, max_period_num)
    elif params['segment'] == 1:
        test_sentences_list = split_by_period(articles, max_token_size, max_period_num)
    elif params['segment'] == 2:
        test_sentences_list = split_by_regex(articles, max_token_size)
    
    test_article_id_list = []    
    for idx, article in enumerate(test_sentences_list):
        test_article_id_list.extend([idx for _ in range(len(article))])
    test_num_sent = sum([len(article) for article in test_sentences_list])
    print('num of test sentences:', test_num_sent)

    # tokenizing
    print('Tokenizing...')
    tokenizer = BertTokenizer.from_pretrained(params['tokenizer'], do_lower_case=True)
    tokenized_test_ids, tokenized_test_texts = tokenize_to_ids(tokenizer, tag2idx, test_sentences_list, None)   # tokenize to every single word
    print('num of tokenized sentences', len(tokenized_test_texts))

    # Padding
    print('Padding...')
    input_ids = padding_sequences(max_token_size, tokenized_test_ids, pad_val=0.0)
    test_mask = [[float(word != 0.0) for word in sent] for sent in input_ids]

    # Dataset
    print('Making datasets...')
    test_masks = torch.tensor(test_mask)
    test_data = TensorDataset(input_ids, test_masks)
    test_loader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=params['batchsize'], num_workers=8, pin_memory=True)

    return tokenized_test_texts, test_article_id_list, test_loader