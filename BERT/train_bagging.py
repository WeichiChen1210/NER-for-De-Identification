import os
import time
import logging
from tqdm import tqdm, trange
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler

from transformers import BertConfig, BertForTokenClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import KFold
from seqeval.metrics import f1_score, recall_score, precision_score

from utils import create_dataset
from parameters import get_args

def evaluate(args, tag_values, model, val_loader):
    model.eval()

    validation_loss_values = []
    eval_loss = 0
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
    return validation_loss_values, eval_loss, eval_prec, eval_recall, eval_f1

def train_step(model, train_loader, optimizer, scheduler):
    total_loss = 0

    # Training loop   
    train_iterator = tqdm(train_loader, desc='Training:', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    for batch in train_iterator:
        batch = tuple(t.cuda(non_blocking=True) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        model.zero_grad()
        
        # This will return the loss (rather than the model output) because we have provided the 'labels'.
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        # get the loss
        loss = outputs[0]
        
        loss.backward()
        total_loss += loss.item()

        # Clip the norm of the gradient to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=args.max_grad_norm)
        
        optimizer.step()
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_loader)
    
    return avg_train_loss

def train(args):
    (tag_values, tag2idx), (input_ids, tags, attention_mask), _ = create_dataset(args, False)
    attention_mask = np.array(attention_mask)

    # K-Fold
    kf = KFold(n_splits=args.k, shuffle=True, random_state=1)   # construct KFold object
    for fold_idx, (train_index, test_index) in enumerate(kf.split(input_ids)):  # for every fold
        args.logging('----- {}-fold -----'.format(fold_idx+1))

        # Dataset
        print('Making datasets...')
        # get dataset using the indexs
        tr_inputs, val_inputs = input_ids[train_index], input_ids[test_index]
        tr_tags, val_tags = tags[train_index], tags[test_index]
        tr_masks, val_masks = attention_mask[train_index], attention_mask[test_index]

        # convert to tensor
        tr_masks, val_masks = torch.tensor(tr_masks), torch.tensor(val_masks)

        # make pytorch dataset, each iteration will output a batch of tuples(inputs, masks, tags)
        train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
        train_loader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=args.batchsize, num_workers=8, pin_memory=True)

        val_data = TensorDataset(val_inputs, val_masks, val_tags)
        val_loader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=args.batchsize, num_workers=8, pin_memory=True)

        # model
        print('Building model...')
        model = BertForTokenClassification.from_pretrained(
            args.pre_model,
            num_labels=len(tag2idx),
            output_attentions=False,
            output_hidden_states=False,
        )
        model.cuda()

        # optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)
        # Total number of training steps is number of batches * number of epochs.
        total_steps = len(train_loader) * args.epoch
        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup * total_steps, num_training_steps=total_steps)
        # scheduler = None
        
        """
        FULL_FINETUNING = True
        if FULL_FINETUNING:
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.0}]
        else:
            param_optimizer = list(model.classifier.named_parameters())
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
        """
        
        loss_values = []    # Store the average loss after each epoch so we can plot them.
        best_f1 = 0.0
        args.logging('----------\nStart training...')
        for epoch in range(args.epoch):
            tic = time.time()
            print('---------- Epoch {} ----------'.format(epoch+1))
            model.train()
            
            # train
            avg_train_loss = train_step(model, train_loader, optimizer, scheduler)
            loss_values.append(avg_train_loss)# Store the loss value for plotting the learning curve.
            
            # validate
            _, eval_loss, eval_prec, eval_recall, eval_f1 = evaluate(args, tag_values, model, val_loader)
            args.logging("Epoch {} - Average train loss: {:.4f}".format(epoch+1, avg_train_loss))
            args.logging("Epoch {} - Validation loss: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1-Score: {:.4f}".format(epoch+1, eval_loss, eval_prec, eval_recall, eval_f1))

            if epoch > 0 and epoch % 5 == 0 and eval_f1 > best_f1:
                torch.save({'epoch': epoch+1, 'state_dict': model.state_dict(), 'tag2idx': tag2idx}, os.path.join(args.save_path, '{}_fold_best.pth'.format(fold_idx+1)))
                best_f1 = eval_f1
            torch.save({'epoch': epoch+1, 'state_dict': model.state_dict(), 'tag2idx': tag2idx}, os.path.join(args.save_path, '{}_fold_checkpoint.pth'.format(fold_idx+1)))

            args.logging('epoch time: %.2f\n----------'%(time.time() - tic))
        
        args.logging('End training at time: %s'%(time.ctime()))
        args.logging('Best F1: %.3f'%(best_f1))
    
if __name__ == "__main__":
    args = get_args()
    
    args.save_path = os.path.join(args.save_path, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(args.save_path)
    # save parameters
    with open(os.path.join(args.save_path, 'params.txt'), 'w') as f:
        for key, val in vars(args).items():
            f.write("%s: %s\n"%(key, val))

    # logging
    logging.basicConfig(level=logging.DEBUG,
                        filename=os.path.join(args.save_path, 'log.log'),
                        filemode='w',
                        format='%(asctime)s %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    args.logging = logging.info

    print('--Training mode--')
    train(args)