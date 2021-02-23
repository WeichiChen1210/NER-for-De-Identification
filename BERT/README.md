# DE-ID with NER Task by BERT
The code uses BERT to do NER task, specifically the tokenizing and classification.

## Requirements
+ tqdm
+ numpy
+ torch
+ transformers
+ keras
+ sklearn
+ seqeval

Simply install from requirements.txt:
```command
$pip install -r requirements.txt
```
## Data Processing
The original data is a txt file including a lot of articles.

Fisrt we have to use the function ```create_dataset``` in [utils.py](./utils.py) to prepare dataset:
    
1. Load data using ```loadInputFile```
2. Process articles into sentences by three segmentation methods
3. Tokenize and padding for bert input
4. Make Dataset and DataLoader

## Models
We adopt the pretrained BERT model from transformers.

## Training Methods
There are four methods for training BERT:
    
1. [Normal training](./train.py)
2. [Bagging](./train_bagging.py): Apply bagging and k-fold to generate k models to ensemble their results
3. [Weighted training](./train_weighted.py): Apply weighted cross entropy loss for the output of Bert model
4. [Bagging + Weighted training](./train_bagging_weighted.py): Combine above

## Usage
1. Training
Arguments:
+ ```--gpu-no```: gpu to use
+ ```--batchsize```: batch size
+ ```--epoch```: training epochs
+ ```--lr```: initial learning rate
+ ```--weight-decay```: weight decay for optimizer
+ ```--max-grad-num```: for preventing exploding gradients
+ ```--max-period-num```: the max number of periods for cutting a sentence
+ ```--max-token-size```: max token size or sentence length in BERT model
+ ```--segment```: the way to segment article and sentences
+ ```--bert-model```: the BERT pretrained model for tokenizer and classification
+ ```--save-path```: the path to save models and files, the program will automatically generate a foler named by current date and time under the given path

Examples:
+ Training using batch size 64, max token size 256, max period num 5, lr 5e-3 for epoch 5:
    ```command
    $ python main.py --train --batchsize 64 --max-period-num 5 --max-token-size 256 --lr 0.00003 --epoch 5
    ```
2. Inferencing
Arguments:
+ ```--gpu-no```: gpu to use
+ ```--batchsize```: batch
+ ```--max-period-num```: the max number of periods for cutting a sentence
+ ```--max-token-size```: max token size or sentence length in BERT model
+ ```--segment```: the way to segment article and sentences
+ ```--bert-model```: the BERT pretrained model for tokenizer and classification
+ ```--load-path```: folder path to load the models and files for inferencing
+ ```--best```: choose the best checkpoint or not

Examples:
+ Inferencing using model from path '```./output/test/```'(the program automatically append '```checkpoint.pth```' and will load the hyper-parameters from the ```params.txt```)
    ```command
    $ python main.py --inference --load-path ./output/test/
    ```
    Then the output file call ```output.tsv``` will appear under the folder of ```--load-path```
