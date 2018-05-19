import sys

import pandas
import time

sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')

import argparse
import os
from config import BASE_PATH
from model.configs import NLI_BASELINE
from model.models import train_pairmodel
from util.load_embeddings import load_word_vectors, merged_embeddings

"""
Example with only the pretrained embeddings on the three_common dataset
$> python model/snli.py  --pretrained glove.6B.300d.txt --pretrained_dim 300 --dataset three


Example with pretrained embeddings on the one_common dataset
with the p_avg 250dim embeddings
$> python model/snli.py  --pretrained glove.6B.300d.txt --pretrained_dim 300 --dataset one --na p_avg/250.txt --na_dim 250


Example with pretrained embeddings on the one_common dataset
with the p_9 50dim embeddings
$> python model/snli.py  --pretrained glove.6B.300d.txt --pretrained_dim 300 --dataset one --na p_9/50.txt --na_dim 50



"""

config = NLI_BASELINE

parser = argparse.ArgumentParser()
# add arguments ########################################
parser.add_argument('--pretrained', nargs='?',
                    required=True,
                    # default="glove.6B.300d.txt",
                    help='filename for the pretrained word embeddings.')
parser.add_argument('--pretrained_dim', nargs='?', type=int,
                    required=True,
                    # default=300,
                    help='dimension of pretrained word embeddings.')
parser.add_argument('--na', nargs='?',
                    help='filename for the neural activation word embeddings.')
parser.add_argument('--na_dim', nargs='?', type=int,
                    help='dimension of neural activation word embeddings.')
parser.add_argument('--dataset', nargs='?',
                    required=True,
                    # default="three",
                    help='choose one from ["normal", "one", "two", "three"].')
args = parser.parse_args()

print(args)

#######################################

WORD_VECTORS = os.path.join(BASE_PATH, "embeddings", args.pretrained)
WORD_VECTORS_DIMS = args.pretrained_dim

DATASET = args.dataset

########################################################
# PREPARE FOR DATA
########################################################
# load word embeddings
print("loading word embeddings...")

if args.na is not None:

    WORD_VECTORS_NA = os.path.join(BASE_PATH,
                                   "data/neural_activations/{}".format(args.na))
    WORD_VECTORS_DIMS_NA = args.na_dim

    word2idx, idx2word, embeddings = merged_embeddings(WORD_VECTORS,
                                                       WORD_VECTORS_DIMS,
                                                       WORD_VECTORS_NA,
                                                       WORD_VECTORS_DIMS_NA)

else:
    word2idx, idx2word, embeddings = load_word_vectors(WORD_VECTORS,
                                                       WORD_VECTORS_DIMS)

if args.na is None:
    name = "SNLI_{}_{}".format(DATASET, args.pretrained)
else:
    name = "SNLI_{}_{}_{}".format(DATASET, args.pretrained, args.na)

name = name.replace("/", "_")
name = name.replace(".txt", "")

trainer = train_pairmodel(name, DATASET, embeddings, word2idx, config)

# create a name for the log file
log_name = time.strftime("%Y-%m-%d_%H:%M")
log_name = name + "_" + log_name

print("Training...")
for epoch in range(config["epochs"]):
    trainer.model_train()
    trainer.model_eval()
    print()

    scores = pandas.DataFrame(trainer.scores)
    scores.to_csv("{}.csv".format(log_name), sep=',', encoding='utf-8')

    trainer.checkpoint.check()

    if trainer.early_stopping.stop():
        print("Early stopping...")
        break
