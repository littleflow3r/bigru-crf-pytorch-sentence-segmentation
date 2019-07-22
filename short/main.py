#coding:utf-8
import sys
import os
import json
import time
import logging
import argparse
from collections import defaultdict
from tqdm import tqdm

import utils
from model import BiLSTM_CRF

import torch
import torch.nn as nn
import torch.optim as optim

def set_logger(log_path):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

def count_elapsed_time(start, end):
    elapsed_time = end - start
    e_min = int(elapsed_time)/60
    e_sec = int(elapsed_time - (elapsed_time*60))
    return e_min, e_sec

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def main():
    cmd = argparse.ArgumentParser('Training!')
    cmd.add_argument('--id', required=True, default="train1", help='id of the training (string)')
    cmd.add_argument('--gpu', type=int, default=0, help='set to 1 if using GPU')
    cmd.add_argument("--epoch", type=int, default=10, help='epochs')

    args = cmd.parse_args(sys.argv[2:])

    set_logger(os.path.join(str(args.id)+'_train.log'))

    # import random
    # random.seed(config['seed'])
    torch.manual_seed(1)
    # torch.cuda.manual_seed(config['seed'])

    gpu = True if args.gpu==1 else False
    device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
    print (device)

    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    EMBEDDING_DIM = 5
    HIDDEN_DIM = 4

    # Make up some training data
    training_data = [(
        "the wall street journal reported today that apple corporation made money".split(),
        "B I I I O O O B I O O".split()
    ), (
        "georgia tech is a university in georgia".split(),
        "B I O O O O B".split()
    )]

    #create word and tag vocabulary
    wtoi = {} 
    for sentence, tags in training_data:
        for word in sentence:
            if word not in wtoi:
                wtoi[word] = len(wtoi)
    ttoi = {"B":0, "I":1, "O":2, START_TAG:3, STOP_TAG:4}

    model = BiLSTM_CRF(len(wtoi), ttoi, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    with torch.no_grad():
        precheck_sent = prepare_sequence(training_data[0][0], wtoi)
        precheck_tags = torch.tensor([ttoi[t] for t in training_data[0][1]], dtype=torch.long)
        print ('best path before training', model(precheck_sent))

    for epoch in range(args.epoch):  # again, normally you would NOT do 300 epochs, it is toy data
        for sentence, tags in training_data:
            model.zero_grad()
            sentence_in = prepare_sequence(sentence, wtoi)
            targets = torch.tensor([ttoi[t] for t in tags], dtype=torch.long)
            loss = model.neg_log_likelihood(sentence_in, targets) #forward pass
            loss.backward() #compute the loss, gradients, and update parameter by calling optimizer.step()
            optimizer.step()

    # Check predictions after training
    with torch.no_grad():
        precheck_sent = prepare_sequence(training_data[0][0], wtoi)
        print('best path after training',model(precheck_sent))


if __name__ == "__main__":
    start = time.time()
    if len(sys.argv) > 1:
        if sys.argv[1] == 'train':
            main()
        elif sys.argv[1] == 'test':
            test_main()
        elif sys.argv[1] == 'predict':
            predict()
    else:
        print('Usage: {0} [train] [test] [predict] [options]'.format(sys.argv[0]), file=sys.stderr)

    end = time.time()
    e_min, e_sec = count_elapsed_time(start, end)
    logging.info (f'#Total elapsed time: {e_min}m {e_sec}s')
