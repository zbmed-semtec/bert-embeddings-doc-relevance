import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from torch.optim import AdamW
from datasets import load_dataset
from pooling import PoolingWithDropout
from torch.utils.data import DataLoader
from dataset import CreateFineTuneDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sentence_transformers import evaluation, losses, models
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from sentence_transformers import SentenceTransformer, InputExample, SentenceTransformerTrainer, SentenceTransformerTrainingArguments


logging.basicConfig(filename='training_biobert_base_mnr_cl_2.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s') 

class EarlyStoppingCallback(TrainerCallback):
    """
    Early stopping callback for training to monitor the evaluation metric (`eval_loss`) and
    stop the training if no improvement is observed after a specified number of evaluation
    steps (patience).
    
    Parameters
    ----------
    patience : int, optional
        Number of evaluation steps with no improvement before stopping training (default is 1).
    min_delta : float, optional
        Minimum change in the monitored metric to qualify as an improvement (default is 0.0).
    """   
    def __init__(self, scheduler, patience=1, min_delta=0.0):
        self.scheduler = scheduler
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = None
        self.epochs_without_improvement = 0

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        metric = state.log_history[-1].get('eval_loss')

        if metric is None:
            return

        self.scheduler.step(metric)

        if self.best_metric is None or metric < self.best_metric - self.min_delta:
            self.best_metric = metric
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        current_lr = optimizer.param_groups[0]['lr']
        if self.epochs_without_improvement >= self.patience and current_lr <= self.min_lr:
            control.should_training_stop = True
            logging.info("Early stopping triggered.")

class TuneBert:
    """
    Class to fine-tune the BERT model for a given dataset when relevance
    scores are available.

    Parameters
    ----------
    train_dataset_path : str
        Path to the CSV file containing the training dataset.
    valid_dataset_path : str
        Path to the CSV file containing the validation dataset.
    save_dir : str
        Path to save the fine-tuned model.
    model_name : str
        Name of the pre-trained model.
    loss_func : str
        Loss function to use during training ('contrastive', 'mnr', or 'softmax').
    dropout : float
        Dropout value for regularization.
    """
    def __init__(self, train_dataset_path: str, valid_dataset_path: str, 
                save_dir: str, model_name: str, loss_func: str, droput: float):

        self.train_data = load_dataset("csv", data_files=train_dataset_path)['train']
        self.valid_data = load_dataset("csv", data_files=valid_dataset_path)['train']

        word_embedding_dimension = 768
        pooling_mode = 'mean'
        dropout_prob = dropout

        transformer = models.Transformer(model_name)
        pooling_layer = PoolingWithDropout(word_embedding_dimension, pooling_mode, dropout_prob)

        self.model = SentenceTransformer(modules=[transformer, pooling_layer])
        
        self.loss_func = loss_func
        
        if loss_func == 'contrastive':
            self.train_loss = losses.ContrastiveLoss(self.model)
        if loss_func == 'mnr':
            self.train_loss = losses.MultipleNegativesRankingLoss(self.model)
        if loss_func == 'softmax':
            self.train_loss = losses.SoftmaxLoss(self.model, self.model.get_sentence_embedding_dimension(), num_labels=3)


    def train(self, save_dir, batch_size, epochs):
        batch_size = batch_size
        epochs = epochs
        args = SentenceTransformerTrainingArguments(
                output_dir = save_dir,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs = epochs,
                eval_strategy = 'epoch',
                learning_rate = 2e-5,
                warmup_ratio = 0.1,
                lr_scheduler_type = 'reduce_lr_on_plateau',
                do_train=True,
                do_eval=True
            	)

        optimizer = AdamW(self.model.parameters(), lr=args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

        trainer = SentenceTransformerTrainer(
            model = self.model,
            args = args,
            train_dataset = self.train_data,
            eval_dataset = self.valid_data,
            loss = self.train_loss
        )        

        self.model.save(save_dir)
        
        logging.info(trainer.train())
        for log in trainer.state.log_history:
            if 'eval_loss' in log:
                logging.info(f"Epoch {log['epoch']}: eval_loss = {log['eval_loss']}")
            logging.info(log)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune BioBpipERT')
    parser.add_argument('-m', '--model_name', type=str, required=True,
                        help='model')                  
    parser.add_argument('-o', '--save_train', type=str, required=True,
                        help='path to save the model')
    parser.add_argument('-e', '--epochs', type=int, default=2,
                        help='number of epochs to train')
    parser.add_argument('-b', '--batch_size', type=int, default=8,
                        help='batch size for training')
    parser.add_argument('-l', '--loss_func', type=str, default='MNRLoss',
                        help='loss function to be used for training')
    parser.add_argument('-c', '--classes', type=int, default=3,
                        help='class distribution to be used (either 2 or 3)')
    parser.add_argument('-d', '--dropout', type=float, default=0.5,
                        help='dropout value')                 
    args = parser.parse_args()

    data = CreateFineTuneDataset(loss_func=args.loss_func, classes=args.classes)

    input_train_dataset_path = f'data/Split_Dataset/Data/input_{args.loss_func.lower()}_text_train.csv'
    input_test_dataset_path =  f'data/Split_Dataset/Data/input_{args.loss_func.lower()}_text_test.csv'
    input_valid_dataset_path =  f'data/Split_Dataset/Data/input_{args.loss_func.lower()}_text_valid.csv'

    tune = TuneBert(train_dataset_path= input_train_dataset_path,
                    valid_dataset_path = input_valid_dataset_path,
                    save_dir = args.save_train,
                    model_name=args.model_name,
                    loss_func=args.loss_func,
                    dropout=args.dropout)
    tune.train(args.save_train, args.batch_size, args.epochs)