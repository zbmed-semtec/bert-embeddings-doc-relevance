import argparse
import os
import sys
# sys.path.insert(0, '../')
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers import evaluation, losses, models
from sentence_transformers.util import cos_sim
from datasets import load_dataset
from dataset import CreateFineTuneDataset


logging.basicConfig(filename='training.log',  # Specify the log file name
                    level=logging.INFO,       # Set the logging level
                    format='%(asctime)s - %(levelname)s - %(message)s') 

# from generate_embeds.datamodule import DataPreprocess

class TuneBert:
    """
        class to finetune the bert model for a given dataset when relavance
        scores are available.
    """
    def __init__(self, train_dataset_path: str,
                valid_dataset_path, model_name: str, loss_func: str
                 ):
        """
            Initialize the class with the dataset and the relavance scores
            as a pandas dataframe. Prepares the data for training. Loads the
            model and prepares the dataloader and loss function.

        Args:
            train_dataset_path (str): path to the dataset to be used for training.
                                Expects the data to have three columns: text1, text2 and
                                label.
            test_data_path (str): path to the dataset to be used for evaluation. 
                                Expects the data to have two columns: .
            model_name (str, optional): name of the model to be used.
            loss_func (str, optional): loss function to be used for training.
                                        ('CosiineSimilarityLoss'/'MNRLoss')
                                        Defaults to MNRLoss.
        """
        self.train_data = load_dataset("csv", data_files=train_dataset_path)['train']
        print(self.train_data)
        # self.valid_data = load_dataset("csv", data_files=valid_dataset_path)['train']
        # print(self.valid_data)

        self.model = SentenceTransformer(model_name)

        self.loss_func = loss_func
        
        if loss_func == 'CosineSimilarityLoss':
            self.train_loss = losses.CosineSimilarityLoss(self.model)
        if loss_func == 'MNR':
            self.train_loss = losses.MultipleNegativesRankingLoss(self.model)
        if loss_func == 'Softmax':
            self.train_loss = losses.SoftmaxLoss(self.model, self.model.get_sentence_embedding_dimension(), num_labels=3)


    def prepare_data(self):
        data = []
        abstracts = self.text_data['text'].to_numpy()
        pmid_column = self.text_data['PMID'].to_numpy()
        for pmid1,pmid2,relv in tqdm(self.rel_data.to_numpy(),
                                        desc='Preparing train-data'):
            text1 = abstracts[np.where(pmid_column == pmid1)[0]]
            text2 = abstracts[np.where(pmid_column == pmid2)[0]]
            if self.loss_func == 'CosineSimilarityLoss':
                if relv == 0: label = 0.3
                if relv == 1: label = 0.6
                if relv == 2: label  = 0.9
                if len(text1)>0 and len(text2)>0:
                    text1 = text1.tolist()[0]
                    text2 = text2.tolist()[0]
                    data.append(InputExample(texts=[text1, text2],
                                                label=label))
                else:
                    continue
        
    def prepare_data(self):
        data = []
        abstracts = self.text_data['text'].to_numpy()
        pmid_column = self.text_data['PMID'].to_numpy()
        for pmid1,pmid2,relv in tqdm(self.rel_data.to_numpy(),
                                        desc='Preparing train-data'):
            text1 = abstracts[np.where(pmid_column == pmid1)[0]]
            text2 = abstracts[np.where(pmid_column == pmid2)[0]]
            if self.loss_func == 'CosineSimilarityLoss':
                if relv == 0: label = 0.3
                if relv == 1: label = 0.6
                if relv == 2: label  = 0.9
                if len(text1)>0 and len(text2)>0:
                    text1 = text1.tolist()[0]
                    text2 = text2.tolist()[0]
                    data.append(InputExample(texts=[text1, text2],
                                                label=label))
                else:
                    continue

    def train(self, save_dir, batch_size, epochs):
        batch_size = batch_size
        epochs = epochs
        args = SentenceTransformerTrainingArguments(
                output_dir = save_dir,
                per_device_train_batch_size=batch_size,
                # per_device_eval_batch_size=batch_size,
                num_train_epochs = epochs,
                # eval_strategy = 'epoch',
                learning_rate = 2e-5,
                warmup_ratio = 0.1,
                # do_train=True,
                # do_eval=True
            	)

        trainer = SentenceTransformerTrainer(
            model = self.model,
            args = args,
            train_dataset = self.train_data,
            # eval_dataset = self.valid_data,
            loss = self.train_loss
        )        

        logging.info(trainer.train())

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Finetune BioBERT')
    parser.add_argument('-i', '--input_train_dataset_path', type=str, required=True,
                        help='path to the dataset')
    parser.add_argument('-v', '--input_valid_dataset_path', type=str, required=True,
                        help='path to the dataset')
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
    args = parser.parse_args()

    # data = CreateFineTuneDataset(loss_func=args.loss_func, classes=args.classes)

    input_train_dataset_path = f'input_{args.loss_func.lower()}_text_train.csv'
    input_test_dataset_path =  f'input_{args.loss_func.lower()}_text_train.csv'
    input_valid_dataset_path =  f'input_{args.loss_func.lower()}_text_train.csv'

    tune = TuneBert(train_dataset_path= args.input_train_dataset_path,
                    valid_dataset_path = args.input_valid_dataset_path,
                    model_name=args.model_name,
                    loss_func=args.loss_func)
    tune.train(args.save_train, args.batch_size, args.epochs)