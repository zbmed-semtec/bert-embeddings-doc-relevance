"""
This script contains code to fine-tune a pre-trained transformer model on a
new dataset.

author: Vishnu Vardhan Dadi
credits: [Leyla Jael Castro, Dietrich Rebholz-Schuhmann]
copyright: GENERAL PUBLIC LICENSE Version 3, 29 June 2007

maintainer: Vishnu Vardhan Dadi
"""

import argparse
import os
import sys
sys.path.insert(0, '../')

import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers import evaluation, losses

from generate_embeds.datamodule import DataPreprocess


MODEL_NAME = 'dmis-lab/biobert-base-cased-v1.1'

class TuneBert:
    """
        class to finetune the bert model for a given dataset when relavance
        scores are available.
    """
    def __init__(self, dataset_path: str, rel_data_path: str,
                 model_name: str = MODEL_NAME, batch_size: int = 8,
                 loss_func: str = 'MNRLoss'):
        """
            Initialize the class with the dataset and the relavance scores
            as a pandas dataframe. Prepares the data for training. Loads the
            model and prepares the dataloader and loss function.

        Args:
            dataset_path (str): path to the dataset to be used for training.
                                Expects the data to have two columns: PMID and
                                abstract.
            rel_data_path (str): path to the relevance data. Expects the data
                                to have three columns: PMID1, PMID2 and Relavance.
            model_name (str, optional): name of the model to be used.
                                        Defaults to bio_bert_large_cased.
            batch_size (int, optional): batch size for training. Defaults to 8.
            loss_func (str, optional): loss function to be used for training.
                                        ('CosiineSimilarityLoss'/'MNRLoss')
                                        Defaults to MNRLoss.
        """

        # loaidng and preparing the data
        data_preprocess = DataPreprocess(dataset_path)
        self.text_data = data_preprocess.text_normalize()
        self.rel_data = pd.read_csv(rel_data_path, sep='\t')
        self.train_data = self.prepare_data()
        self.train_dataloader = DataLoader(self.train_data, shuffle=True,
                                           batch_size= batch_size)

        # creating the model object
        self.model = SentenceTransformer(model_name)

        # preparing the loss function
        self.loss_func = loss_func
        if loss_func == 'CosineSimilarityLoss':
            self.train_loss = losses.CosineSimilarityLoss(self.model)
        if loss_func == 'MNRLoss':
            self.train_loss = losses.MultipleNegativesRankingLoss(self.model)

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

            elif self.loss_func == 'MNRLoss': # Multiple negatives ranking loss
                if relv == 2:
                    label = 1
                    if len(text1)>0 and len(text2)>0:
                        text1 = text1.tolist()[0]
                        text2 = text2.tolist()[0]
                        data.append(InputExample(texts=[text1, text2],
                                                    label=label))
                else:
                    continue

        return data

    def train(self,  save_dir: str, epochs: int = 2):
        """
            Trains the model for the given number of epochs and saves the
            model to the disk.
        Args:
            save_dir (str): path to save the model.
            epochs (int, optional): number of epochs to train. Defaults to 2.
            warmup_steps (int, optional): number of warmup steps. Defaults to 100.
            evaluation_steps (int, optional): number of evaluation steps.
                                                Defaults to 500.
        """

        warmup_steps = int(len(self.train_dataloader) * epochs * 0.1)
        self.model.fit(train_objectives = [(self.train_dataloader, self.train_loss)],
                        epochs=epochs,
                        warmup_steps = warmup_steps,
                        output_path = save_dir)

    def evaluate(self, save_path: str):
        """
            Evaluates the model on the test data with
        Args:
            save_path (str): path to save the evaluation results.
        """
        evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
                            self.train_data, name='train', batch_size=8,
                                show_progress_bar=True, write_csv=True)

        evaluator(self.model, output_path=save_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Finetune BioBERT')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='path to the dataset')
    parser.add_argument('--rel_data_path', type=str, required=True,
                        help='path to the relevance data')
    parser.add_argument('--save_train', type=str, required=True,
                        help='path to save the model')
    parser.add_argument('--save_eval', type=str, default=os.getcwd(),
                        help='path to save the evaluation results')
    parser.add_argument('--epochs', type=int, default=2,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size for training')
    parser.add_argument('--loss_func', type=str, default='MNRLoss',
                        help='loss function to be used for training')
    args = parser.parse_args()

    tune = TuneBert(dataset_path=args.dataset_path,
                    rel_data_path=args.rel_data_path,
                    batch_size=args.batch_size,
                    loss_func=args.loss_func)
    tune.train(save_dir=args.save_dir, epochs=args.epochs)