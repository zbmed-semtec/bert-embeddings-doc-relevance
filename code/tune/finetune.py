"""
This script contains code to fine-tune a pre-trained transformer model on a
new dataset.

author: Vishnu Vardhan Dadi
credits: [Leyla Jael Castro, Dietrich Rebholz-Schuhmann]
copyright: GENERAL PUBLIC LICENSE Version 3, 29 June 2007

maintainer: Vishnu Vardhan Dadi
"""

import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers import evaluation, losses


MODEL_NAME = 'dmis-lab/biobert-large-cased-v1.1'

class TuneBert:
    """
        class to finetune the bert model for a given dataset when relavance
        scores are available.
    """
    def __init__(self, dataset_path: str, rel_data_path: str,
                 model_name: str = MODEL_NAME, batch_size: int = 8):
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
        """

        self.data = pd.read_csv(dataset_path, sep='\t')
        self.rel_data = pd.read_csv(rel_data_path, sep='\t')
        self.sentences1 = []
        self. sentences2 = []
        self.scores = []
        train_data = self.prepare_data()
        self.train_dataloader = DataLoader(train_data, shuffle=True,
                                           batch_size= batch_size)
        self.model = SentenceTransformer(model_name)
        self.train_loss = losses.CosineSimilarityLoss(self.model)
        self.evaluator = evaluation.EmbeddingSimilarityEvaluator(self.sentences1,
                                                self.sentences2, self.scores)

    def prepare_data(self):
        """ Prepares the data for training as per the sentence transformer
            package requirements.
            More info: https://www.sbert.net/docs/training/overview.html
        Returns:
            data (InputExample): list of InputExample objects
        """
        data = []
        abstracts = self.data['abstract'].to_numpy()
        pmid_column =self.data['PMID'].to_numpy()
        for pmid1,pmid2,relv in tqdm(self.rel_data.to_numpy()):
            text1 = abstracts[np.where(pmid_column == pmid1)[0]]
            text2 = abstracts[np.where(pmid_column == pmid2)[0]]
            if relv == 0: label = 0.3
            if relv == 1: label = 0.6
            if relv == 2: label  = 0.9
            if len(text1)>0 and len(text2)>0:
                text1 = text1.tolist()[0]
                text2 = text2.tolist()[0]
                data.append(InputExample(texts=[text1, text2],
                                            label=label))
                self.sentences1.append(text1)
                self.sentences2.append(text2)
                self.scores.append(label)
            else:
                continue

        return data

    def train(self,  save_dir: str, epochs: int = 2, warmup_steps: int = 100,
              evaluation_steps: int = 500):
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

        self.model.fit(train_objectives = [(self.train_dataloader, self.train_loss)],
                        epochs=epochs,
                        warmup_steps = warmup_steps,
                        evaluator = self.evaluator,
                        evaluation_steps = evaluation_steps,
                        output_path = save_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Finetune BioBERT')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='path to the dataset')
    parser.add_argument('--rel_data_path', type=str, required=True,
                        help='path to the relevance data')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='path to save the model')
    parser.add_argument('--epochs', type=int, default=2,
                        help='number of epochs to train')
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help='number of warmup steps')
    parser.add_argument('--evaluation_steps', type=int, default=500,
                        help='number of evaluation steps')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size for training')
    args = parser.parse_args()

    tune = TuneBert(dataset_path=args.dataset_path,
                    rel_data_path=args.rel_data_path,
                    batch_size=args.batch_size)
    tune.train(save_dir=args.save_dir, epochs=args.epochs,
                warmup_steps=args.warmup_steps,
                evaluation_steps=args.evaluation_steps)

