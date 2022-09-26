""" This python file contains code to generate sentence embeddings for the
preprocessed text using the sentence transformers package.

author: Vishnu Vardhan Dadi
credits: [Leyla Jael Castro, Dietrich Rebholz-Schuhmann]
copyright: GENERAL PUBLIC LICENSE Version 3, 29 June 2007

maintainer: Vishnu Vardhan Dadi
"""
import os
import warnings
import argparse

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from datamodule import CustomDataset
from utils import get_device


def generate_embeddings(data: CustomDataset, model: SentenceTransformer,
                        save_path: str, batch_size: int = 64,
                        return_df: bool = False)->pd.DataFrame:
    """ Generates the embeddings for the data and saves it in a pickle file

    Args:
        data (CustomDataset): CustomDataset object containing the data to be
        embedded model (SentenceTransformer): SentenceTransformer object
        containing the model to be used.

        model (SentenceTransformer): SentenceTransformer object containing the
                                     model to be used.
        save_path (str): path to save the embeddings in a pickle file

        batch_size (int, optional): batch size to be used for generating the
                                    embeddings. Defaults to 64. Applies only
                                    if GPU is available.

        return_df (bool, optional): whether to return the dataframe containing
                                    two columns: PMID and embedding. Defaults
                                    to False.

    Returns:
        pd.DataFrame: pandas dataframe or None if return_df is False
    """
    df = pd.DataFrame(columns=['PMID', 'embedding'])
    print("*"*50)
    for pmid, text in tqdm(data, desc='Generating embeddings'):
        embed = model.encode(text, batch_size = batch_size, device = get_device())
        df = pd.concat([df, pd.DataFrame({'PMID':[pmid], 'embedding':[embed]})])
        df.sort_values(by='PMID', inplace=True, ascending=True)
    df.to_pickle(save_path)
    if return_df:
        return df


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='path to the pre-processed data')
    parser.add_argument('--model_name', type=str, help='name of the model to be used')
    parser.add_argument('--save_path', type=str, help='path to save the embeddings')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size to be used')

    args = parser.parse_args()

    MODEL_NAME = args.model_name
    DATA_PATH = args.data_path
    TREC_DATA_PATH = args.data_path
    BATCH_SIZE = args.batch_size
    SAVE_PATH = args.save_path

    sbert_model = SentenceTransformer(MODEL_NAME)
    data = CustomDataset(DATA_PATH)
    generate_embeddings(data= data, model= sbert_model, save_path= SAVE_PATH,
                        batch_size= BATCH_SIZE)



