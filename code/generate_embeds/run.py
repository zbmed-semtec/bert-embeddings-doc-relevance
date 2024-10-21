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
import logging
from .datamodule import CustomDataset
from .utils import get_device


def generate_embeddings(data: CustomDataset, model: SentenceTransformer,
                        save_path: str,
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
    results = []
    device = get_device()
    
    for pmid, text in tqdm(data, desc='Generating embeddings', leave=False):
        embed = model.encode(text, device=device, show_progress_bar=False)
        results.append({'PMID': pmid, 'embedding': embed})
    logging.info("Generated embeddings")

    df = pd.DataFrame(results)
    df.sort_values(by='PMID', inplace=True, ascending=True)
    logging.info("Saved Embeddings")
    df.to_pickle(save_path)
    if return_df:
        return df