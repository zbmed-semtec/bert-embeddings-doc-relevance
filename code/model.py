""" This python file contains code to generate sentence embeddings for the
preprocessed text using the sentence transformers package.

author: Vishnu Vardhan Dadi
credits: [Leyla Jael Castro, Dietrich Rebholz-Schuhmann]
copyright: GENERAL PUBLIC LICENSE Version 3, 29 June 2007

maintainer: Vishnu Vardhan Dadi, Lukas Geist
"""
import pickle as pkl

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from datamodule import CustomDataset


MODEL_NAME = 'dmis-lab/biobert-large-cased-v1.1'
RELISH_DATA_PATH = '../data/Input/RELISH/TSV/sample.tsv'
TREC_DATA_PATH = '../data/Input/TREC/TSV/sample.tsv'
BATCH_SIZE = 64



def generate_embeddings(data: CustomDataset, model: SentenceTransformer,
                        save_path: str, return_df: bool = False):
    """ Generates the embeddings for the data and saves it in a pickle file

    Args:
        data (CustomDataset): CustomDataset object containing the data to be
        embedded model (SentenceTransformer): SentenceTransformer object
        containing the model to be used.

        save_path (str): path to save the pickle file. Defaults to False.

    Returns:
        pd.DataFrame: pandas dataframe or None if return_df is False
    """
    df = pd.DataFrame(columns=['PMID', 'embedding'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("*"*50)
    print('Generating embeddings...')
    for pmid, text in tqdm(data):
        embed = model.encode(text, batch_size = BATCH_SIZE, device = device)
        df = pd.concat([df, pd.DataFrame({'PMID':[pmid], 'embedding':[embed]})])
    pkl.dump(df, open(save_path, 'wb'))
    if return_df:
        return df


if __name__ == '__main__':
    sbert_model = SentenceTransformer(MODEL_NAME)
    relish_data = CustomDataset(RELISH_DATA_PATH)
    trec_data = CustomDataset(TREC_DATA_PATH)
    generate_embeddings(relish_data, sbert_model,
                                '../data/Output/RELISH/relish_embeddings.pkl')
    generate_embeddings(trec_data, sbert_model,
                                    '../data/Output/TREC/trec_embeddings.pkl')



