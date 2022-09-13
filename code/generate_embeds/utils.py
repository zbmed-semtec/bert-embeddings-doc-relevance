""" This python file contain utility functions to use embeddings.

author: Vishnu Vardhan Dadi
credits: [Leyla Jael Castro, Dietrich Rebholz-Schuhmann]
copyright: GENERAL PUBLIC LICENSE Version 3, 29 June 2007

maintainer: Vishnu Vardhan Dadi, Lukas Geist
"""

import os
import warnings
import pickle as pkl
from typing import List

import torch
import numpy as np
from tqdm import tqdm
from numba import njit
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search, cos_sim

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def read_pickle(file_path: str):
    """ Reads the pickle file and return the data"""
    with open(file_path, 'rb') as file:
        data = pkl.load(file)
    return data


def cosine_similarity_matrix(pkl_path: str, save_path: str, return_matrix: bool = False):
    """ Generates the cosine similarity matrix for the embeddings in the pickle

    Args:
        pkl_path (str): path to the pickle file containing the embeddings
        save_path (str): path to save the cosine similarity matrix
        return_matrix (bool, optional): whether to return the matrix or not.
                                        Defaults to False.

    Returns:
        Matrix: torch.Tensor or None if return_matrix is False
    """
    data = read_pickle(pkl_path)
    pmids = data['PMID']
    matrix  = torch.zeros((len(pmids), len(pmids)))
    for i, pmid1 in enumerate(tqdm(pmids)):
        embed1 = data.loc[data['PMID'] == pmid1, 'embedding'].values[0].reshape(1, -1)
        for j, pmid2 in enumerate(pmids):
            embed2 = data.loc[data['PMID'] == pmid2, 'embedding'].values[0].reshape(1, -1)
            matrix[i, j] =  cos_sim(embed1, embed2)

    pkl.dump(matrix, open(save_path, 'wb'))
    if return_matrix:
        return matrix


@njit(fastmath=True)
def cosine_similarity_numba(u:np.ndarray, v:np.ndarray):
    """ Computes the cosine similarity between two
        vectors using numba.

    Args:
        u (np.ndarray): vector 1
        v (np.ndarray): vector 2

    Returns:
        _type_: float - cosine similarity
    """
    assert(u.shape[0] == v.shape[0])

    uv = 0
    uu = 0
    vv = 0
    for i in range(u.shape[0]):
        uv += u[i]*v[i]
        uu += u[i]*u[i]
        vv += v[i]*v[i]
    cos_theta = 1
    if uu!=0 and vv!=0:
        cos_theta = uv/np.sqrt(uu*vv)
    return cos_theta


def query_similar_pmids(pkl_file_path: str, queries: List, model_name: str, top_k: int = 2):
    """ Finds the top k similar pmids for the given queries.

    Args:
        pkl_file_path (str): path to the pickle file containing the embeddings
        queries (List): list of queries to find the similar pmids
        model_name (str): name of the model to be used
        top_k (int, optional): number of similar pmids to be returned. Defaults to 2.

    Returns:
        List: list of similar pmids
    """
    data = read_pickle(pkl_file_path)
    embedder = SentenceTransformer(model_name)
    embeds = data['embedding'].tolist()
    similari_pmids = []
    for query in queries:
        q = embedder.encode(query,convert_to_tensor=True, device =torch.device('cpu'))
        similars = semantic_search(q, embeds, top_k=top_k)[0]
        similari_pmids.append(similars)
        print("query:", query)
        for similar in similars:
            id = similar["corpus_id"]
            score = similar["score"]
            pmid = data.iloc[id]['PMID']
            print(f'pmid:{pmid}|| score:{score}')
        print("-"*50)
    return similari_pmids

def get_device():
    """ Returns GPU if available else returns CPU.
        Prints GPU info if GPU is available."""

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_current = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_current)
        device = torch.device('cuda')
        print(f'Found {device_count} GPU(s). \
              Using GPU {device_current}.    \
              Device name: {device_name}')
    else:
        device = torch.device('cpu')
        print('No GPU found. Using CPU.')
    return device


if __name__ == '__main__':
    device = get_device()