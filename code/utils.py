""" This python file contain utility functions to use embeddings.

author: Vishnu Vardhan Dadi
credits: [Leyla Jael Castro, Dietrich Rebholz-Schuhmann]
copyright: GENERAL PUBLIC LICENSE Version 3, 29 June 2007

maintainer: Vishnu Vardhan Dadi, Lukas Geist
"""
import pickle as pkl
from typing import List

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search



def read_pickle(file_path:str):
    """ Reads the pickle file and return the data"""
    with open(file_path, 'rb') as f:
        data = pkl.load(f)
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
    for i, pmid1 in enumerate(pmids):
        embed1 = data.loc[data['PMID'] == pmid1, 'embedding'].values[0].reshape(1, -1)
        for j, pmid2 in enumerate(pmids):
            embed2 = data.loc[data['PMID'] == pmid2, 'embedding'].values[0].reshape(1, -1)
            matrix[i, j] =  torch.cosine_similarity(embed1, embed2)

    pkl.dump(matrix, open(save_path, 'wb'))
    if return_matrix:
        return matrix

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
        q = embedder.encode(query,convert_to_tensor=True, device=torch.device('cpu'))
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
