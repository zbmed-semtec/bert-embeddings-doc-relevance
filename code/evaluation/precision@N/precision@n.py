""" 
This python script contains the code to compute the precision@N matrix for the
TREC and RELISH datasets.

author: Vishnu Vardhan Dadi
credits: [Leyla Jael Castro, Dietrich Rebholz-Schuhmann]
copyright: GENERAL PUBLIC LICENSE Version 3, 29 June 2007
"""

from multiprocessing import Pool
from typing import List
import argparse
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from numba import njit
from tqdm import tqdm


@njit(fastmath=True)
def get_cosine_similarity(
    u: np.ndarray,
    v: np.ndarray
    )->np.float64:
    """ Computes the cosine similarity between two vectors using numba.
        The cosine similarity is defined as:

                Cos(x, y) = x . y / ||x|| * ||y||

    Args:
        u (np.ndarray): vector 1 of shape (n,) which is embedding1.
        v (np.ndarray): vector 2 of shape (n,) which is embedding2.

    Returns:
        np.float64: cosine similarity between u and v vectors
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


@njit(fastmath=True)
def cosine_similarity_row(
    embeds: np.array,
    row_idx: int,
    n: int 
    )->np.array:
    """ Indentifies the top n most similar pmids in each 
        row of the cosine similarity matrix.

    Args:
        embeds (np.array): array of embeddings for each pmid in the dataset
        row_idx (int): row index of the cosine similarity matrix
        n (int): number of top n similar pmids to retrieve

    Returns:
        np.array: idx of the pmid with the top n most similar pmids
    """
    cs_row = np.zeros((embeds.shape[0],), dtype=np.float32)
    idxs = np.zeros((n), dtype=np.int64)
    for i in range(embeds.shape[0]):
        if i == row_idx:
            embed1 = embeds[i]
            for j in range(embeds.shape[0]):
                embed2 = embeds[j]
                cs_row[j] = get_cosine_similarity(embed1, embed2)
            break


    topn =  np.sort(cs_row)[::-1][1:n+1]

    # finding indexes of top n values in cs_row
    for i in range(n):
        idxs[i] = np.where(cs_row == topn[i])[0][0]

    return idxs

def calculate_tps(
    pmids: np.array,
    rel_pmid1_col: np.array,
    rel_pmid2_col: np.array,
    rel_label_col: np.array,
    embeds: np.array,
    N: int,
    relvance: int = 2
    )->tuple(int, list):
    """ 
    
    Args:
        pmids (np.array): array of all pmids in the dataset
        rel_pmid1_col (np.array): array of pmids in the first column of the relevance file
        rel_pmid2_col (np.array): array of pmids in the second column of the relevance file
        rel_label_col (np.array): array of labels in the relevance column of the relevance file
        embeds (np.array): array of embeddings for each pmid in the dataset
        N (int): number of top n similar pmids to retrieve
        relvance (int): relevance level to consider. Defaults to 2.
        
    Returns:
        tuple(int, list): tuple with column n and presicion array w.r.t to n
    """

    p_array = np.zeros((pmids.shape[0],), dtype=np.float32)
    for i in tqdm(range(len(pmids)),total=len(pmids), desc=f'calc p@{N}'):

        tp =0
        pmid1 = pmids[i]
        topn_idxs = cosine_similarity_row(embeds, row_idx=i, n=N)
        for i in topn_idxs:
            pmid2 = pmids[i]

            if pmid1 == pmid2:
                continue

            pmid1_idxs = np.where(rel_pmid1_col == pmid1)[0]
            pmid2_idxs = np.where(rel_pmid2_col == pmid2)[0]
            common_idx = np.intersect1d(pmid1_idxs, pmid2_idxs)

            if not common_idx:
                pmid1_idxs = np.where(rel_pmid2_col == pmid1)[0]
                pmid2_idxs = np.where(rel_pmid1_col == pmid2)[0]
                common_idx = np.intersect1d(pmid1_idxs, pmid2_idxs)

            if common_idx:
                rel_label = rel_label_col[common_idx[0]]
                if rel_label == relvance:
                    tp += 1
        p_array[i] = tp/N
    return (N,p_array)



def create_precision_matrix(
    embed_path: str,
    rel_path: str,
    save_path: str,
    dataset: str = 'TREC',
    n_values: list = [ 5, 10, 15, 20, 25, 50]
    )->None:
    """ Loads the embeddings and relevance data and calculates the precision@N
       and saves the results in a tsv/pkl file.

    Args:
        embed_path (str): path to the embeddings file
        rel_path (str): path to the three column relevance file
        save_path (str): path to save the precision@N matrix
        dataset (str, optional): Datasets: TREC/RELISH. Defaults to 'TREC'.
        n_values (list, optional): N values to calcuate the precision at.
                                     Defaults to [ 5, 10, 15, 20, 25, 50].
    """

    # load embeddings data frame
    data = pd.read_pickle(embed_path)# compression='gzip')
    pmids  = np.array(data['PMID'].tolist(), dtype=np.int64)
    embeds = np.array(data['embedding'].tolist(), dtype=np.float32)

    # load relevance data frame
    if dataset == 'RELISH':
        rel_data = pd.read_csv(rel_path, sep='\t',
                                 names=['PMID1','PMID2','relevance'])

    elif dataset == 'TREC':
        rel_data = pd.read_csv(rel_path, sep='\t')
    rel_pmid1_col = rel_data['PMID1'].to_numpy(dtype=np.int64)
    rel_pmid2_col = rel_data['PMID2'].to_numpy(dtype=np.int64)
    if dataset == 'TREC':
        rel_label_col = rel_data['Rel-d2d'].to_numpy(dtype=np.int64)
    if dataset == 'RELISH':
        rel_label_col = rel_data['relevance'].to_numpy(dtype=np.int64)

    # precision data frame setup
    pn_df = pd.DataFrame()
    pn_df['PMID'] = pmids

    relevance = 2
    items = [(pmids, rel_pmid1_col, rel_pmid2_col, rel_label_col, embeds, N, relevance) for N in n_values]

    with Pool() as pool:
        results = list(pool.starmap(calculate_tps, items))

    for n,p_list in results:
        pn_df['p@'+str(n)] = p_list


    avg_list = pn_df.mean(axis=0).tolist()
    pn_df.loc['avg'] = avg_list

    if '.tsv' in save_path:
        pn_df.to_csv(save_path, sep='\t')

    if '.pkl' in save_path:
        pn_df.to_pickle(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_path', type=str, required=True,
                        help='path to the embeddings file')

    parser.add_argument('--rel_path', type=str, required=True,
                        help='path to the three column relevance file')

    parser.add_argument('--save_path', type=str, required=True,
                        help='path to save the precision@N matrix')

    parser.add_argument('--dataset', type=str, default='TREC',
                        help='Datasets: TREC/RELISH')
    
    parser.add_argument('--n_values', type=List, default=[ 5, 10, 15, 20, 25, 50],
                        help='N values to calcuate the precision at')

    args = parser.parse_args()
    
    create_precision_matrix(embed_path=args.embed_path,
                            rel_path=args.rel_path,
                            save_path=args.save_path,
                            dataset=args.dataset,
                            n_values=args.n_values)