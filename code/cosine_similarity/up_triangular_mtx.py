"""
 This code is to create a upper triangular cosine similarity matrix for all pmids.

author: Vishnu Vardhan Dadi
credits: [Leyla Jael Castro, Dietrich Rebholz-Schuhmann]
copyright: GENERAL PUBLIC LICENSE Version 3, 29 June 2007
"""

import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import njit


@njit(fastmath=True)
def cosine_similarity_numba(u: np.ndarray, v: np.ndarray)->np.float64:
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

def create_cs_matrix(embeds_path: str, save_path: str, compression: bool = True,
                        return_df: bool = False)->pd.DataFrame:
    """ Creates a upper tringular cosine similarity matrix for all pmid pairs
        in the given embeddings file. Saves the matrix as a pickle file.

    Args:
        embeds_path (str): Path to the embeddings file with two columns:
                            'PMID' and 'embedding'.
        save_path (str): Path to save the cosine similarity matrix.
        compression (bool, optional): Whether to compress the matrix or not.
                                      Defaults to True.
        return_df (bool, optional): Whether to return the matrix as a dataframe
                                    or not. Defaults to False.

    Returns:
        pd.DataFrame: Cosine similarity matrix as a dataframe.
                        If return_df is True.
    """
    data = pd.read_pickle(embeds_path)

    pmids = data['PMID'].values
    embeds = data['embedding'].values
    similarity_matrix = np.zeros((len(pmids), len(pmids)))

    for i in tqdm(range(len(pmids)), desc='Computing cosine similarity'):
        embed1 = embeds[i]
        for j in range(len(pmids)):
            embed2 = embeds[j]
            if j<i:
                similarity_matrix[i, j] = round(cosine_similarity_numba(
                                            np.array(embed1),np.array(embed2)), 4)

    df = pd.DataFrame(similarity_matrix, columns=pmids, index=pmids)

    if compression:
        df.to_pickle(save_path, compression='gzip')

    else:
        df.to_pickle(save_path)

    if return_df:
        return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeds_path', type=str, required=True,
                        help='Path to the embeddings file with two columns: \
                            "PMID" and "embedding".')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to save the cosine similarity matrix.')
    parser.add_argument('--compression', type=bool, default=True,
                        help='Whether to compress the matrix or not. Defaults to True.')
    args = parser.parse_args()

    create_cs_matrix(args.embeds_path, args.save_path,
                        args.compression)