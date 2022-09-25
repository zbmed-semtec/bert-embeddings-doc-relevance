"""
This script is a modified version of the code in the following link to support
the current requirements of this repository.
source: https://github.com/zbmed-semtec/medline-preprocessing/tree/main/code/Cosine_Similarity

author: Vishnu Vardhan Dadi
credits: [Rohitha Ravinder, Leyla Jael Castro, Dietrich Rebholz-Schuhmann]
copyright: GENERAL PUBLIC LICENSE Version 3, 29 June 2007

maintainer: Vishnu Vardhan Dadi
"""

import argparse as ap
import pandas as pd
import numpy as np
from tqdm import tqdm
from numba import njit


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

class CosineSimilarity:
    """
    Creates a 4 column matrix by appending cosine similarity scores for all existing pairs
    of PMIDs to the Relevance matrix.
    """
    def __init__(self, embeddings_path: str, rel_matrix_path: str,
                dataset: str = 'TREC'):
        """ Reads the embeddings and relevance matrix from the given paths and
            loads them as a pandas dataframes. Initializes the 4 column matrix
            dataframe.

        Args:
            embeddings_path (str): Path to the embeddings file with two columns
                                   'PMID' and 'embedding'.

            rel_matrix_path (str): Path to the relevance matrix file with three
                                    columns 'PMID1', 'PMID2' and 'Relevance'.

            dataset (str, optional): Name of the dataset. Defaults to 'TREC'.
        """

        if '.pkl' in embeddings_path:
            self.embeds = pd.read_pickle(embeddings_path)
        elif '.tsv' in embeddings_path:
            self.embeds = pd.read_csv(embeddings_path, sep='\t')

        if '.pkl' in rel_matrix_path:
            self.relavance_matrix = pd.read_pickle(rel_matrix_path)
        elif '.tsv' in rel_matrix_path:
            self.relavance_matrix = pd.read_csv(rel_matrix_path, sep='\t')

        self.four_column_matrix = pd.DataFrame()

        if dataset == 'TREC':
            self.four_column_matrix['Rel-d2d'] = self.relavance_matrix['Rel-d2d']
        elif dataset == 'RELISH':
            self.four_column_matrix['Relevance'] = self.relavance_matrix['Relevance']

    def create_relavance_matrix(self, save_dir: str):
        """ Function that handles the creation of the 4 column matrix and saves
            it to the given directory as a pickle/tsv file.

        Args:
            save_dir (str): path to the directory to save the 4 column matrix.
        """
        cs_list = []
        pmid1_list = []
        pmid2_list = []
        rel_list = []
        embedding_column = self.embeds['embedding'].to_numpy()
        pmid_column = self.embeds['PMID'].to_numpy()
        for pmid1, pmid2, rel in tqdm(self.relavance_matrix.to_numpy(),
                                        total=self.relavance_matrix.shape[0]):
            embed1 = embedding_column[np.where(pmid_column == pmid1)[0]]
            embed2 = embedding_column[np.where(pmid_column == pmid2)[0]]
            if not embed1.size == 0 and not embed2.size == 0:
                cs_list.append(round(cosine_similarity_numba(embed1[0], embed2[0]),4))
                pmid1_list.append(pmid1)
                pmid2_list.append(pmid2)
                rel_list.append(rel)
            else:
                continue

        self.four_column_matrix['PMID1'] =  pmid1_list
        self.four_column_matrix['PMID2'] =  pmid2_list
        self.four_column_matrix['Rel-d2d'] =  rel_list
        self.four_column_matrix['Cosine Similarity'] = cs_list

        if '.pkl' in save_dir:
            self.four_column_matrix.to_pickle(save_dir, compression='infer')
        elif '.tsv' in save_dir:
            self.four_column_matrix.to_csv(save_dir, sep='\t', index=False)

if __name__ == '__main__':

    parser = ap.ArgumentParser()
    parser.add_argument('-e', '--embeddings_path', type=str,
                        help='Path to the embeddings file with two columns \
                                "PMID" and "embedding".')
    parser.add_argument('-r', '--rel_matrix_path', type=str,
                        help='Path to the relevance matrix file with three \
                                columns "PMID1", "PMID2" and "Relevance".')
    parser.add_argument('-s', '--save_dir', type=str,
                        help='Path to the directory to save the 4 column matrix.')

    parser.add_argument('-d', '--dataset', type=str, default='TREC',
                        help='Name of the dataset. Defaults to "TREC".')

    args = parser.parse_args()

    mat = CosineSimilarity(embeddings_path=args.embeddings_path,
                            rel_matrix_path=args.rel_matrix_path,
                            dataset=args.dataset)

    mat.create_relavance_matrix(save_dir=args.save_dir)

