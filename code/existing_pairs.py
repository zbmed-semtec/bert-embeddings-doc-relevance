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

from utils import cosine_similarity_numba


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

        self.four_column_matrix = pd.DataFrame(columns=['PMID1', 'PMID2', 'Rel-d2d'])
        self.four_column_matrix['PMID1'] = self.relavance_matrix['PMID1']
        self.four_column_matrix['PMID2'] = self.relavance_matrix['PMID2']

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
        embedding_column = self.embeds['embedding'].to_numpy()
        pmid_column = self.embeds['PMID'].to_numpy()
        for pmid1, pmid2, _ in tqdm(self.relavance_matrix.to_numpy(),
                                        total=self.relavance_matrix.shape[0]):
            embed1 = embedding_column[np.where(pmid_column == pmid1)[0]]
            embed2 = embedding_column[np.where(pmid_column == pmid2)[0]]
            if not embed1.size == 0 and not embed2.size == 0:
                cs_list.append(cosine_similarity_numba(embed1[0], embed2[0]))
            else:
                cs_list.append('')

        self.four_column_matrix['Cosine Similarity'] = cs_list

        if '.pkl' in save_dir:
            self.four_column_matrix.to_pickle(save_dir, compression='infer')
        elif '.tsv' in save_dir:
            self.four_column_matrix.to_csv(save_dir, sep='\t')

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

