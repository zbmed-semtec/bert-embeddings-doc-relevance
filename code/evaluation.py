import pickle as pkl
import numpy as np
import pandas as pd


class PrecisionN:
    def __init__(self, mat_path: str, tsv_path: str,
                    embed_path: str):

        with open(mat_path, 'rb') as file:
            self.similarity_matrix = pkl.load(file)

        self.tsv_data = pd.read_csv(tsv_path, sep='\t')
        self.embed_df = pd.read_pickle(embed_path)

    def sort_








if __name__ == '__main__':
    mat_path = "../trec_similarity_matrix_up.pkl"
    trec_repurposed = '../trec_repurposed_matrix.tsv'
    embeds_pkl = '../data/Output/TREC/trec_embeddings.pkl'
    precision_n = PrecisionN(mat_path, trec_repurposed, embeds_pkl)
    print(precision_n.similarity_matrix)
    print(precision_n.tsv_data.head())