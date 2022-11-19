
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from numba import njit, prange
from numba_progress import ProgressBar


@njit(fastmath=True, nogil=True, parallel=True)
def get_cosine_similarity(u: np.ndarray, v: np.ndarray)->np.float64:
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
    for i in prange(u.shape[0]):
        uv += u[i]*v[i]
        uu += u[i]*u[i]
        vv += v[i]*v[i]
    cos_theta = 1
    if uu!=0 and vv!=0:
        cos_theta = uv/np.sqrt(uu*vv)
    return cos_theta


@njit(fastmath=True, nogil=True, parallel=True)
def cosine_similarity_row(embeds: np.array, row_idx: int, n: int = None):
    cs_row = np.zeros((embeds.shape[0],), dtype=np.float32)
    idxs = np.zeros((n), dtype=np.int64)
    for i in prange(embeds.shape[0]):
        if i == row_idx:
            embed1 = embeds[i]
            for j in prange(embeds.shape[0]):
                embed2 = embeds[j]
                cs_row[j] = get_cosine_similarity(embed1, embed2)
            break


    topn =  np.sort(cs_row)[::-1][1:n+1]

    # finding indexes of top n values in cs_row
    for i in prange(n):
        idxs[i] = np.where(cs_row == topn[i])[0][0]

    return idxs

@njit(fastmath=True, nogil=True, parallel=True)
def intersection_idx(array1, array2):
    idxs = np.zeros(len(array1), dtype=np.int32)
    for i in prange(len(array1)):
        idx = i
        elem1 = array1[i]
        for elem2 in array2:
            if elem1 == elem2:
                idxs[idx] = elem1
    # remove zeros from idxs
    idxs = idxs[idxs != 0]
    return idxs

@njit(fastmath=True, nogil=True, parallel=True)
def calculate_tps(pmids: np.array,
                  rel_pmid1_col: np.array,
                  rel_pmid2_col: np.array,
                  rel_label_col: np.array,
                  embeds: np.array,
                  N: int,
                  relvance: int,
                  progress_proxy: ProgressBar):

    p_array = np.zeros((pmids.shape[0],), dtype=np.float32)
    for i in prange(len(pmids)):

        tp =0
        pmid1 = pmids[i]
        topn_idxs = cosine_similarity_row(embeds, row_idx=i, n=N)
        for i in topn_idxs:
            pmid2 = pmids[i]

            if pmid1 == pmid2:
                continue

            pmid1_idxs = np.where(rel_pmid1_col == pmid1)[0]
            pmid2_idxs = np.where(rel_pmid2_col == pmid2)[0]
            common_idx = intersection_idx(pmid1_idxs, pmid2_idxs)

            if not common_idx:
                pmid1_idxs = np.where(rel_pmid2_col == pmid1)[0]
                pmid2_idxs = np.where(rel_pmid1_col == pmid2)[0]
                common_idx = intersection_idx(pmid1_idxs, pmid2_idxs)

            if common_idx:
                rel_label = rel_label_col[common_idx[0]]
                if rel_label == relvance:
                    tp += 1
        p_array[i] = tp/N
        progress_proxy.update(1)
    return p_array


def create_precision_matrix(embed_path: str, rel_path: str, save_path: str,
                            dataset: str = 'TREC',
                            n_values: list = [ 5, 10, 15, 20, 25, 50]):

    # load embeddings data frame
    data = pd.read_pickle(embed_path)
    pmids  = np.array(data['PMID'].tolist(), dtype=np.int64)
    embeds = np.array(data['embedding'].tolist(), dtype=np.float32)

    # load relevance data frame
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
    for n in n_values:
        print(f'Calculating precision@{n}...')
        with ProgressBar(total=embeds.shape[0]) as progress_proxy:
            p_array = calculate_tps(pmids, rel_pmid1_col, rel_pmid2_col,
                                    rel_label_col, embeds, n, relevance,
                                    progress_proxy)
        pn_df[f'P@{n}'] = p_array

    avg_list = pn_df.mean(axis=0).tolist()
    pn_df.loc['avg'] = avg_list

    if '.tsv' in save_path:
        pn_df.to_csv(save_path, sep='\t')

    if '.pkl' in save_path:
        pn_df.to_pickle(save_path)



if __name__ == '__main__':
    embed_path= 'path to embeds'
    rel_path = 'path to relavance file'
    save_path = 'path to save precision matrix'

    create_precision_matrix(embed_path, rel_path, save_path)