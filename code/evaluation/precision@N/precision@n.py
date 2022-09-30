
import argparse as ap
from traceback import print_tb
import warnings
import concurrent.futures

import pandas as pd
from tqdm import tqdm
import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

class PrecisionN:
    """
        This Class represents the precision@N evaluation method.
    """
    def __init__(self, mat_path: str, rel_path: str, relavance: str = None):
        """ Reads the cosine similarity matrix and the relavance data into a
            pandas dataframe. Initializes the precision@n dataframe and class
            variables.

        Args:
            mat_path (str): path to the cosine similarity matrix
            tsv_path (str): path to the relavance data
            relavance (str): if 'simplified' argument is passed, relavance
                             column is conisdered as (1==2).
        """
        # n values to find precision at
        self.n_at = [5,10,15,20,25,50]

        # read the cosine similarity matrix as a pandas dataframe
        if ".tsv" in mat_path:
            self.mat_data = pd.read_csv(mat_path, sep='\t')

        if ".pkl" in mat_path:
            try:
                self.mat_data = pd.read_pickle(mat_path)
            except:
                self.mat_data = pd.read_pickle(mat_path, compression='gzip')
            else:
                self.mat_data = pd.read_pickle(mat_path,compression='infer')


        # read the relavance data as a pandas dataframe
        if ".tsv" in rel_path:
            self.tsv_data = pd.read_csv(rel_path, sep='\t')

        if ".pkl" in rel_path:
            try:
                self.tsv_data = pd.read_pickle(rel_path)
            except:
                self.tsv_data = pd.read_pickle(rel_path, compression='infer')
            else:
                self.tsv_data = pd.read_pickle(rel_path, compression='gzip')

        # initialize the precision@n dataframe
        self.pn_df = pd.DataFrame(columns=['PMID'])
        for n in self.n_at:
            self.pn_df['p@'+str(n)] = []

        if relavance == 'simplified':
            self.relv = 1 or 2
        else:
            self.relv = 2

        # class variables to handle matrix data
        self.pn_df['PMID'] = self.mat_data.index.tolist()
        self.headers = self.mat_data.columns.tolist()
        self.sort_rows()

    def sort_rows(self):
        """ Function to sort all rows in the cosine similarity matrix in descending
            order."""

        for idx,row in tqdm(enumerate(self.mat_data.values), total=len(self.mat_data),
                        desc="Sorting matrix rows"):
            self.mat_data.iloc[idx] = np.sort(row)[::-1]

    def find_relavance(self, pmid1 : str, pmid2 : str):
        tsv_pmid1 = self.tsv_data['PMID1'].to_numpy()
        tsv_pmid2 = self.tsv_data['PMID2'].to_numpy()
        tsv_rel = self.tsv_data['Rel-d2d'].to_numpy()
        return tsv_rel[np.where((tsv_pmid1 == pmid1) & (tsv_pmid2 == pmid2))]


    def find_topn(self, n: int):
        """ Finds the precision at each 'n' value passes to the function and
            adds it to the initialised precision@n dataframe.

        Args:
            n (int):  precision @ n value to find
        """
        p_list = []
        for index,row in tqdm(zip(self.mat_data.index, self.mat_data.values),
                                total= len(self.mat_data.index),
                                 desc=f"Finding precision@{n}"):

            tp = 0
            pmid1 = index
            topn = row[1:n+1]
            for value in topn:
                pmid2 = self.headers[np.where(row==value)[0][0]]
                if pmid1 == pmid2:
                    continue
                if self.tsv_data.loc[(self.tsv_data['PMID1'] == pmid1) &
                                    (self.tsv_data['PMID2'] == pmid2)]['Rel-d2d'].values == self.relv:
                    tp += 1

            try:
                precision = tp/n
            except ZeroDivisionError as error:
                print(error)
                precision = 0.0

            p_list.append(precision)

        self.pn_df['p@'+str(n)] = p_list


        # for index in tqdm(self.indices, desc="Finding precision@"+str(n)):
        #     tp = 0
        #     pmid1 = index
        #     row =
        #     row = sorted(self.mat_data.loc[index].values.tolist(), reverse=True)
        #     topn_values = row[1:n+1]
        #     for value in topn_values:
        #         pmid2 = self.headers[row.index(value)]
        #         if self.tsv_data.loc[(self.tsv_data['PMID2'] == pmid2) &
        #             (self.tsv_data['PMID1'] == pmid1)]['Rel-d2d'].values == self.relv:

        #             tp += 1

        #     # to avoid division by zero error,
        #     # if no relevant documents are found,
        #     # precision is set to 0
        #     try:
        #         self.pn_df['p@'+str(n)].loc[index] = tp/n
        #     except ZeroDivisionError:
        #         self.pn_df['p@'+str(n)].loc[index] = 0.0

    def create_precision_matrix(self, save_path: str):
        """ Computes the precision at each n value and saves the dataframe to
            pickle/tsv file.

        Args:
            save_path (str): path to save the precision matrix
        """
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(self.find_topn, self.n_at)

        # adding avg precision to the dataframe
        avg_list = self.pn_df.mean(axis=0).tolist()
        self.pn_df.loc['avg'] = avg_list

        if ".pkl" in save_path:
            self.pn_df.to_pickle(save_path)
        if ".tsv" in save_path:
            self.pn_df.to_csv(save_path, sep='\t', index=False)

if __name__ == '__main__':
    aps = ap.ArgumentParser()
    aps.add_argument("-m", "--matrix_path", help="path to the cosine similarity matrix", required=True)
    aps.add_argument("-r", "--relavance_path", help="path to the relavance data", required=True)
    aps.add_argument("-s", "--save_path", help="path to save the precision matrix", required=True)
    aps.add_argument("-rel", "--relavance", help="if 'simplified' argument is passed, \
                                    relavance column is conisdered as (1==2)", default=None)

    args = aps.parse_args()

    precision_n = PrecisionN(mat_path = args.matrix_path,
                             rel_path = args.relavance_path,
                             relavance= args.relavance)
    precision_n.n_at = [5,10,15,20,25,50] # pass only list, if wanted new set of n values
    precision_n.create_precision_matrix(save_path = args.save_path)