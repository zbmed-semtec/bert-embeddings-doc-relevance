

import pandas as pd
from tqdm import tqdm


class PrecisionN:
    """ Class for precision@n evaluation mehtod.
    """
    def __init__(self, mat_path: str, tsv_path: str, relavance: str = None):
        """ Initializes the class with the matrix and tsv data paths

        Args:
            mat_path (str): path to the matrix file
            tsv_path (str): path to the tsv file
            relavance (str): if 'simplified' argument is passed, relavance
                             column is conisdered as (1==2).
        """

        if ".tsv" in mat_path:
            self.mat_data = pd.read_csv(mat_path, sep='\t')
        if ".pkl" in mat_path:
            self.mat_data = pd.read_pickle(mat_path)
        self.tsv_data = pd.read_csv(tsv_path, sep='\t')

        self.n_at = [5,10,15,20,25,50]

        #df for storing the precision@n values
        self.pn_df = pd.DataFrame(columns=['PMID'])

        if relavance == 'simplified':
            self.relv = 1 or 2
        else:
            self.relv = 2


    def find_topn(self, n: int):
        """ Finds the top n precision values for each PMID

        Args:
            n (int):  precision @ n value to find

        Returns:
            list: [(pmid1,precision@n),(pmid2,precision@n),...,(pmidn,precision@n)]
        """

        precision_list = []

        # handling matrix data
        indices = self.mat_data.index.tolist()
        self.pn_df['PMID'] = indices
        headers = self.mat_data.columns.tolist()

        for index in tqdm(indices, desc="Finding precision@"+str(n)):
            tp = 0
            visited_pairs = [] # to avoid duplicate pairs
            row = sorted(self.mat_data.loc[index].values.tolist(), reverse=True)
            topn_values = row[1:n]
            pmid1 = index
            for value in topn_values:
                pmid2 = headers[row.index(value)]
                # find relevance of pmid2 in pmid1 in tsv data
                if (pmid1,pmid2) or  (pmid2,pmid1) not in visited_pairs:
                    visited_pairs.append((pmid1,pmid2))
                    visited_pairs.append((pmid2,pmid1))
                    if self.tsv_data.loc[(self.tsv_data['PMID2'] == pmid2) &
                                            (self.tsv_data['PMID1'] == pmid1)]['Rel-d2d'].values == self.relv:
                        tp += 1

            try:
                precision_list.append(tp/n)
            except ZeroDivisionError:
                precision_list.append(0.0)

        return list(zip(indices, precision_list))

    def create_precision_matrix(self, save_path: str):
        """ Creates the precision matrix for each n value and saves it to the
            save_path

        Args:
            save_path (str): path to save the precision matrix
        """

        for n in self.n_at:
            self.pn_df['p@'+str(n)] = [x[1] for x in self.find_topn(n)]

        # adding avg precision to the dataframe
        avg_list = self.pn_df.mean(axis=0).tolist()
        self.pn_df.loc['avg'] = avg_list

        if ".pkl" in save_path:
            self.pn_df.to_pickle(save_path)
        if ".tsv" in save_path:
            self.pn_df.to_csv(save_path, sep='\t', index=False)
