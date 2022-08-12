
import pandas as pd


class PrecisionN:
    def __init__(self, mat_path: str, tsv_path: str):
        """ Initializes the class with the matrix and tsv data paths

        Args:
            mat_path (str): path to the matrix file
            tsv_path (str): path to the tsv file
        """

        if ".tsv" in mat_path:
            self.mat_data = pd.read_csv(mat_path, sep='\t')
        if ".pkl" in mat_path:
            self.mat_data = pd.read_pickle(mat_path)
        self.tsv_data = pd.read_csv(tsv_path, sep='\t')

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
        headers = self.mat_data.columns.tolist()

        for index in indices:
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
                                            (self.tsv_data['PMID1'] == pmid1)]['Rel-d2d'].values[0] == 2:
                        tp += 1

            try:
                precision_list.append(tp/n)
            except ZeroDivisionError:
                precision_list.append(0.0)

        return list(zip(indices, precision_list))

if __name__ == '__main__':
    mat_path = "../trec_similarity_matrix_up.pkl"
    trec_repurposed = '../trec_repurposed_matrix.tsv'
    precision_n = PrecisionN(mat_path, trec_repurposed)
    print(precision_n.find_topn(10))