"""
This code contains a wrapper class for the distribution analysis method.

Source: https://github.com/zbmed-semtec/medline-preprocessing/tree/main/code/Distribution_Analysis
docs: https://github.com/zbmed-semtec/medline-preprocessing/tree/main/docs/Distribution_Analysis

author: Vishnu Vardhan Dadi
copyright: GENERAL PUBLIC LICENSE Version 3, 29 June 2007
"""

import argparse as ap

import pandas as pd

import counting_table as ct
import ROC_curve as roc


class DistributionAnalysis:
    """ Wrapper class for the DistributionAnalysis"""
    def __init__(self, four_col_mat: str, save_path: str, dataset: str = "TREC",
                    repurposed: bool = False):

        self.four_col_mat = four_col_mat
        self.dataset = dataset
        self.repurposed = repurposed
        self.save_path = save_path

        self.table: pd.DataFrame
        self.roc_vals: pd.DataFrame
        self.AUC: float

    def gen_count_table(self):
        """ Generates a count table from the four column matrix, creates a
            counting distribution plot and saves it to the save_path.
        """
        rel_mat = ct.load_relevance_matrix(self.four_col_mat)

        self.table = ct.create_counting_table(data=rel_mat, dataset=self.dataset,
                                            repurposed=self.repurposed)

        save_dir = self.save_path + "/" + self.dataset + "_count_distribution_plot.jpg"
        ct.plot_graph(self.table, dataset = self.dataset, repurposed=self.repurposed,
                         normalize = True, output_path= save_dir)

    def gen_roc_curve(self):
        """ Generates a ROC curve from the four column matrix, creates a
            ROC curve plot and saves it to the save_path.
        """

        self.roc_vals = roc.generate_roc_values(self.table, dataset=self.dataset,
                                                repurposed=self.repurposed)

        save_dir = self.save_path + "/" + self.dataset + "_ROC_curve.jpg"

        roc.draw_roc_curve(self.roc_vals, draw_auc=True, show_figure = True,
                            output_path= save_dir)

        self.AUC = round(roc.calculate_auc(self.roc_vals), 4)

    def get_AUC(self)-> float:
        return self.AUC

    def get_count_table(self)-> pd.DataFrame:
        return self.table

    def get_roc_values(self)-> pd.DataFrame:
        return self.roc_vals



if __name__ == '__main__':

    aps = ap.ArgumentParser()
    aps.add_argument("-f", "--four_col_mat", required=True,
                    help="path to the four column matrix")
    aps.add_argument("-s", "--save_path", required=True,
                    help="path to save the plots")
    aps.add_argument("-d", "--dataset", required=False, default="TREC",
                    help="dataset name: 'TREC'/'RELISH'")
    aps.add_argument("-r", "--repurposed", required=False, default=False,
                    help="repurposed dataset: True/False")

    args = aps.parse_args()

    da = DistributionAnalysis(four_col_mat=args.four_col_mat,
                                save_path=args.save_path,
                                dataset=args.dataset,
                                repurposed=args.repurposed)

    da.gen_count_table()
    da.gen_roc_curve()
    print("*"*50)
    print(f'Area under the ROC curve: {da.get_AUC()}')
    print("*"*50)


