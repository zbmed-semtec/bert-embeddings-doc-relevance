"""
Source: https://github.com/zbmed-semtec/medline-preprocessing/blob/main/code/Distribution_Analysis/counting_table.py

"""

import math
import sys

import numpy as np
import pandas as pd
import logging

from typing import Tuple
from matplotlib import pyplot as plt


logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

__version__ = "0.2.3"
__author__ = "Guillermo Rocamora Pérez"
__author__ = "Muhammad Talha"

def verify_dataset(dataset: str, repurposed: bool) -> None:
    """
    Verifies if the dataset name provided is valid. Either RELISH or TREC. If
    the dataset is RELISH, it checks that the repurposed argument is set to
    False.

    Parameters
    ----------
    dataset : str
        The input dataset.
    repurposed : bool
        Whether the TREC dataset is simplified or repurposed.
    """
    if not dataset in ["RELISH", "TREC"]:
        logger.error("The dataset must be either TREC or RELISH.")
        sys.exit("Invalid dataset provided.")
    if dataset == "RELISH" and repurposed == True:
        logger.warning('There is no "repurposed" option for RELISH dataset.')

def verify_matrix_columns(data: pd.DataFrame) -> None:
    """
    Verifies if the input relevance matrix has the required column names for
    the program to work properly. The required fields are:

    * PMID 1: specified with "PMID1" or "PMID Reference".
    * PMID 2: specified with "PMID2" or "PMID Assessed".
    * Relevance: specified with "relevance", "Relevance", "Group", "Rel-d2d" or
      "Relevance Assessment".
    * Cosine Similarity: specified with "Cosine Similarity".

    Parameters
    ----------
    data : pd.DataFrame
        _description_
    """
    valid_pmid1 = ["PMID1", "PMID Reference"]
    valid_pmid2 = ["PMID2", "PMID Assessed"]

    if not any(valid_str in data.columns for valid_str in valid_pmid1):
        logger.error('No valid PMID1 column found in the specified file.')
        sys.exit("Invalid relevance matrix")
    if not any(valid_str in data.columns for valid_str in valid_pmid2):
        logger.error('No valid PMID2 column found in the specified file.')
        sys.exit("Invalid relevance matrix")

    valid_relevance = ["relevance", "Relevance", "Group", "Rel-d2d", "Relevance Assessment"]
    if not any(valid_str in data.columns for valid_str in valid_relevance):
        logger.error('No valid Relevance column found in the specified file.')
        sys.exit("Invalid relevance matrix")

    if not "Cosine Similarity" in data.columns:
        logger.error("No valid Cosine Similarity column found in the specified file.")
        sys.exit("Invalid relevance matrix")

def load_relevance_matrix(input_path: str) -> pd.DataFrame:
    """
    Reads a .TSV file containing the relevance matrix. Three columns are
    needed: PMID1, PMID2 and relevance (for RELISH), Group (for TREC
    simplified) or Rel-d2d (for TREC repurposed).

    Parameters
    ----------
    input_path: str
        File path to the Relevance Matrix.

    Returns
    -------
    data: pd.DataFrame
        Dataframe with at least three columns: PMID 1, PMID 2, and either
         relevance (for RELISH), Group (for TREC simplified) or Rel-d2d (for
         TREC repurposed).
    """
    if not input_path.endswith(".tsv"):
        logging.warning("A tab separated value (.TSV) file is recommended.")

    data = pd.read_csv(input_path, sep = "\t")
    verify_matrix_columns(data)

    # It is important to rename the columns so that from now on, the relevance
    # matrix has: PMID1, PMID2 and relevance/Group/Rel-d2d
    data.rename(columns = {"PMID Reference": "PMID1", "PMID Assessed": "PMID2", "Relevance Assessment": "relevance", "Relevance": "relevance"}, inplace = True)
    return data

def count_entries(data: pd.DataFrame, interval: float, dataset: str = "RELISH", repurposed: bool = False) -> dict:
    """
    Counts the number of Relevance Assessments or Groups for a given value of
    Cosine Similarity.

    Parameters
    ----------
    data: pd.DataFrame
        Input dataframe with 4 columns: PMID 1, PMID 2, relevance/group and
        Cosine Similarity.
    interval: float
        Value of Cosine Similarity to count the entries.
    dataset: str, optional
        String to determine the dataset. Must be either RELISH or TREC, by
        default "RELISH".
    repurposed: bool, optional
        Boolean to determine whether the data is from the TREC repurposed
        file or not.

    Returns
    -------
    counter: dict
        Dictionary containing the counts for each relevance/group.
    """
    verify_dataset(dataset, repurposed)

    if dataset == "RELISH":
        filtered_df = data[data["Cosine Similarity"] == interval]["relevance"]
        counter = {0: sum(filtered_df == 0), 1: sum(filtered_df == 1), 2: sum(filtered_df == 2)}
    elif dataset == "TREC" and repurposed == False:
        filtered_df = data[data["Cosine Similarity"] == interval]["Group"]
        counter = {'A': sum(filtered_df == 'A'), 'B': sum(filtered_df == 'B'), 'C': sum(filtered_df == 'C')}
    elif dataset == "TREC" and repurposed == True:
        filtered_df = data[data["Cosine Similarity"] == interval]["Rel-d2d"]
        counter = {0: sum(filtered_df == 0), 1: sum(filtered_df == 1), 2: sum(filtered_df == 2)}

    return counter

def create_counting_table(data: pd.DataFrame, dataset: str = "RELISH", repurposed: bool = False) -> pd.DataFrame:
    """
    Creates the "counting table" from a given Relevance matrix.

    Parameters
    ----------
    data: pd.DataFrame
        Input dataframe with 4 columns: PMID 1, PMID 2, relevance/group and
        Cosine Similarity.
    dataset: str, optional
        String to determine the dataset. Must be either RELISH or TREC, by
        default "RELISH".
    repurposed: bool, optional
        Boolean to determine whether the data is from the TREC repurposed
        file or not.

    Returns
    -------
    counting_df: pd.DataFrame
        DataFrame of the counting table generated.
    """
    verify_dataset(dataset, repurposed)

    if dataset == "RELISH" or (dataset == "TREC" and repurposed == True):
        counting_df = pd.DataFrame({"Cosine Interval":  np.round(np.linspace(0, 1, 101), 2).tolist(), "2s": 0, "1s": 0, "0s": 0})

        for i, row in counting_df.iterrows():
            interval = row["Cosine Interval"]
            interval_counts = count_entries(data, interval, dataset = dataset,
                                                        repurposed = repurposed)

            counting_df.at[i, "2s"] = interval_counts[2]
            counting_df.at[i, "1s"] = interval_counts[1]
            counting_df.at[i, "0s"] = interval_counts[0]

    elif dataset == "TREC" and repurposed == False:
        counting_df = pd.DataFrame({"Cosine Interval":  np.round(np.linspace(0, 1, 101), 2).tolist(),
                                    "As": 0, "Bs": 0, "Cs": 0})

        for i, row in counting_df.iterrows():
            interval = row["Cosine Interval"]
            interval_counts = count_entries(data, interval, dataset = dataset,
                                                        repurposed = repurposed)

            counting_df.at[i, "As"] = interval_counts['A']
            counting_df.at[i, "Bs"] = interval_counts['B']
            counting_df.at[i, "Cs"] = interval_counts['C']

    return counting_df

def hp_create_counting_table(data: pd.DataFrame, dataset: str = "RELISH", repurposed: bool = False) -> pd.DataFrame:
    """
    Creates the "counting table" from a given Relevance matrix in the
    hyperparameter optimization process. The main difference from
    create_counting_table() function is that the relevant groups (either 2s and
    1s or As and Bs) are joined together to discriminate between relevant and
    non-relevant publications.

    Parameters
    ----------
    data: pd.DataFrame
        Input dataframe with 4 columns: PMID 1, PMID 2, relevance/group and
        Cosine Similarity.
    dataset: str, optional
        String to determine the dataset. Must be either RELISH or TREC, by
        default "RELISH".
    repurposed: bool, optional
        Boolean to determine whether the data is from the TREC repurposed
        file or not.

    Returns
    -------
    counting_df: pd.DataFrame
        DataFrame of the counting table generated.
    """
    verify_dataset(dataset, repurposed)

    if dataset == "RELISH" or (dataset == "TREC" and repurposed == True):
        counting_df = pd.DataFrame({"Cosine Interval":  np.round(np.linspace(0, 1, 101), 2).tolist(), "2s": 0, "0s": 0})

        for i, row in counting_df.iterrows():
            interval = row["Cosine Interval"]
            interval_counts = count_entries(data, interval, dataset, repurposed)

            counting_df.at[i, "2s"] = interval_counts[2] + interval_counts[1]
            counting_df.at[i, "0s"] = interval_counts[0]
    elif dataset == "TREC" and repurposed == False:
        counting_df = pd.DataFrame({"Cosine Interval":  np.round(np.linspace(0, 1, 101), 2).tolist(),
                                    "As": 0, "Cs": 0})

        for i, row in counting_df.iterrows():
            interval = row["Cosine Interval"]
            interval_counts = count_entries(data, interval, dataset, repurposed)

            counting_df.at[i, "As"] = interval_counts['A'] + interval_counts['B']
            counting_df.at[i, "Cs"] = interval_counts['C']

    return counting_df


def plot_graph(data: pd.DataFrame, dataset: str = "RELISH", repurposed: bool = False, normalize: bool = False, show_figure: bool = True, output_path: str = None, best_cosine: float = None) -> None:
    """
    Plots the graph of "Relevance counting" against the "Cosine intervals" for
    the number of the different relevances/groups found in the input counting
    table.

    Parameters
    ----------
    data: pd.DataFrame
        DataFrame containing the counting table.
    dataset: str, optional
        String to determine the dataset. Must be either RELISH or TREC, by
        default "RELISH".
    repurposed: bool, optional
        Boolean to determine whether the data is from the TREC repurposed file
        or not.
    normalize: bool, optional
        Boolean to determine whether to normalize the plotted histograms so
        that the sum adds up to one, by default False.
    show_figure: bool, optional
        Boolean to determine whether to print the pyplot figure, by default
        True.
    output_path : str, optional
        If an output path is given, the figure will be saved, by default None.
    best_cosine: float, optional
        If provided, an orange vertical line will be drawn at that cosine
        similarity. It is used to plot the best cosine interval to split
        between relevance and non-relevant.
    """
    verify_dataset(dataset, repurposed)
    intervals = data["Cosine Interval"].values.tolist()

    fig = plt.figure(figsize=(4, 3), dpi = 200, facecolor="w", edgecolor="k")
    if dataset == "RELISH" or (dataset == "TREC" and repurposed == True):
        two_points = data["2s"].values.tolist()
        zero_points = data["0s"].values.tolist()

        if normalize:
            two_points = [i/sum(two_points) for i in two_points]
            zero_points = [i/sum(zero_points) for i in zero_points]

        plt.plot(intervals, two_points, 'r', label='2 counts')

        if "1s" in data.columns:
            one_points = data["1s"].values.tolist()
            if normalize:
                one_points = [i/sum(one_points) for i in one_points]
            plt.plot(intervals, one_points, 'b', label='1 counts')

        plt.plot(intervals, zero_points, 'g', label='0 counts')
    elif dataset == "TREC":
        two_points = data["As"].values.tolist()
        zero_points = data["Cs"].values.tolist()

        if normalize:
            two_points = [i/sum(two_points) for i in two_points]
            zero_points = [i/sum(zero_points) for i in zero_points]

        plt.plot(intervals, two_points, 'r', label='A counts')

        if "Bs" in data.columns:
            one_points = data["Bs"].values.tolist()
            if normalize:
                one_points = [i/sum(one_points) for i in one_points]
            plt.plot(intervals, one_points, 'b', label='B counts')
        plt.plot(intervals, zero_points, 'g', label='C counts')

    plt.xlabel("Cosine intervals")
    plt.ylabel("Relevance counting")

    if(best_cosine):
        plt.axvline(x = best_cosine, color = "orange", label = "Cut-off")

    plt.legend()
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, facecolor="white")

    if show_figure:
        plt.show()
    plt.close()

def save_table(counting_df: pd.DataFrame, output_path: str) -> None:
    """
    Saves the counting table into .TSV format.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the counting table.
    output_path : str
        Output path to where the counting table will be saved.
    """
    counting_df.to_csv(output_path, index=False, sep = "\t")

def load_table(input_path: str) -> pd.DataFrame:
    """
    Reads the counting table stored in .TSV format.

    Parameters
    ----------
    input_path : str
        Input path to where the counting table is located.

    Returns
    -------
    data : pd.DataFrame
        DataFrame containing the counting table.
    """
    if not input_path.endswith(".tsv"):
        logging.warning("A tab separated value (.TSV) file is recommended.")

    data = pd.read_csv(input_path, sep = "\t")
    return data