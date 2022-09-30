"""
Source: https://github.com/zbmed-semtec/medline-preprocessing/blob/main/code/Distribution_Analysis/ROC_curve.py
"""

import sys
import math
import pandas as pd

import logging
import numpy as np

from matplotlib import pyplot as plt
from typing import List

import counting_table as ct

__version__ = "0.2.1"
__author__ = "Guillermo Rocamora Pérez"
__author__ = "Muhammad Talha"

def count_TP(i: int, data: pd.DataFrame, dataset: str = "RELISH", repurposed: bool = False) -> int:
    """
    Counts the true positives for a given counting table and cut point.
    Parameters
    ----------
    i: int
        Index of the Cosine Interval to calculate the metric.
    data: pd.DataFrame
        DataFrame containing the counting table.
    dataset: str, optional
        String to determine the dataset. Must be either RELISH or TREC, by
        default "RELISH".
    repurposed: bool, optional
        Boolean to determine whether the data is from the TREC repurposed
        file or not.
    Returns
    -------
    int
        True positives calculated from a counting table given a cut point.
    """
    # If data is from the RELISH or TREC-repurposed file
    if dataset == "RELISH" or (dataset == "TREC" and repurposed == True):
        return sum(data["2s"][i:])
    # If data is from the TREC-simplified file
    elif dataset == "TREC" and repurposed == False:
        return sum(data["As"][i:])

def count_FP(i: int, data: pd.DataFrame, dataset: str = "RELISH", repurposed: bool = False) -> int:
    """
    Counts the false positives for a given counting table and cut point.
    Parameters
    ----------
    i: int
        Index of the Cosine Interval to calculate the metric.
    data: pd.DataFrame
        DataFrame containing the counting table.
    dataset: str, optional
        String to determine the dataset. Must be either RELISH or TREC, by
        default "RELISH".
    repurposed: bool, optional
        Boolean to determine whether the data is from the TREC repurposed
        file or not.
    Returns
    -------
    int
        False positives calculated from a counting table given a cut point.
    """
    # If data is from the RELISH or TREC-repurposed file
    if dataset == "RELISH" or (dataset == "TREC" and repurposed == True):
        return sum(data["0s"][i:])
    # If data is from the TREC-simplified file
    elif dataset == "TREC" and repurposed == False:
        return sum(data["Cs"][i:])


def count_FN(i: int, data: pd.DataFrame, dataset: str = "RELISH", repurposed: bool = False) -> int:
    """
    Counts the false negatives for a given counting table and cut point.
    Parameters
    ----------
    i: int
        Index of the Cosine Interval to calculate the metric.
    data: pd.DataFrame
        DataFrame containing the counting table.
    dataset: str, optional
        String to determine the dataset. Must be either RELISH or TREC, by
        default "RELISH".
    repurposed: bool, optional
        Boolean to determine whether the data is from the TREC repurposed
        file or not.
    Returns
    -------
    int
        False negatives calculated from a counting table given a cut point.
    """
    # If data is from the RELISH or TREC-repurposed file
    if dataset == "RELISH" or (dataset == "TREC" and repurposed == True):
        return sum(data["2s"][:i])
    # If data is from the TREC-simplified file
    elif dataset == "TREC" and repurposed == False:
        return sum(data["As"][:i])

def count_TN(i: int, data: pd.DataFrame, dataset: str = "RELISH", repurposed: bool = False) -> int:
    """
    Counts the true negatives for a given counting table and cut point.
    Parameters
    ----------
    i: int
        Index of the Cosine Interval to calculate the metric.
    data: pd.DataFrame
        DataFrame containing the counting table.
    dataset: str, optional
        String to determine the dataset. Must be either RELISH or TREC, by
        default "RELISH".
    repurposed: bool, optional
        Boolean to determine whether the data is from the TREC repurposed
        file or not.
    Returns
    -------
    int
        True negatives calculated from a counting table given a cut point.
    """
    # If data is from the RELISH or TREC-repurposed file
    if dataset == "RELISH" or (dataset == "TREC" and repurposed == True):
        return sum(data["0s"][:i])
    # If data is from the TREC-simplified file
    elif dataset == "TREC" and repurposed == False:
        return sum(data["Cs"][:i])


def calculate_TPR(i: int, data: pd.DataFrame, dataset: str = "RELISH", repurposed: bool = False) -> float:
    """
    Calculates the true positive rate for a given counting table and cut point.
    Parameters
    ----------
    i: int
        Index of the Cosine Interval to calculate the metric.
    data: pd.DataFrame
        DataFrame containing the counting table.
    dataset: str, optional
        String to determine the dataset. Must be either RELISH or TREC, by
        default "RELISH".
    repurposed: bool, optional
        Boolean to determine whether the data is from the TREC repurposed
        file or not.
    Returns
    -------
    float
        True positive rate for a given cut-off point in the counting table.
    """
    TP = count_TP(i, data, dataset, repurposed)
    FN = count_FN(i, data, dataset, repurposed)
    return(TP/(TP+FN))

def calculate_FPR(i: int, data: pd.DataFrame, dataset: str = "RELISH", repurposed: bool = False) -> float:
    """
    Calculates de false positive rate for a given counting table and cut point.
    Parameters
    ----------
    i: int
        Index of the Cosine Interval to calculate the metric.
    data: pd.DataFrame
        DataFrame containing the counting table.
    dataset: str, optional
        String to determine the dataset. Must be either RELISH or TREC, by
        default "RELISH".
    repurposed: bool, optional
        Boolean to determine whether the data is from the TREC repurposed
        file or not.
    Returns
    -------
    float
        False positive rate for a given cut-off point in the counting table.
    """
    FP = count_FP(i, data, dataset, repurposed)
    TN = count_TN(i, data, dataset, repurposed)
    return(FP/(FP+TN))

def generate_roc_values(counting_table: pd.DataFrame, dataset: bool = "RELISH", repurposed: bool = False) -> pd.DataFrame:
    """
    Generates from a counting table the TPR and FPR values required to plot the
    ROC curve.

    Parameters
    ----------
    counting_table : pd.DataFrame
        DataFrame containing the counting table.
    dataset: str, optional
        String to determine the dataset. Must be either RELISH or TREC, by
        default "RELISH".
    repurposed: bool, optional
        Boolean to determine whether the data is from the TREC repurposed
        file or not.

    Returns
    -------
    pd.DataFrame
        DataFrame containing counting table filled with the ROC curve values.
    """
    ct.verify_dataset(dataset, repurposed)

    counting_table["TPR"] = 0
    counting_table["FPR"] = 0

    for i, row in counting_table.iterrows():
        row["TPR"] = calculate_TPR(i, counting_table, dataset, repurposed)
        row["FPR"] = calculate_FPR(i, counting_table, dataset, repurposed)

        counting_table.iloc[i] = row

    return counting_table

def calculate_best_index(counting_table: pd.DataFrame) -> int:
    """
    Calculates the index of the optimal cut-off point to discriminate between two
    categories.

    Parameters
    ----------
    counting_table : pd.DataFrame
        DataFrame containing counting table filled with the ROC curve values.

    Returns
    -------
    int
        Index of the counting table for the optimal cut-off point.
    """
    if not "TPR" in counting_table.columns or not "FPR" in counting_table.columns:
        logging.error("The counting table must contain the TPR and FPR values. Please, use the function generate_roc_values() before.")
        sys.exit("ROC curve values not found in the counting table.")

    distance = []
    for _, row in counting_table.iterrows():
        ideal_p = [0, 1]
        current_p = [row["FPR"], row["TPR"]]
        distance.append(math.dist(ideal_p, current_p))

    return np.argmin(distance)

def calculate_best_cosine_interval(counting_table: pd.DataFrame) -> float:
    """
    Calculates the optimal Cosine Interval cut-off point to discriminate between two
    categories.

    Parameters
    ----------
    counting_table : pd.DataFrame
        DataFrame containing counting table filled with the ROC curve values.

    Returns
    -------
    float
        Optimal cut-off point for the given counting table.
    """
    best_index = calculate_best_index(counting_table)
    return counting_table["Cosine Interval"][best_index]

def integrate_curve(x: List[float], y: List[float]) -> float:
    """
    Calculates the area under the curve by approximating each interval as a
    rectangle.

    Parameters
    ----------
    x : List[float]
        X values of the curve.
    y : List[float]
        Y values of the curve.

    Returns
    -------
    float
        Calculated area under the ROC curve.
    """
    sm = 0
    for i in range(1, len(x)):
        h = x[i] - x[i-1]
        sm += h * (y[i-1] + y[i]) / 2

    return sm

def calculate_auc(counting_table: pd.DataFrame) -> float:
    """
    Calculates the area under the ROC curve.

    Parameters
    ----------
    counting_table : pd.DataFrame
        DataFrame containing counting table filled with the ROC curve values.

    Returns
    -------
    float
        Calculated area under the ROC curve.
    """
    if not "TPR" in counting_table.columns or not "FPR" in counting_table.columns:
        logging.error("The counting table must contain the TPR and FPR values. Please, use the function generate_roc_values() before.")
        sys.exit("ROC curve values not found in the counting table.")
    return integrate_curve(counting_table["FPR"].tolist()[::-1], counting_table["TPR"].tolist()[::-1])

def draw_roc_curve(counting_table: pd.DataFrame, draw_auc: bool = True, show_figure: bool = True, output_path: str = None) -> None:
    """
    Plots the ROC curve and the best cutoff point. Optionally, it also displays
    the area under the curve.

    Parameters
    ----------
    counting_table : pd.DataFrame
        DataFrame containing counting table filled with the ROC curve values.
    draw_auc : bool, optional
        Boolean to determine whether to show the area under the curve, by default True.
    show_figure : bool, optional
        Boolean to determine whether to print the pyplot figure, by default
        True.
    output_path : str, optional
        If an output path is given, the figure will be saved, by default None.
    """
    if not "TPR" in counting_table.columns or not "FPR" in counting_table.columns:
        logging.error("The counting table must contain the TPR and FPR values. Please, use the function generate_roc_values() before.")
        sys.exit("ROC curve values not found in the counting table.")
    tpr = counting_table["TPR"].values.tolist()
    fpr = counting_table["FPR"].values.tolist()

    best_row = counting_table.iloc[calculate_best_index(counting_table)]
    fig = plt.figure(figsize=(4, 3), dpi = 200, facecolor="w", edgecolor="k")
    plt.plot(fpr, tpr, linewidth=1, markersize = 2, c='blue', marker="D", mec='k', mfc = "k")
    plt.plot(best_row["FPR"], best_row["TPR"], "or", markersize = 3)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC curve")

    if draw_auc:
        auc = calculate_auc(counting_table)
        ax = plt.gca()
        plt.text(0.02, 0.95, f"AUC = {auc:.2f}", fontsize = 9, horizontalalignment='left', verticalalignment='top', transform = ax.transAxes)

    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, facecolor="white")

    if show_figure:
        plt.show()
    plt.close()