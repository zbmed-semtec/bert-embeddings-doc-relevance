import pandas as pd
import numpy as np


def create_text_datasets():
    main_file = pd.read_csv('data/RELISH_documents_2022628.tsv', sep='\t')
    test_file = np.load("dataset/test_normal.npy", allow_pickle=True)
    pmids = [np.int64(line[0]) for line in test_file]
    filtered_main_file = main_file[main_file['PMID'].isin(pmids)]
    filtered_main_file.to_csv("dataset/input_test_text_data.tsv", sep='\t', index=False)
