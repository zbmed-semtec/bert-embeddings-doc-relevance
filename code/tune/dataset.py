import os
import sys
import pandas as pd

class CreateFineTuneDataset():
    """
    Class to prepare datasets for fine-tuning a BERT model with different loss functions
    by transforming document pairs and relevance labels into a format compatible with 
    specific loss function requirements.
    """
    def __init__(self, loss_func: str, classes: int):
        """
        Initializes the CreateFineTuneDataset with specified loss function and class configuration.
        Loads data, creates text mappings, and formats datasets according to loss type.

        Args:
            loss_func (str): The loss function type ('softmax', 'mnr', or 'contrastive').
            classes (int): Number of relevance classes (2 or 3).
        
        """
        self.lossfunction = loss_func
        self.classes = classes
        self.main_df = None
        self.train_df = None
        self.test_df = None
        self.eval_df = None
        self.pmid_dict = None

        self.load_data()
        self.create_loss_specific_data()
    
    def load_data(self):
        """
        Loads datasets and initializes the PMIDs to text mapping.
        Calls read_files to read individual data files and then maps text based on document IDs.
        """
        self.read_files()
        self.pmid_dict = self.map_text()

    def read_files(self):
        """
        Reads in the main dataset and relevance datasets for training, testing, and evaluation.
        """
        input_file = 'data/Split_Dataset/Data/relish_documents.tsv'
        train_ground_truth = 'data/Split_Dataset/Ground_truth/train.tsv'
        test_ground_truth = 'data/Split_Dataset/Ground_truth/test.tsv'
        valid_ground_truth = 'data/Split_Dataset/Ground_truth/valid.tsv'

        self.main_df = pd.read_csv(input_file, sep='\t')
        self.train_df = pd.read_csv(train_ground_truth, sep='\t')
        self.test_df = pd.read_csv(test_ground_truth, sep='\t')
        self.eval_df = pd.read_csv(valid_ground_truth, sep='\t')
        

    def map_text(self):
        """
        Maps each document ID (PMID) to its corresponding combined title and abstract text.
        Returns:
            dict: Dictionary mapping PMIDs to combined text strings.
        """
        self.main_df['combined_text'] = self.main_df['title'] + " " + self.main_df['abstract']
        self.main_df['PMID'] = self.main_df['PMID'].astype(int)
        return dict(zip(self.main_df['PMID'], self.main_df['combined_text']))

    def create_loss_specific_data(self):
        """
        Prepares and saves loss-specific data by calling the appropriate method
        based on the specified loss function (softmax, mnr, or contrastive).
        """
        if self.lossfunction == 'softmax':
            self.create_softmax_loss_data()
        elif self.lossfunction == 'mnr':
            return self.create_mnr_loss_data()
        elif self.lossfunction == 'contrastive':
            return self.create_contrastive_loss_data()

    def create_softmax_loss_data(self):
        """
        Creates and saves softmax-compatible data for training, testing, and evaluation.
        Each dataset includes text pairs and relevance labels for softmax classification.
        """
        self.train_df['text1'] = self.train_df['PMID1'].map(self.pmid_dict)
        self.train_df['text2'] = self.train_df['PMID2'].map(self.pmid_dict)

        self.test_df['text1'] = self.test_df['PMID1'].map(self.pmid_dict)
        self.test_df['text2'] = self.test_df['PMID2'].map(self.pmid_dict)

        self.eval_df['text1'] = self.eval_df['PMID1'].map(self.pmid_dict)
        self.eval_df['text2'] = self.eval_df['PMID2'].map(self.pmid_dict)


        self.train_df[['text1', 'text2', 'Relevance']].rename(columns={'Relevance': 'label'}).to_csv(('data/Split_Dataset/Data/input_softmax_text_train.csv'), index=False)
        self.test_df[['text1', 'text2', 'Relevance']].rename(columns={'Relevance': 'label'}).to_csv(('data/Split_Dataset/Data/input_softmax_text_test.csv'), index=False)
        self.eval_df[['text1', 'text2', 'Relevance']].rename(columns={'Relevance': 'label'}).to_csv(('data/Split_Dataset/Data/input_softmax_text_valid.csv'), index=False)


    def create_mnr_loss_data(self):
        """
        Creates and saves Multiple Negatives Ranking (MNR)-compatible data for training and evaluation.
        Filters pairs by relevance score and maps them to anchor-positive pairs for MNR training.
        """
        if self.classes == 3:
            filtered_train_df = self.train_df[self.train_df['Relevance'] == 2].copy()
            filtered_train_df['anchor'] = filtered_train_df['PMID1'].map(self.pmid_dict)
            filtered_train_df['positive'] = filtered_train_df['PMID2'].map(self.pmid_dict)
            filtered_train_df[['anchor', 'positive']].to_csv('data/Split_Dataset/Data/input_mnr_text_train.csv', index=False)

            filtered_eval_df = self.eval_df[self.eval_df['Relevance'] == 2].copy()
            filtered_eval_df['anchor'] = filtered_eval_df['PMID1'].map(self.pmid_dict)
            filtered_eval_df['positive'] = filtered_eval_df['PMID2'].map(self.pmid_dict)
            filtered_eval_df[['anchor', 'positive']].to_csv('data/Split_Dataset/Data/input_mnr_text_valid.csv', index=False)

        elif self.classes == 2:
            filtered_train_df = self.train_df[self.train_df['Relevance'].isin([2, 1])].copy()
            filtered_train_df['anchor'] = filtered_train_df['PMID1'].map(self.pmid_dict)
            filtered_train_df['positive'] = filtered_train_df['PMID2'].map(self.pmid_dict)
            filtered_train_df[['anchor', 'positive']].to_csv('data/Split_Dataset/Data/input_mnr_text_train.csv', index=False)

            filtered_eval_df = self.eval_df[self.eval_df['Relevance'] == 2].copy()
            filtered_eval_df['anchor'] = filtered_eval_df['PMID1'].map(self.pmid_dict)
            filtered_eval_df['positive'] = filtered_eval_df['PMID2'].map(self.pmid_dict)
            filtered_eval_df[['anchor', 'positive']].to_csv('data/Split_Dataset/Data/input_mnr_text_valid.csv', index=False)

    def create_contrastive_loss_data(self):
        """
        Creates and saves contrastive loss-compatible data for training and evaluation.
        Transforms pairs to include binary labels indicating relevance for contrastive training.
        """
        output_train_rows = []
        for _, row in self.train_df.iterrows():
            pmid1_text = self.pmid_dict.get(row['PMID1'], '')
            pmid2_text = self.pmid_dict.get(row['PMID2'], '')
            label = 1 if (self.classes == 3 and row['Relevance'] == 2) or (self.classes == 2 and row['Relevance'] in [1, 2]) else 0
            output_train_rows.append({'text1': pmid1_text, 'text2': pmid2_text, 'label': label})

        output_train_df = pd.DataFrame(output_train_rows)
        output_train_df.to_csv('data/Split_Dataset/Data/input_contrastive_text_train.csv', index=False)

        output_eval_rows = []
        for _, row in self.eval_df.iterrows():
            pmid1_text = self.pmid_dict.get(row['PMID1'], '')
            pmid2_text = self.pmid_dict.get(row['PMID2'], '')
            label = 1 if (self.classes == 3 and row['Relevance'] == 2) or (self.classes == 2 and row['Relevance'] in [1, 2]) else 0
            output_eval_rows.append({'text1': pmid1_text, 'text2': pmid2_text, 'label': label})

        output_eval_df = pd.DataFrame(output_eval_rows)
        output_eval_df.to_csv('data/Split_Dataset/Data/input_contrastive_text_valid.csv', index=False)