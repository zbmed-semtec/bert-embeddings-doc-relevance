""" This python file contains code for the DataModules that will preprocess the
data and returns it in the form of torch dataloaders.

author: Vishnu Vardhan Dadi
credits: [Leyla Jael Castro, Dietrich Rebholz-Schuhmann]
copyright: GENERAL PUBLIC LICENSE Version 3, 29 June 2007

maintainer: Vishnu Vardhan Dadi, Lukas Geist
"""
import os
import re

import pandas as pd
import nltk
from torch.utils.data.dataset import Dataset
import xmltodict



class DataPreprocess:
    """
        Class to preprocess the documents in the corpus
    """
    def __init__(self, file_path: str):
        """ Initializes the class

        Args:
            file_path (str): File path if tsv or, directory path if dir of xml
            files
        """
        self.path = file_path

    def read_data(self)->pd.DataFrame:
        """ Reads the data from the file path

        Returns:
            df: pandas dataframe
        """
        if self.path.endswith('.tsv'):
            data = pd.read_csv(self.path, sep='\t')
            data = data.dropna()
            # combine the title and abstract
            data['text'] = data['title'] + ' ' + data['abstract']
            data = data.drop(['title','abstract'],axis=1)
            return data
        # if path is a directory
        elif os.path.isdir(self.path):
            dict = self.parse_xmls(self.path)
            data = pd.DataFrame.from_dict(dict,orient='index')
            data['PMID'] = data.index
            data = data.reset_index(drop=True)
            data['text'] = data['title'] + ' ' + data['abstract']
            data = data.drop(['title','abstract'],axis=1)
            return data

    @staticmethod
    def parse_xmls(xml_dir:str)->dict:
        """ Fuction to parse the xml file into a dictionary

        Args:
            xml_dir (str) : path to the xml files directory

        Returns:
            {"pmid1":{"title":title,"abstract":abstract},"pmid2"...} :
            dictionary of the xml files,with PMID as key and title and abstract
            as values

        """
        def generate_dict(file_path:str):
            dic = {}
            for file in os.listdir(file_path):
                if file.endswith('.xml'):
                    with open(os.path.join(xml_dir,file),'r') as f:
                        xml_dict = xmltodict.parse(f.read())
                        pmid = xml_dict["collection"]["document"]["id"]
                        title = xml_dict["collection"]["document"]["passage"][0]["text"]
                        abstract = xml_dict["collection"]["document"]["passage"][-1]["text"]
                        dic[pmid] = {'title':title,'abstract':abstract}
            return dic

        if os.path.isdir(xml_dir):
            dict = generate_dict(xml_dir)
            return dict


    @staticmethod
    def remove_stopwords(data:pd.DataFrame)->pd.DataFrame:
        """ Removes the stopwords from the data

        Args:
            data: pandas dataframe
        Returns:
            data: pandas dataframe
        """
        nltk.download('stopwords')
        stop_words = nltk.corpus.stopwords.words('english')
        data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
        return data

    @staticmethod
    def remove_white_space(data:pd.DataFrame)->pd.DataFrame:
        """ Removes the white spaces from the text

        Args:
            data: pandas dataframe
        Returns:
            data: pandas dataframe
        """
        data['text'] = data['text'].apply(lambda x: re.sub('\s+', ' ', x))
        return data

    @staticmethod
    def remove_punctuation(data:pd.DataFrame)->pd.DataFrame:
        """ Removes the punctuation from the data

        Args:
            data: pandas dataframe
        Returns:
            data: pandas dataframe
        """
        data['text'] = data['text'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))
        return data

    @staticmethod
    def stemming(data:pd.DataFrame)->pd.DataFrame:
        """ Stemming the data

        Args:
            data: pandas dataframe
        Returns:
            data: pandas dataframe
        """
        ps = nltk.stem.PorterStemmer()
        data['text'] = data['text'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split()]))
        return data

    def text_normalize(self)->pd.DataFrame:
        """ Normalizes the data

        Args:
            data (pd.DataFrame): pandas dataframe

        Returns:
            data: pandas dataframe
        """

        data = self.read_data()
        data = self.remove_stopwords(data)
        data = self.remove_white_space(data)
        #data = self.remove_punctuation(data)
        #data = self.stemming(data)
        return data

class CustomDataset(Dataset):
    def __init__(self, data_file_path:str):
        super().__init__()
        self.df = DataPreprocess(data_file_path).text_normalize()

    def __len__(self)->int:
        """ Returns number of rows in the dataset
        """
        return len(self.df)

    def __getitem__(self, idx:int):
        """ Returns the PMID and text at the index"""
        pmid = self.df.iloc[idx]['PMID']
        text = self.df.iloc[idx]['text']
        return pmid, text




