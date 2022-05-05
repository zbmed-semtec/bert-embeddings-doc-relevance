import pandas as pd
import nltk
import re
from torch.utils.data.dataset import Dataset


class DataPreprocess:
    """
        Class to preprocess the documents in the corpus
    """
    def __init__(self,tsv_file_path:str):
        self.path = tsv_file_path
        
    def read_data(self):
        """ Reads the data from the tsv file 

        Returns:
            df: pandas dataframe
        """
        data = pd.read_csv(self.path, sep='\t')
        data = data.dropna()
        # combine the title and abstract
        data['text'] = data['title'] + ' ' + data['abstract']
        data = data.drop(['title','abstract'],axis=1)
        return data
    
    def remove_stopwords(self,data:pd.DataFrame):
        """ Removes the stopwords from the data

        Args:
            data: pandas dataframe
        Returns:
            data: pandas dataframe
        """
        stop_words = nltk.corpus.stopwords.words('english')
        data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
        return data
    
    def remove_white_space(self,data:pd.DataFrame):
        """ Removes the white spaces from the text

        Args:
            data: pandas dataframe
        Returns:
            data: pandas dataframe
        """
        data['text'] = data['text'].apply(lambda x: re.sub('\s+', ' ', x))
        return data
        
    def remove_punctuation(self,data:pd.DataFrame):
        """ Removes the punctuation from the data

        Args:
            data: pandas dataframe
        Returns:
            data: pandas dataframe
        """
        data['text'] = data['text'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))
        return data
    
    def stemming(self,data:pd.DataFrame):
        """ Stemming the data

        Args:
            data: pandas dataframe
        Returns:
            data: pandas dataframe
        """
        ps = nltk.stem.PorterStemmer()
        data['text'] = data['text'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split()]))
        return data
    
    def text_normalize(self):
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

    def __len__(self):
        """ Returns number of rows in the dataset
        """
        return len(self.df)
    
    def __getitem__(self, idx:int):
        """ Returns the PMID and text at the index"""
        pmid = self.df.iloc[idx]['PMID']
        text = self.df.iloc[idx]['text']
        return pmid, text


if __name__ == '__main__':
    data = CustomDataset('../data/Input/RELISH/TSV/sample.tsv')
    
    for idx,(pmid, text) in enumerate(data):
        print(pmid, text)
        print("*"*50)
        if idx == 5:
            break
    
    

