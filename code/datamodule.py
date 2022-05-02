import pandas as pd
import nltk
import re


class DataPreprocess:
    """
        Class to preprocess the documents in the corpus
    """
    def __init__(self,tsv_file_path):
        self.path = tsv_file_path
        
    def read_data(self):
        """ Reads the data from the tsv file 

        Returns:
            df: pandas dataframe
        """
        data = pd.read_csv(self.path, sep='\t')
        data = data.dropna()
        return data
    
    def remove_stopwords(self,data):
        """ Removes the stopwords from the data

        Args:
            data: pandas dataframe
        Returns:
            data: pandas dataframe
        """
        stop_words = nltk.corpus.stopwords.words('english')
        data['title'] = data['title'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
        data["abstract"] = data["abstract"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
        return data
    
    def remove_white_space(self,data):
        """ Removes the white spaces from the data

        Args:
            data: pandas dataframe
        Returns:
            data: pandas dataframe
        """
        data['title'] = data['title'].apply(lambda x: re.sub('\s+', ' ', x))
        data['abstract'] = data['abstract'].apply(lambda x: re.sub('\s+', ' ', x))
        return data
    
    def remove_punctuation(self,data):
        """ Removes the punctuation from the data

        Args:
            data: pandas dataframe
        Returns:
            data: pandas dataframe
        """
        data['title'] = data['title'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))
        data['abstract'] = data['abstract'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))
        return data
    
    def stemming(self,data):
        """ Stemming the data

        Args:
            data: pandas dataframe
        Returns:
            data: pandas dataframe
        """
        ps = nltk.stem.PorterStemmer()
        data['title'] = data['title'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split()]))
        data['abstract'] = data['abstract'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split()]))
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
        #data = self.remove_white_space(data)
        #data = self.remove_punctuation(data)
        data = self.stemming(data)
        return data
        
        
if __name__ == "__main__":
    RELISH_DATA_PATH = "../data/Input/RELISH/TSV/sample.tsv"
    TREC_DATA_PATH = "../data/Input/TREC/TSV/sample.tsv"
    dp_relish = DataPreprocess(RELISH_DATA_PATH)
    dp_trec = DataPreprocess(TREC_DATA_PATH)
    data_relish = dp_relish.text_normalize()
    data_trec = dp_trec.text_normalize()
    
    # save processed dataframe to pickle file
    data_relish.to_pickle("../data/Output/RELISH/sample.pkl")
    data_trec.to_pickle("../data/Output/TREC/sample.pkl")
    # load pickle file
    relish_data = pd.read_pickle("../data/Output/RELISH/sample.pkl")
    trec_data = pd.read_pickle("../data/Output/TREC/sample.pkl")
    print(relish_data.head())
    print(trec_data.head())
    
    

