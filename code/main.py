""" This python file contains code that shows fucniton usage to perfrom semantic search
and returns closer embedings, finding the cosine similarity.

author: Vishnu Vardhan Dadi
credits: [Leyla Jael Castro, Dietrich Rebholz-Schuhmann]
copyright: GENERAL PUBLIC LICENSE Version 3, 29 June 2007

maintainer: Vishnu Vardhan Dadi, Lukas Geist
"""

from utils import cosine_similarity_matrix, query_similar_pmids

MODEL_NAME = 'dmis-lab/biobert-large-cased-v1.1'
PKL_FILE_PATH = '../data/Output/TREC/trec_embeddings.pkl'
SAVE_PKL_FILE_PATH = '../data/Output/TREC/trec_similarity_matrix.pkl'


if __name__ == '__main__':
    # create cosine similarity matrix
    cosine_matrix = cosine_similarity_matrix(pkl_path = PKL_FILE_PATH,
                                            save_path = SAVE_PKL_FILE_PATH,
                                            return_matrix = True)

    # # find the closest pmids for below queries
    # queries = ['Investigating global trends in paraquat intoxication research \
    #            from 1962 to 2015 using bibliometric analysis.',
    #            'Environmental implications for disaster preparedness: lessons \
    #            learnt from the Indian Ocean Tsunami.',
    #            'Trends in and contributions to entrepreneurship research: a   \
    #             broad review of literature from 1996 to June 2012.']

    # pmids = query_similar_pmids(pkl_file_path = PKL_FILE_PATH, queries = queries,
    #                             model_name = MODEL_NAME, top_k = 2)



#eXfgznqxZ7PmPNF