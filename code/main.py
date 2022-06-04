import pickle as pkl
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search
import torch
from sklearn.metrics.pairwise import cosine_similarity


# reads the pickle file and return the data
def read_pickle(file_path:str):
    with open(file_path, 'rb') as f:
        data = pkl.load(f)
    return data

def create_similarity_matrix():
    pass
    


if __name__ == '__main__':
    file_path = '../data/Output/RELISH/relish_embeddings.pkl'
    data = read_pickle(file_path)
    embedder = SentenceTransformer('dmis-lab/biobert-large-cased-v1.1')
    queries = ['Investigating global trends in paraquat intoxication research from 1962 to 2015 using bibliometric analysis.',
               'Environmental implications for disaster preparedness: lessons learnt from the Indian Ocean Tsunami.',
               'Trends in and contributions to entrepreneurship research: a broad review of literature from 1996 to June 2012.']
    embeds = data['embedding'].tolist()
    for query in queries:
        q = embedder.encode(query,convert_to_tensor=True, device=torch.device('cpu'))
        similars = semantic_search(q, embeds, top_k=2)[0]
        print("query:", query)
        for similar in similars:
            id = similar["corpus_id"]
            score = similar["score"]
            pmid = data.iloc[id]['PMID']
            print(f'pmid:{pmid}|| score:{score}')
        print("-"*50)
    
    