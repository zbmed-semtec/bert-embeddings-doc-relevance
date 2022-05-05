from sentence_transformers import SentenceTransformer
import pandas as pd

# get the first abstract
df = pd.read_csv('../data/Input/RELISH/TSV/sample.tsv', sep='\t')

abst1 = df.iloc[0]['abstract']
abst2 = df.iloc[1]['abstract']
query = [abst1+abst2]

#tokenized_query = tokenizer(query,padding=True,return_tensors="pt")


#embed = model(tokenized_query)

sbert_model = SentenceTransformer('dmis-lab/biobert-large-cased-v1.1')

embed = sbert_model.encode(query)
print(embed.shape)