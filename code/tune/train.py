"""
status: under development
"""

import argparse as ap
import yaml

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from datamodule import CustomDataset



# constants for this module
MODEL_NAME = 'dmis-lab/biobert-base-cased-v1.1'
DATA_PATH = '../data/Input/TREC/TSV/sample.tsv'

# initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
data = CustomDataset(DATA_PATH)

# initialize dictionary to store tokenized sentences
tokens = {'input_ids': [], 'attention_mask': []}

# loop through the data and tokenize each sentence
for pmid, text in tqdm(data):
    token = tokenizer.encode(text, max_length=128,
                                truncation=True, padding='max_length',
                                return_tensors='pt')

    tokens['input_ids'].append(token['input_ids'][0])
    tokens['attention_mask'].append(token['attention_mask'][0])


# passing the tokenized sentences to the model
# and get the embeddings
outputs = model(**tokens)
embeddings = outputs.last_hidden_state


if __name__ == "__main__":
    # load yaml file with parameters
    arg = ap.ArgumentParser()
    arg.add_argument("-c", "--config", required=True, help="config file path")
    args = vars(arg.parse_args())
    config_file = args["config"]

    with open(config_file, 'r') as file:
        try:
            config = yaml.safe_load(file)

        except yaml.YAMLError as exc:
            print(exc)
            exit(1)







