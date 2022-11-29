# Finetuning BERT model for document relevance

This directory contains code for finetuning BERT model for document relevance, and follows a supervised learning approach. This is a wrapper code that uses the finetuining functionalty provided by the [sentence-transformers]() library.  

## Data 
The datasets used for experiments are:

- [Text Retrieval Conference TREC (TREC)](https://trec.nist.gov/)
- [RELISH](https://doi.org/10.1093/database/baz138)

To know more about the datasets used in this project and to get these datasets, please refer to the [medline-preprocessing](https://github.com/zbmed-semtec/medline-preprocessing) repository.

To work with other datasets, the input data should contain the following columns:

- `PMID`: PubMed ID of the document
- `title`: Title of the document
- `abstract`: Abstract of the document

The input data should be in a tsv format or a folder containing multiple xml files.

## Data Preprocessing
TREC and RELISH corpora contain pubmed-IDs (PMIDS) of the scientific articles and their corresponding titles and abstracts. To get the full text of the articles, use the [medline-preprocessing](https://github.com/zbmed-semtec/medline-preprocessing) repository. The full text of the articles is stored in the `data` folder of the `medline-preprocessing` repository.

BERT models can be categorized into two types:
 - `Cased`: Cased models are case-sensitive which means the text contain upper and lower case letters. All the special characters are also kept for trainig the model. For example, `Hello World` and `hello world` are different tokens for cased models.

 - `Uncased`: Uncased models are not case-insensitive which means the text is converted to lower case before tokenization. Removing or keeping the special characters is also a choice for uncased models.

As both TREC and RELISH corpora contain titles and abstracts of the scientific articles, there is no need of much preprocessing. Scientific documents won't contain any gebberish text or unneccessary special characters which may require careful preprocessing. So we use the `Cased` models for the experiments. The `Cased` models are better suited for biomedical domain as the biomedical domain contains a lot of abbreviations and acronyms.

The only preprocessing steps applied to these corpora are:
- combining the titles and abstracts of the articles as a single text

- Removing unnecessary white spaces and new lines

## Relevance file
For a supervised learnign approach we need ground truth relevance labels. TREC relevance file and RELSIH relevance file are used for this purpose. This is a tab separated file with 3 columns. The first and second column contains PMIDs and the third column contian relevance label representig the relevance of the PMID in the first column to the PMID in the second column. The relevance label are 0,1 and 2. Here,

- `0` means not relevant
- `1` means parially relevant
- `2` means relevant 

You can find the relevance file for TREC [here](https://github.com/zbmed-semtec/medline-preprocessing/blob/main/data/input/RELISH/RELISH.tsv) and RELSIH [here](https://github.com/zbmed-semtec/medline-preprocessing/blob/main/data/input/TREC/TREC.tsv). The relevance score for RELISH is obtained from the [RELISH relevance json](https://github.com/zbmed-semtec/medline-preprocessing/blob/main/data/input/RELISH/RELISH_v1.json) file. 

The relevance files are expected to in tsv format. The first and second column should contain the PMIDs and the third column should contain the relevance labels.

## Models 

This code should support all the models provided by the [sentence-transformers](https://www.sbert.net/#sentencetransformers-documentation) and [Hugetransformers](https://huggingface.co/models) libraries beacuse it uses the finetuning functionality provided by these libraries.

By default the code uses the `dmis-lab/biobert-base-cased-v1.1` model.


## Loss Functions 

This code supports the following loss functions: 

- `CosineSimilarityLoss`
- `TripletMarginLoss`

These loss function code snippets are used from the [sentence-transformers](https://www.sbert.net/#sentencetransformers-documentation) library.

### [Cosine Similarity Loss](https://www.sbert.net/docs/package_reference/losses.html#cosinesimilarityloss)(CSL)

In cosine similarity loss, the cosine similarity between the query and the Negative document is maximized, while the cosine similarity between the query and the positive document is minimized. The cosine similarity between the embedding pairs is compared with the ground truth relavance score (In our case we get it from the RELISH relevance file). Below table shows the relevance scores from the RELISH relevance document mapped to the cosine similarity scores as per the loss function described in sentence-transformers paper.

| **Relevance** | **Score** | **Cosine Similarity label** |
|:--------------------:|:------------------:|:------------------------------------:|
| High                 | 2                  | 0.9                                  |
| Partial              | 1                  | 0.6                                  |
| Low                  | 0                  | 0.3                                  |


The loss function is defined as:

 $v = M(s1)$ 

 $u = M(s2)$

 $S =  (u * v)/(||u|| * ||v||)$

 $CSL = ||GT - S||_2 $

Here,
- `v` and `u` = embeddings
- `M` = model
- `S` = cosine similarity
- `GT` = ground truth relevance score
- `s1` and `s2` = query and document pair
- CSL = cosine similarity loss (Means squared error between the ground truth and the cosine similarity)


## [Multiple Negative Ranking Loss](https://www.sbert.net/docs/package_reference/losses.html#multiplenegativesrankingloss) (MNRL)

In multiple negative ranking loss, the query and the positive document are compared with multiple negative documents. The loss is calculated as the sum of the maximum margin loss between the query and the positive document and the maximum margin loss between the query and the negative documents. The maximum margin loss is calculated as the difference between the cosine similarity between the query and the positive document and the cosine similarity between the query and the negative document. You can read more about this [here](https://arxiv.org/pdf/1705.00652.pdf).

The loss function is defined as:

$v = M(s1)$ 

$u = M(s2)$

$S =  (u * v)/(||u|| * ||v||)$

$MNRL = max(0, margin - S_{pos} + S_{neg})$

Here,
- `v` and `u` = embeddings
- `M` = model
- `S` = cosine similarity
- `S_pos` = cosine similarity between the query and the positive document
- `S_neg` = cosine similarity between the query and the negative document
- `margin` = margin value 
- `s1` and `s2` = query and document pair
- MNRL = multiple negative ranking loss 

## To Run the code 

### Install the requirements

    pip install -r requirements.txt

### Run the code from termina

```
python3 finetune.py --dataset_path <path to the dataset(tsv/xml-files)>
                    --rel_data_path <path to the relevance file(Three column tsv file)>
                    --save_train <path to save finetuned model>
                    --epochs <number of epochs to finetune, default is 2>
                    --batch_size <batch size, default is 8>
                    --loss_func <loss function to be used for training, default MNRLoss>
```

To change the model, modify the `MODEL_NAME` variable in the `finetune.py` file.





