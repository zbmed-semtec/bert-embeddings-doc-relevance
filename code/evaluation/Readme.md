# Evaluation of the doc-to-doc relevance task with BERT embeddings

Three different evaluation approaches are used to find the best model to perform the doc-to-doc relevance task on TREC and RELISH datasets. The evaluation approaches are:

-  [Distribution analysis](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/docs/Distribution_Analysis#distribution-analysis)
- [Discounted cumulative gain method](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/code/Evaluation#calculation-of-doc-2-doc-normalised-discounted-cumulative-gain-ndcgn)
- [Precision@N method](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/code/precision%40N#create-precision-matrix)


**Note:** This folder contains wrapper scripts to run the evaluation of the doc-to-doc relevance task with BERT embeddings. The original evaluation scripts are available at [medline-preprocessing](https://github.com/zbmed-semtec/medline-preprocessing) repository. The scripts in this folder are adapted to work with the BERT embeddings.


## Distribution analysis


Table with five columns:
- **Model**: Name of the model used to generate the embeddings.
- **Tuned**: If the model was tuned or not.
- **Loss-function**: Loss function used to fine-tune the model.
- **Epochs**: Number of epochs used to fine-tune the model.
- **AUC**: value of the area under the curve (AUC) of the distribution analysis.

The loss fucntions used for the fine-tuning are:

- **MNL:** Multiple NegativeRanking Loss
- **CSL:** Cosine Similarity Loss

**TREC-repurposed:** The table below shows the results of the distribution analysis for the TREC-repurposed dataset.

| Model                    | Fine-tuning | Loss-Function | Epochs | AUC  |
|:------------------------:|:-----------:|:-------------:|:------:|:----:|
| biobert-base-cased-v1.1  | No          | -             | -      | 0.77 |
| biobert-base-cased-v1.1  | Yes         | MNL           | 2      | 0.81 |
| biobert-base-cased-v1.1  | Yes         | MNL           | 4      | 0.43 |
| biobert-large-cased-v1.1 | No          | -             | -      | 0.78 |
| biobert-large-cased-v1.1 | Yes         | MNL           | 2      | 0.48 |
| scibert-scivocab-cased   | No          | -             | -      | 0.78 |
| scibert-scivocab-cased   | Yes         | MNL           | 2      | 0.81 |

**TREC-simplified:** The table below shows the results of the distribution analysis for the TREC-simplified dataset.

| Model                    | Fine-tuning | Loss-Function | Epochs | AUC  |
|:------------------------:|:-----------:|:-------------:|:------:|:----:|
| biobert-base-cased-v1.1  | No          | -             | -      | 0.75 |
| biobert-base-cased-v1.1  | Yes         | MNL           | 2      | 0.54 |
| biobert-large-cased-v1.1 | No          | -             | -      | 0.73 |
| scibert-scivocab-cased   | No          | -             | -      | 0.76 |
| scibert-scivocab-cased   | Yes         | MNL           | 2      | 0.45 |


**RELISH:** The table below shows the results of the distribution analysis for the RELISH dataset.

| Model                    | Fine-tuning | Loss-Function | Epochs | AUC  |
|:------------------------:|:-----------:|:-------------:|:------:|:----:|
| biobert-base-cased-v1.1  | No          | -             | -      | 0.61 |
| biobert-base-cased-v1.1  | Yes         | MNL           | 2      | 0.53 |
| biobert-base-cased-v1.1  | Yes         | MNL           | 4      | 0.54 |
| biobert-base-cased-v1.1  | Yes         | CSL           | 2      | 1.0  |
| biobert-large-cased-v1.1 | No          | -             | -      | 0.60 |
| biobert-large-cased-v1.1 | Yes         | MNL           | 2      | 0.57 |
| biobert-large-cased-v1.1 | Yes         | MNL           | 4      | 0.54 |
| scibert-scivocab-cased   | No          | -             | -      | 0.65 |
| scibert-scivocab-cased   | Yes         | MNL           | 2      | 0.57 |
| scibert-scivocab-cased   | Yes         | MNL           | 4      | 0.52 |

## Discounted cumulative gain method

The following tables show the results of the [nDCG@N](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/code/Evaluation) evaluation approach, when applied on to "BERT doc2doc relevance" technique. 
These results are calculated for the different models used in this approach to obtain the optimal model for each dataset used in this work.

The below table contains following columns:

- **Model:** Model used to obtain these results.
- **nDCG@5 (AVG):** Normalized Discounted Cumulative Gain (nDCG) score for the top 5 articles retrieved.
- **nDCG@10 (AVG):** Normalized Discounted Cumulative Gain (nDCG) score for the top 10 articles retrieved.
- **nDCG@15 (AVG):** Normalized Discounted Cumulative Gain (nDCG) score for the top 15 articles retrieved.
- **nDCG@20 (AVG):** Normalized Discounted Cumulative Gain (nDCG) score for the top 20 articles retrieved.
- **nDCG@25 (AVG):** Normalized Discounted Cumulative Gain (nDCG) score for the top 25 articles retrieved.
- **nDCG@50 (AVG):** Normalized Discounted Cumulative Gain (nDCG) score for the top 50 articles retrieved.


**TREC-repurposed:** The table below shows the results of the nDCG@N evaluation for the TREC-repurposed dataset.

| Model         | nDCG@5 (AVG) | nDCG@10 (AVG) | nDCG@15 (AVG) | nDCG@20 (AVG) | nDCG@25 (AVG) | nDCG@50 (AVG) |
|:-------------:|:------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| BioBERT-base  | 0.519        | 0.510         | 0.506         | 0.505         | 0.505         | 0.513         |
| BioBERT-large | 0.510        | 0.502         | 0.497         | 0.496         | 0.495         | 0.503         |
| SciBERT       | 0.524        | 0.514         | 0.510         | 0.509         | 0.509         | 0.516         |


**RELISH:** The table below shows the results of the nDCG@N evaluation for the RELISH dataset.

| Model         | nDCG@5 (AVG) | nDCG@10 (AVG) | nDCG@15 (AVG) | nDCG@20 (AVG) | nDCG@25 (AVG) | nDCG@50 (AVG) |
|:-------------:|:------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| BioBERT-base  | 0.647        | 0.629         | 0.629         | 0.638         | 0.653         | 0.770         |
| BioBERT-large | 0.634        | 0.619         | 0.622         | 0.632         | 0.648         | 0.765         |
| SciBERT       | 0.646        | 0.629         | 0.632         | 0.640         | 0.655         | 0.771         |

## Precision@N method

The following tables show the results of the precision@N evaluation approach, when applied on to "BERT doc2doc relevance" technique. 

The below table contains following columns:

- **Model:** Model used to obtain these results.
- **Dataset:** Dataset used to obtain these results. (TREC-repurposed, TREC-simplified, RELISH)
- **Precision@5:** Precision score for the top 5 articles retrieved.
- **Precision@10:** Precision score for the top 10 articles retrieved.
- **Precision@15:** Precision score for the top 15 articles retrieved.
- **Precision@20:** Precision score for the top 20 articles retrieved.
- **Precision@25:** Precision score for the top 25 articles retrieved.
- **Precision@50:** Precision score for the top 50 articles retrieved.


|Model     |Dataset     |p@5                 |p@10                |p@15                |p@20                 |p@25                |p@50                |
|:--------:|:----------:|:------------------:|:------------------:|:------------------:|:-------------------:|:------------------:|:------------------:|
|bert-base | relish.pkl |0.0042|0.0060| 0.0066 |0.0074|0.0086|0.0104|
|bert-base |trec-rep.pkl|0.0121|0.0134|0.0142|0.0145 |0.0146|0.0152|
|bert-large| relish.pkl |0.0037|0.0054|0.0060|0.0068 |0.0079|0.0097|
|bert-large|trec-rep.pkl|0.0114 |0.0134 |0.0137|0.0140|0.0143|0.0148|
| bert-sci | relish.pkl |0.0041|0.0060|0.0066|0.0074 |0.0085|0.01026|
| bert-sci |trec-rep.pkl|0.0130|0.0136|0.0141|0.0150 |0.0150|0.0155|


