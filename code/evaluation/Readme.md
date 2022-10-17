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
| biobert-large-cased-v1.1 | No          | -             | -      | 0.73 |
| scibert-scivocab-cased   | No          | -             | -      | 0.76 |


**RELISH:** The table below shows the results of the distribution analysis for the RELISH dataset.

| Model                    | Fine-tuning | Loss-Function | Epochs | AUC  |
|:------------------------:|:-----------:|:-------------:|:------:|:----:|
| biobert-base-cased-v1.1  | No          | -             | -      | 0.61 |
| biobert-base-cased-v1.1  | Yes         | MNL           | 2      | 0.53 |
| biobert-base-cased-v1.1  | Yes         | MNL           | 4      | 0.54 |
| biobert-base-cased-v1.1  | Yes         | CSL           | 2      | 1.0  |
| biobert-large-cased-v1.1 | No          | -             | -      | 0.60 |
| biobert-large-cased-v1.1 | Yes         | MNL           | 2      | 0.57 |
| scibert-scivocab-cased   | No          | -             | -      | 0.65 |
| scibert-scivocab-cased   | Yes         | MNL           | 2      | 0.57 |




