# Create precision matrix

This folder contains the code to generate precision@N matrix and this
readme file explains this method and usage of the code.

## About the Precision@N method

These are the steps for the calculation of the Precision@N matrix.

1. Inputs are **cosine similarity matrix** and **relavance scores** (tsv file) for the document to document pairs in the corpus.
**Cosine similarity matrix**:
An example of the cosine similarity matrix is shown below.
In our case we created upper triangular similarity matrix for the TREC and RELISH corpus.

|       | pmid1 | pmid2 | pmid3 | pmid4 | pmid5 |
| ----- | ----- | ----- | ----- | ----- | ----- |
| pmid1 | 1     | 0.78  | 0.56  | 0.92  | 0.99  |
| pmid2 | 0     | 1     | 0.56  | 0.74  | 0.23  |
| pmid3 | 0     | 0     | 1     | 0.45  | 0.94  |
| pmid4 | 0     | 0     | 0     | 1     | 0.56  |
| pmid5 | 0     | 0     | 0     | 0     | 1     |

**Relavance scores file**:
This is the file that contains the relevance scores for the document to document pairs which are annotated by human annotators. So this file serves as a ground truth for the precision@N method.

This is an example of the relevance scores file:
(Note: These scores are dummy scores for the purpose of this example)

| PMID1 | PMID2  | Rel-d2d |
| ----- | ------ | ------- |
| 9800  | 7546   | 2       |
| 9854  | 14597  | 2       |
| 10245 | 7856   | 1       |
| 4589  | 3254   | 0       |
| 458   | 14789  | 0       |

Here, relevance scores and their meaning are as follows:
- 2: Relevant
- 1: partly relevant
- 0: Not relevant

2. We take cosine similarity of each document with respect to every other document (i.e each row in the cosine similarity matrix), and sort this row in descending order.

**Example:**

|       | pmid1 | pmid5 | pmid4 | pmid2 | pmid3 |
| ----- | ----- | ----- | ----- | ----- | ----- |
| pmid1 | 1     | 0.99  | 0.92  | 0.78  | 0.56  |

3. We then take the top N similar documents in the sorted cosine simialrity matrix row (N is the precision parameter) and count the True positives with respect to the relevance scores.

**Example:**

Lets say N is 3 and from the below example we take the top 3 similar documents.

|       | pmid1 | pmid5 | pmid4 | pmid2 | pmid3 |
| ----- | ----- | ----- | ----- | ----- | ----- |
| pmid1 | 1     | 0.99  | 0.92  | 0.78  | 0.56  |


So here we consider pmid5, pmid4, pmid2 as the top 3 similar documents. we dont consider pmid1 as they are same documents.

4. We consider that both documents are relavant or is a True Positive, if each similar document (top n) has a relevance score of 2 with respect to the the relavance scores file. Formula for precision is:

$Precision = \frac{tp}{N}$

where,
tp = number of True Positives
N = precision parameter (top N similar documents)

**Example:**
For N = 3, we have 2 True Positives (pmid5 and pmid4) in this example.

| PMID1 | PMID2 | Cosine Similarity | Rel -d2d | TP      |
| ----- | ----- | ----------------- | -------- | ------- |
| pmid1 | pmid5 | 0.99              | 2        | 1       |
| pmid1 | pmid4 | 0.92              | 2        | 1       |
| pmid1 | pmid2 | 0.78              | 0        | 0       |
|       |       |                   |          | tps = 2 |

$precision = \frac{2}{3} = 0.66$

5. We repeat the above steps for all the document to document pairs in the corpus.
6. we calculate the average precision for each N value and add it to the precision@N matrix.

The example of final precision@N matrix is shown below.

|       | p@5 | p@10 | p@15 | p@20 | p@25 | p@50 |
| ----- | --- | ---- | ---- | ---- | ---- | ---- |
| pmid1 | #   | #    | #    | #    | #    | #    |
| pmid2 | #   | #    | #    | #    | #    | #    |
| pmid3 | #   | #    | #    | #    | #    | #    |
| pmid4 | #   | #    | #    | #    | #    | #    |
| pmid5 | #   | #    | #    | #    | #    | #    |
| AVG   | #   | #    | #    | #    | #    | #    |

## To run the code

Run the following command in the terminal:

``` 
python3 precision@n.py --embed_path <path to embeddings file>
                       --rel_path <path to relevance scores file>
                       --save_path <path to save the precision@N matrix>
                       --dataset <dataset name: TREC/RELISH.>    
                       --n_values <precision parameters.>  
```

where,
- embed_path: path to the embeddings pickle file, with two columns: `pmids` and `embeddings`.
- rel_path: path to the relevance scores tsv file, with three columns: `pmid1`, `pmid2` and `rel-d2d` for TREC or  `relevance` for RELISH datasets.
- save_path: path to save the precision@N matrix. Path should end with either `.pkl` or `.tsv` extension.
- dataset: name of the dataset. It can be either `TREC` or `RELISH`. Default is `TREC`.
-  n_values: precision parameters. It should be a list of integers. Default is `[5, 10, 15, 20, 25, 50]`.