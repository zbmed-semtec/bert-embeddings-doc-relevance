# Cosine similarity matrices

This folder contains the code to compute the cosine similarity matrix and four column relavance matrix.

## Cosine similarity matrix

The input file for this code is the generated embeddings file which has to be in pickle format. This file should follow the below rules:

- File has to be in pickle format.
- This file should contain a pandas dataframe with the following columns:
    - `PMID`: This column should contain the unique pmid of the document.
    - `embedding`: This column should contain the embedding of the document.
- The embeddings should be in the form of a numpy array.

A sample input dataframe should look like this:

| **PMID** | **embedding** |
|------|-----------|
| 1023  | [0.1, 0.2, 0.3, 0.4, 0.5] |
| 9852 | [0.2, 0.3, 0.4, 0.5, 0.6] |
| 11457| [0.3, 0.4, 0.5, 0.6, 0.7] |

To generate [Upper triangular cosine similarity matrix](https://github.com/zbmed-semtec/bert-embeddings-doc-relevance/blob/main/code/cosine_similarity/up_triangular_mtx.py) for a given set of documents, run the following command:

```bash
python3 up_triangular_mtx.py --embeds_path <path to embeddings>
                             --save_path <path to save the matrix>
                             --compression <True/False>
```

The above command will generate a pickle file with the upper triangular cosine similarity matrix.
**Note**: The cosine similarity matrix is a symmetric matrix. So, we are storing only the upper triangular matrix, as lower triangular matrix will also be same. This matrix is also dumped as a pandas dataframe into the pickle file.

To use this code in your python script:

```python

from up_triangular_mtx import create_cs_matrix

# embeddings file path and save file path
embed_file_path = '../data/embeddings.pkl'
save_path = '../data/up_cosine_similarity.pkl'

# create cosine similarity matrix
cs_mat = create_cs_matrix(embeds_path = embed_file_path,
                save_path = save_path,
                compression = False,
                return_df = True)

# print the cosine similarity matrix
print(cs_mat.head())
```
The output is an upper triangular cosine similarity matrix in the form of a pandas dataframe and it looks like this:

|       | pmid1 | pmid2 | pmid3 | pmid4 | pmid5 |
| ----- | ----- | ----- | ----- | ----- | ----- |
| pmid1 | 1     | 0.78  | 0.56  | 0.92  | 0.99  |
| pmid2 | 0     | 1     | 0.56  | 0.74  | 0.23  |
| pmid3 | 0     | 0     | 1     | 0.45  | 0.94  |
| pmid4 | 0     | 0     | 0     | 1     | 0.56  |
| pmid5 | 0     | 0     | 0     | 0     | 1     |

**Note:** The cosine similarity values are stored in `numpy.float32` dtype and values are rounded to 4 decimal places.

## Four column relevance matrix

To generate the [four column relevance matrix](https://github.com/zbmed-semtec/bert-embeddings-doc-relevance/blob/main/code/cosine_similarity/existing_pairs.py) two input files are required:

- The first input file is the generated embeddings file which has to be in pickle format. This file should follow the below rules:
    - File has to be in pickle format.
    - This file should contain a pandas dataframe with the following columns:
        - `PMID`: This column should contain the unique pmid of the document.
        - `embedding`: This column should contain the embedding of the document.
    - The embeddings should be in the form of a numpy array.

        A sample input dataframe should look like this:

        | **PMID** | **embedding** |
        |------|-----------|
        | 1023  | [0.1, 0.2, 0.3, 0.4, 0.5] |
        | 9852 | [0.2, 0.3, 0.4, 0.5, 0.6] |
        | 11457| [0.3, 0.4, 0.5, 0.6, 0.7] |

- The second input file is the three column relavance scores file which can be in either tsv or pkl format. This file should follow the below rules:
    - If the file is in tsv format, it should contain the following columns:
        - `PMID1`: This column should contain the unique pmid of the first document.
        - `PMID2`: This column should contain the unique pmid of the second document.
        - `Relavance/Rel-d2d`: This column should contain the relevance score between the two documents.
                    The score should be in the range of 0 to 2. 0 means the two documents are not related.
                    1 means the two documents are partially related and 2 means the two documents are highly related.

        A sample input dataframe should look like this:

        | **PMID1** | **PMID2** | **Rel-d2d**|
        |-----------|-----------|-----------|
        |     1023  | 9852      | 1         |
        | 1023      | 11457     | 2         |
        | 9852      | 11457     | 0         |
        | 9852      | 1023      | 1         |


To generate the four column relevance matrix, run the following command:

```bash
python existing_pairs.py -e <path to embeddings>
                         -r <path to relevance scores>
                         -s <path to save the matrix>
                         -d <dataset name TREC/RELISH>
```

The above command will generate a pickle file with the four column relevance matrix. The output file saved can be a
pickle file or a tsv file and it depends on the extension of the save path. If the save path has a `.pkl` extension,
the output file will be saved as a pickle file. If the save path has a `.tsv` extension, the output file will be saved
**Note**: If the extension is `.pkl`, output is a compressed pickle file. This means while loading the file, you have to use `compression='gzip'` option.
For example:

```python
import pandas as pd

four_col_rel_df = pd.read_pickle('../data/four_col_rel.pkl', compression='gzip')
print(four_col_rel_df.head())
```

To use this code in your python script:

```python

from existing_pairs import CosineSimilarity

# Input and output file paths
embed_file_path = '../data/embeddings.pkl'
rel_file_path = '../data/relevance_scores.pkl'
save_path = '../data/four_col_rel.pkl'
dataset = 'TREC'

# create an object of CosineSimilarity class
four_col_rel_df = CosineSimilarity(embeds_path = embed_file_path,
                rel_path = rel_file_path,
                save_path = save_path,
                dataset = dataset)

# generate the four column relevance matrix
four_col_rel_df.create_relavance_matrix(save_path = save_path)
```

The output matrix will look like this:

| **PMID1** | **PMID2** | **Rel-d2d** | **Cosine Similarity** |
|-----------|-----------|-------------|-----------------------|
| 1023      | 9852      | 1           | 0.7894    |
| 1023      | 11457     | 2           | 0.9635    |
| 9852      | 11457     | 0           | 0.3252    |
| 9852      | 1023      | 1           | 0.7894    |



### Requirements

This code is stable with python 3.6 and higher. The required python packages are listed in the `requirements.txt` file. To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```
