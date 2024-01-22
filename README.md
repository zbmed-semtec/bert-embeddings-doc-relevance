# BERT-embeddings-doc-relevance
An approach exploring and assessing literature-based doc-2-doc recommendations using BERT embeddings, and applying it to RELISH dataset.

## Table of Contents

1. [About](#about)
2. [Input Data](input-data)
3. [Pipeline](#pipeline)
    1. [Data Preprocessing](#data-preprocessing)    
    2. [Models](#models)
    3. [Generate Embeddings](#generate-embeddings)
    4. [Calculate Cosine Similarity](#calculate-cosine-similarity)
    5. [Hyperparameter Optimization](#hyperparameter-optimization)
    6. [Evaluation](#evaluation)
        - [Precision@N](#precisionn)
        - [nDCG@N](#ndcgn)
5. [Getting Started](#getting-started)
6. [Fine-tune BERT](https://github.com/zbmed-semtec/bert-embeddings-doc-relevance/tree/main/code/tune#finetuning-bert-model-for-document-relevance) [**Optional**]




## About

This project uses the [Sentence Transformers pkg](https://www.sbert.net/) for running the experiments with BERT models. Sentence Transformers is a framework for sentence, text and image embeddings. It provides a wide range of pretrained models and state-of-the-art algorithms for semantic search and text classification. The models can be used directly for inference or fine-tuned on your own dataset. The models are available in PyTorch and TensorFlow 2.0. With Sentence Transformers package also supports all the models from [Hugging Face Transformers](https://huggingface.co/transformers/).

## Input Data
The input data for this method consists of preprocessed tokens derived from the RELISH documents. These tokens are stored in the RELISH.npy file, which contains preprocessed arrays comprising PMIDs, document titles, and abstracts. These arrays are generated through an extensive preprocessing pipeline, as elaborated in the [relish-preprocessing repository](https://github.com/zbmed-semtec/relish-preprocessing). Within this preprocessing pipeline, both the title and abstract texts undergo several stages of refinement: structural words are eliminated, text is converted to lowercase, stop words are removed and finally, tokenization is employed, resulting in arrays of individual words.

## Pipeline

This section outlines the process of generating embeddings and evaluating the effectiveness of the approach.


### Data Preprocessing

RELISH corpus contains pubmed-IDs (PMIDS) of the scientific articles and their corresponding titles and abstracts. To get the full text of the articles, use the [relish-preprocessing](https://github.com/zbmed-semtec/relisj-preprocessing) repository.


BERT models can be categorized into two types:
 - `Cased`: Cased models are case-sensitive which means the text contain upper and lower case letters. All the special characters are also kept for trainig the model. For example, `Hello World` and `hello world` are different tokens for cased models.

 - `Uncased`: Uncased models are not case-insensitive which means the text is converted to lower case before tokenization. Removing or keeping the special characters is also a choice for uncased models.

As the RELISH corpus contains titles and abstracts of the scientific articles, there is no need of much preprocessing. Scientific documents won't contain any gibberish text or unneccessary special characters which may require careful preprocessing. So we use the `Cased` models for the experiments. The `Cased` models are better suited for biomedical domain as the biomedical domain contains a lot of abbreviations and acronyms.

The only preprocessing steps applied to this corpus are:
- combining the titles and abstracts of the articles as a single text

- Removing unnecessary white spaces and new lines

### Models

Two state-of-the-art BERT models relevant to the biomedical domain are used to run the experiments in this project. The models used for experiments are:
1. [**BioBERT**](https://doi.org/10.1093/bioinformatics/btz682)
    BioBERT is a pre-trained language representation model for the biomedical domain. It is based on the BERT model, but trained on a large corpus of biomedical text. BioBERT model is pretrained on PubMed corpus or PMC corpus or both. In this project, we use the BioBERT model pretrained on PubMed corpus by [DMIS-Lab](https://dmis.korea.ac.kr/).
    - [dmis-lab/biobert-base-cased-v1.1](https://huggingface.co/dmis-lab/biobert-base-cased-v1.1)

    - [dmis-lab/biobert-large-cased-v1.1](https://huggingface.co/dmis-lab/biobert-large-cased-v1.1)

2. [**SciBERT**](https://doi.org/10.48550/arXiv.1903.10676)
    SciBERT is a pretrained language model based on BERT but trained on a large corpus of scientific text. Model trained on 1.14M papers from semantic scholar (18% - Computer science, 82% - Biomedical domain). corpus size - 3.17B Tokens.
    - [allenai/scibert_scivocab_cased](https://huggingface.co/allenai/scibert_scivocab_cased)


## Generate Embeddings


Here, we utilize the pre-trained BioBERT and SciBERT models from the Sentence Transformer library for embedding generation without fine-tuning. The script employs a mean pooling strategy for embedding generation, where the mean of the embeddings of individual tokens is computed.  The vector size of the embeddings is determined by the chosen pre-trained mode. Command-line arguments within the script offer flexibility in customizing model, data path, save path, and batch size for generating the embeddings associated with PMID values. More a detailed explanation on the execution of the code, please refer to this [documentation](code/generate_embeds/Readme.md).

## Calculate Cosine Similarity
To assess the similarity between two documents within the RELISH corpus, we employ the Cosine Similarity metric. This process enables the generation of a 4-column matrix containing cosine similarity scores for existing pairs of PMIDs within our corpus. For a more detailed explanation of the process, please refer to this [documentation](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/code/Cosine_Similarity).

## Hyperparameter Optimization
*To be written*

## Evaluation

### Precision@N

In order to evaluate the effectiveness of this approach, we make use of Precision@N. Precision@N measures the precision of retrieved documents at various cutoff points (N).We generate a Precision@N matrix for existing pairs of documents within the RELISH corpus, based on the original RELISH JSON file. The code determines the number of true positives within the top N pairs and computes Precision@N scores. The result is a Precision@N matrix with values at different cutoff points, including average scores. For detailed insights into the algorithm, please refer to this [documentation](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/code/Precision%40N_existing_pairs).


### nDCG@N

Another metric used is the nDCG@N (normalized Discounted Cumulative Gain). This ranking metric assesses document retrieval quality by considering both relevance and document ranking. It operates by using a TSV file containing relevance and cosine similarity scores, involving the computation of DCG@N and iDCG@N scores. The result is an nDCG@N matrix for various cutoff values (N) and each PMID in the corpus, with detailed information available in the [documentation](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/code/Evaluation).

## Getting Started


To get started with this project, follow these steps:

### Step 1: GPU Verification & NVIDIA Driver Setup for Virtual Machine

In case you are using a virtual machine, run the following set of commands to check if the virtual machine has a GPU, and if it is present, install the NVIDIA driver along with the necessary commands to verify GPU information.

```
# Update package information
sudo apt-get update

# Check for GPU presence
lspci | grep -i nvidia

# Install NVIDIA driver (if GPU is present)
sudo ubuntu-drivers autoinstall

# Reboot the system to apply changes
sudo reboot

# Verify GPU information using nvidia-smi
nvidia-smi
```

**NOTE:** A GPU is not mandatory; its presence is optional and primarily serves to speed up the execution during the generation of embeddings.

### Step 2: Clone the Repository
Clone the repository to your machine using the following command:

###### Using HTTP:

`git clone https://github.com/zbmed-semtec/bert-embeddings-doc-relevance.git`

###### Using SSH:
Ensure you have set up SSH keys in your GitHub account.

`git clone git@github.com:zbmed-semtec/bert-embeddings-doc-relevance.git`


### Step 3: Create a virtual environment and install dependencies

To create a virtual environment within your repository, run the following command:

```
python3 -m venv .venv 
source .venv/bin/activate   # On Windows, use '.venv\Scripts\activate' 
```

To confirm if the virtual environment is activated and check the location of yourPython interpreter, run the following command:

```
which python    # On Windows command prompt, use 'where python'
                # On Windows PowerShell, use 'Get-Command python'
```
The code is stable with python 3.6 and higher. The required python packages are listed in the requirements.txt file. To install the required packages, run the following command:

```
pip install -r requirements.txt
```

To deactivate the virtual environment after running the project, run the following command:

```
deactivate
```

### Step 4: Generate Embeddings

To generate the embeddings, run the following command:

```bash
python code/generate_embeds/run.py --data_path <path to the data folder>
                                    --save_path <path to the folder where the embeddings should be saved>
                                    --model_name <name of the model to use>
                                    --batch_size <batch size, default 64>
```

Arguments:
+ `data_path` - Path to the folder containing the input data. The input data should be a tsv file.
+ `save_path` - Path to the folder where the embeddings should be saved.
+ `model_name` - Name of the model to use. This should support most of the pretained models from the [transformers](https://huggingface.co/transformers/pretrained_models.html) library. Models used in this project can be found [here](https://github.com/zbmed-semtec/bert-embeddings-doc-relevance#models).
+ `batch_size` - Batch size to use. Default is 64. This is only relevant if GPU is available. (64 is a good default for a GPU with 4GB memory).

The `model_name` parameter can be assigned to any Sentence Transformer-based BERT model of choice. In our case, we have used the following models:

+ allenai/scibert_scivocab_cased
+ dmis-lab/biobert-base-cased-v1.1
+ dmis-lab/biobert-large-cased-v1.1



### Step 5: Calculate Cosine Similarity

To generate the cosine similarity matrix and execute this [script](/code/cosine_similarity/existing_pairs.py), run the following command:

```bash
python3 /code/cosine_similarity/existing_pairs.py -e <path to embeddings> -r <path to relevance scores>  -s <path to save the matrix>  -d <dataset name TREC/RELISH>
```

The above command will generate a pickle file with the four column relevance matrix. The output file saved can be a
pickle file or a tsv file and it depends on the extension of the save path. If the save path has a `.pkl` extension,
the output file will be saved as a pickle file. If the save path has a `.tsv` extension, the output file will be saved

**Note**: If the extension is `.pkl`, output is a compressed pickle file. This means while loading the file, you have to use `compression='gzip'` option.



### Step 6: Hyperparameter Optimization

**_To be written_**

### Step 7: Precision@N
In order to calculate the Precision@N scores and execute this [script](/code/evaluation/precision@N/precision.py), run the follwing command:

```
python3 code/evaluation/precision@N/precision.py [-c COSINE FILE PATH]  [-o OUTPUT PATH]
```

You must pass the following two arguments:

+ -c/ --cosine_file_path: path to the 4-column cosine similarity existing pairs RELISH file: (tsv file)
+ -o/ --output_path: path to save the generated precision matrix: (tsv file)

For example, if you are running the code from the code folder and have the cosine similarity TSV file in the data folder, run the precision matrix creation for the first hyperparameter as:

```
python3 code/evaluation/precision@N/precision.py -c data/cosine_similarity_0.tsv -o data/precision_fasttext_0.tsv
```


### Step 8: nDCG@N
In order to calculate nDCG scores and execute this [script](/code/evaluation/calculate_gain.py), run the following command:

```
python3 code/evaluation/calculate_gain.py [-i INPUT]  [-o OUTPUT]
```

You must pass the following two arguments:

+ -i / --input: Path to the 4 column cosine similarity existing pairs RELISH TSV file.
+ -o/ --output: Output path along with the name of the file to save the generated nDCG@N TSV file.

For example, if you are running the code from the code folder and have the 4 column RELISH TSV file in the data folder, run the matrix creation for the first hyperparameter as:

```
python3 code/evaluation/calculate_gain.py -i data/cosine_similarity_0.tsv -o data/ndcg_fasttext_0.tsv
```


### Step 9: Compile Results

In order to compile the average result values for Precison@ and nDCG@N and generate a single TSV file each, please use this [script](code/evaluation/show_avg.py).

You must pass the following two arguments:

+ -i / --input: Path to the directory consisting of all the precision matrices/gain matrices.
+ -o/ --output: Output path along with the name of the file to save the generated compiled Precision@N / nDCG@N TSV file.


If you are running the code from the code folder, run the compilation script as:

```
python3 code/evaluation/show_avg.py -i data/output/gain_matrices/ -o data/output/results_gain.tsv
```

NOTE: Please do not forget to put a `'/'` at the end of the input file path.

## Fine tuning BERT for Document Relevance

This optional step involves fine-tuning BERT models for document relevance using a supervised learning approach. The provided [code](code/tune/finetune.py) utilizes the fine-tuning functionality from the sentence-transformers library. The fine tuning is conducted with respect to the RELISH dataset, with instructions on obtaining and preprocessing data. The code supports various models and loss functions, providing flexibility for experimentation. To run the code, refer to the instructions provided [here](code/tune/README.md).

