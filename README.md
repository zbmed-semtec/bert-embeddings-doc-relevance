# BERT-embeddings-doc-relevance
An approach exploring and assessing literature-based doc-2-doc recommendations using BERT models with its application to the RELISH dataset. The dataset used is the RELISH Corpus, an expert-curated collection of biomedical literature consisting of pairwise document assessments. 
The workflow involves two main steps: First, BERT models are used without any fine-tuning, generating document embeddings to assess document recommendations. Second, the BERT models are fine-tuned on a specific training set derived from the RELISH dataset, and the resulting model is used to generate document recommendations on a separate test set. The performance of both the pretrained and fine-tuned models is then compared to assess the impact of fine-tuning on the quality of document recommendations.

## Table of Contents

1. [About](#about)
2. [Input Data](input-data)
3. [Models](#models)
4. [Pipeline](#pipeline)
    1. [Without Fine-tuning](#without-fine-tuning)
        - [Input Text Preprocessing](#input-text-preprocessing)    
        - [Generaing Embeddings using Pre-trained Models](#generating-embeddings-using-pre-trained-models)
    2. [With Fine-tuning](#with-fine-tuning)
        - [Input Text Preprocessing](#input-text-preprocessing-1)
        - [Fine-tuning Procedure](#fine-tuning-procedure)
        - [Generating Embeddings using Fine-tuned Models](#generating-embeddings-using-fine-tuned-models)
    3. [Calculate Cosine Similarity](#calculate-cosine-similarity)
    4. [Evaluation](#evaluation)
        - [Precision@N](#precisionn)
        - [nDCG@N](#ndcgn)
5. [Getting Started](#getting-started)


## About

This project uses the [Sentence Transformers pkg](https://www.sbert.net/) to generate embeddings and fine-tune BERT models for document recommendation. The framework is designed to support various pre-trained BERT-based models, enabling fine-tuning and embedding generation for relevance scoring and ranking.


## Input Data
The input data for this method includes preprocessed tokens derived from the RELISH documents, a specialized database curated by experts for benchmarking document similarity in biomedical literature. The RELISH dataset comprises a JSON file containing PubMed IDs (PMIDs) along with document-to-document relevance assessments categorized as "relevant (2)," "partial (1)," or "irrelevant (0)." 

The titles and abstracts of the associated articles were retrieved and stored in a TSV file. This TSV file is divided into training, validation and test sets based on specific criteria detailed [here](https://github.com/zbmed-semtec/relish-preprocessing?tab=readme-ov-file#splitting-the-data). These splits are then saved as three separate TSV files.

Additionally, the ground truth relevance assessments are used to evaluate the accuracy of the doc-2-doc recommendations, ensuring that the method's results align with expert judgments.

## Models

Three state-of-the-art BERT models relevant to the biomedical domain are used to run the experiments in this project:

1. [**BioBERT**](https://doi.org/10.1093/bioinformatics/btz682)
    BioBERT is a pre-trained language representation model for the biomedical domain. It is based on the BERT model, but trained on a large corpus of biomedical text. BioBERT model is pretrained on PubMed corpus or PMC corpus or both. In this project, we use the BioBERT model pretrained on PubMed corpus by [DMIS-Lab](https://dmis.korea.ac.kr/).
    - [dmis-lab/biobert-base-cased-v1.1](https://huggingface.co/dmis-lab/biobert-base-cased-v1.1)

    - [dmis-lab/biobert-large-cased-v1.1](https://huggingface.co/dmis-lab/biobert-large-cased-v1.1)

2. [**SciBERT**](https://doi.org/10.48550/arXiv.1903.10676)
    SciBERT is a pre-trained language model based on BERT but trained on a large corpus of scientific text. Model is trained on 1.14M papers from semantic scholar (18% - Computer science, 82% - Biomedical domain). corpus size - 3.17B Tokens.
    - [allenai/scibert_scivocab_cased](https://huggingface.co/allenai/scibert_scivocab_cased)

3. [**SPECTER**](https://arxiv.org/abs/2004.07180)
    SPECTER is a pre-trained language model based on BERT trained using a Siamese network architecture that leverages citation information to improve embedding quality for scientific documents.
    - [allenai/specter](https://huggingface.co/allenai/specter)


## Pipeline

This section describes the process of generating embeddings and evaluating the effectiveness of the approach for BERT models, both with and without fine-tuning.

## Without fine-tuning

### Input Text Preprocessing

When using BERT models without fine-tuning, we directly input the test dataset into the model. The dataset, stored as a TSV file with columns [PMID | Title | Abstract], undergoes preprocessing to prepare the text for embedding generation.

To ensure consistency, the titles and abstracts of each article are always combined into a single text string, separated by the BERT '[SEP]' token. The following preprocessing approaches are explored:
 
+ **None**: No pre-processing is applied on the dataset
+ **wspn**: White spaces, new lines, and punctuation are removed from the dataset.
+ **wspnss**: White spaces, new lines, punctuation, and stop words are removed from the dataset.

Note: Tokenization is inherently handled by the BERT model and does not require manual intervention during preprocessing.


### Generating Embeddings Using Pre-trained Models

In this step, we leverage pre-trained BERT models from the Sentence Transformer library to generate embeddings without fine-tuning. The script applies a mean pooling strategy, where the embeddings of individual tokens are averaged to create a fixed-size vector representation for each document. The embedding vector size is determined by the specific pre-trained model selected. The script provides command-line arguments, allowing users to customize the model, input data path, output save location, and batch size when generating embeddings associated with PMID values.


## With fine-tuning

### Input Text Preprocessing

The input text preprocessing for fine-tuning BERT models involves preparing the dataset by combining document titles and abstracts and transforming document pairs with relevance labels into a format compatible with various loss functions. The preprocessing pipeline uses a TSV file containing columns for PMID, Title, and Abstract to map each document's unique ID to its complete text. The Ground truth files are used as a basis.

Titles and abstracts are merged into a single string, separated by a space, creating a "combined text" for each document. This mapping is stored in a dictionary, allowing efficient pairing of documents based on their relevance scores.

Depending on the specific loss function—such as softmax, multiple negatives ranking (MNR), or contrastive loss—the script prepares data differently and store them as a CSV file:

+ **Softmax Loss:** Produces text pairs with corresponding relevance labels for classification tasks. The dataset is formatted as `[text1 | text2 | label]`, where `label` represents the relevance score that as per the RELISH relevance annotations (0, 1 or 2).

+ **MNR Loss:** Focuses on anchor-positive pairs for ranking, filtering pairs based on relevance thresholds with the class distribution dictating the treatment of relevance scores:

    - For 2 classes: Both partially relevant and fully relevant documents are grouped as positive with respect to the anchor.
    - For 3 classes: Only fully relevant documents are considered positive for the anchor. 
    
    The dataset is structured as `[anchor | positive]`. 


+ **Contrastive Loss:** Converts relevance scores into binary labels, preparing the dataset for similarity-based learning. 

    - Label 1: Indicates a positive pair (e.g., documents that are either partially relevant or fully relevant for 2 classes, and only fully relevant for 3 classes).
    - Label 0: Indicates a negative pair (e.g., irrelevant documents). 

    The dataset is structured as `[text1 | text2 | label]`. 

### Fine-tuning Procedure

The fine-tuning process used involves adapting pre-trained BERT-based models for document-to-document by training it on labeled data.

#### 1. Model Architecture
The fine-tuning begins by loading a pre-trained transformer model (e.g., BioBERT, SciBERT or SPECTER) and attaching a pooling layer with dropout for regularization.

#### 2. Datasets
Training and validation datasets prepared as CSV files are used, formatted with text pairs and labels based on the chosen loss function (contrastive, MNR, or softmax). The validation dataset (eval_dataset) is used during training to monitor the model's performance after each epoch and the validation loss (eval_loss) is recorded in the log file.

#### 3. Optimization Techniques

+ Learning Rate Adjustment: A ReduceLROnPlateau scheduler lowers the learning rate if validation loss stagnates, avoiding overfitting.
+ Dropout Regularization: Applied during pooling to mitigate overfitting.
+ AdamW Optimizer: Efficient weight updates with weight decay for regularization.

#### 4. Training Parameters
+ Batch Size: 8 – Number of samples processed in one pass.
+ Learning Rate: 2×10<sup>5</sup> with warm-up – Small learning rate with initial warm-up.
+ Epochs: 5 – Number of times the model processes the entire dataset.

The fine-tuned models are then saved and used for generating embeddings on the test dataset and for downstream evaluation. 

## Generating Embeddings using Fine-tuned models

In this step, we utilize the fine-tuned BERT models from the Sentence Transformer library for embedding generation. The process follows the same script as the one for generating embeddings with pre-trained models, with the only difference being the use of the fine-tuned model.

## Calculate Cosine Similarity
To assess the similarity between two documents within the RELISH corpus, we employ the Cosine Similarity metric. This process enables the generation of a 4-column matrix containing cosine similarity scores for existing pairs of PMIDs within our corpus. For a more detailed explanation of the process, please refer to this [documentation](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/code/Cosine_Similarity).


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

# Install ubuntu-drivers
sudo apt install ubuntu-drivers-common
 
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

Change to the repository directory

```
cd bert-embeddings-doc-relevance
```

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
The code is stable with python 3.9 and higher. The required python packages are listed in the requirements.txt file. To install the required packages, run the following command:

```
pip install -r requirements.txt
```

To deactivate the virtual environment after running the project, run the following command:

```
deactivate
```

### Step 4: Dataset

Use the `Download_Dataset.sh` script to download the Split Dataset by running the following commands:

```
chmod + Download_Dataset.sh
./Download_Dataset.sh
```

This script makes sure that the necessary folders are created and the files are downloaded in the corresponding folders as shown below:

```
📦 /bert-embeddings-doc-relevance
└─ data
     └─ Split_Dataset
          ├─ Data
          |  ├─ relish_documents.tsv
          |  ├─ input_train_text_data.tsv
          |  ├─ input_test_text_data.tsv
          |  └─ input_valid_text_data.tsv
          └─ Ground_truth
             ├─ train.tsv
             └─ test.tsv
             └─ valid.tsv
```

### Step 5: Generate BERT Embeddings

### Without Fine Tuning

To generate embeddings without fine-tuning the BERT models, run the following command:

```
python code/process.py [-i INPUT_DATA_PATH] [-g GROUND_TRUTH_PATH] [-p PREPROCESS] [-m MODEL_NAME] [-b BATCH_SIZE] [-c CLASSES]
```

You must pass the following arguments:

+ -i/--input_data_path - Path to the input text test dataset. The input data should be a tsv file.
+ -g/--ground_truth_path - Path to the test ground truth TSV file.
+ -p/--preprocess - Specifies the preprocessing to be applied on the dataset.
    - 0: No preprocessing (None)
    - 1: Remove whitespace and punctuation (wspn)
    - 2: Remove whitespace, punctuation, and stop words (wspnss)
+ -m/--model_name - Name of the model to use. This should support most of the pretained models from the [transformers](https://huggingface.co/transformers/pretrained_models.html) library. Models used in this project can be found [here](https://github.com/zbmed-semtec/bert-embeddings-doc-relevance#models).
+ -b/--batch_size - Batch size to use. Default is 16.
+ -c/--classes - Class distribution to use. (Integer 2 or 3/ Default value is 3)

The `model_name` parameter can be assigned to any Sentence Transformer-based BERT model of choice. In our case, we have used the following models:

+ allenai/scibert_scivocab_cased
+ dmis-lab/biobert-base-cased-v1.1
+ dmis-lab/biobert-large-cased-v1.1
+ allenai/specter

The `process.py` script is designed to handle the preprocessing of the test dataset, generate embeddings, compute cosine similarity, and evaluate metrics such as precision@N and ndcg@N. It saves all the necessary files based on the selected class distribution.

For example, running the following command:

```
python3 code/process.py -i data/Split_Data/Data/input_test_text_data.tsv -g data/Ground_truth/test.tsv -p none -m allenai/specter -b 16 -c 2
```

Would result in the output being stored in a newly created folder named output_2 with the following structure:

```
output_2
├─ embeddings
│  ├─ bert_embeddings_specter_none_pretrained.pkl
├─ evaluation
|  ├─ bert_specter_dcg_none_2.tsv
|  ├─ bert_specter_idcg_none_2.tsv
|  └─ bert_specter_ndcg_none_2.tsv
|  └─ bert_specter_precision_none_2.tsv
└─ bert_specter_none_2.log
```

### With Fine Tuning

To generate embeddings with fine-tuning the BERT models, run the following command first:

```
python code/tune/finetune.py [-m MODEL_NAME] [-o SAVE_MODEL] [-e EPOCHS]  [-b BATCH_SIZE] [-l LOSS_FUNC] [-c CLASSES] [-d DROPOUT]
```


You must pass the following arguments:

+ -m/--model_name - Name of the model to use. This should support most of the pretained models from the [transformers](https://huggingface.co/transformers/pretrained_models.html) library. Models used in this project can be found [here](https://github.com/zbmed-semtec/bert-embeddings-doc-relevance#models).
+ -o/--save_model - Path to save the fine-tuned BERT model
+ -e/--epochs - Number of epochs to be used during the fine-tuning process.
+ -b/--batch_size - Batch size to use. Default is 16.
+ -l/--loss_func - Loss function to be used. (softmax, mnr, contrastive). Loss functions used in this project can be found [here](https://github.com/zbmed-semtec/bert-embeddings-doc-relevance#input-text-preprocessing-1)/. 
+ -c/--classes - Class distribution to use. (Integer 2 or 3/ Default value is 3)
+ -d/--dropout - Dropout value to be used for regularizsation (Default value is 0.5)

Once the fine-tuning is complete, the next step is to used the fine-tuned model and execute the `process.py` script as described above.
