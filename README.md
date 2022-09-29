# bert-embeddings-doc-relevance
An approach exploring and assessing literature-based doc-2-doc recommendations using BERT embeddings, and applying it to TREC and RELISH datasets

This project uses [Sentence Transformers pkg](https://www.sbert.net/) for running the experiments with BERT models. Sentence Transformers is a framework for sentence, text and image embeddings. It provides a wide range of pretrained models and state-of-the-art algorithms for semantic search and text classification. The models can be used directly for inference or fine-tuned on your own dataset. The models are available in PyTorch and TensorFlow 2.0. With Sentence Transformers package also supports all the models from [Hugging Face Transformers](https://huggingface.co/transformers/).

### Contents

1. [Introduction](https://github.com/zbmed-semtec/bert-embeddings-doc-relevance#bert-embeddings-doc-relevance)
    - [Datasets](https://github.com/zbmed-semtec/bert-embeddings-doc-relevance#bert-embeddings-doc-relevance)
    - [Models](https://github.com/zbmed-semtec/bert-embeddings-doc-relevance#models)
    - [Data Preprocessing](https://github.com/zbmed-semtec/bert-embeddings-doc-relevance#data-preprocessing)
2. [Cosine Similarity](https://github.com/zbmed-semtec/bert-embeddings-doc-relevance/tree/main/code/cosine_similarity#cosine-similarity-matrices)
    - [Generate Cosine Similarity Matrix](https://github.com/zbmed-semtec/bert-embeddings-doc-relevance/tree/main/code/cosine_similarity#cosine-similarity-matrix)
    - [Four column relevance matrix](https://github.com/zbmed-semtec/bert-embeddings-doc-relevance/tree/main/code/cosine_similarity#four-column-relevance-matrix)
3. [Generate Embeddings](https://github.com/zbmed-semtec/bert-embeddings-doc-relevance/tree/main/code/generate_embeds#generate-bert-embeddings)
4. [Fine-tune BERT]()
5. [Evaluation]()


### Datasets
The datasets used for experiments are:

- [Text Retrieval Conference TREC (TREC)](https://trec.nist.gov/)
- [RELISH](https://doi.org/10.1093/database/baz138)

To know more about the datasets used in this project and to get these datasets, please refer to the [medline-preprocessing](https://github.com/zbmed-semtec/medline-preprocessing) repository.

### Models

Two state-of-the-art BERT models relevant to the biomedical domain are used to run the experiments in this project. The models used for experiments are:
1. [**BioBERT**](https://doi.org/10.1093/bioinformatics/btz682)
    BioBERT is a pre-trained language representation model for the biomedical domain. It is based on the BERT model, but trained on a large corpus of biomedical text. BioBERT model is pretrained on PubMed corpus or PMC corpus or both. In this project, we use the BioBERT model pretrained on PubMed corpus by [DMIS-Lab](https://dmis.korea.ac.kr/).
    - [dmis-lab/biobert-base-cased-v1.1](https://huggingface.co/dmis-lab/biobert-base-cased-v1.1)

    - [dmis-lab/biobert-large-cased-v1.1](https://huggingface.co/dmis-lab/biobert-large-cased-v1.1)

2. [**SciBERT**](https://doi.org/10.48550/arXiv.1903.10676)
    SciBERT is a pretrained language model based on BERT but trained on a large corpus of scientific text. Model trained on 1.14M papers from semantic scholar (18% - Computer science, 82% - Biomedical domain). corpus size - 3.17B Tokens.
    - [allenai/scibert-scivocab-cased](https://huggingface.co/allenai/scibert_scivocab_cased)

### Data Preprocessing

TREC and RELISH corpora contain pubmed-IDs (PMIDS) of the scientific articles and their corresponding titles and abstracts. To get the full text of the articles, use the [medline-preprocessing](https://github.com/zbmed-semtec/medline-preprocessing) repository. The full text of the articles is stored in the `data` folder of the `medline-preprocessing` repository.

BERT models can be categorized into two types:
 - `Cased`: Cased models are case-sensitive which means the text contain upper and lower case letters. All the special characters are also kept for trainig the model. For example, `Hello World` and `hello world` are different tokens for cased models.

 - `Uncased`: Uncased models are not case-insensitive which means the text is converted to lower case before tokenization. Removing or keeping the special characters is also a choice for uncased models.

As both TREC and RELISH corpora contain titles and abstracts of the scientific articles, there is no need of much preprocessing. Scientific documents won't contain any gebberish text or unneccessary special characters which may require careful preprocessing. So we use the `Cased` models for the experiments. The `Cased` models are better suited for biomedical domain as the biomedical domain contains a lot of abbreviations and acronyms.

The only preprocessing steps applied to these corpora are:
- combining the titles and abstracts of the articles as a single text
- Removing the stop words
- Removing unnecessary white spaces and new lines

