
# Generate BERT Embeddings

This folder contains the code to generate BERT embeddings for the documents in the corpus. This project uses [Sentence Transformers pkg](https://www.sbert.net/) for running the experiments with BERT models. Sentence Transformers is a framework for sentence, text and image embeddings. It provides a wide range of pretrained models and state-of-the-art algorithms for semantic search and text classification. The models can be used directly for inference or fine-tuned on your own dataset. The models are available in PyTorch and TensorFlow 2.0. With Sentence Transformers package also supports all the models from [Hugging Face Transformers](https://huggingface.co/transformers/).

### Datasets

The datasets used for experiments are:

- [Text Retrieval Conference TREC (TREC)](https://trec.nist.gov/)
- [RELISH](https://doi.org/10.1093/database/baz138)

To know more about the datasets used in this project, please refer to the [medline-preprocessing](https://github.com/zbmed-semtec/medline-preprocessing) repository.

### Models

Two state-of-the-art BERT models relevant to the biomedical domain are used to run the experiments in this project. The models used for experiments are:
1. [**BioBERT**](https://doi.org/10.1093/bioinformatics/btz682)
    BioBERT is a pre-trained language representation model for the biomedical domain. It is based on the BERT model, but trained on a large corpus of biomedical text. BioBERT model is pretrained on PubMed corpus or PMC corpus or both.In this project, we use the BioBERT model pretrained on PubMed corpus by [DMIS-Lab](https://dmis.korea.ac.kr/).
    - [dmis-lab/biobert-base-cased-v1.1](https://huggingface.co/dmis-lab/biobert-base-cased-v1.1)

    - [dmis-lab/biobert-large-cased-v1.1](https://huggingface.co/dmis-lab/biobert-large-cased-v1.1)

2. [**SciBERT**](https://doi.org/10.48550/arXiv.1903.10676)
    SciBERT is a pretrained language model based on BERT but trained on a large corpus of scientific text. Model trained on 1.14M papers from semantic scholar (18% - Computer science, 82% - Biomedical domain). corpus size - 3.17B Tokens.
    - [allenai/scibert-scivocab-cased](https://huggingface.co/allenai/scibert_scivocab_cased)

### Data Preprocessing



