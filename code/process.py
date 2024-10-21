import os
import warnings
import argparse

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from generate_embeds.datamodule import CustomDataset
from generate_embeds.run import generate_embeddings
from generate_embeds.utils import get_device
from cosine_similarity.existing_pairs import CosineSimilarity
from evaluation.precision import precision
from evaluation.gain import calculate_gain

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_data_path', type=str, help='path to the pre-processed data')
    parser.add_argument('-g', '--ground_truth_data', type=str, help='path to the ground truth test TSV file')
    parser.add_argument('-p', '--preprocess', type=str, help='preprocessing steps: 0 no preprocessing, 1 for removing white space and punctuations and 2 for removing white space and punctuations and stop words')
    parser.add_argument('-m', '--model_name', type=str, help='name of the model to be used')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='batch size to be used')
    parser.add_argument('-c', '--classes', type=int, help='class distribution to be used')

    args = parser.parse_args()

    permissions = 0o755  # This sets permissions to rwxr-xr-x

    # 1) Define the directory for storing pipeline outputs
    output_directory = f"output_{args.classes}"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    os.chmod(output_directory, permissions)

    # 2) Define the Directory for storing embeddings
    embeddings_directory = f"output_{args.classes}/embeddings"
    if not os.path.exists(embeddings_directory):
        os.makedirs(embeddings_directory)
    os.chmod(embeddings_directory, permissions)

    # 3) Define the directory for storing evaluation results
    results_directory = f"output_{args.classes}/evaluation"
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    os.chmod(results_directory, permissions)

    # 4) Define the log file
    safe_model_name = args.model_name.replace('/', '_')
    log_file = f"output_{args.classes}/bert_{safe_model_name}_{args.classes}.log"
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

    # 5) Load the BERT model
    sbert_model = SentenceTransformer(args.model_name)

    # 6) Preprocess the dataset based on the preprocess steps
    data = CustomDataset(args.input_data_path, args.preprocess)

    # 7) Generate embeddings for dataset
    embeddings_file = os.path.join(embeddings_directory, f"bert_embeddings_{safe_model_name}_{args.preprocess}_pretrained.pkl")
    generate_embeddings(data, sbert_model, embeddings_file)

    # 8) Compute cosine similarity and save matrix
    cosine_sim_matrix = CosineSimilarity(embeddings_file, args.ground_truth_data)
    cosine_similarity_file = os.path.join(results_directory, f"bert_{safe_model_name}_{args.preprocess}_cosine_.tsv")
    cosine_sim_matrix.create_relavance_matrix(cosine_similarity_file)
    logging.info("Cosine Similarity Matrix saved")

    # 9) Generate and save the precision matrix
    ref_pmids, data = precision.read_file(cosine_similarity_file)
    matrix = precision.generate_matrix(ref_pmids, data, args.classes)
    precision_file = os.path.join(results_directory, f"bert_{safe_model_name}_precision_{args.preprocess}_{args.classes}.tsv")
    precision.write_to_tsv(ref_pmids, matrix, precision_file, data)
    logging.info("Precision@N Matrix saved")

    # 10) Generate and save the DCG and IDCG matrices
    dcg_file = os.path.join(results_directory, f"bert_{safe_model_name}_dcg_{args.preprocess}_{args.classes}.tsv")
    idcg_file = os.path.join(results_directory, f"bert_{safe_model_name}_idcg_{args.preprocess}_{args.classes}.tsv")
    ndcg_file = os.path.join(results_directory, f"bert_{safe_model_name}_ndcg_{args.preprocess}_{args.classes}.tsv") 

    sim_matrix = calculate_gain.load_cosine_sim_matrix(cosine_similarity_file)
    calculate_gain.get_dcg_matrix(sim_matrix, dcg_file)
    calculate_gain.get_identity_dcg_matrix(sim_matrix, idcg_file)
    all_pmids, ndcg_matrix = calculate_gain.fill_ndcg_scores(
        dcg_file, idcg_file)
    calculate_gain.write_to_tsv(all_pmids, ndcg_matrix, ndcg_file)
    logging.info("DCG, IDCG, and NDCG matrices saved")