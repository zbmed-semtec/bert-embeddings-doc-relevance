import json
import os
import torch
import torch.nn as nn
from sentence_transformers import models

class PoolingWithDropout(nn.Module):
    """
    A pooling layer with dropout applied to the sentence embedding.

    Parameters
    ----------
    word_embedding_dimension : int
        The dimension of the word embeddings.
    pooling_mode : str
        The type of pooling operation ('mean').
    dropout_prob : float, optional
        The dropout probability for the sentence embedding (default is 0.5).
    """
    def __init__(self, word_embedding_dimension, pooling_mode, dropout_prob=0.5):
        super(PoolingWithDropout, self).__init__()
        self.pooling = models.Pooling(word_embedding_dimension, pooling_mode=pooling_mode)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_mode = pooling_mode
        self.dropout_prob = dropout_prob

    def forward(self, features):
        pooled_output = self.pooling(features)
        pooled_output['sentence_embedding'] = self.dropout(pooled_output['sentence_embedding'])
        return pooled_output

    def save(self, save_dir: str, **kwargs):
        self.pooling.save(save_dir)
        
        config = {
            "word_embedding_dimension": self.word_embedding_dimension,
            "pooling_mode": self.pooling_mode,
            "dropout_prob": self.dropout_prob,
        }
        with open(os.path.join(save_dir, "config.json"), "w") as fOut:
            json.dump(config, fOut, indent=4)

    @classmethod
    def load(cls, load_dir) -> "PoolingWithDropout":
        with open(os.path.join(load_dir, "config.json")) as fIn:
            config = json.load(fIn)

        return cls(
            word_embedding_dimension=config["word_embedding_dimension"],
            pooling_mode=config["pooling_mode"],
            dropout_prob=config["dropout_prob"]
        )