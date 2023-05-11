import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer, util

SBERT_MODEL = 'NbAiLab/nb-sbert-base'


def sbert():
    return SentenceTransformer(SBERT_MODEL)

def get_cos(sbert_model, sents):
    embeddings = sbert_model.encode(sents, convert_to_tensor=True)
    cos_scores = util.cos_sim(embeddings, embeddings).numpy()
    return cos_scores

