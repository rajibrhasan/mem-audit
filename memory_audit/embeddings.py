"""Share the CPU encoder within a fact worker, without changing its outputs."""
from functools import lru_cache


@lru_cache(maxsize=2)
def encoder(model):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model, device="cpu")
