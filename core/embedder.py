"""
Pluggable text embedding backend.

Backends:
  "sentence-transformers"  real semantic embeddings (all-MiniLM-L6-v2).
                           This is the default and what your paper numbers
                           should be generated with.
  "tfidf"                  lightweight lexical fallback (scikit-learn only).
                           Useful if sentence-transformers is unavailable.

Both expose the same interface -- fit(corpus) and embed(texts) -> np.ndarray --
so the detector core and sensors never need to know which one is active.
"""

from __future__ import annotations
import numpy as np

# "auto" tries sentence-transformers and silently falls back to tfidf.
EMBEDDER_BACKEND = "auto"
EMBED_DIM = 64  # only used by the tfidf fallback


class TfidfEmbedder:
    """Lexical fallback embedder. Development use only -- not for paper numbers."""

    def __init__(self, n_components: int = EMBED_DIM):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        self.name = "tfidf"
        self.n_components = n_components
        self._vectorizer = TfidfVectorizer()
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._fitted = False

    def fit(self, corpus: list[str]) -> None:
        from sklearn.decomposition import TruncatedSVD

        n = min(self.n_components, max(2, len(corpus) - 1))
        if n != self.n_components:
            self._svd = TruncatedSVD(n_components=n, random_state=42)
        X = self._vectorizer.fit_transform(corpus)
        self._svd.fit(X)
        self._fitted = True

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            self.fit(texts)
        vecs = self._svd.transform(self._vectorizer.transform(texts))
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


class SentenceTransformerEmbedder:
    """Real semantic embedder. Use this for your actual experiments."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.name = f"sentence-transformers:{model_name}"
        self._model = SentenceTransformer(model_name)

    def fit(self, corpus: list[str]) -> None:
        pass  # pretrained; nothing to fit

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, normalize_embeddings=True))


def get_embedder(backend: str | None = None):
    backend = backend or EMBEDDER_BACKEND

    if backend == "auto":
        try:
            return SentenceTransformerEmbedder()
        except Exception as exc:  # noqa: BLE001
            print(f"[embedder] sentence-transformers unavailable ({exc}); "
                  f"falling back to TF-IDF. Do not report these numbers.")
            return TfidfEmbedder()

    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder()
    if backend == "tfidf":
        return TfidfEmbedder()
    raise ValueError(f"Unknown embedder backend: {backend}")
