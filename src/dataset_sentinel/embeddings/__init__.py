"""Optional embedding providers used to re-rank near-duplicate candidates.

Embeddings are never compared all-pairs. The near-duplicate detector generates
candidate pairs with perceptual hashes and, when a provider is enabled, keeps
only pairs whose embeddings are close (cosine similarity above
``policy.near_duplicate.embedding.min_cosine``). This is a heuristic that
tightens or widens the hash stage; it is not a claim of semantic duplicate
detection: two different photos of the same road can be similar without being
an invalid overlap.

Providers:

* ``builtin`` (always available, numpy only): a 288-dimensional descriptor of
  colour distribution and coarse gradient structure computed from a 32x32
  thumbnail. Colour-aware, unlike the grayscale dHash.
* ``torchvision`` (``pip install "dataset-sentinel[embeddings]"``): ImageNet
  ResNet-18 penultimate features on CPU. Experimental.

Third-party providers register through the ``dataset_sentinel.embeddings``
entry-point group.
"""

from .base import EmbeddingProvider, cosine_similarity, get_provider

__all__ = ["EmbeddingProvider", "cosine_similarity", "get_provider"]
