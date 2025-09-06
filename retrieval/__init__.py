"""Retrieval package initializer.

Expose a small public surface used by tests.
"""
from .gnn_utils import compute_ghost_node_embeddings

__all__ = ["compute_ghost_node_embeddings"]
