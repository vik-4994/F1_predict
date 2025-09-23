# FILE: src/training/models/__init__.py
"""
Models package: expose common ranker constructors.
"""

from .mlp_ranker import Ranker as MLPRanker, make_model as make_mlp_ranker

__all__ = [
    "MLPRanker",
    "make_mlp_ranker",
]
