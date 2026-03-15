"""
- global logging setup
- metric aggregation and reporting
- evaluation artifact orchestration
- embedding and training trend plotting
- tabular data persistence
- contrastive learning losses
- pseudo-labeling utilities
"""

from __future__ import annotations

from .logger import setup_global_logger
from .losses import NeighborhoodAwareContrastiveLoss
from .orchestrator import evaluate_predictions, save_evaluation_bundle
from .plotting import plot_embedding_projection, plot_seed_metric_trend, plot_training_history
from .pseudo_labeler import PseudoLabeler
from .reporting import aggregate_seed_results, save_benchmark_table
from .utils import PRIMARY_METRICS, save_frame, to_frame


__all__ = [
    # Logging
    "setup_global_logger",
    # Losses
    "NeighborhoodAwareContrastiveLoss",
    # Orchestrator
    "evaluate_predictions",
    "save_evaluation_bundle",
    # Plotting
    "plot_embedding_projection",
    "plot_seed_metric_trend",
    "plot_training_history",
    # Pseudo-Labeling
    "PseudoLabeler",
    # Reporting
    "aggregate_seed_results",
    "save_benchmark_table",
    # Utilities
    "PRIMARY_METRICS",
    "save_frame",
    "to_frame"
]
