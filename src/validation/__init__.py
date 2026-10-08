"""
Модуль валидации модели.
"""

from .metrics import compute_ner_metrics, extract_labels_from_tensors
from .validator import NERValidator, validate_model
from .model_registry import ModelRegistry

__all__ = [
    'compute_ner_metrics',
    'extract_labels_from_tensors',
    'NERValidator',
    'validate_model',
    'ModelRegistry'
]