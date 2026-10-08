"""
Модуль обучения модели.
"""

from .model_factory import create_model, get_device
from .trainer import train_model, NERDataset, train_from_batches

__all__ = [
    'create_model',
    'get_device',
    'train_model',
    'NERDataset',
    'train_from_batches',
]