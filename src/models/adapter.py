"""Локальная загрузка архитектур и единый выход: BIO-ID, не эталонные метки."""
import json
from pathlib import Path

import torch
from . import bert
from src.data_preparation.bio_decoder import validate_offsets


def is_bilstm_crf(model):
    return getattr(getattr(model, 'config', None), 'model_type', None) == 'bilstm_crf'


def token_mask_from_samples(samples, width, device='cpu'):
    """Маска берётся только из координат, labels в этой функции недоступны."""
    mask = torch.zeros((len(samples), width), dtype=torch.bool, device=device)
    for row, sample in enumerate(samples):
        offsets = validate_offsets(sample['offset_mapping'], sample.get('text'))
        if len(offsets) > width:
            raise ValueError('Координаты длиннее модельной последовательности')
        mask[row, :len(offsets)] = torch.tensor([span != (0, 0) for span in offsets],
                                               dtype=torch.bool, device=device)
    return mask


def predict_token_labels(model, input_ids, attention_mask, token_mask=None):
    if is_bilstm_crf(model):
        if token_mask is None:
            raise ValueError('Для CRF-предсказания нужна независимая маска из координат')
        return model(input_ids, attention_mask=attention_mask,
                     token_mask=token_mask, decode=True).predictions
    return model(input_ids, attention_mask=attention_mask).logits.argmax(dim=-1)


def load_local_model(path):
    path = Path(path)
    config_path = path / 'config.json'
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding='utf-8'))
        if config.get('model_type') == 'bilstm_crf':
            from .bilstm_crf import load_model
            return load_model(path)
    # Обычные HF-модели и ранее сохранённые BERT-комплекты.
    return bert.load_model(path)
