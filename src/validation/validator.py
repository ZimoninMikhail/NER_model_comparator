"""
Модуль для валидации модели на данных.
"""

import torch
from pathlib import Path
from typing import Dict, Any, List
from src.models.adapter import load_local_model, predict_token_labels, token_mask_from_samples

from src.device import resolve_device
from src.data_preparation.bio_decoder import decode_bio

from .metrics import compute_ner_metrics
from .document_predictions import collect_document_predictions


class NERValidator:
    """
    Валидатор для NER модели.
    """
    
    def __init__(self, model_path: Path, device: str = "auto"):
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_dir():
            raise ValueError(f"Локальная папка модели не найдена: {self.model_path}")
        self.device = resolve_device(device)
        self.model = None
        self.id2label = None
        self._load_model()
    
    def _load_model(self):
        self.model = load_local_model(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        self.id2label = self.model.config.id2label
        print(f"Модель загружена: {self.model_path}")
        print(f"  num_labels: {len(self.id2label)}")
    
    def validate(
        self,
        prepared_storage,
        batch_paths: List[Path],
        ignore_label: int = -100
    ) -> Dict[str, Any]:
        """
        Валидация на нескольких prepared батчах.
        
        Args:
            prepared_storage: Экземпляр PreparedDataStorage
            batch_paths: Список путей к prepared батчам
            ignore_label: ID игнорируемых токенов
            
        Returns:
            Словарь с метриками
        """
        
        decoded_documents = []
        # Схема модели фиксирована, а не выводится из встреченных предсказаний.
        entity_types = None
        for batch_path in batch_paths:
            batch_data = prepared_storage.load_batch(batch_path)
            tensors, samples = batch_data['tensors'], batch_data.get('samples')
            if samples is None or len(samples) != len(tensors['input_ids']):
                raise ValueError('Нет samples.json для валидации: пересоздайте prepared-данные')
            types = batch_data['dataset_info'].get('entity_types')
            if types is None:
                types = sorted({name[2:] for name in self.id2label.values() if name.startswith(('B-', 'I-'))})
            if entity_types is None:
                entity_types = list(types)
            elif list(types) != entity_types:
                raise ValueError('Батчи используют разные схемы типов')
            for sample in samples:
                if 'gold_entities' not in sample:
                    raise ValueError('Нет исходного эталона gold_entities: пересоздайте prepared-данные')
            with torch.no_grad():
                predictions = predict_token_labels(self.model, tensors['input_ids'].to(self.device),
                    tensors['attention_mask'].to(self.device),
                    token_mask_from_samples(samples, tensors['input_ids'].shape[1], self.device)).cpu().numpy()
            for row, sample in enumerate(samples):
                offsets = sample['offset_mapping']
                if len(offsets) > predictions.shape[1]:
                    raise ValueError('Координаты длиннее предсказанной последовательности')
                attention = tensors['attention_mask'][row, :len(offsets)]
                predicted = [int(p) if int(mask) else ignore_label
                    for p, mask in zip(predictions[row, :len(offsets)], attention)]
                decoded_documents.append((sample, decode_bio(predicted, offsets, sample['text'],
                        self.id2label, ignore_label=ignore_label)))
        if not decoded_documents:
            raise ValueError('Нет документов для оценки')
        decoded_documents = collect_document_predictions(decoded_documents)
        metrics = compute_ner_metrics(decoded_documents, entity_types)
        metrics['entity_predictions'] = decoded_documents
        print(f"Валидация: {len(decoded_documents)} документов; "
              f"macro-F1={metrics['macro_f1']:.4f}; micro-F1={metrics['micro_f1']:.4f}")
        return metrics


def validate_model(
    model_path: Path,
    prepared_storage,
    batch_paths: List[Path],
    device: str = "auto"
) -> Dict[str, Any]:
    """
    Функция валидации модели на нескольких батчах.
    
    Args:
        model_path: Путь к модели
        prepared_storage: Экземпляр PreparedDataStorage
        batch_paths: Список путей к prepared батчам для валидации
        device: Устройство
        
    Returns:
        Словарь с метриками
    """
    validator = NERValidator(model_path, device)
    return validator.validate(prepared_storage, batch_paths)