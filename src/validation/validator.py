"""
Модуль для валидации модели на данных.
"""

import torch
from pathlib import Path
from typing import Dict, Any, List
from transformers import AutoModelForTokenClassification

from .metrics import compute_ner_metrics, extract_labels_from_tensors


class NERValidator:
    """
    Валидатор для NER модели.
    """
    
    def __init__(self, model_path: Path, device: str = "cpu"):
        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        self.id2label = None
        self._load_model()
    
    def _load_model(self):
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
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
        import torch.nn.functional as F
        
        all_true_labels = []
        all_pred_labels = []
        total_samples = 0
        
        target_length = 0
        for batch_path in batch_paths:
            batch_data = prepared_storage.load_batch(batch_path)
            tensors = batch_data['tensors']
            current_length = tensors['input_ids'].shape[1]
            if current_length > target_length:
                target_length = current_length
        
        print(f"  Целевая длина для валидации: {target_length}")
        
        for batch_path in batch_paths:
            batch_data = prepared_storage.load_batch(batch_path)
            tensors = batch_data['tensors']
            
            input_ids = tensors['input_ids']
            attention_mask = tensors['attention_mask']
            labels = tensors['labels'].cpu().numpy()
            
            current_length = input_ids.shape[1]
            if current_length < target_length:
                pad_len = target_length - current_length
                input_ids = F.pad(input_ids, (0, pad_len), value=0)
                attention_mask = F.pad(attention_mask, (0, pad_len), value=0)
            
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                predictions = outputs.logits.cpu().numpy()
            
            true_labels, pred_labels = extract_labels_from_tensors(
                predictions, labels, self.id2label, ignore_label
            )
            
            all_true_labels.extend(true_labels)
            all_pred_labels.extend(pred_labels)
            total_samples += len(input_ids)
        
        print(f"Валидация завершена. Обработано {total_samples} образцов из {len(batch_paths)} батчей.")
        
        return compute_ner_metrics(all_true_labels, all_pred_labels)


def validate_model(
    model_path: Path,
    prepared_storage,
    batch_paths: List[Path],
    device: str = "cpu"
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