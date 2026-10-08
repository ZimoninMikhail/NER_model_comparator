"""
Модуль для расчёта метрик NER.
"""

import numpy as np
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from typing import Dict, Any, List


def compute_ner_metrics(
    true_labels: List[List[str]],
    pred_labels: List[List[str]]
) -> Dict[str, Any]:
    """
    Вычисляет метрики для NER.
    
    Args:
        true_labels: Истинные метки для каждого токена
        pred_labels: Предсказанные метки для каждого токена
        
    Returns:
        Словарь с метриками
    """
    return {
        'accuracy': accuracy_score(true_labels, pred_labels),
        'precision': precision_score(true_labels, pred_labels, zero_division=0),
        'recall': recall_score(true_labels, pred_labels, zero_division=0),
        'f1': f1_score(true_labels, pred_labels, zero_division=0),
        'report': classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
    }


def extract_labels_from_tensors(predictions, labels, id2label, ignore_label=-100):
    """
    Извлекает метки из тензоров предсказаний.
    
    Args:
        predictions: Тензор предсказаний модели (logits)
        labels: Тензор истинных меток
        id2label: Словарь {ID: метка}
        ignore_label: ID игнорируемых токенов
        
    Returns:
        true_labels, pred_labels
    """
    preds = np.argmax(predictions, axis=2)
    
    true_labels = []
    pred_labels = []
    
    for pred, label in zip(preds, labels):
        true = []
        pred_list = []
        for p, l in zip(pred, label):
            if l != ignore_label:
                true.append(id2label[l])
                pred_list.append(id2label[p])
        true_labels.append(true)
        pred_labels.append(pred_list)
    
    return true_labels, pred_labels