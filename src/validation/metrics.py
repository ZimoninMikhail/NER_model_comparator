"""
Модуль для расчёта метрик NER.
"""

import numpy as np
from src.data_preparation.bio_decoder import normalize_bio_labels
from typing import Dict, Any, List


def compute_ner_metrics(documents, entity_types):
    """Exact-match entity metrics. Macro усредняется только по типам с support > 0.

    documents: [{doc_id, true_entities, pred_entities, text?}].
    Сущность: {start, end, type} или {spans: [{start, end}], type}.
    Повторные одинаковые сущности внутри документа считаются один раз.
    """
    from numbers import Integral
    if (not isinstance(entity_types, (list, tuple)) or not entity_types
            or any(not isinstance(t, str) or not t for t in entity_types)
            or len(set(entity_types)) != len(entity_types)):
        raise ValueError('Требуется фиксированная схема уникальных типов')
    schema = set(entity_types)
    counts = {kind: {'tp': 0, 'fp': 0, 'fn': 0} for kind in entity_types}
    seen_documents = set()
    for doc in documents:
        doc_id = doc.get('doc_id')
        if not isinstance(doc_id, (str, int)) or isinstance(doc_id, bool) or doc_id in seen_documents:
            raise ValueError('Идентификаторы документов должны быть заданы и уникальны')
        seen_documents.add(doc_id)
        def entities(key):
            if key not in doc:
                raise ValueError(f'Не задано поле {key} для {doc_id}')
            result = set()
            for item in doc[key]:
                kind = item.get('type')
                if kind not in schema:
                    raise ValueError(f'Тип {kind} отсутствует в схеме оценки')
                if 'spans' in item:
                    if len(item['spans']) != 1:
                        raise ValueError('Оценщик ожидает непрерывные сущности')
                    start, end = item['spans'][0].get('start'), item['spans'][0].get('end')
                else:
                    start, end = item.get('start'), item.get('end')
                if any(not isinstance(v, Integral) or isinstance(v, bool) for v in (start,end)) or not 0 <= start < end:
                    raise ValueError('Некорректные границы оцениваемой сущности')
                if 'text' in doc and end > len(doc['text']):
                    raise ValueError('Сущность выходит за границы документа')
                result.add((int(start), int(end), kind))
            return result
        gold, predicted = entities('true_entities'), entities('pred_entities')
        for _, _, kind in gold & predicted: counts[kind]['tp'] += 1
        for _, _, kind in predicted - gold: counts[kind]['fp'] += 1
        for _, _, kind in gold - predicted: counts[kind]['fn'] += 1
    def scores(tp, fp, fn):
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        return precision, recall, f1
    per_type = {}
    for kind, c in counts.items():
        precision, recall, f1 = scores(**c)
        per_type[kind] = {**c, 'precision': precision, 'recall': recall, 'f1': f1,
                          'support': c['tp'] + c['fn'], 'predicted': c['tp'] + c['fp']}
    macro_types = [kind for kind in entity_types if per_type[kind]['support'] > 0]
    totals = {key: sum(c[key] for c in counts.values()) for key in ('tp','fp','fn')}
    precision, recall, f1 = scores(**totals)
    return {'metrics_version': 1, 'primary_metric': 'macro_f1',
        'match': 'exact_span_and_type', 'entity_types': list(entity_types),
        'macro_f1': sum(per_type[t]['f1'] for t in macro_types) / len(macro_types) if macro_types else 0.0,
        'macro_defined': bool(macro_types), 'macro_types': macro_types,
        'excluded_macro_types': [t for t in entity_types if t not in macro_types],
        'micro_precision': precision, 'micro_recall': recall, 'micro_f1': f1,
        'per_type': per_type, 'counts': totals, 'num_documents': len(seen_documents)}


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
    
    if preds.shape != labels.shape:
        raise ValueError("Размеры предсказаний и меток не совпадают")
    for pred, label in zip(preds, labels):
        true = []
        pred_list = []
        normalized = normalize_bio_labels([ignore_label if int(l) == ignore_label else int(p)
                                          for p, l in zip(pred, label)], id2label, ignore_label)
        mapping = {int(k): v for k, v in id2label.items()}
        for p, l in zip(normalized, label):
            if l != ignore_label:
                true.append(mapping[int(l)])
                pred_list.append(p)
        true_labels.append(true)
        pred_labels.append(pred_list)
    
    return true_labels, pred_labels