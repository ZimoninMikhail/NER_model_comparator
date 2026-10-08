"""
Модуль для фильтрации документов по порогам качества.
"""

from typing import List, Dict, Any, Optional


def filter_document_by_quality(
    doc: Dict[str, Any],
    min_consistency: float = 0.95,
    min_entities: int = 1
) -> tuple[bool, Dict[str, Any]]:
    """
    Проверяет, проходит ли документ пороги качества.
    
    Args:
        doc: Документ с полем 'quality'
        min_consistency: Минимальная доля консистентных сущностей
        min_entities: Минимальное количество сущностей
        
    Returns:
        (keep, reason) - keep=True если документ проходит фильтрацию
    """
    quality = doc.get('quality', {})
    
    consistency_ratio = quality.get('consistency_ratio', 1.0)
    total_entities = quality.get('total_entities', 0)
    
    if consistency_ratio < min_consistency:
        return False, {'reason': 'low_consistency', 'value': consistency_ratio, 'threshold': min_consistency}
    
    if total_entities < min_entities:
        return False, {'reason': 'too_few_entities', 'value': total_entities, 'threshold': min_entities}
    
    return True, {}


def filter_batch_by_quality(
    batch: List[Dict[str, Any]],
    min_consistency: float = 0.95,
    min_entities: int = 1,
    remove_empty_docs: bool = True
) -> List[Dict[str, Any]]:
    """
    Фильтрует батч по порогам качества.
    
    Args:
        batch: Список документов (с полем 'quality')
        min_consistency: Минимальная доля консистентных сущностей
        min_entities: Минимальное количество сущностей
        remove_empty_docs: Удалять ли документы, не прошедшие фильтрацию
        
    Returns:
        Отфильтрованный список документов
    """
    filtered_batch = []
    removed_docs = []
    removal_reasons = {
        'low_consistency': [],
        'too_few_entities': []
    }
    
    for doc in batch:
        keep, reason = filter_document_by_quality(doc, min_consistency, min_entities)
        
        if keep or not remove_empty_docs:
            filtered_batch.append(doc)
        else:
            removed_docs.append(doc.get('doc_id'))
            if reason.get('reason') == 'low_consistency':
                removal_reasons['low_consistency'].append({
                    'doc_id': doc.get('doc_id'),
                    'value': reason['value']
                })
            elif reason.get('reason') == 'too_few_entities':
                removal_reasons['too_few_entities'].append({
                    'doc_id': doc.get('doc_id'),
                    'value': reason['value']
                })
    
    if filtered_batch:
        filtered_batch[0]['_quality_filter_stats'] = {
            'original_docs': len(batch),
            'filtered_docs': len(filtered_batch),
            'removed_docs': removed_docs,
            'removal_reasons': removal_reasons,
            'thresholds': {
                'min_consistency': min_consistency,
                'min_entities': min_entities
            }
        }
    
    print(f"  Фильтрация по качеству: {len(batch)} -> {len(filtered_batch)} документов")
    if removal_reasons['low_consistency']:
        print(f"    Отброшено по низкой консистентности: {len(removal_reasons['low_consistency'])}")
    if removal_reasons['too_few_entities']:
        print(f"    Отброшено по малому количеству сущностей: {len(removal_reasons['too_few_entities'])}")
    
    return filtered_batch


def get_quality_filter_statistics(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Собирает статистику о фильтрации по качеству для метаданных.
    
    Args:
        batch: Отфильтрованный батч
        
    Returns:
        Словарь со статистикой
    """
    if not batch:
        return {}
    
    stats = batch[0].get('_quality_filter_stats', {})
    
    total_entities = sum(doc.get('entities_count', 0) for doc in batch)
    consistent = sum(doc.get('quality', {}).get('consistent_entities', 0) for doc in batch)
    
    entity_type_distribution = {}
    for doc in batch:
        for ent in doc.get('parsed_entities', []):
            ent_type = ent.get('type')
            if ent_type:
                entity_type_distribution[ent_type] = entity_type_distribution.get(ent_type, 0) + 1
    
    return {
        'quality_thresholds': stats.get('thresholds', {'min_consistency': 0.95, 'min_entities': 1}),
        'docs_before_filter': stats.get('original_docs', len(batch)),
        'docs_after_filter': len(batch),
        'docs_removed': len(stats.get('removed_docs', [])),
        'removed_doc_ids': stats.get('removed_docs', []),
        'removed_low_consistency': len(stats.get('removal_reasons', {}).get('low_consistency', [])),
        'removed_few_entities': len(stats.get('removal_reasons', {}).get('too_few_entities', [])),
        'total_entities_after_filter': total_entities,
        'consistent_entities_after_filter': consistent,
        'final_consistency_ratio': round(consistent / total_entities, 4) if total_entities > 0 else 1.0,
        'entity_type_distribution': entity_type_distribution  # добавлено
    }