"""
Модуль для проверки качества данных (консистентность сущностей).
"""

from typing import List, Dict, Any, Tuple


def check_entity_consistency(
    text: str, 
    entity: Dict[str, Any]
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Проверяет консистентность сущности.
    
    Для сущности с несколькими spans проверяет каждый.
    
    Args:
        text: Полный текст документа
        entity: Сущность с полями 'spans' и 'text'
        
    Returns:
        (is_consistent, list_of_errors)
        is_consistent: True если все spans корректны
        errors: Список ошибок для каждого span
    """
    errors = []
    entity_text = entity.get('text', '')
    spans = entity.get('spans', [])
    
    for i, span in enumerate(spans):
        start = span.get('start', 0)
        end = span.get('end', 0)
        
        if start < 0 or end > len(text):
            errors.append({
                'span_index': i,
                'error': 'out_of_bounds',
                'start': start,
                'end': end,
                'text_length': len(text)
            })
            continue
        
        extracted = text[start:end]
        if extracted != entity_text:
            errors.append({
                'span_index': i,
                'error': 'text_mismatch',
                'start': start,
                'end': end,
                'expected': entity_text,
                'actual': extracted
            })
    
    return len(errors) == 0, errors


def check_document_quality(
    doc: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Проверяет качество всех сущностей в документе.
    
    Args:
        doc: Документ с полями 'text' и 'parsed_entities'
        
    Returns:
        Словарь с метриками качества документа
    """
    text = doc.get('text', '')
    entities = doc.get('parsed_entities', [])
    
    consistent_count = 0
    inconsistent_count = 0
    all_errors = []
    
    for entity in entities:
        is_consistent, errors = check_entity_consistency(text, entity)
        if is_consistent:
            consistent_count += 1
        else:
            inconsistent_count += 1
            all_errors.extend(errors)
    
    total_entities = len(entities)
    consistency_ratio = consistent_count / total_entities if total_entities > 0 else 1.0
    
    result = doc.copy()
    result['quality'] = {
        'total_entities': total_entities,
        'consistent_entities': consistent_count,
        'inconsistent_entities': inconsistent_count,
        'consistency_ratio': consistency_ratio,
        'has_errors': inconsistent_count > 0,
        'errors': all_errors if inconsistent_count > 0 else []
    }
    
    return result


def calculate_batch_quality(
    batch: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Вычисляет метрики качества для всего батча.
    
    Args:
        batch: Список документов
        
    Returns:
        Словарь с агрегированными метриками
    """
    total_entities = 0
    total_consistent = 0
    total_inconsistent = 0
    docs_with_errors = 0
    all_errors = []
    
    for doc in batch:
        quality = doc.get('quality', {})
        total_entities += quality.get('total_entities', 0)
        total_consistent += quality.get('consistent_entities', 0)
        total_inconsistent += quality.get('inconsistent_entities', 0)
        
        if quality.get('has_errors', False):
            docs_with_errors += 1
            all_errors.extend(quality.get('errors', []))
    
    overall_consistency = total_consistent / total_entities if total_entities > 0 else 1.0
    docs_error_ratio = docs_with_errors / len(batch) if batch else 0
    
    return {
        'total_entities': total_entities,
        'consistent_entities': total_consistent,
        'inconsistent_entities': total_inconsistent,
        'consistency_ratio': round(overall_consistency, 4),
        'total_docs': len(batch),
        'docs_with_errors': docs_with_errors,
        'docs_error_ratio': round(docs_error_ratio, 4),
        'error_count': len(all_errors),
        'errors_sample': all_errors[:10]
    }