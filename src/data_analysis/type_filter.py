"""
Модуль для фильтрации сущностей по типам (география).
"""

from typing import List, Dict, Any, Set

DEFAULT_GEO_TYPES = {
    "COUNTRY",
    "CITY", 
    "LOCATION",
    "STATE_OR_PROV",
    "DISTRICT"
}


def filter_entities_by_type(
    entities: List[Dict[str, Any]], 
    allowed_types: Set[str]
) -> List[Dict[str, Any]]:
    """
    Оставляет только сущности разрешённых типов.
    
    Args:
        entities: Список сущностей (каждая с полем 'type')
        allowed_types: Множество разрешённых типов
        
    Returns:
        Отфильтрованный список сущностей
    """
    return [e for e in entities if e.get('type') in allowed_types]


def filter_document_geo(
    doc: Dict[str, Any],
    allowed_types: Set[str] = None
) -> Dict[str, Any]:
    """
    Фильтрует документ, оставляя только географические сущности.
    
    Args:
        doc: Документ с полем 'parsed_entities'
        allowed_types: Разрешённые типы (по умолчанию DEFAULT_GEO_TYPES)
        
    Returns:
        Документ с отфильтрованными сущностями
    """
    if allowed_types is None:
        allowed_types = DEFAULT_GEO_TYPES
    
    parsed_entities = doc.get('parsed_entities', [])
    original_count = len(parsed_entities)
    
    filtered_entities = filter_entities_by_type(parsed_entities, allowed_types)
    
    result = doc.copy()
    result['parsed_entities'] = filtered_entities
    result['entities_count'] = len(filtered_entities)
    result['original_entities_count'] = original_count
    result['removed_entities_count'] = original_count - len(filtered_entities)
    
    removed_types = {}
    for e in parsed_entities:
        t = e.get('type')
        if t not in allowed_types:
            removed_types[t] = removed_types.get(t, 0) + 1
    result['removed_entity_types'] = removed_types
    
    return result


def filter_batch_geo(
    batch: List[Dict[str, Any]],
    allowed_types: Set[str] = None,
    remove_empty_docs: bool = True
) -> List[Dict[str, Any]]:
    """
    Фильтрует весь батч, оставляя только географические сущности.
    
    Args:
        batch: Список документов
        allowed_types: Разрешённые типы
        remove_empty_docs: Удалять ли документы без сущностей после фильтрации
        
    Returns:
        Отфильтрованный список документов
    """
    filtered_batch = []
    removed_docs = []
    
    for doc in batch:
        filtered_doc = filter_document_geo(doc, allowed_types)
        
        if remove_empty_docs and filtered_doc['entities_count'] == 0:
            removed_docs.append(filtered_doc.get('doc_id'))
            continue
        
        filtered_batch.append(filtered_doc)
    
    if filtered_batch:
        filtered_batch[0]['_filter_stats'] = {
            'original_docs': len(batch),
            'filtered_docs': len(filtered_batch),
            'removed_docs': removed_docs,
            'allowed_types': list(allowed_types or DEFAULT_GEO_TYPES)
        }
    
    print(f"  Фильтрация по типам: {len(batch)} -> {len(filtered_batch)} документов")
    if removed_docs:
        print(f"    Удалено документов без гео-сущностей: {len(removed_docs)}")
    
    return filtered_batch


def get_filter_statistics(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Собирает статистику о фильтрации для метаданных.
    
    Args:
        batch: Отфильтрованный батч
        
    Returns:
        Словарь со статистикой
    """
    if not batch:
        return {}
    
    stats = batch[0].get('_filter_stats', {})
    
    removed_types = {}
    for doc in batch:
        for t, count in doc.get('removed_entity_types', {}).items():
            removed_types[t] = removed_types.get(t, 0) + count
    
    total_before = sum(doc.get('original_entities_count', doc.get('entities_count', 0)) for doc in batch)
    total_after = sum(doc.get('entities_count', 0) for doc in batch)
    
    return {
        'allowed_types': stats.get('allowed_types', list(DEFAULT_GEO_TYPES)),
        'original_docs': stats.get('original_docs', len(batch)),
        'filtered_docs': len(batch),
        'removed_docs': stats.get('removed_docs', []),
        'removed_docs_count': len(stats.get('removed_docs', [])),
        'total_entities_before': total_before,
        'total_entities_after': total_after,
        'removed_by_type': removed_types
    }