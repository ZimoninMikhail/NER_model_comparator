"""
Модуль для расчёта базовых метапараметров сырого батча
"""

from typing import List, Dict, Any
from collections import Counter


def calculate_batch_metadata(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Рассчитывает метапараметры для сырого батча.
    
    Returns:
        Dict с полями:
        - total_characters: общее количество символов
        - total_entities: общее количество сущностей
        - entity_type_distribution: распределение по типам
    """
    total_chars = 0
    total_entities = 0
    entity_type_counter = Counter()
    
    for doc in batch:
        text = doc.get('text', '')
        total_chars += len(text)
        
        entities = doc.get('entities', '')
        if not entities:
            continue
        
        if isinstance(entities, list):
            total_entities += len(entities)
            for entity in entities:
                if isinstance(entity, dict):
                    entity_type = entity.get('type', entity.get('label', 'unknown'))
                    entity_type_counter[entity_type] += 1
                elif isinstance(entity, str):
                    parts = entity.split('\t')
                    if len(parts) >= 2:
                        type_span = parts[1].split()
                        if type_span:
                            entity_type_counter[type_span[0]] += 1
        
        elif isinstance(entities, str):
            lines = [l.strip() for l in entities.split('\n') if l.strip()]
            total_entities += len(lines)
            for line in lines:
                parts = line.split('\t')
                if len(parts) >= 2:
                    type_span = parts[1].split()
                    if type_span:
                        entity_type_counter[type_span[0]] += 1
    
    return {
        'total_characters': total_chars,
        'total_entities': total_entities,
        'entity_type_distribution': dict(entity_type_counter)
    }