"""
Модуль для парсинга сущностей из формата NEREL в структурированный вид.
"""

from typing import List, Dict, Any, Union, Optional
import re


def parse_span(span_str: str) -> Dict[str, int]:
    """
    Парсит один интервал вида "start end" или "start;end".
    
    Args:
        span_str: Строка с интервалом (например "0 5" или "895;909")
        
    Returns:
        Словарь с полями start, end
    """
    span_str = span_str.replace(';', ' ')
    parts = span_str.split()
    
    if len(parts) != 2:
        raise ValueError(f"Неверный формат интервала: {span_str}")
    
    try:
        start = int(parts[0])
        end = int(parts[1])
    except ValueError as e:
        raise ValueError(f"start/stop должны быть числами: {parts[0]}, {parts[1]}") from e
    
    return {'start': start, 'end': end}


def parse_entity(entity_str: str) -> Dict[str, Any]:
    """
    Парсит одну строку сущности в словарь.
    
    Форматы:
    - Обычный: "<id>\t<type> <start> <stop>\t<text>"
      Пример: "123\tCOUNTRY 0 5\tРоссия"
    
    - С несколькими интервалами: "<id>\t<type> <start1> <stop1>;<start2> <stop2>\t<text>"
      Пример: "T23\tEVENT 895 909;947 957\tсобытие"
    
    Args:
        entity_str: Строка с сущностью
        
    Returns:
        Словарь с полями: id, type, spans, text, is_multiple
    """
    if not entity_str or not entity_str.strip():
        raise ValueError("Пустая строка сущности")
    
    parts = entity_str.strip().split('\t')
    
    if len(parts) != 3:
        raise ValueError(
            f"Неверный формат сущности. Ожидается 3 части, получено {len(parts)}: {entity_str}"
        )
    
    entity_id = parts[0]
    span_part = parts[1]
    text = parts[2]
    
    span_tokens = span_part.split()
    if len(span_tokens) < 3:
        raise ValueError(
            f"Неверный формат span. Ожидается 'TYPE start stop', получено: {span_part}"
        )
    
    entity_type = span_tokens[0]
    interval_strings = span_tokens[1:]
    
    all_intervals = []
    current_interval = []
    
    for token in interval_strings:
        if ';' in token:
            sub_tokens = token.split(';')
            for sub in sub_tokens:
                current_interval.append(sub)
                if len(current_interval) == 2:
                    all_intervals.append(current_interval)
                    current_interval = []
        else:
            current_interval.append(token)
            if len(current_interval) == 2:
                all_intervals.append(current_interval)
                current_interval = []
    
    if current_interval:
        raise ValueError(f"Неполный интервал: {current_interval}")
    
    spans = []
    for interval in all_intervals:
        try:
            start = int(interval[0])
            end = int(interval[1])
            spans.append({'start': start, 'end': end})
        except ValueError as e:
            raise ValueError(f"start/stop должны быть числами: {interval[0]}, {interval[1]}") from e
    
    if not spans:
        raise ValueError(f"Не найдено ни одного интервала: {span_part}")
    
    return {
        'id': entity_id,
        'type': entity_type,
        'spans': spans,
        'text': text,
        'is_multiple': len(spans) > 1
    }


def parse_entities(entities_data: Union[str, List[str]]) -> List[Dict[str, Any]]:
    """
    Парсит сущности из строки или списка строк.
    
    Поддерживает два формата:
    - Строка с разделителями \n
    - Список строк
    
    Args:
        entities_data: Строка с сущностями или список строк
        
    Returns:
        Список словарей с сущностями
    """
    entities = []
    
    if not entities_data:
        return entities
    
    if isinstance(entities_data, str):
        lines = [line.strip() for line in entities_data.split('\n') if line.strip()]
        for line in lines:
            try:
                entities.append(parse_entity(line))
            except ValueError as e:
                print(f"Предупреждение: пропущена сущность из-за ошибки: {e}")
                continue
        return entities
    
    if isinstance(entities_data, list):
        for item in entities_data:
            if not item:
                continue
            try:
                entities.append(parse_entity(item))
            except ValueError as e:
                print(f"Предупреждение: пропущена сущность из-за ошибки: {e}")
                continue
        return entities
    
    raise TypeError(f"entities_data должен быть str или list, получен {type(entities_data)}")


def parse_document_entities(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Добавляет в документ поле 'parsed_entities' с распарсенными сущностями.
    
    Args:
        doc: Документ с полем 'entities'
        
    Returns:
        Документ с добавленным полем 'parsed_entities'
    """
    entities_data = doc.get('entities', '')
    parsed = parse_entities(entities_data)
    
    result = doc.copy()
    result['parsed_entities'] = parsed
    result['entities_count'] = len(parsed)
    
    multiple_count = sum(1 for e in parsed if e['is_multiple'])
    result['multiple_entities_count'] = multiple_count
    
    return result


def parse_batch_entities(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Применяет parse_document_entities ко всем документам батча.
    
    Args:
        batch: Список документов
        
    Returns:
        Список документов с добавленным полем 'parsed_entities'
    """
    return [parse_document_entities(doc) for doc in batch]


def get_entity_types(entities: List[Dict[str, Any]]) -> List[str]:
    """
    Возвращает список типов сущностей.
    
    Args:
        entities: Список сущностей
        
    Returns:
        Список типов
    """
    return [e['type'] for e in entities]


def get_entity_type_counts(entities: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Возвращает словарь с подсчётом типов сущностей.
    
    Args:
        entities: Список сущностей
        
    Returns:
        Словарь {тип: количество}
    """
    counts = {}
    for e in entities:
        t = e['type']
        counts[t] = counts.get(t, 0) + 1
    return counts


def get_total_span_count(entities: List[Dict[str, Any]]) -> int:
    """
    Возвращает общее количество интервалов (spans) во всех сущностях.
    Полезно для статистики.
    
    Args:
        entities: Список сущностей
        
    Returns:
        Общее количество интервалов
    """
    return sum(len(e['spans']) for e in entities)
    