"""Чтение BRAT/NEREL без потери исходных аннотаций и ошибок."""
from copy import deepcopy


def parse_span(span_str):
    parts = span_str.split()
    if len(parts) != 2:
        raise ValueError(f'Неверный интервал: {span_str}')
    try:
        return {'start': int(parts[0]), 'end': int(parts[1])}
    except ValueError:
        raise ValueError(f'Координаты должны быть целыми: {span_str}') from None


def parse_entity(entity_str):
    if not isinstance(entity_str, str):
        raise ValueError('Аннотация должна быть строкой')
    parts = entity_str.rstrip('\r\n').split('\t', 2)
    if len(parts) != 3 or not parts[0]:
        raise ValueError('Ожидается ID<TAB>TYPE start end<TAB>text')
    descriptor = parts[1].split(maxsplit=1)
    if len(descriptor) != 2:
        raise ValueError('Не заданы тип и интервалы')
    spans = [parse_span(part) for part in descriptor[1].split(';')]
    return {'id': parts[0], 'type': descriptor[0], 'spans': spans,
            'text': parts[2], 'is_multiple': len(spans) > 1, 'source_annotation': entity_str}


def parse_entities(entities_data, doc_id=None, return_errors=False):
    if entities_data is None or entities_data == '':
        lines = []
    elif isinstance(entities_data, str):
        lines = entities_data.splitlines()
    elif isinstance(entities_data, list):
        lines = entities_data
    else:
        lines = [entities_data]
    parsed, errors = [], []
    for index, line in enumerate(lines):
        if isinstance(line, str) and not line.strip():
            continue
        try:
            parsed.append(parse_entity(line))
        except ValueError as error:
            entity_id = line.split('\t', 1)[0] if isinstance(line, str) else None
            errors.append({'doc_id': doc_id, 'entity_id': entity_id,
                'annotation_index': index, 'source_annotation': deepcopy(line),
                'error': 'parse_error', 'message': str(error)})
    return (parsed, errors) if return_errors else parsed


def parse_document_entities(doc):
    result = deepcopy(doc)
    parsed, errors = parse_entities(doc.get('entities', []), doc.get('doc_id'), True)
    result.update(original_annotations=deepcopy(doc.get('entities', [])),
        original_parsed_entities=deepcopy(parsed), parsed_entities=parsed,
        parse_errors=errors, entities_count=len(parsed),
        multiple_entities_count=sum(e['is_multiple'] for e in parsed))
    return result


def parse_batch_entities(batch):
    return [parse_document_entities(doc) for doc in batch]


def get_entity_types(entities):
    return [e['type'] for e in entities]


def get_entity_type_counts(entities):
    from collections import Counter
    return dict(Counter(get_entity_types(entities)))


def get_total_span_count(entities):
    return sum(len(e['spans']) for e in entities)
