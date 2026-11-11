"""Строгая проверка аннотаций. Разрывная разметка сама по себе корректна."""
from copy import deepcopy

VALIDATION_VERSION = 2


def check_entity_consistency(text, entity, allowed_types=None):
    errors = []
    if allowed_types is not None and entity.get('type') not in allowed_types:
        errors.append({'error': 'unknown_type', 'type': entity.get('type')})
    spans = entity.get('spans')
    if not isinstance(spans, list) or not spans:
        return False, errors + [{'error': 'empty_spans'}]
    previous_end = -1
    for i, span in enumerate(spans):
        if not isinstance(span, dict):
            errors.append({'error': 'invalid_span', 'span_index': i})
            continue
        start, end = span.get('start'), span.get('end')
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
            errors.append({'error': 'invalid_bounds', 'span_index': i, 'start': start, 'end': end})
            continue
        if start < previous_end:
            errors.append({'error': 'unordered_or_overlapping_spans', 'span_index': i})
        previous_end = end
    if not errors:
        # BRAT представляет разрывную сущность соединением фрагментов через пробел.
        extracted = ' '.join(text[s['start']:s['end']] for s in spans)
        if extracted != entity.get('text'):
            errors.append({'error': 'text_mismatch', 'expected': entity.get('text'), 'actual': extracted})
    return not errors, errors


def check_document_quality(doc, allowed_types=None):
    result = deepcopy(doc)
    errors = deepcopy(doc.get('parse_errors', []))
    consistent = 0
    for entity in doc.get('parsed_entities', []):
        valid, issues = check_entity_consistency(doc.get('text', ''), entity, allowed_types)
        consistent += int(valid)
        for issue in issues:
            errors.append({'doc_id': doc.get('doc_id'), 'entity_id': entity.get('id'), **issue})
    entities = doc.get('parsed_entities', [])
    # Повторные ID означают неоднозначное происхождение аннотации.
    ids = [e.get('id') for e in entities]
    if len(ids) != len(set(ids)):
        errors.append({'doc_id': doc.get('doc_id'), 'entity_id': None, 'error': 'duplicate_entity_id'})
    total = len(entities) + len(doc.get('parse_errors', []))
    result['quality'] = {'validation_version': VALIDATION_VERSION,
        'total_entities': total, 'consistent_entities': consistent,
        'inconsistent_entities': total - consistent,
        'consistency_ratio': consistent / total if total else 1.0,
        'has_errors': bool(errors), 'errors': errors}
    return result


def calculate_batch_quality(batch):
    total = sum(d.get('quality', {}).get('total_entities', 0) for d in batch)
    consistent = sum(d.get('quality', {}).get('consistent_entities', 0) for d in batch)
    errors = [e for d in batch for e in d.get('quality', {}).get('errors', [])]
    return {'total_entities': total, 'consistent_entities': consistent,
        'inconsistent_entities': total - consistent,
        'consistency_ratio': consistent / total if total else 1.0,
        'total_docs': len(batch), 'docs_with_errors': sum(d['quality']['has_errors'] for d in batch),
        'error_count': len(errors), 'errors_sample': errors[:10]}
