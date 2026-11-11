"""Детерминированный выбор плоских аннотаций без изменения исходных."""
from copy import deepcopy

PREPROCESSING_VERSION = 3


def flatten_document(doc, priority):
    if doc.get('quality', {}).get('has_errors'):
        raise ValueError('Повреждённый документ нельзя преобразовывать в обучающий')
    rank = {name: i for i, name in enumerate(priority)}
    if len(rank) != len(priority):
        raise ValueError('Повторяющиеся типы в приоритете')
    result = deepcopy(doc)
    original = deepcopy(doc.get('parsed_entities', []))
    result['original_parsed_entities'] = deepcopy(doc.get('original_parsed_entities', original))
    candidates, excluded = [], []
    for entity in original:
        if entity['type'] not in rank:
            raise ValueError(f'Тип отсутствует в приоритете: {entity["type"]}')
        if len(entity['spans']) > 1:
            excluded.append({'entity': deepcopy(entity), 'reason': 'discontinuous'})
        elif len(entity['spans']) == 1:
            candidates.append(entity)
        else:
            raise ValueError('Пустые интервалы: сначала выполните проверку разметки')
    def key(entity):
        span = entity['spans'][0]
        return (-(span['end'] - span['start']), span['start'], rank[entity['type']],
                str(entity.get('id', '')), entity.get('text', ''))
    chosen, seen = [], set()
    for entity in sorted(candidates, key=key):
        span = entity['spans'][0]
        signature = (span['start'], span['end'], entity['type'])
        if signature in seen:
            excluded.append({'entity': deepcopy(entity), 'reason': 'duplicate'})
            continue
        seen.add(signature)
        conflict = next((other for other in chosen if
            span['start'] < other['spans'][0]['end'] and other['spans'][0]['start'] < span['end']), None)
        if conflict is not None:
            other_span = conflict['spans'][0]
            reason = 'type_priority' if span == other_span else 'overlap'
            excluded.append({'entity': deepcopy(entity), 'reason': reason,
                             'selected_entity_id': conflict.get('id')})
        else:
            chosen.append(entity)
    chosen.sort(key=lambda e: (e['spans'][0]['start'], e['spans'][0]['end'], e['type']))
    result.update(parsed_entities=deepcopy(chosen), flat_entities=deepcopy(chosen),
        entities_count=len(chosen), excluded_annotations=excluded,
        preprocessing_version=PREPROCESSING_VERSION)
    return result
