"""Сборка неперекрывающихся фрагментов для оценки исходных документов."""
from copy import deepcopy


def collect_document_predictions(records):
    groups = {}
    for sample, predicted in records:
        key = sample['doc_id']
        if 'chunk_count' not in sample:
            if key in groups:
                raise ValueError(f'Повторный документ: {key}')
            groups[key] = {'document': {'doc_id': key, 'text': sample['text'],
                'true_entities': sample['gold_entities'], 'pred_entities': list(predicted)},
                'count': 1, 'indexes': {0}, 'intervals': []}
            continue
        count, index = sample['chunk_count'], sample['chunk_index']
        if count < 1 or not 0 <= index < count:
            raise ValueError('Некорректный номер фрагмента')
        gold = sample['document_gold_entities']
        if key not in groups:
            groups[key] = {'document': {'doc_id': key, 'text': sample['text'],
                'true_entities': deepcopy(gold), 'pred_entities': []},
                'count': count, 'indexes': set(), 'intervals': []}
        group = groups[key]
        if (group['count'] != count or index in group['indexes'] or
                group['document']['text'] != sample['text'] or
                group['document']['true_entities'] != gold):
            raise ValueError(f'Несогласованные или повторные фрагменты: {key}')
        start, end = sample['chunk_start'], sample['chunk_end']
        if not 0 <= start <= end <= len(sample['text']):
            raise ValueError('Некорректные границы фрагмента')
        if any(not start <= entity['start'] < entity['end'] <= end for entity in predicted):
            raise ValueError('Предсказание выходит за границы фрагмента')
        group['indexes'].add(index)
        group['intervals'].append((index, start, end))
        group['document']['pred_entities'].extend(predicted)
    result = []
    for key, group in groups.items():
        if group['indexes'] != set(range(group['count'])):
            raise ValueError(f'Неполный набор фрагментов документа: {key}')
        intervals = sorted(group['intervals'])
        if any(a[2] > b[1] for a, b in zip(intervals, intervals[1:])):
            raise ValueError(f'Перекрывающиеся фрагменты: {key}')
        group['document']['pred_entities'].sort(key=lambda e: (e['start'], e['end'], e['type']))
        result.append(group['document'])
    return result
