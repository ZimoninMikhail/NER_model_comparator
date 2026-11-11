"""Сводка потерь разметки; исходный = сохранённый + все причины исключения."""
from collections import Counter


def new_split_report(entity_types):
    return {'documents_total': 0, 'documents_saved': 0, 'documents_damaged': 0,
        'documents_without_entities': 0, 'parse_errors': 0, 'annotation_corrections': [],
        'entities_before': {t: 0 for t in entity_types},
        'entities_after': {t: 0 for t in entity_types}, 'excluded_by_reason': {},
        'excluded_by_type': {}, 'errors_by_reason': {}}


def add_batch_report(report, checked, flattened, damaged):
    report['documents_total'] += len(checked)
    report['documents_saved'] += len(flattened)
    report['documents_damaged'] += len(damaged)
    report['parse_errors'] += sum(len(d.get('parse_errors', [])) for d in checked)
    for doc in checked:
        report['annotation_corrections'].extend(doc.get('annotation_corrections', []))
        for entity in doc.get('parsed_entities', []):
            t = entity['type']
            report['entities_before'][t] = report['entities_before'].get(t, 0) + 1
        for error in doc['quality']['errors']:
            reason = error['error']
            report['errors_by_reason'][reason] = report['errors_by_reason'].get(reason, 0) + 1
    removed = [item for d in flattened for item in d['excluded_annotations']]
    removed.extend({'entity': e, 'reason': 'damaged_document'} for d in damaged for e in d.get('parsed_entities', []))
    for item in removed:
        t, reason = item['entity']['type'], item['reason']
        report['excluded_by_reason'][reason] = report['excluded_by_reason'].get(reason, 0) + 1
        by_type = report['excluded_by_type'].setdefault(t, {})
        by_type[reason] = by_type.get(reason, 0) + 1
    for doc in flattened:
        report['documents_without_entities'] += int(not doc['parsed_entities'])
        for entity in doc['parsed_entities']:
            t = entity['type']
            report['entities_after'][t] = report['entities_after'].get(t, 0) + 1
    report['loss_by_type'] = {t: count - report['entities_after'].get(t, 0)
                               for t, count in report['entities_before'].items()}
