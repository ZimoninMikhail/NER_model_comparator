"""Отделение повреждённых документов; документы без сущностей сохраняются."""


def filter_document_by_quality(doc):
    quality = doc.get('quality')
    if quality is None:
        raise ValueError('Документ не прошёл quality_checker')
    return (False, {'reason': 'damaged_document'}) if quality['has_errors'] else (True, {})


def partition_batch_by_quality(batch):
    accepted, damaged = [], []
    for doc in batch:
        (accepted if filter_document_by_quality(doc)[0] else damaged).append(doc)
    return accepted, damaged


def filter_batch_by_quality(batch):
    return partition_batch_by_quality(batch)[0]


def get_quality_filter_statistics(batch):
    return {'docs_after_filter': len(batch),
            'total_entities_after_filter': sum(len(d.get('parsed_entities', [])) for d in batch)}
