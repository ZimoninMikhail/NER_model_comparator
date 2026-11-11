"""Сохранённый приоритет из проверенных аннотаций train. Без dev/test."""
import hashlib
import json
from pathlib import Path
from collections import Counter
from src.data_analysis.parser import parse_batch_entities
from src.data_collection.brat_loader import source_descriptor
from src.data_analysis.quality_checker import check_document_quality, VALIDATION_VERSION
from src.data_analysis.quality_filter import partition_batch_by_quality
from src.data_analysis.annotation_corrections import (
    load_correction_catalog, correction_metadata, apply_annotation_corrections,
)


def canonical_hash(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def document_signature(doc):
    return canonical_hash({k: doc.get(k) for k in ('doc_id', 'text', 'entities', 'source_files') if k in doc})


def train_signature(documents):
    # Сортировка обеспечивает независимость от порядка документов и границ батчей.
    return canonical_hash(sorted(documents, key=lambda d: str(d['doc_id'])))


def checked_batch(batch, entity_types, correction_catalog=None):
    parsed = parse_batch_entities(batch)
    if correction_catalog is not None:
        parsed = [apply_annotation_corrections(doc, correction_catalog) for doc in parsed]
    return [check_document_quality(doc, set(entity_types)) for doc in parsed]


def build_type_priority(config, loader):
    catalog = load_correction_catalog(config['data']['dataset_name'], config['data'].get('corrections_path'))
    applied_corrections = []
    loader.load_train()
    frequencies = Counter({t: 0 for t in config['data']['entity_types']})
    documents, damaged_ids = [], []
    for batch in loader:
        for doc in batch:
            documents.append({'doc_id': doc['doc_id'], 'sha256': document_signature(doc)})
        checked = checked_batch(batch, config['data']['entity_types'], catalog)
        applied_corrections.extend(item for doc in checked for item in doc.get('annotation_corrections', []))
        accepted, damaged = partition_batch_by_quality(checked)
        damaged_ids.extend(doc['doc_id'] for doc in damaged)
        for doc in accepted:
            # Учитываются и корректные разрывные сущности: это исходная разметка.
            frequencies.update(e['type'] for e in doc['parsed_entities'])
    if not documents:
        raise ValueError('Train пуст: невозможно определить приоритет')
    ids = [str(d['doc_id']) for d in documents]
    if len(set(ids)) != len(ids):
        raise ValueError('Повторяющиеся идентификаторы train')
    payload = {'annotation_corrections': correction_metadata(catalog),
        'applied_corrections': sorted(applied_corrections, key=lambda c: (c['correction_id'], str(c['doc_id']))),
        'format_version': 1, 'validation_version': VALIDATION_VERSION,
        'dataset_name': config['data']['dataset_name'],
        'dataset_revision': config['data'].get('dataset_revision'),
        'dataset_fingerprint': getattr(getattr(loader, 'dataset', None), '_fingerprint', None),
        'entity_types': list(config['data']['entity_types']),
        'train_documents': sorted(documents, key=lambda d: str(d['doc_id'])),
        'train_signature': train_signature(documents),
        'damaged_document_ids': sorted(damaged_ids, key=str),
        'frequencies': dict(frequencies),
        'priority': sorted(frequencies, key=lambda t: (frequencies[t], t))}
    if source_descriptor(config) is not None:
        payload['source'] = source_descriptor(config)
        payload['source']['schema_sha256'] = getattr(loader, 'schema_sha256', None)
    payload['sha256'] = canonical_hash(payload)
    return payload


def validate_type_priority(payload, config):
    if not isinstance(payload, dict):
        raise ValueError('Файл приоритетов должен содержать JSON-объект')
    expected = {'annotation_corrections': correction_metadata(load_correction_catalog(config['data']['dataset_name'], config['data'].get('corrections_path'))),
        'format_version': 1, 'validation_version': VALIDATION_VERSION,
        'dataset_name': config['data']['dataset_name'],
        'dataset_revision': config['data'].get('dataset_revision'),
        'entity_types': config['data']['entity_types']}
    if source_descriptor(config) is not None:
        source = payload.get('source', {})
        if not isinstance(source, dict) or any(source.get(k) != v for k, v in source_descriptor(config).items()):
            raise ValueError('Несовместимый источник в файле приоритетов')
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f'Несовместимый файл приоритетов: {key}. Создайте новый файл через type-statistics')
    data = {k: v for k, v in payload.items() if k != 'sha256'}
    if payload.get('sha256') != canonical_hash(data):
        raise ValueError('Нарушена контрольная сумма файла приоритетов')
    frequencies = payload.get('frequencies', {})
    if set(frequencies) != set(expected['entity_types']) or any(type(v) is not int or v < 0 for v in frequencies.values()):
        raise ValueError('Некорректные частоты типов')
    if payload.get('priority') != sorted(frequencies, key=lambda t: (frequencies[t], t)):
        raise ValueError('Приоритет не соответствует частотам типов')
    docs = payload.get('train_documents')
    if not isinstance(docs, list) or not docs or any(not isinstance(d, dict) or 'doc_id' not in d or 'sha256' not in d for d in docs):
        raise ValueError('Не сохранён состав train')
    if payload.get('train_signature') != train_signature(docs):
        raise ValueError('Некорректная подпись train')
    return payload


def load_type_priority(config):
    path = Path(config['data']['type_priority_path'])
    if not path.is_file():
        raise ValueError(f'Нет файла приоритетов {path}. Выполните: python run.py -m type-statistics')
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise ValueError(f'Ошибка чтения приоритетов: {error}') from error
    return validate_type_priority(payload, config)


def save_type_priority(payload, config):
    path = Path(config['data']['type_priority_path'])
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if load_type_priority(config) == payload:
            return path
        raise ValueError('Файл приоритетов уже существует и отличается; задайте новый type_priority_path')
    with path.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    return path
