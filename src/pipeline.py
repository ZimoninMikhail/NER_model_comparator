"""Версионируемая подготовка данных и отдельный запуск обучения."""
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from transformers import set_seed
from src.device import resolve_device
from src.config import validate_config
from src.data_collection.loader import StreamingDataLoader
from src.data_collection.brat_loader import BratDataLoader, source_descriptor
from src.data_collection.storage import RawDataStorage
from src.data_collection.metadata_calculator import calculate_batch_metadata
from src.data_analysis.type_statistics import (
    checked_batch, build_type_priority, load_type_priority, save_type_priority,
    document_signature, train_signature, validate_type_priority,
)
from src.data_analysis.quality_filter import partition_batch_by_quality
from src.data_analysis.annotation_corrections import load_correction_catalog, correction_metadata
from src.data_analysis.preparation_report import new_split_report, add_batch_report
from src.data_preparation.annotation_flattener import flatten_document, PREPROCESSING_VERSION
from src.data_analysis.processed_data_storage import ProcessedDataStorage
from src.data_preparation.tokenizer_setup import load_tokenizer
from src.data_preparation.bio_encoder import create_label_mapping, encode_batch
from src.data_preparation.document_chunker import CHUNKING_VERSION
from src.data_preparation.prepared_data_storage import PreparedDataStorage
from src.training.trainer import train_from_batches
from src.validation.validator import validate_model
from src.validation.model_registry import ModelRegistry


def _new_version():
    return datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + uuid4().hex[:8]


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def _loader(config):
    data = config['data']
    if data.get('source', {}).get('format') == 'brat':
        return BratDataLoader(data['source']['path'], data['dataset_id'], data['entity_types'],
                              data['source']['commit'], batch_size=data['batch_size'], shuffle=False)
    return StreamingDataLoader(batch_size=config['data']['batch_size'],
        dataset_name=config['data']['dataset_name'], shuffle=False,
        revision=config['data'].get('dataset_revision'))


def calculate_type_statistics(config):
    config = validate_config(config)
    payload = build_type_priority(config, _loader(config))
    path = save_type_priority(payload, config)
    print(f'Приоритет типов сохранён: {path}')
    return str(path)


def prepare_data(config):
    """Использует готовый приоритет. Никогда не пересчитывает частоты."""
    config = validate_config(config)
    catalog = load_correction_catalog(config['data']['dataset_name'], config['data'].get('corrections_path'))
    priority = load_type_priority(config)  # Проверяем до создания каталогов и загрузки корпуса.
    version = _new_version()
    version_dir = Path(config['paths']['processed']) / version
    version_dir.mkdir(parents=True, exist_ok=False)
    _write_json(version_dir / 'type_priority.json', priority)
    _write_json(version_dir / 'annotation_corrections.json', catalog)
    manifest = {'format_version': 2, 'version': version, 'status': 'preparing',
        'preprocessing_version': PREPROCESSING_VERSION, 'config': config, 'splits': {},
        'annotation_corrections': correction_metadata(catalog),
        'source': priority.get('source'),
        'priority_sha256': priority['sha256'], 'report': 'preparation_report.json'}
    _write_json(version_dir / 'manifest.json', manifest)
    reports = {}
    try:
        loader = _loader(config)
        if priority.get('source') and priority['source'].get('schema_sha256') != getattr(loader, 'schema_sha256', None):
            raise ValueError('annotation.conf изменился: создайте новый файл приоритетов')
        for split in ('train', 'dev', 'test'):
            getattr(loader, 'load_' + split)()
            raw = RawDataStorage(str(Path(config['paths']['raw']) / version / split))
            processed = ProcessedDataStorage(str(version_dir / split))
            batch_names, signatures = [], []
            report = new_split_report(config['data']['entity_types'])
            reports[split] = report
            quarantine_path = version_dir / (split + '_damaged.jsonl')
            with quarantine_path.open('w', encoding='utf-8') as quarantine:
                for batch in loader:
                    # Даже тестовые/внешние загрузчики получают однозначную принадлежность split.
                    batch = [{**doc, 'split': split} for doc in batch]
                    if split == 'train':
                        signatures.extend({'doc_id': d['doc_id'], 'sha256': document_signature(d)} for d in batch)
                    raw_path = raw.save_batch(batch, calculate_batch_metadata(batch))
                    checked = checked_batch(batch, config['data']['entity_types'], catalog)
                    accepted, damaged = partition_batch_by_quality(checked)
                    final = [flatten_document(doc, priority['priority']) for doc in accepted]
                    add_batch_report(report, checked, final, damaged)
                    for doc in damaged:
                        quarantine.write(json.dumps(doc, ensure_ascii=False) + '\n')
                    path = processed.save_batch(final, {'source_batch': raw_path.name,
                        'source_path': str(raw_path), 'dataset_version': version, 'split': split,
                        'preprocessing_version': PREPROCESSING_VERSION,
                        'priority_sha256': priority['sha256'],
                        'entity_type_distribution': dict(Counter(
                            e['type'] for d in final for e in d['parsed_entities']))})
                    if path is not None:
                        batch_names.append(path.name)
            manifest['splits'][split] = batch_names
            if split == 'train' and train_signature(signatures) != priority['train_signature']:
                raise ValueError('Состав или содержимое train изменились. Создайте новый файл приоритетов')
        if any(not manifest['splits'][split] for split in ('train', 'dev', 'test')):
            raise ValueError('После проверки один из split пуст; см. отчёт и *_damaged.jsonl')
        manifest['status'] = 'ready'
    except Exception as error:
        manifest['status'] = 'failed'
        manifest['error'] = str(error)
        raise
    finally:
        _write_json(version_dir / 'preparation_report.json', {
            'preprocessing_version': PREPROCESSING_VERSION,
            'priority_sha256': priority['sha256'], 'annotation_corrections': correction_metadata(catalog),
            'status': manifest['status'], 'splits': reports})
        _write_json(version_dir / 'manifest.json', manifest)
    print(f'Подготовлена версия данных: {version}')
    return version


def load_dataset_version(config, version):
    """Читает только явно выбранный завершённый набор, без fallback на latest."""
    if not version or Path(version).name != version or version in ('.', '..'):
        raise ValueError('Укажите ID версии данных без компонентов пути')
    root = Path(config['paths']['processed']) / version
    path = root / 'manifest.json'
    if not path.is_file():
        raise ValueError(f'Версия данных не найдена: {version}')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('status') != 'ready' or manifest.get('version') != version or manifest.get('format_version') != 2:
        raise ValueError('Версия данных не завершена или формат не поддерживается')
    saved = manifest['config']['data']
    for key in ('dataset_id', 'dataset_name', 'dataset_version', 'dataset_revision', 'entity_types'):
        if saved.get(key) != config['data'].get(key):
            raise ValueError(f'Конфигурация не соответствует версии данных: data.{key}')
    if source_descriptor({'data': saved}) != source_descriptor(config):
        raise ValueError('Источник не соответствует версии данных')
    if manifest['config'].get('annotation_policy') != config.get('annotation_policy'):
        raise ValueError('Конфигурация annotation_policy не соответствует версии данных')
    if manifest.get('preprocessing_version') != PREPROCESSING_VERSION:
        raise ValueError('Версия алгоритма подготовки не поддерживается')
    catalog = load_correction_catalog(config['data']['dataset_name'], config['data'].get('corrections_path'))
    saved_catalog = json.loads((root / 'annotation_corrections.json').read_text(encoding='utf-8'))
    if (manifest.get('annotation_corrections') != correction_metadata(catalog)
            or saved_catalog != catalog):
        raise ValueError('Поправки не соответствуют версии данных')
    priority = validate_type_priority(json.loads((root / 'type_priority.json').read_text(encoding='utf-8')), config)
    if manifest.get('source') != priority.get('source'):
        raise ValueError('Источник в манифесте не соответствует приоритетам')
    if manifest.get('priority_sha256') != priority['sha256']:
        raise ValueError('Приоритет не соответствует манифесту данных')
    for split in ('train', 'dev', 'test'):
        names = manifest.get('splits', {}).get(split, [])
        if not names:
            raise ValueError(f'Пустой split: {split}')
        for name in names:
            if Path(name).name != name or name in ('.', '..') or not (root / split / name).is_dir():
                raise ValueError(f'Отсутствует батч split {split}: {name}')
    return root, manifest


def train(config, dataset_version):
    """Токенизирует выбранную общую версию и обучает модель, не загружая корпус."""
    config = validate_config(config)
    if config['model']['architecture'] == 'gliner':
        raise ValueError('GLiNER используется без дообучения; train недоступен')
    root, manifest = load_dataset_version(config, dataset_version)
    device = resolve_device(config['device'])
    set_seed(config['seed'])
    architecture = config['model']['architecture']
    tokenizer_name = config['model']['tokenizer_name']
    tokenizer = load_tokenizer(tokenizer_name, config['data']['max_seq_length'])
    label2id, id2label = create_label_mapping(config['data']['entity_types'])
    info = {'label2id': label2id, 'id2label': id2label, 'num_labels': len(label2id),
            'entity_types': config['data']['entity_types'],
            'tokenizer_name': tokenizer_name, 'architecture': architecture, 'max_length': config['data']['max_seq_length'],
            'chunking': {'version': CHUNKING_VERSION, 'strategy': 'sentences_then_tokens', 'overlap': 0,
                         'protect_gold_entities': True}}
    if architecture == 'bilstm_crf':
        if tokenizer.pad_token_id is None:
            raise ValueError('Токенизатор BiLSTM-CRF должен иметь pad_token_id')
        info.update(vocab_size=len(tokenizer), padding_idx=tokenizer.pad_token_id,
                    special_token_ids=[token for token in (tokenizer.cls_token_id, tokenizer.sep_token_id)
                                       if token is not None])
    # Отдельный каталог производных тензоров на каждый запуск: общая версия неизменна.
    prepared_root = Path(config['paths']['prepared']) / dataset_version / _new_version()
    split_paths = {}
    chunking_report = {}
    for split in ('train', 'dev'):
        source = ProcessedDataStorage(str(root / split))
        destination = PreparedDataStorage(str(prepared_root / split))
        paths = []
        statistics = {'documents': 0, 'chunks': 0, 'split_documents': 0, 'token_boundaries': 0}
        for name in manifest['splits'][split]:
            batch = source.load_batch(root / split / name)
            if not batch:
                raise ValueError(f'Пустой батч: {split}/{name}')
            encoded = encode_batch(batch, tokenizer, label2id, config['data']['max_seq_length'])
            statistics['documents'] += len(batch)
            statistics['chunks'] += len(encoded)
            statistics['split_documents'] += sum(s['chunk_count'] > 1 and s['chunk_index'] == 0 for s in encoded)
            statistics['token_boundaries'] += sum(s['boundary'] == 'token' for s in encoded)
            paths.append(destination.save_batch(encoded,
                {'source_batch': name, 'dataset_version': dataset_version, 'split': split}, info))
        split_paths[split] = paths
        chunking_report[split] = statistics
    _write_json(prepared_root / 'chunking_report.json', {'policy': info['chunking'], 'splits': chunking_report})
    training_storage = PreparedDataStorage(str(prepared_root / 'train'))
    model_path, version, loss_history = train_from_batches(
        prepared_storage=training_storage, batch_paths=split_paths['train'],
        eval_batch_paths=split_paths['dev'], seed=config['seed'], device=device,
        model_name=config['model']['name'], models_dir=config['paths']['models'],
        architecture=architecture, model_parameters=config['model'].get('parameters'),
        **config['training'])
    tokenizer.save_pretrained(model_path)
    _write_json(Path(model_path) / 'model_runtime.json', {
        'architecture': architecture, 'tokenizer_name': tokenizer_name,
        'max_length': config['data']['max_seq_length'], 'entity_types': config['data']['entity_types'],
        'dataset_version': dataset_version})
    metrics = validate_model(Path(model_path), training_storage, split_paths['dev'], device=device)
    training_metadata_path = Path(model_path) / 'training_metadata.json'
    selection = json.loads(training_metadata_path.read_text(encoding='utf-8')).get('selection', {}) if training_metadata_path.is_file() else {}
    metadata = {'architecture': architecture, 'dataset_version': dataset_version, 'preprocessing_version': manifest['preprocessing_version'],
                'config': config, 'source': manifest.get('source'), 'device': device, 'selection': selection, 'prepared_path': str(prepared_root),
                'chunking': {'policy': info['chunking'], 'splits': chunking_report},
                'train_batches': len(split_paths['train']), 'val_batches': len(split_paths['dev'])}
    ModelRegistry(config['paths']['models']).register_model(model_path, metrics, metadata)
    # Реестр уже преобразует numpy-типы; используем тот же конвертер для отчёта.
    from src.validation.model_registry import convert_to_serializable
    report = {'version': version, 'model_path': str(model_path), 'metrics': metrics,
              'loss_history': loss_history, **metadata}
    _write_json(Path(model_path) / 'experiment.json', convert_to_serializable(report))
    return report
