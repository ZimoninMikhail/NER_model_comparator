"""Загрузка и проверка конфигурации; относительные пути привязаны к проекту."""
from copy import deepcopy
from pathlib import Path
import math
import json
import re
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / 'config/config.yaml'
DATASET_PROFILES = PROJECT_ROOT / 'config/datasets'
DATASET_KEYS = ('dataset_id', 'dataset_name', 'dataset_version', 'dataset_revision',
                'source', 'entity_types', 'type_priority_path', 'corrections_path')


def _read_yaml(path):
    try:
        with Path(path).open(encoding='utf-8') as stream:
            value = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        raise ValueError(f'Ошибка YAML-конфигурации: {error}') from error
    if not isinstance(value, dict):
        raise ValueError('Конфигурация должна быть YAML-словарём')
    return value


def resolve_dataset_profile(config, dataset=None):
    config = deepcopy(config)
    selected = dataset or config.get('dataset')
    if selected is None:
        return config  # Старые явные конфигурации остаются доступными.
    if selected not in ('nerel', 'nerel-bio'):
        raise ValueError(f'Датасет {selected} пока не поддерживается; доступны nerel и nerel-bio')
    profile = _read_yaml(DATASET_PROFILES / (selected + '.yaml'))['data']
    overrides = deepcopy(config.get('data', {}))
    if dataset is not None and selected != config.get('dataset'):
        for key in DATASET_KEYS:
            overrides.pop(key, None)
    source = {**profile.get('source', {}), **overrides.get('source', {})}
    config['data'] = {**profile, **overrides, 'source': source}
    config['dataset'] = selected
    return config


def config_from_dataset_manifest(config, version):
    """CLI train/evaluate без --dataset получает схему из выбранного обработанного набора.

    Источник при обучении не открывается; исходники могут быть перенесены.
    """
    if not version or Path(version).name != version or version in ('.', '..'):
        raise ValueError('Укажите ID версии данных без компонентов пути')
    manifest_path = Path(config['paths']['processed']) / version / 'manifest.json'
    if not manifest_path.is_file():
        raise ValueError(f'Версия данных не найдена: {version}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    result = deepcopy(config)
    saved = manifest['config']['data']
    for key in DATASET_KEYS:
        result['data'].pop(key, None)
        if key in saved:
            result['data'][key] = deepcopy(saved[key])
    result['dataset'] = manifest['config'].get('dataset')
    return validate_config(result)



def validate_config(config):
    config = deepcopy(config)
    config.setdefault('device', 'auto')
    if config['device'] not in ('auto', 'cpu', 'cuda'):
        raise ValueError('device: допустимы auto, cpu, cuda')
    for section in ('model', 'data', 'training', 'paths'):
        if not isinstance(config.get(section), dict):
            raise ValueError(f'Не задан раздел конфигурации: {section}')
    for section, key in [('model', 'name'), ('data', 'dataset_name')]:
        if not isinstance(config[section].get(key), str) or not config[section][key].strip():
            raise ValueError(f'{section}.{key}: требуется непустая строка')
    architecture = config['model'].setdefault('architecture', 'bert')
    if architecture not in ('bert', 'bilstm_crf', 'gliner'):
        raise ValueError('model.architecture: допустимы bert, bilstm_crf и gliner')
    config['model'].setdefault('tokenizer_name', config['model']['name'] if architecture == 'bert' else None)
    if architecture != 'gliner' and (not isinstance(config['model']['tokenizer_name'], str) or not config['model']['tokenizer_name'].strip()):
        raise ValueError('model.tokenizer_name: требуется имя или путь токенизатора')
    if architecture == 'bilstm_crf':
        parameters = config['model'].setdefault('parameters', {})
        if not isinstance(parameters, dict):
            raise ValueError('model.parameters должен быть словарём')
        if set(parameters) - {'embedding_dim', 'hidden_size', 'num_layers', 'dropout'}:
            raise ValueError('Неизвестный параметр BiLSTM-CRF')
        for key, default in [('embedding_dim',128),('hidden_size',128),('num_layers',1)]:
            parameters.setdefault(key, default)
            if type(parameters[key]) is not int or parameters[key] <= 0:
                raise ValueError(f'model.parameters.{key}: требуется положительное целое число')
        parameters.setdefault('dropout', 0.2)
        value = parameters['dropout']
        if isinstance(value, bool) or not isinstance(value, (float,int)) or not 0 <= value < 1:
            raise ValueError('model.parameters.dropout должен лежать в [0,1)')
    types = config['data'].get('entity_types')
    if not isinstance(types, list) or not types or any(not isinstance(t, str) or not t for t in types):
        raise ValueError('data.entity_types: требуется непустой список типов')
    if len(set(types)) != len(types):
        raise ValueError('data.entity_types: повторяющиеся типы')
    if architecture == 'gliner':
        model = config['model']
        path = Path(model['name']).expanduser()
        model['name'] = str((PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve())
        threshold = model.setdefault('threshold', 0.5)
        if (isinstance(threshold, bool) or not isinstance(threshold, (float,int))
                or not math.isfinite(threshold) or not 0 <= threshold <= 1):
            raise ValueError('model.threshold: требуется число от 0 до 1')
        mapping = model.get('type_mapping')
        if (not isinstance(mapping, dict) or set(mapping) != set(types)
                or any(not isinstance(label,str) or not label.strip() or label != label.strip() for label in mapping.values())
                or len(set(mapping.values())) != len(mapping)):
            raise ValueError('model.type_mapping: нужны уникальные названия GLiNER для всех типов выбранного датасета')
        # Порядок prompt закреплён схемой датасета, а не порядком YAML-ключей.
        model['type_mapping'] = {kind:mapping[kind] for kind in types}
    for section, key in [('data', 'batch_size'), ('data', 'max_seq_length'),
                         ('training', 'batch_size'), ('training', 'num_epochs'),
                         ('training', 'logging_steps')]:
        value = config[section].get(key)
        if type(value) is not int or value <= 0:
            raise ValueError(f'{section}.{key}: требуется положительное целое число')
    for key in ('warmup_steps',):
        value = config['training'].get(key)
        if type(value) is not int or value < 0:
            raise ValueError(f'training.{key}: требуется неотрицательное целое число')
    if type(config.get('seed')) is not int or config['seed'] < 0:
        raise ValueError('seed: требуется неотрицательное целое число')
    for key in ('learning_rate', 'weight_decay'):
        value = config['training'].get(key)
        try:
            if isinstance(value, bool):
                raise ValueError()
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f'training.{key}: требуется конечное число') from None
        if not math.isfinite(value):
            raise ValueError(f'training.{key}: требуется конечное число')
        config['training'][key] = value
        if value < 0 or (key == 'learning_rate' and value == 0):
            raise ValueError(f'training.{key}: недопустимое значение')
    revision = config['data'].get('dataset_revision')
    if revision is not None and (not isinstance(revision, str) or not revision.strip()):
        raise ValueError('data.dataset_revision: требуется непустая строка или null')
    priority_path = config['data'].get('type_priority_path')
    if not isinstance(priority_path, str) or not priority_path:
        raise ValueError('data.type_priority_path: требуется путь')
    path = Path(priority_path).expanduser()
    config['data']['type_priority_path'] = str((PROJECT_ROOT / path).resolve()
        if not path.is_absolute() else path.resolve())
    source = config['data'].get('source')
    if source is not None:
        if not isinstance(source, dict) or source.get('format') != 'brat':
            raise ValueError('data.source.format: поддерживается brat')
        if config['data'].get('dataset_id') not in ('nerel', 'nerel-bio'):
            raise ValueError('data.dataset_id: неизвестный локальный датасет')
        if config.get('dataset') not in (None, config['data']['dataset_id']):
            raise ValueError('Выбранный профиль и data.dataset_id не совпадают')
        for key in ('path', 'repository', 'commit', 'subdirectory'):
            if not isinstance(source.get(key), str) or not source[key].strip():
                raise ValueError(f'data.source.{key}: требуется непустая строка')
        if not re.fullmatch(r'[0-9a-f]{40}', source['commit']):
            raise ValueError('data.source.commit: требуется полный SHA commit')
        if config['data'].get('dataset_revision') != source['commit']:
            raise ValueError('dataset_revision не соответствует source.commit')
        if not isinstance(config['data'].get('dataset_version'), str):
            raise ValueError('data.dataset_version: требуется строка')
        path = Path(source['path']).expanduser()
        source['path'] = str((PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve())
    corrections_path = config['data'].get('corrections_path')
    if corrections_path is not None:
        if not isinstance(corrections_path, str) or not corrections_path:
            raise ValueError('data.corrections_path: требуется путь или null')
        path = Path(corrections_path).expanduser()
        config['data']['corrections_path'] = str((PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve())
    expected_policy = {'overlap': 'longest_first',
        'same_span_priority': 'train_frequency_ascending',
        'final_tie': 'alphabetical', 'discontinuous': 'exclude'}
    if config.get('annotation_policy') != expected_policy:
        raise ValueError('annotation_policy: поддерживаются только согласованные правила')
    if config['data']['max_seq_length'] < 3:
        raise ValueError('data.max_seq_length: слишком малая длина')
    if config.get('evaluation') != {'primary_metric': 'macro_f1', 'match': 'exact_span_and_type'}:
        raise ValueError('evaluation: поддерживается macro_f1 с точными границами и типом')
    config['training'].pop('eval_steps', None)
    config['training'].pop('save_steps', None)
    config['model']['num_labels'] = 1 + 2 * len(types)
    config['paths'].setdefault('reports', 'reports')
    config['paths'].setdefault('predictions', 'data/predictions')
    config['paths'].setdefault('visualization', str(Path(config['paths']['reports']) / 'visualization'))
    benchmark = config.setdefault('benchmark', {})
    if not isinstance(benchmark, dict):
        raise ValueError('benchmark должен быть словарём')
    benchmark.setdefault('warmup_runs', 2)
    benchmark.setdefault('repeats', 5)
    for key, minimum in [('warmup_runs', 0), ('repeats', 1)]:
        if type(benchmark[key]) is not int or benchmark[key] < minimum:
            raise ValueError(f'benchmark.{key}: требуется целое число >= {minimum}')
    for key in ('raw', 'processed', 'prepared', 'models', 'reports', 'predictions', 'visualization'):

        value = config['paths'].get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f'paths.{key}: требуется путь')
        path = Path(value).expanduser()
        config['paths'][key] = str((PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve())
    return config


def load_config(path=None, dataset=None):
    return validate_config(resolve_dataset_profile(_read_yaml(path or DEFAULT_CONFIG), dataset))
