"""Точечные проверенные поправки. Raw и оригинальные аннотации неизменны."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import yaml
from .parser import parse_entity

CATALOG_PATH = Path(__file__).resolve().parents[2] / 'config/nerel_annotation_corrections.yaml'


def text_signature(text):
    # HF заменил один NBSP обычным пробелом; длина и координаты сохраняются.
    return hashlib.sha256(text.replace('\u00a0', ' ').encode('utf-8')).hexdigest()


def load_correction_catalog(dataset_name, path=None):
    if dataset_name not in ('iluvvatar/NEREL', 'NEREL') and path is None:
        return {'version': 0, 'dataset_name': dataset_name, 'corrections': []}
    with (Path(path) if path is not None else CATALOG_PATH).open(encoding='utf-8') as stream:
        catalog = yaml.safe_load(stream)
    if (not isinstance(catalog, dict) or catalog.get('dataset_name') not in (dataset_name, 'iluvvatar/NEREL' if dataset_name == 'NEREL' else dataset_name)
            or catalog.get('version') != 1 or catalog.get('text_hash_rule') != 'sha256_utf8_nbsp_to_space'
            or not isinstance(catalog.get('corrections'), list)):
        raise ValueError('Некорректный каталог поправок NEREL')
    ids = set()
    for rule in catalog['corrections']:
        if rule['id'] in ids:
            raise ValueError('Повторный ID поправки')
        ids.add(rule['id'])
        before, after = parse_entity(rule['before']), parse_entity(rule['after'])
        if before['id'] != after['id'] or before['id'] != rule['entity_id']:
            raise ValueError('Поправка не должна менять ID сущности')
    return catalog


def correction_metadata(catalog):
    digest = hashlib.sha256(json.dumps(catalog, ensure_ascii=False, sort_keys=True,
                                      separators=(',', ':')).encode('utf-8')).hexdigest()
    return {'version': catalog['version'], 'sha256': digest}


def apply_annotation_corrections(doc, catalog):
    """После парсинга, перед проверкой. Совпадение по полному тексту и аннотации.

    Внутренний doc_id зависит от порядка загрузки, поэтому не используется
    для поиска документа. Неожиданное содержимое целевой аннотации — ошибка.
    """
    result = deepcopy(doc)
    result.setdefault('annotation_corrections', [])
    signature = text_signature(doc.get('text', ''))
    for rule in catalog['corrections']:
        if signature != rule['text_sha256']:
            continue
        matches = [e for e in result.get('parsed_entities', []) if e['id'] == rule['entity_id']]
        if len(matches) != 1:
            raise ValueError(f'Поправка {rule["id"]}: не найдена единственная целевая аннотация')
        current = matches[0]
        before, after = parse_entity(rule['before']), parse_entity(rule['after'])
        fields = ('id', 'type', 'spans', 'text', 'is_multiple')
        if all(current.get(k) == after[k] for k in fields):
            continue  # Уже исправленный исходник или повторное применение.
        if any(current.get(k) != before[k] for k in fields):
            raise ValueError(f'Поправка {rule["id"]}: исходная аннотация изменилась; нужна проверка')
        text = result['text']
        for span in after['spans']:
            if not 0 <= span['start'] < span['end'] <= len(text):
                raise ValueError(f'Поправка {rule["id"]}: неверные новые границы')
        actual = ' '.join(text[s['start']:s['end']] for s in after['spans'])
        if actual != after['text']:
            raise ValueError(f'Поправка {rule["id"]}: новый текст не соответствует документу')
        for field in fields:
            current[field] = deepcopy(after[field])
        # source_annotation, original_annotations и original_parsed_entities остаются исходными.
        result['annotation_corrections'].append({'correction_id': rule['id'],
            'doc_id': doc.get('doc_id'), 'entity_id': rule['entity_id'],
            'before': rule['before'], 'after': rule['after'],
            'reason': rule['reason'], 'source_url': rule['source_url']})
    return result
