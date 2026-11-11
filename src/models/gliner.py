"""Предобученная span-модель GLiNER: загрузка и общий формат сущностей.

Модуль не обучает модель и не преобразует предсказания в BIO.
Разбиение документов выполняется уровнем выше; здесь запрещено обрезание.
"""
from collections.abc import Mapping
from math import isfinite
from numbers import Integral, Real
from pathlib import Path

import torch
from gliner import GLiNER
from src.device import resolve_device

DEFAULT_MODEL = 'urchade/gliner_multi-v2.1'


def load_model(model_name_or_path=DEFAULT_MODEL, device='auto', *,
               local_files_only=True, revision=None, cache_dir=None):
    """Загрузить готовую GLiNER, по умолчанию без скачивания.

    Для первого скачивания явно передайте local_files_only=False. revision
    позволяет зафиксировать commit Hub. Полный комплект save_pretrained
    содержит токенизатор и encoder_config и читается автономно.
    """
    device = resolve_device(device)
    if not isinstance(model_name_or_path, (str, Path)) or not str(model_name_or_path).strip():
        raise ValueError('Укажите имя или локальную папку GLiNER')
    path = Path(model_name_or_path).expanduser()
    is_path = isinstance(model_name_or_path, Path) or str(model_name_or_path).startswith(('/', '.', '~'))
    if is_path and not path.is_dir():
        raise ValueError(f'Локальная папка GLiNER не найдена: {path}')
    if path.is_dir():
        # Старый официальный snapshot без encoder_config/tokenizer не является
        # автономным комплектом, даже если веса уже скачаны.
        if local_files_only:
            import json
            config_path = path / 'gliner_config.json'
            if not config_path.is_file():
                raise ValueError(f'Нет gliner_config.json: {path}')
            config = json.loads(config_path.read_text(encoding='utf-8'))
            if not config.get('encoder_config') or not (path / 'tokenizer_config.json').is_file():
                raise ValueError('Неполный локальный комплект GLiNER: нужны encoder_config и '
                                 'токенизатор. Сохраните загруженную модель через save_pretrained')
        model_name_or_path = str(path)
    elif local_files_only:
        # Флаг сторонней библиотеки не запрещает сетевой fallback её энкодера.
        raise ValueError('Для автономной GLiNER укажите полный локальный комплект; '
                         'для первого скачивания разрешите local_files_only=False')
    model = GLiNER.from_pretrained(
        str(model_name_or_path), revision=revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        local_files_only=local_files_only, map_location='cpu', strict=True)
    if model.config.span_mode == 'token_level' or model.config.labels_encoder:
        raise ValueError('Этот адаптер поддерживает span-based GLiNER с одним энкодером')
    model.to(device)
    model.eval()
    return model


def _validate_type_mapping(type_mapping):
    if not isinstance(type_mapping, Mapping) or not type_mapping:
        raise ValueError('Нужен непустой словарь типов: тип датасета → название GLiNER')
    for kind, label in type_mapping.items():
        if any(not isinstance(s, str) or not s.strip() or s != s.strip() for s in (kind, label)):
            raise ValueError('Типы и названия GLiNER должны быть непустыми строками без крайних пробелов')
    if len(set(type_mapping.values())) != len(type_mapping):
        raise ValueError('Названия GLiNER должны однозначно соответствовать типам датасета')


def _input_dimensions(model, text, labels, max_length=None):
    """Проверить оба ограничения GLiNER: слова и subtokens вместе с типами."""
    processor = model.data_processor
    tokens = [token for token, _, _ in processor.words_splitter(text)]
    inputs, _ = processor.prepare_inputs([tokens], [labels])
    if processor.preprocess_text:
        inputs = processor.prepare_texts(inputs)
    tokenizer = processor.transformer_tokenizer
    encoded = tokenizer(inputs, is_split_into_words=True, truncation=False, padding=False)
    limits = [tokenizer.model_max_length]
    encoder_config = model.config.encoder_config
    positions = getattr(encoder_config, 'max_position_embeddings', None)
    if isinstance(positions, int) and positions > 0:
        limits.append(positions)
    if max_length is not None:
        limits.append(max_length)
    limit = min(limits)
    length = len(encoded['input_ids'][0])
    return len(tokens), model.config.max_len, length, limit


def _check_input_length(model, text, labels):
    words, max_words, length, limit = _input_dimensions(model, text, labels)
    if words > max_words:
        raise ValueError(f'Текст длиннее лимита GLiNER: {words} > {max_words} слов; передайте фрагменты')
    if length > limit:
        raise ValueError(f'Текст и названия типов превышают лимит энкодера GLiNER: '
                         f'{length} > {limit} токенов; передайте фрагменты')


def predict_entities(model, text, type_mapping, threshold=0.5):
    """Вернуть плоские сущности в координатах исходного текста [start, end).

    type_mapping: например {'CITY': 'city', 'COUNTRY': 'country'}.
    Порог фиксируется конфигурацией эксперимента, не выбирается по test.
    """
    if not isinstance(text, str):
        raise ValueError('Текст должен быть строкой')
    _validate_type_mapping(type_mapping)
    if (isinstance(threshold, bool) or not isinstance(threshold, Real)
            or not isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError('Порог GLiNER должен быть числом от 0 до 1')
    if not text.strip():
        return []
    labels = list(type_mapping.values())
    _check_input_length(model, text, labels)
    model.eval()
    with torch.no_grad():
        predictions = model.predict_entities(text, labels, threshold=float(threshold),
                                             flat_ner=True, multi_label=False)
    inverse = {label: kind for kind, label in type_mapping.items()}
    entities = {}
    for item in predictions:
        start, end, label, score = (item.get(k) for k in ('start', 'end', 'label', 'score'))
        if label not in inverse:
            raise ValueError(f'GLiNER вернул неизвестный тип: {label}')
        if (any(isinstance(v, bool) or not isinstance(v, Integral) for v in (start, end))
                or not 0 <= start < end <= len(text)):
            raise ValueError('GLiNER вернул некорректные координаты сущности')
        if item.get('text') != text[start:end]:
            raise ValueError('Текст сущности GLiNER не совпадает с исходными координатами')
        if (isinstance(score, bool) or not isinstance(score, Real)
                or not isfinite(score) or not 0 <= score <= 1):
            raise ValueError('GLiNER вернул некорректную оценку уверенности')
        key = (int(start), int(end), inverse[label])
        entity = {'start': int(start), 'end': int(end), 'type': inverse[label],
                  'text': text[start:end], 'score': float(score)}
        if key not in entities or entity['score'] > entities[key]['score']:
            entities[key] = entity
    return [entities[key] for key in sorted(entities)]
