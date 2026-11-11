"""Общие правила BIO для инференса и валидации; без моделей и файлов."""
from numbers import Integral


def validate_offsets(offset_mapping, text=None):
    offsets, previous_end = [], 0
    for pair in offset_mapping:
        if len(pair) != 2 or any(not isinstance(x, Integral) or isinstance(x, bool) for x in pair):
            raise ValueError('Координаты токенов должны быть парами целых чисел')
        start, end = map(int, pair)
        if (start, end) != (0, 0):
            if start < previous_end or start < 0 or end <= start or (text is not None and end > len(text)):
                raise ValueError(f'Некорректные или пересекающиеся координаты токена: {(start, end)}')
            previous_end = end
        offsets.append((start, end))
    return offsets


def normalize_bio_labels(labels, id2label=None, ignore_label=-100):
    """Одиночный I и смена типа трактуются как B. Неизвестные метки — ошибка."""
    result, current = [], None
    mapping = {int(k): v for k, v in id2label.items()} if id2label is not None else None
    for value in labels:
        if isinstance(value, Integral) and int(value) == ignore_label:
            result.append(None)
            current = None
            continue
        if isinstance(value, Integral):
            if mapping is None or int(value) not in mapping:
                raise ValueError(f'Неизвестный ID BIO-метки: {value}')
            label = mapping[int(value)]
        else:
            label = value
        if label == 'O':
            result.append('O')
            current = None
        elif isinstance(label, str) and len(label) > 2 and label[:2] in ('B-', 'I-'):
            kind = label[2:]
            prefix = 'B-' if label.startswith('B-') or current != kind else 'I-'
            result.append(prefix + kind)
            current = kind
        else:
            raise ValueError(f'Некорректная BIO-метка: {label!r}')
    return result


def decode_bio(labels, offset_mapping, text, id2label=None, return_spans=True, ignore_label=-100):
    """Восстановить сущности. B всегда начинает новую; служебные токены разрывают цепочку."""
    offsets = validate_offsets(offset_mapping, text)
    if len(labels) != len(offsets):
        raise ValueError('Количество меток и координат токенов не совпадает')
    # Предсказания на специальных токенах не должны влиять на соседние I-метки.
    effective = [ignore_label if span == (0, 0) else label for label, span in zip(labels, offsets)]
    normalized = normalize_bio_labels(effective, id2label, ignore_label)
    entities, current = [], None
    for label, (start, end) in zip(normalized, offsets):
        if label is None or label == 'O':
            if current is not None:
                entities.append(current)
                current = None
            continue
        kind = label[2:]
        if label.startswith('B-') or current is None or current['type'] != kind:
            if current is not None:
                entities.append(current)
            current = {'type': kind, 'start': start, 'end': end, 'text': text[start:end]}
        else:
            current['end'] = end
            current['text'] = text[current['start']:end]
    if current is not None:
        entities.append(current)
    if not return_spans:
        entities = [{k: v for k, v in entity.items() if k not in ('start', 'end')} for entity in entities]
    return entities
