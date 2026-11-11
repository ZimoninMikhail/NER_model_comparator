"""Неперекрывающиеся фрагменты: предложения, затем границы токенов.

Токенизация документа выполняется один раз. Координаты всегда относятся
к исходному тексту; исходная строка не нормализуется.
"""
import re
from bisect import bisect_right
from numbers import Integral
from .bio_decoder import validate_offsets

CHUNKING_VERSION = 1
# Консервативный разделитель: конец фразы перед прописной буквой/цифрой
# или перенос строки. Это эвристика, не лингвистический анализатор.
_SENTENCE_END = re.compile(r'[.!?…]+[»”"\)\]]*(?=\s+[А-ЯЁA-Z0-9«“"(])|\n+')
_ABBREVIATION = re.compile(r'(?:\b[А-ЯЁA-Z]|\b(?:г|ул|д|им|рис|стр|т|п|др|пр|см))\.$', re.I)


def sentence_ends(text):
    ends = []
    for match in _SENTENCE_END.finditer(text):
        if match.group().startswith('.') and _ABBREVIATION.search(text[:match.end()]):
            continue
        ends.append(match.end())
    return ends


def chunk_document(text, tokenizer, max_length=512, entities=None):
    """Вернуть модельные входы и глобальные offsets без потерь токенов.

    entities задаётся только при подготовке размеченных данных. Для инференса
    границы определяются исключительно текстом. Длинная сущность — ошибка.
    """
    if not isinstance(max_length, Integral) or isinstance(max_length, bool) or max_length < 3:
        raise ValueError('max_length должен быть целым числом не меньше 3')
    special_count = tokenizer.num_special_tokens_to_add(pair=False)
    capacity = max_length - special_count
    if capacity < 1:
        raise ValueError('Лимит не вмещает текст и служебные токены')
    encoding = tokenizer(text, add_special_tokens=False, truncation=False,
                         padding=False, return_offsets_mapping=True, return_tensors=None)
    ids = list(encoding['input_ids'])
    offsets = validate_offsets(encoding['offset_mapping'], text)
    if len(ids) != len(offsets) or any(span == (0, 0) for span in offsets):
        raise ValueError('Некорректная токенизация без служебных токенов')
    protected = set()
    for entity in entities or []:
        spans = entity.get('spans', [])
        if len(spans) != 1:
            raise ValueError('Для разбиения нужна плоская непрерывная разметка')
        start, end = spans[0]['start'], spans[0]['end']
        indexes = [i for i, (a, b) in enumerate(offsets) if a < end and start < b]
        if not indexes or offsets[indexes[0]][0] != start or offsets[indexes[-1]][1] != end:
            raise ValueError(f'Границы сущности {entity.get("type")} [{start}, {end}) не совпадают с токенами')
        if len(indexes) > capacity:
            raise ValueError(f'Сущность {entity.get("type")} [{start}, {end}) длиннее вместимости '
                             f'фрагмента: {len(indexes)} > {capacity} токенов')
        protected.update(range(indexes[0] + 1, indexes[-1] + 1))
    token_ends = [b for a, b in offsets]
    preferred = {bisect_right(token_ends, end) for end in sentence_ends(text)}
    preferred.add(len(ids))
    chunks, position = [], 0
    while position < len(ids) or not chunks:
        limit = min(position + capacity, len(ids))
        end = _select_chunk_end(position, limit, preferred, protected)
        content = ids[position:end]
        model_ids = tokenizer.build_inputs_with_special_tokens(content)
        # Шаблон отличает добавленные токены от [UNK]/[MASK] в самом тексте.
        template = tokenizer.build_inputs_with_special_tokens([-1] * len(content))
        mask = [int(token != -1) for token in template]
        if len(model_ids) != len(mask) or len(model_ids) > max_length or mask.count(0) != len(content):
            raise ValueError('Несовместимая схема служебных токенов токенизатора')
        real_offsets = iter(offsets[position:end])
        chunk_offsets = [(0, 0) if special else next(real_offsets) for special in mask]
        chunks.append({'input_ids': model_ids, 'attention_mask': [1] * len(model_ids),
            'offset_mapping': chunk_offsets, 'token_start': position, 'token_end': end,
            'chunk_start': offsets[position][0] if ids else 0,
            'chunk_end': offsets[end - 1][1] if ids else 0,
            'boundary': 'document_end' if end == len(ids) else
                        ('sentence' if end in preferred else 'token')})
        position = end
    return chunks


def _select_chunk_end(position, limit, preferred, protected=(), fits=None):
    """Общий выбор границы: предложение, иначе последняя допустимая позиция."""
    def allowed(end):
        return end not in protected and (fits is None or fits(end))
    for end in sorted((end for end in preferred if position < end <= limit), reverse=True):
        if allowed(end):
            return end
    end = limit
    while end > position and not allowed(end):
        end -= 1
    if end <= position and limit > position:
        raise ValueError('Не удалось выбрать границу фрагмента: единица текста не помещается в лимит')
    return end


def chunk_spans(text, offsets, capacity, fits=None):
    """Тот же выбор границ для моделей со своей единицей текста (GLiNER — слова).

    fits(start, end) проверяет полный вход, включая названия типов.
    Возвращает неперекрывающиеся интервалы исходного текста.
    """
    if type(capacity) is not int or capacity < 1:
        raise ValueError('Вместимость фрагмента должна быть положительным целым числом')
    offsets = validate_offsets(offsets, text)
    if any(a == b or (i and a < offsets[i-1][1]) for i, (a,b) in enumerate(offsets)):
        raise ValueError('Для разбиения нужны непустые неперекрывающиеся единицы текста')
    if not offsets:
        return []
    ends = [b for a,b in offsets]
    preferred = {bisect_right(ends, end) for end in sentence_ends(text)}
    preferred.add(len(offsets))
    result, position = [], 0
    while position < len(offsets):
        start = offsets[position][0]
        check = None if fits is None else lambda end: fits(start, offsets[end-1][1])
        end = _select_chunk_end(position, min(position+capacity,len(offsets)), preferred, fits=check)
        result.append({'chunk_start':start, 'chunk_end':offsets[end-1][1]})
        position = end
    return result
