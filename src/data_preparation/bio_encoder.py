"""Строгое преобразование плоской непрерывной разметки в BIO."""
from copy import deepcopy
from numbers import Integral
from .bio_decoder import validate_offsets


def create_label_mapping(entity_types):
    if not entity_types or any(not isinstance(t, str) or not t for t in entity_types):
        raise ValueError('Требуется непустой список типов')
    if len(set(entity_types)) != len(entity_types):
        raise ValueError('Типы не должны повторяться')
    names = ['O'] + [prefix + kind for kind in entity_types for prefix in ('B-', 'I-')]
    return {name: i for i, name in enumerate(names)}, dict(enumerate(names))


def generate_bio_labels(offset_mapping, entities, label2id, ignore_label=-100):
    offsets = validate_offsets(offset_mapping)
    if 'O' not in label2id:
        raise ValueError('В схеме отсутствует O')
    labels = [ignore_label if span == (0, 0) else label2id['O'] for span in offsets]
    ordered = []
    for entity in entities:
        kind = entity.get('type')
        if f'B-{kind}' not in label2id or f'I-{kind}' not in label2id:
            raise ValueError(f'Тип отсутствует в BIO-схеме: {kind}')
        spans = entity.get('spans', [])
        if len(spans) != 1:
            raise ValueError('BIO-кодировщик принимает только одну непрерывную область сущности')
        start, end = spans[0].get('start'), spans[0].get('end')
        if any(not isinstance(v, Integral) or isinstance(v, bool) for v in (start, end)) or not 0 <= start < end:
            raise ValueError('Некорректные границы сущности')
        ordered.append((int(start), int(end), kind))
    ordered.sort()
    if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
        raise ValueError('Обнаружены пересекающиеся сущности: сначала выполните flattening')
    for start, end, kind in ordered:
        indexes = [i for i, (a, b) in enumerate(offsets) if (a, b) != (0, 0) and a < end and start < b]
        if not indexes or offsets[indexes[0]][0] != start or offsets[indexes[-1]][1] != end:
            raise ValueError(f'Границы сущности {kind} [{start}, {end}) не совпадают с токенами')
        if any(not start <= offsets[i][0] < offsets[i][1] <= end for i in indexes):
            raise ValueError('Токен пересекает границу сущности')
        if indexes != list(range(indexes[0], indexes[-1] + 1)):
            raise ValueError('Сущность пересекает служебный токен')
        for position, index in enumerate(indexes):
            labels[index] = label2id[('B-' if position == 0 else 'I-') + kind]
    return labels


def encode_document_chunks(doc, tokenizer, label2id, max_length=512):
    """Один или несколько образцов с эталоном полного исходного документа."""
    import json
    from .document_chunker import chunk_document, CHUNKING_VERSION
    text = doc.get('text', '')
    entities = doc.get('parsed_entities', [])
    try:
        chunks = chunk_document(text, tokenizer, max_length, entities)
    except ValueError as exc:
        raise ValueError(f'Документ {doc.get("doc_id")}: {exc}') from exc
    # Проверяем всю схему и пересечения, не токенизируя документ повторно.
    offsets = [span for chunk in chunks for span in chunk['offset_mapping'] if span != (0, 0)]
    generate_bio_labels(offsets, entities, label2id)
    result = []
    for index, chunk in enumerate(chunks):
        gold = [entity for entity in entities
                if chunk['chunk_start'] <= entity['spans'][0]['start']
                and entity['spans'][0]['end'] <= chunk['chunk_end']]
        result.append({**chunk, 'doc_id': doc.get('doc_id'), 'text': text,
            'split': doc.get('split'),
            'sample_id': json.dumps([doc.get('doc_id'), index], ensure_ascii=False),
            'chunk_index': index, 'chunk_count': len(chunks), 'chunking_version': CHUNKING_VERSION,
            'document_gold_entities': deepcopy(entities), 'gold_entities': deepcopy(gold),
            'labels': generate_bio_labels(chunk['offset_mapping'], gold, label2id)})
    return result


def encode_document(doc, tokenizer, label2id, max_length=512):
    """API одного образца. Для длинных документов используйте encode_document_chunks."""
    chunks = encode_document_chunks(doc, tokenizer, label2id, max_length)
    if len(chunks) != 1:
        raise ValueError('Документ длиннее одного фрагмента; используйте encode_document_chunks')
    return chunks[0]


def encode_batch(batch, tokenizer, label2id, max_length=512):
    return [chunk for doc in batch
            for chunk in encode_document_chunks(doc, tokenizer, label2id, max_length)]
