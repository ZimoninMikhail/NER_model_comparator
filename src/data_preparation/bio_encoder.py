"""
Модуль для генерации BIO-меток на основе токенов и сущностей.
"""

from typing import List, Dict, Any, Tuple


def create_label_mapping(entity_types: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Создаёт маппинг BIO-меток → ID и обратно.
    
    Args:
        entity_types: Список типов сущностей (COUNTRY, CITY, ...)
        
    Returns:
        label2id: {метка: ID}
        id2label: {ID: метка}
    """
    label2id = {'O': 0}
    id2label = {0: 'O'}
    
    idx = 1
    for entity_type in entity_types:
        label2id[f'B-{entity_type}'] = idx
        id2label[idx] = f'B-{entity_type}'
        idx += 1
        label2id[f'I-{entity_type}'] = idx
        id2label[idx] = f'I-{entity_type}'
        idx += 1
    
    return label2id, id2label


def generate_bio_labels(
    offset_mapping: List[Tuple[int, int]],
    entities: List[Dict[str, Any]],
    label2id: Dict[str, int],
    ignore_label: int = -100
) -> List[int]:
    """
    Генерирует BIO-метки для токенов документа.
    
    Args:
        offset_mapping: Список (start, end) для каждого токена
        entities: Список сущностей (каждая с полем spans и type)
        label2id: Маппинг метки → ID
        ignore_label: ID для игнорируемых токенов ([CLS], [SEP], ...)
        
    Returns:
        Список ID меток для каждого токена
    """
    num_tokens = len(offset_mapping)
    labels = [ignore_label] * num_tokens
    
    for token_idx, (start, end) in enumerate(offset_mapping):
        if start == 0 and end == 0:
            continue
        
        best_label = 'O'
        for entity in entities:
            entity_type = entity.get('type')
            
            for span in entity.get('spans', []):
                entity_start = span['start']
                entity_end = span['end']
                
                if start >= entity_start and end <= entity_end:
                    if start == entity_start:
                        best_label = f'B-{entity_type}'
                    else:
                        best_label = f'I-{entity_type}'
                    break
            
            if best_label != 'O':
                break
        
        labels[token_idx] = label2id.get(best_label, ignore_label)
    
    return labels


def encode_document(
    doc: Dict[str, Any],
    tokenizer,
    label2id: Dict[str, int],
    max_length: int = 512
) -> Dict[str, Any]:
    """
    Преобразует один документ в BIO-формат.
    
    Args:
        doc: Документ с полями text, parsed_entities
        tokenizer: Токенизатор BERT
        label2id: Маппинг меток
        max_length: Максимальная длина
        
    Returns:
        Словарь с input_ids, attention_mask, labels
    """
    text = doc.get('text', '')
    entities = doc.get('parsed_entities', [])
    
    encoding = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_offsets_mapping=True,
        return_tensors=None
    )
    
    labels = generate_bio_labels(
        encoding['offset_mapping'],
        entities,
        label2id
    )
    
    return {
        'doc_id': doc.get('doc_id'),
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'labels': labels
    }


def encode_batch(
    batch: List[Dict[str, Any]],
    tokenizer,
    label2id: Dict[str, int],
    max_length: int = 512
) -> List[Dict[str, Any]]:
    """
    Преобразует батч документов в BIO-формат.
    
    Args:
        batch: Список документов с полями text, parsed_entities
        tokenizer: Токенизатор BERT
        label2id: Маппинг меток
        max_length: Максимальная длина
        
    Returns:
        Список документов с полями input_ids, attention_mask, labels
    """
    encoded_batch = []
    
    for doc in batch:
        encoded = encode_document(doc, tokenizer, label2id, max_length)
        encoded_batch.append(encoded)
    
    return encoded_batch