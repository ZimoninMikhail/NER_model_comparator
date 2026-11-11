"""Точечные поправки NEREL: происхождение, защита и интеграция подготовки."""
from copy import deepcopy

import pytest
import yaml
from transformers import BertTokenizerFast

from src.data_analysis import annotation_corrections as corrections
from src.data_analysis.parser import parse_entity
from src.data_analysis.type_statistics import checked_batch
from src.data_preparation.annotation_flattener import flatten_document
from src.data_preparation.bio_encoder import create_label_mapping, encode_batch


@pytest.fixture
def cases(tmp_path):
    catalog = corrections.load_correction_catalog('iluvvatar/NEREL')
    documents = []
    for index, rule in enumerate(catalog['corrections']):
        after = parse_entity(rule['after'])
        text = ' ' * after['spans'][0]['start'] + after['text'] + '.'
        rule['text_sha256'] = corrections.text_signature(text)
        documents.append({'doc_id': f'train:{index}', 'text': text, 'entities': [rule['before']]})
    path = tmp_path/'corrections.yaml'
    path.write_text(yaml.safe_dump(catalog, allow_unicode=True), encoding='utf-8')
    return catalog, documents, path


@pytest.mark.parametrize('index', [0, 1])
def test_verified_boundary_changes_preserve_originals_and_tokenize(cases, tmp_path, index):
    catalog, documents, _ = cases
    source = documents[index]
    before = deepcopy(source)
    checked = checked_batch([source], ['ORGANIZATION', 'TIME'], catalog)[0]
    assert not checked['quality']['has_errors']
    fixed = checked['parsed_entities'][0]
    assert fixed['text'] == parse_entity(catalog['corrections'][index]['after'])['text']
    assert fixed['spans'][0]['end'] == parse_entity(source['entities'][0])['spans'][0]['end'] + 1
    assert checked['original_annotations'] == source['entities']
    assert checked['original_parsed_entities'][0]['text'] == parse_entity(source['entities'][0])['text']
    assert fixed['source_annotation'] == source['entities'][0]
    flat = flatten_document(checked, ['ORGANIZATION', 'TIME'])
    assert flat['original_parsed_entities'] == checked['original_parsed_entities']
    assert flat['annotation_corrections'][0]['before'] == source['entities'][0]
    assert flat['annotation_corrections'][0]['after'] == catalog['corrections'][index]['after']
    assert source == before
    vocab = tmp_path/'vocab.txt'
    vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nсочинского\nХудожественного\nмузея\nв\n16\nчасов\n25\nминут\n.\n')
    tokenizer = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=False)
    assert encode_batch([flat], tokenizer, create_label_mapping(['ORGANIZATION', 'TIME'])[0])


