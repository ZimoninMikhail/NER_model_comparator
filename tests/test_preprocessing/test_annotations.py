from copy import deepcopy
from itertools import permutations
import unittest

from src.data_analysis.parser import parse_document_entities
from src.data_analysis.quality_checker import check_document_quality
from src.data_analysis.quality_filter import partition_batch_by_quality
from src.data_preparation.annotation_flattener import flatten_document


def entity(identifier, kind, start, end, text):
    return {'id': identifier, 'type': kind, 'spans': [{'start': start, 'end': end}],
            'text': text, 'is_multiple': False}


class AnnotationTests(unittest.TestCase):
    def test_parser_preserves_source_and_records_identifiers(self):
        doc = {'doc_id': 'train:1', 'entities': ['T1\tCITY 0 6\tМосква', 'T2\tCITY nope 4\tx']}
        parsed = parse_document_entities(doc)
        self.assertEqual(parsed['original_annotations'], doc['entities'])
        self.assertEqual(parsed['parse_errors'][0]['entity_id'], 'T2')
        self.assertEqual(parsed['parse_errors'][0]['doc_id'], 'train:1')
        self.assertEqual(parsed['parse_errors'][0]['source_annotation'], doc['entities'][1])
        self.assertEqual(len(parsed['parsed_entities']), 1)


    def test_unknown_type_is_quarantined_and_empty_document_kept(self):
        bad = check_document_quality(parse_document_entities({'doc_id': 'bad', 'text': 'x',
            'entities': ['T1\tUNKNOWN 0 1\tx']}), {'CITY'})
        empty = check_document_quality(parse_document_entities({'doc_id': 'empty', 'text': 'нет', 'entities': []}), {'CITY'})
        good, damaged = partition_batch_by_quality([bad, empty])
        self.assertEqual([d['doc_id'] for d in good], ['empty'])
        self.assertEqual([d['doc_id'] for d in damaged], ['bad'])

    def test_rare_type_wins_same_span_and_result_is_order_independent(self):
        items = [entity('T1','CITY',0,6,'Москва'), entity('T2','LOCATION',0,6,'Москва'),
                 entity('T3','CITY',1,5,'оскв')]
        original = deepcopy(items)
        for order in permutations(items):
            flat = flatten_document({'parsed_entities': list(order)}, ['LOCATION', 'CITY'])
            self.assertEqual([e['id'] for e in flat['parsed_entities']], ['T2'])
            self.assertEqual(sorted(e['reason'] for e in flat['excluded_annotations']), ['overlap','type_priority'])
        self.assertEqual(items, original)

    def test_longer_wins_even_if_type_is_common(self):
        flat = flatten_document({'parsed_entities': [entity('a','CITY',1,3,'xx'),
                entity('b','LOCATION',0,5,'xxxxx')]}, ['CITY','LOCATION'])
        self.assertEqual(flat['parsed_entities'][0]['id'], 'b')


