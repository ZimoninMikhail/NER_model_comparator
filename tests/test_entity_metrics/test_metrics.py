import unittest
import json
from src.validation.metrics import compute_ner_metrics


def e(kind='CITY', start=0, end=6):
    return {'type':kind,'start':start,'end':end}


def doc(gold,pred,identifier='x'):
    return {'doc_id':identifier,'true_entities':gold,'pred_entities':pred}


class EntityMetricsTests(unittest.TestCase):
    def test_perfect_match(self):
        metrics=compute_ner_metrics([doc([e()],[e()])],['CITY'])
        self.assertEqual(metrics['macro_f1'],1)
        self.assertEqual(metrics['counts'],{'tp':1,'fp':0,'fn':0})
        self.assertNotIn('f1',metrics)
        json.dumps(metrics)

    def test_partial_boundaries_do_not_count(self):
        metrics=compute_ner_metrics([doc([e()],[e(end=5)])],['CITY'])
        self.assertEqual(metrics['counts'],{'tp':0,'fp':1,'fn':1})


    def test_rare_type_has_equal_weight(self):
        gold=[e(start=i*10,end=i*10+6) for i in range(9)] + [e('COUNTRY',100,106)]
        metrics=compute_ner_metrics([doc(gold,gold[:-1])],['CITY','COUNTRY'])
        self.assertEqual(metrics['macro_f1'],0.5)
        self.assertAlmostEqual(metrics['micro_f1'],18/19)
        self.assertEqual(metrics['per_type']['COUNTRY']['support'],1)


