import unittest


from src.data_preparation.bio_encoder import create_label_mapping, generate_bio_labels
from src.data_preparation.bio_decoder import decode_bio


def entity(kind, start, end):
    return {'type': kind, 'spans': [{'start': start, 'end': end}]}


class BioTests(unittest.TestCase):
    def setUp(self):
        self.l2i, self.i2l = create_label_mapping(['CITY','COUNTRY'])

    def test_roundtrip_multitoken_and_adjacent_same_type(self):
        text = 'Нижний НовгородМосква'
        offsets = [(0,0),(0,6),(7,10),(10,15),(15,21),(0,0)]
        labels = generate_bio_labels(offsets,[entity('CITY',0,15),entity('CITY',15,21)],self.l2i)
        self.assertEqual(labels,[-100,1,2,2,1,-100])
        result = decode_bio(labels,offsets,text,self.i2l)
        self.assertEqual([(e['start'],e['end']) for e in result],[(0,15),(15,21)])
        self.assertEqual(result[0]['text'],'Нижний Новгород')


    def test_orphan_i_starts_entity(self):
        self.assertEqual(decode_bio(['I-CITY'],[(0,6)],'Москва')[0]['type'],'CITY')


