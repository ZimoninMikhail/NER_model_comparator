"""Локальные BRAT-источники: данные, происхождение и совместимость этапов."""
import hashlib
import shutil
from unittest.mock import patch

import pytest

from src import pipeline
from src.config import load_config, validate_config
from src.data_analysis.type_statistics import build_type_priority


@pytest.fixture
def local(tmp_path):
    config = load_config()
    source = tmp_path / 'source'
    source.mkdir()
    (source/'SOURCE_COMMIT.txt').write_text(config['data']['source']['commit'])
    (source/'annotation.conf').write_text('[entities]\nCITY\nPERSON\n[relations]\nConnected Arg1:CITY, Arg2:PERSON\n')
    for split in ('train','dev','test'):
        folder=source/split;folder.mkdir()
        (folder/'b.txt').write_text('Москва\u00a0Иван\n', encoding='utf-8')
        (folder/'b.ann').write_text('T1\tCITY 0 6\tМосква\nT2\tPERSON 7 11\tИван\nR1\tConnected Arg1:T1 Arg2:T2\n', encoding='utf-8')
        (folder/'a.txt').write_text('Без сущностей', encoding='utf-8')
        (folder/'a.ann').write_text('', encoding='utf-8')
    config['data'].update(entity_types=['CITY','PERSON'],type_priority_path=str(tmp_path/'priority.json'))
    config['data']['source']['path']=str(source)
    for key in config['paths']:config['paths'][key]=str(tmp_path/key)
    return validate_config(config),source


def reader(config):return pipeline._loader(config)


def test_real_local_reader_no_network_preserves_annotations_ids_and_offsets(local):
    config,source=local
    with patch('socket.socket.connect',side_effect=AssertionError('network forbidden')):
        loader=reader(config);loader.load_train();docs=[d for batch in loader for d in batch]
    assert [d['source_doc_id'] for d in docs]==['a','b']
    assert len(loader)==2
    doc=docs[1]
    assert doc['doc_id']=='nerel:train:b'
    assert doc['text']=='Москва\u00a0Иван\n'
    assert len(doc['entities'])==2 and len(doc['source_annotations'])==3
    assert doc['source_annotations'][-1].startswith('R1')
    assert doc['source_files']['annotations_sha256']==hashlib.sha256((source/'train/b.ann').read_bytes()).hexdigest()
    assert doc['text'][7:11]=='Иван'
    assert docs[0]['entities']==[]
    loader.reset();assert len(list(loader)[0])==2
    loader.load_dev();assert next(loader)[1]['doc_id']=='nerel:dev:b'


def test_statistics_uses_only_train_and_is_reproducible(local):
    config,source=local
    shutil.rmtree(source/'dev');shutil.rmtree(source/'test')
    first=build_type_priority(config,reader(config));second=build_type_priority(config,reader(config))
    assert first==second
    assert first['frequencies']=={'CITY':1,'PERSON':1}
    assert first['source']['schema_sha256']==hashlib.sha256((source/'annotation.conf').read_bytes()).hexdigest()


