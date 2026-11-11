"""Общий оценщик, явный split, сохранение отчётов и отсутствие изменения моделей."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import yaml
import run
from src import pipeline as stages
from src.config import load_config, PROJECT_ROOT
from src.validation.evaluator import validate_documents
from src.validation.evaluation_reports import comparison_groups, load_evaluation_reports
from src.validation.model_registry import ModelRegistry


def entity(start,end,kind):
    return {'type':kind,'spans':[{'start':start,'end':end}]}


def document(identifier='doc',text='Москва Россия',gold=None,split='dev'):
    gold = [entity(0,6,'CITY'),entity(7,13,'COUNTRY')] if gold is None else gold
    return {'doc_id':identifier,'text':text,'flat_entities':deepcopy(gold),
            'parsed_entities':deepcopy(gold),'split':split,'preprocessing_version':3,
            'quality':{'has_errors':False}}


def test_common_exact_metrics_and_empty_documents_without_gold_leakage():
    docs=[document(),document('empty','Омск',[])]
    storage=SimpleNamespace(load_batch=Mock(return_value=docs))
    predict=Mock(side_effect=[[{'start':0,'end':6,'type':'CITY'}],
                              [{'start':0,'end':4,'type':'CITY'}]])
    result=validate_documents(SimpleNamespace(predict=predict),storage,[Path('batch')],['CITY','COUNTRY'],'dev')
    assert result['macro_f1']==pytest.approx(1/3)
    assert result['micro_f1']==pytest.approx(.5)
    assert result['num_documents']==2
    assert result['per_type']['COUNTRY']['support']==1
    assert [call.args for call in predict.call_args_list]==[('Москва Россия',),('Омск',)]
    assert len(result['corpus_sha256'])==64


@pytest.fixture
def prepared_experiment(tmp_path,tiny_model):
    config=load_config(PROJECT_ROOT/'config/gliner.yaml')
    config['device']='cpu'
    config['data'].update(entity_types=['CITY'],type_priority_path=str(tmp_path/'priority.json'))
    config['model'].update(type_mapping={'CITY':'city'},threshold=0.,name=str(tmp_path/'bundle'))
    tiny_model.save_pretrained(config['model']['name'])
    for key in config['paths']:config['paths'][key]=str(tmp_path/key)
    class Loader:
        def load_train(self):self.split='train'
        def load_dev(self):self.split='dev'
        def load_test(self):self.split='test'
        def __iter__(self):
            yield [{'doc_id':self.split+':gold','text':'Москва','entities':['T1\tCITY 0 6\tМосква'],'batch_id':0},
                   {'doc_id':self.split+':empty','text':'Россия','entities':[],'batch_id':0}]
    with patch.object(stages,'_loader',side_effect=lambda _:Loader()):
        stages.calculate_type_statistics(config)
        version=stages.prepare_data(config)
    return config,version


def test_real_cli_dev_and_test_are_separate_and_do_not_change_data_or_latest_model(prepared_experiment,tmp_path):
    config,version=prepared_experiment
    before={p:p.read_bytes() for p in Path(config['paths']['processed']).rglob('*') if p.is_file()}
    metrics={'macro_f1':.1,'micro_f1':.1,'micro_precision':.1,'micro_recall':.1,'entity_types':['CITY'],'num_documents':2,'match':'exact_span_and_type'}
    registry=ModelRegistry(config['paths']['models'])
    registry.register_model(str(tmp_path/'models'/'original'),metrics,
                            {'dataset_version':version,'architecture':'bert','preprocessing_version':3})
    registry_before=registry.registry_path.read_bytes()
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(config))
    with patch('socket.socket.connect',side_effect=AssertionError('Network forbidden')), \
         patch.object(stages,'train_from_batches') as trainer,patch.object(stages,'_loader') as source:
        reports=[run.main(['-m','evaluate','--config',str(path),'--dataset-version',version,
                           '--split',split,'--device','cpu','-mt']) for split in ('dev','test')]
        summary_path=run.main(['-m','summary','--config',str(path),'-mt','-c'])
        trainer.assert_not_called();source.assert_not_called()
    assert registry.registry_path.read_bytes()==registry_before
    for p,data in before.items():assert p.read_bytes()==data
    assert reports[0]['evaluation_id']!=reports[1]['evaluation_id']
    for split,report in zip(('dev','test'),reports):
        assert report['split']==split and report['fine_tuning'] is False
        assert report['metrics']['num_documents']==2
        assert report['metrics']['per_type']['CITY']['support']==1
        assert report['model_files_sha256'] and report['threshold']==0.
        predictions=json.loads(Path(report['predictions_path']).read_text())
        assert all(doc['doc_id'].startswith(split+':') for doc in predictions)
        assert Path(report['report_path']).exists()
    summary=json.loads(Path(summary_path).read_text())
    assert summary['models']['latest_version']=='original'
    assert {item['split'] for item in summary['evaluations']}=={'dev','test'}
    evaluations=load_evaluation_reports(config['paths']['reports'])
    groups=comparison_groups(registry.registry['versions'],evaluations)
    assert sorted(len(g['models']) for g in groups)==[1,2]
    assert all(any('GLiNER' in row['display_name'] for row in group['models']) for group in groups)


def test_different_schemas_and_versions_are_not_compared_together():
    base={'version':'a','timestamp':'','metadata':{'architecture':'bert','dataset_version':'v1',
         'preprocessing_version':3},'metrics':{'macro_f1':.1,'micro_f1':.2,'entity_types':['CITY'],
         'match':'exact_span_and_type','num_documents':2}}
    other=deepcopy(base);other['version']='b';other['metadata']['dataset_version']='v2'
    third=deepcopy(base);third['version']='c';third['metrics']['entity_types']=['PERSON']
    assert len(comparison_groups([base,other,third],[]))==3


@pytest.mark.parametrize('architecture',['bert','bilstm_crf'])
def test_saved_token_models_use_same_test_gold_without_training(prepared_experiment,architecture,tmp_path):
    from transformers import AutoTokenizer, BertConfig, BertForTokenClassification
    from src.models.bilstm_crf import BiLSTMCRFConfig, BiLSTMCRFModel
    from src.validation.evaluator import evaluate_model
    config,version=prepared_experiment
    tokenizer=AutoTokenizer.from_pretrained(config['model']['name'],local_files_only=True)
    config['model']={'architecture':architecture,'name':'local','tokenizer_name':'local'}
    mapping={'O':0,'B-CITY':1,'I-CITY':2}
    parameters=dict(vocab_size=len(tokenizer),num_labels=3,label2id=mapping,
                    id2label={v:k for k,v in mapping.items()})
    if architecture=='bert':
        model=BertForTokenClassification(BertConfig(**parameters,hidden_size=8,
            num_hidden_layers=1,num_attention_heads=2,intermediate_size=16,max_position_embeddings=16))
    else:
        model=BiLSTMCRFModel(BiLSTMCRFConfig(**parameters,embedding_dim=8,hidden_size=4,
            dropout=0.,max_seq_length=8,padding_idx=tokenizer.pad_token_id))
    folder=Path(config['paths']['models'])/architecture
    model.save_pretrained(folder);tokenizer.save_pretrained(folder)
    (folder/'model_runtime.json').write_text(json.dumps({'architecture':architecture,
        'dataset_version':version,'max_length':8,'entity_types':['CITY']}))
    registry=ModelRegistry(config['paths']['models']);registry.register_model(str(folder))
    registry_before=registry.registry_path.read_bytes()
    before={p:p.read_bytes() for p in folder.iterdir() if p.is_file()}
    path=tmp_path/'evaluate.yaml';path.write_text(yaml.safe_dump(config))
    with patch('socket.socket.connect',side_effect=AssertionError('Network forbidden')):
        with pytest.raises(ValueError,match='model-version'):
            evaluate_model(config,version,'test')
        report=run.main(['-m','evaluate','--config',str(path),'--dataset-version',version,
                         '--split','test','--model-version',architecture,'--device','cpu','-mt'])
    assert report['architecture']==architecture and report['model_version']==architecture
    assert report['metrics']['num_documents']==2
    assert report['metrics']['per_type']['CITY']['support']==1
    assert all(doc['doc_id'].startswith('test:') for doc in json.loads(Path(report['predictions_path']).read_text()))
    assert registry.registry_path.read_bytes()==registry_before
    for p,data in before.items():assert p.read_bytes()==data
    assert any(r['architecture']==architecture for r in load_evaluation_reports(config['paths']['reports']))
