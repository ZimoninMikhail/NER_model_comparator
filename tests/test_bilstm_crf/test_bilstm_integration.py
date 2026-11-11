"""BiLSTM-CRF в общем Trainer, локальных комплектах и CLI; без сетевых моделей."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from transformers import BertTokenizerFast
import yaml

import run
from src.config import load_config
from src.models.adapter import predict_token_labels
from src.models.bilstm_crf import BiLSTMCRFConfig, BiLSTMCRFModel
from src.training.trainer import NERDataset, build_dev_metrics
from src.data_preparation.prepared_data_storage import PreparedDataStorage
from src.validation.validator import NERValidator
from src.validation.model_registry import ModelRegistry
from src.inference.inference_pipeline import InferencePipeline

L2I={'O':0,'B-CITY':1,'I-CITY':2};I2L={v:k for k,v in L2I.items()}


def tiny_model():
    return BiLSTMCRFModel(BiLSTMCRFConfig(vocab_size=10,embedding_dim=4,hidden_size=4,
        num_layers=1,dropout=0.,padding_idx=0,special_token_ids=[2,3],
        id2label=I2L,label2id=L2I,max_seq_length=5))


def sample(split='dev',text='Москва Россия'):
    return {'doc_id':split+':0','split':split,'text':text,'input_ids':[2,5,6,3],
        'attention_mask':[1,1,1,1],'labels':[-100,1,2,-100],
        'offset_mapping':[(0,0),(0,6),(7,13),(0,0)],
        'gold_entities':[{'type':'CITY','spans':[{'start':0,'end':13}]}]}


def info():
    return {'num_labels':3,'entity_types':['CITY'],'label2id':L2I,'id2label':I2L,
        'architecture':'bilstm_crf','vocab_size':10,'padding_idx':0,
        'special_token_ids':[2,3],'tokenizer_name':'local','max_length':5}


def transition_model():
    model=tiny_model().eval()
    with torch.no_grad():
        model.network.classifier.weight.zero_();model.network.classifier.bias.copy_(torch.tensor([4.,3.,0.]))
        for p in model.network.crf.parameters():p.zero_()
        model.network.crf.transitions[1,2]=10.
    return model


def fixture_tokenizer(folder):
    folder.mkdir(exist_ok=True,parents=True)
    vocab=folder/'vocab.txt';vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nМосква\nРоссия\n')
    tokenizer=BertTokenizerFast(vocab_file=str(vocab),do_lower_case=False)
    tokenizer.save_pretrained(folder)
    return tokenizer


def test_validator_and_trainer_metrics_use_crf_decode_not_argmax(tmp_path):
    model=transition_model();folder=tmp_path/'model';model.save_pretrained(folder)
    storage=PreparedDataStorage(str(tmp_path/'prepared'))
    path=storage.save_batch([sample()],{'split':'dev'},info())
    metrics=NERValidator(folder,device='cpu').validate(storage,[path])
    assert metrics['macro_f1']==1
    data=storage.load_batch(path);dataset=NERDataset(data['tensors'],data['samples'],info())
    callback=build_dev_metrics(dataset,I2L,decoded=True)
    ids=dataset.input_ids;mask=dataset.token_mask
    predictions=predict_token_labels(model,ids,dataset.attention_mask,mask).numpy()
    assert callback(SimpleNamespace(predictions=predictions))['macro_f1']==1
    with pytest.raises(ValueError,match='Витерби'):
        callback(SimpleNamespace(predictions=np.zeros((1,4,3))))
    # Валидатор не получает labels и не скрывает ошибку предсказания по эталону.
    data['tensors']['labels'][:]=-100
    with patch.object(storage,'load_batch',return_value=data):
        assert NERValidator(folder,device='cpu').validate(storage,[path])['macro_f1']==1


def test_full_cli_crf_cycle_with_long_and_empty_documents_without_network(tmp_path):
    config=load_config('config/bilstm_crf.yaml')
    tokenizer_dir=tmp_path/'tokenizer';tokenizer=fixture_tokenizer(tokenizer_dir)
    config['model'].update(tokenizer_name=str(tokenizer_dir),parameters={
        'embedding_dim':8,'hidden_size':4,'num_layers':1,'dropout':0.})
    config['data'].update(entity_types=['CITY'],max_seq_length=5,type_priority_path=str(tmp_path/'priority.json'))
    config['training'].update(num_epochs=2,batch_size=2,logging_steps=1,learning_rate=.01)
    source=tmp_path/'source';source.mkdir()
    (source/'annotation.conf').write_text('[entities]\nCITY\n[relations]\n')
    (source/'SOURCE_COMMIT.txt').write_text(config['data']['source']['commit'])
    config['data']['source']['path']=str(source)
    text='Москва '*8
    for split in ['train','dev','test']:
        folder=source/split;folder.mkdir()
        (folder/'long.txt').write_text(text);(folder/'long.ann').write_text('\n'.join(
            f'T{i}\tCITY {7*i} {7*i+6}\tМосква' for i in range(8)))
        (folder/'empty.txt').write_text('');(folder/'empty.ann').write_text('')
    for key in config['paths']:config['paths'][key]=str(tmp_path/key)
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(config))
    args=['--config',str(path)]
    with patch('socket.socket.connect',side_effect=AssertionError('network forbidden')),patch(
            'src.models.bert.AutoModelForTokenClassification.from_pretrained',
            side_effect=AssertionError('BERT weights must not be loaded')):
        run.main(['-m','type-statistics']+args)
        version=run.main(['-m','prepare']+args)
        report=run.main(['-m','train','--dataset-version',version,'--device','cpu','-l','-mt','-p']+args)
        model_version=report['version'];folder=Path(report['model_path'])
        output=run.main(['-m','inference','-t',text,'--model-version',model_version,'--device','cpu','-hi','-p']+args)
        benchmark=run.main(['-m','benchmark','--model-version',model_version,'--device','cpu',
                           '--warmup','0','--repeats','1','-b']+args)
        run.main(['-m','summary','-mt','-c']+args)
        pipeline=InferencePipeline(models_dir=config['paths']['models'],model_version=model_version,device='cpu')
        assert pipeline.max_length==5
        restored=pipeline.predict(text)
    assert report['architecture']=='bilstm_crf'
    assert report['metrics']['num_documents']==2
    assert report['metrics']['per_type']['CITY']['support']==8
    assert benchmark['architecture']=='bilstm_crf'
    assert benchmark['model_name']==model_version
    assert json.loads(Path(output).read_text())[0]['entities']==restored
    assert report['selection']['best_macro_f1']==report['metrics']['macro_f1']
    assert (folder/'model_runtime.json').is_file() and (folder/'tokenizer.json').is_file()
    assert len(list((tmp_path/'visualization/plots').glob('*.png')))>=5
    config_file=json.loads((folder/'config.json').read_text())
    assert config_file['vocab_size']==len(tokenizer)
    assert config_file['model_type']=='bilstm_crf'
    # Выбор версии определяет собственный локальный комплект, не текущую конфигурацию модели.
    registry=ModelRegistry(config['paths']['models']);latest=tmp_path/'models'/'v_other'
    other=tiny_model();other.save_pretrained(latest);fixture_tokenizer(latest)
    registry.register_model(str(latest))
    with patch('socket.socket.connect',side_effect=AssertionError('network forbidden')):
        selected=InferencePipeline(models_dir=config['paths']['models'],model_version=model_version,device='cpu')
    assert selected.model_path==folder.resolve()


