"""Полный цикл CLI с настоящим маленьким BERT и локальным корпусом."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml
from transformers import BertConfig,BertForTokenClassification,BertTokenizerFast
import run
from src.config import load_config


class FullPipelineTests(unittest.TestCase):
    def test_prepare_train_inference_summary_and_benchmark_in_isolated_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            config=load_config();config['device']='cpu'
            config['data'].update(entity_types=['CITY'],max_seq_length=5,
                type_priority_path=str(root/'priority.json'))
            config['training'].update(num_epochs=1,batch_size=1,logging_steps=1,warmup_steps=0)
            for key in config['paths']:config['paths'][key]=str(root/key)
            source=root/'sources'; source.mkdir()
            (source/'annotation.conf').write_text('[entities]\nCITY\n[relations]\n')
            (source/'SOURCE_COMMIT.txt').write_text(config['data']['source']['commit'])
            config['data']['source']['path']=str(source)
            config_path=root/'config.yaml';config_path.write_text(yaml.safe_dump(config))
            vocab=root/'vocab.txt';vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nМосква\n',encoding='utf-8')
            tokenizer=BertTokenizerFast(vocab_file=str(vocab),do_lower_case=False)
            long_text = 'Москва ' * 8
            corpus={split:[{'text':long_text,'entities':[f'T{i}\tCITY {i*7} {i*7+6}\tМосква' for i in range(8)]}]
                    for split in ('train','dev','test')}
            for split, docs in corpus.items():
                folder=source/split;folder.mkdir()
                for i,doc in enumerate(docs):
                    (folder/f'doc_{i}.txt').write_text(doc['text'],encoding='utf-8')
                    (folder/f'doc_{i}.ann').write_text('\n'.join(doc['entities']),encoding='utf-8')
            def tiny(**kwargs):
                return BertForTokenClassification(BertConfig(vocab_size=6,hidden_size=8,
                    num_hidden_layers=1,num_attention_heads=2,intermediate_size=16,
                    max_position_embeddings=16,num_labels=3,label2id=kwargs['label2id'],
                    id2label={int(k):v for k,v in kwargs['id2label'].items()}))
            previous=Path.cwd()
            try:
                os.chdir(root)
                with patch('src.pipeline.load_tokenizer',return_value=tokenizer), \
                     patch('src.training.model_factory.create_model',side_effect=tiny), \
                     patch('socket.socket.connect',side_effect=AssertionError('network forbidden')):
                    args=['--config',str(config_path)]
                    run.main(['-m','type-statistics']+args)
                    version=run.main(['-m','prepare']+args)
                    result=run.main(['-m','train','--dataset-version',version]+args)
                    output=Path(run.main(['-m','inference','-t',long_text]+args))
                    summary=Path(run.main(['-m','summary']+args))
                    textfile=root/'texts.txt';textfile.write_text(long_text + '\n')
                    benchmark=run.main(['-m','benchmark','-f',str(textfile),'--warmup','0','--repeats','1']+args)
                prepared = Path(result['model_path'])/'experiment.json'
                experiment = json.loads(prepared.read_text())
                chunk_report = json.loads((Path(experiment['prepared_path'])/'chunking_report.json').read_text())
                self.assertEqual(chunk_report['splits']['train']['chunks'], 3)
                self.assertEqual(result['metrics']['num_documents'], 1)
                self.assertEqual(result['metrics']['per_type']['CITY']['support'], 8)
                self.assertEqual(output.parent,root/'predictions')
                self.assertEqual(json.loads(summary.read_text())['models']['latest_version'],result['version'])
                self.assertEqual(result['selection']['best_epoch'],1)
                self.assertEqual(benchmark['num_measurements'],1)
                self.assertFalse((root/'inference_data'/'temp_text.txt').exists())
            finally:os.chdir(previous)
