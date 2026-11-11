"""Проверки настоящего сохранения/восстановления checkpoint без скачивания моделей."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import BertConfig, BertForTokenClassification, TrainerCallback

from src.data_preparation.prepared_data_storage import PreparedDataStorage
from src.training.trainer import train_from_batches


INFO={'num_labels':3,'entity_types':['CITY'],
      'label2id':{'O':0,'B-CITY':1,'I-CITY':2}, 'id2label':{0:'O',1:'B-CITY',2:'I-CITY'}}


def sample(split):
    return {'doc_id':split+':0','split':split,'text':'Москва',
        'input_ids':[2,5,3], 'attention_mask':[1,1,1], 'labels':[-100,1,-100],
        'offset_mapping':[(0,0),(0,6),(0,0)],
        'gold_entities':[{'type':'CITY','start':0,'end':6}]}


def tiny_model(**kwargs):
    return BertForTokenClassification(BertConfig(vocab_size=10, hidden_size=8,
        num_hidden_layers=1,num_attention_heads=2,intermediate_size=16,
        max_position_embeddings=16,num_labels=3,
        label2id=INFO['label2id'],id2label=INFO['id2label']))


class ForceEpochPredictions(TrainerCallback):
    """Первая эпоха даёт CITY; вторая намеренно даёт O, либо ту же оценку."""
    def __init__(self, tie=False):
        self.tie=tie
        self.snapshots={}

    def on_epoch_end(self,args,state,control,model=None,**kwargs):
        epoch=int(state.epoch)
        with torch.no_grad():
            model.classifier.weight.zero_()
            # При равной оценке bias отличается — можно определить сохранённую эпоху.
            values=[-10.,10.+epoch,-10.] if epoch==1 or self.tie else [10.,-10.,-10.]
            model.classifier.bias.copy_(torch.tensor(values,device=model.classifier.bias.device))
        self.snapshots[epoch]=model.classifier.bias.detach().cpu().clone()


class CheckpointSelectionTests(unittest.TestCase):
    def fixture(self,tmp):
        storage=PreparedDataStorage(str(Path(tmp)/'prepared'))
        paths={split:storage.save_batch([sample(split)],{'split':split},INFO)
               for split in ('train','dev','test')}
        return storage,paths

    def train(self,tmp,callback):
        storage,paths=self.fixture(tmp)
        original_test={p:p.read_bytes() for p in paths['test'].iterdir()}
        actual_load=storage.load_batch
        def load(path):
            self.assertNotEqual(Path(path),paths['test'],'Test используется при обучении')
            return actual_load(path)
        with patch('src.training.model_factory.create_model',side_effect=tiny_model), \
             patch('src.training.model_factory.get_device',return_value='cpu'), \
             patch.object(storage,'load_batch',side_effect=load):
            path,version,history=train_from_batches(storage,[paths['train']],
                eval_batch_paths=[paths['dev']],models_dir=str(Path(tmp)/'models'),
                num_epochs=2,batch_size=1,learning_rate=0.00001,
                warmup_steps=0,logging_steps=1,callbacks=[callback])
        for p,data in original_test.items():self.assertEqual(p.read_bytes(),data)
        return Path(path)

    def test_worse_last_epoch_restores_and_saves_first_epoch_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            callback=ForceEpochPredictions()
            final=self.train(tmp,callback)
            metadata=json.loads((final/'training_metadata.json').read_text())
            selection=metadata['selection']
            self.assertEqual(selection['best_epoch'],1)
            self.assertEqual(selection['best_macro_f1'],1)
            self.assertEqual([r['eval_macro_f1'] for r in selection['evaluation_history']],[1,0])
            restored=BertForTokenClassification.from_pretrained(final)
            self.assertTrue(torch.equal(restored.classifier.bias.detach(),callback.snapshots[1]))
            self.assertFalse(torch.equal(restored.classifier.bias.detach(),callback.snapshots[2]))
            self.assertFalse(list((Path(tmp)/'models').glob('.temp_checkpoints*')))


    def test_test_split_cannot_be_used_as_dev(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage,paths=self.fixture(tmp)
            with self.assertRaisesRegex(ValueError,'dev'):
                train_from_batches(storage,[paths['train']],eval_batch_paths=[paths['test']])


