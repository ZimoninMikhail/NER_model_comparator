"""GLiNER в общей конфигурации, разбиении и CLI, включая длинный документ."""
import re
from unittest.mock import patch

import pytest
import run
from src.config import PROJECT_ROOT
from src.inference.gliner_predictor import GLiNERPredictor


def test_long_text_uses_both_limits_and_global_offsets(tiny_model):
    tiny_model.config.max_len=4
    tiny_model.data_processor.transformer_tokenizer.model_max_length=10
    text=('  Москва Россия.\n'*20)+' Москва'
    predictor=GLiNERPredictor(tiny_model,{'CITY':'city','COUNTRY':'country'},max_length=10)
    chunks=predictor.split_text(text)
    assert len(chunks)>1
    calls=[]
    def predict(fragment,labels,**kwargs):
        calls.append(fragment)
        return [dict(start=m.start(),end=m.end(),text=m.group(),label='city',score=.9)
                for m in re.finditer('Москва',fragment)]
    with patch.object(tiny_model,'predict_entities',side_effect=predict):
        entities=predictor.predict_single(text)
    assert [e['start'] for e in entities]==[m.start() for m in re.finditer('Москва',text)]
    assert all(e['text']==text[e['start']:e['end']] for e in entities)
    assert len(calls)==len(chunks)
    offsets=[(start,end) for _,start,end in tiny_model.data_processor.words_splitter(text)]
    assert all(sum(c['chunk_start']<=start<end<=c['chunk_end'] for c in chunks)==1
               for start,end in offsets)


@pytest.mark.parametrize('mode',['train','update'])
def test_no_gliner_training_or_update_is_possible(mode):
    argv=['-m',mode,'--config',str(PROJECT_ROOT/'config/gliner.yaml')]
    if mode=='train':argv+=['--dataset-version','not-used']
    with patch.object(run,'run_prepare') as prepare,patch.object(run,'run_train') as train:
        with pytest.raises(SystemExit) as error:run.main(argv)
        assert error.value.code==2
        prepare.assert_not_called();train.assert_not_called()


