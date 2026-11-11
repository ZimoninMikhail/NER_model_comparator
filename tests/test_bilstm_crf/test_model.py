"""Проверки независимой BiLSTM-CRF: маски, градиенты и маленькая задача NER."""

import torch

from src.models.bilstm_crf_core import BiLSTMCRF

LABELS={0:'O',1:'B-CITY',2:'I-CITY'}


def model(**kwargs):
    return BiLSTMCRF(vocab_size=20,num_labels=3,embedding_dim=8,hidden_size=6,
                     dropout=0.,padding_idx=0,special_token_ids=(1,2),id2label=LABELS,**kwargs)


def test_shape_mask_and_gradients_through_all_layers():
    torch.manual_seed(11)
    net=model()
    ids=torch.tensor([[1,3,4,2,0],[1,5,2,0,0]])
    attention=torch.tensor([[1,1,1,1,0],[1,1,1,0,0]])
    tags=torch.tensor([[-100,1,2,-100,-100],[-100,0,-100,-100,-100]])
    output=net(ids,attention,tags)
    assert output.emissions.shape==(2,5,3)
    assert output.predictions.shape==ids.shape
    assert output.token_mask.tolist()==[[False,True,True,False,False],[False,True,False,False,False]]
    assert torch.all(output.predictions[~output.token_mask]==-100)
    assert output.loss.ndim==0 and torch.isfinite(output.loss)
    output.loss.backward()
    for name,parameter in net.named_parameters():
        assert parameter.grad is not None,name
        assert torch.isfinite(parameter.grad).all(),name
    assert net.embedding.weight.grad[0].count_nonzero()==0
    assert net.embedding.weight.grad[1].count_nonzero()==0
    assert net.embedding.weight.grad[2].count_nonzero()==0
    assert net.embedding.weight.grad[3].abs().sum()>0
    assert net.lstm.weight_ih_l0_reverse.grad.abs().sum()>0
    assert net.classifier.weight.grad.abs().sum()>0
    assert net.crf.transitions.grad.abs().sum()>0


