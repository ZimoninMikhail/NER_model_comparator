"""Обёртка BiLSTM-CRF для сохранения checkpoint и общего Trainer."""
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput

from .bilstm_crf_core import BiLSTMCRF


class BiLSTMCRFConfig(PretrainedConfig):
    model_type = 'bilstm_crf'

    def __init__(self, vocab_size=1, embedding_dim=128, hidden_size=128,
                 num_layers=1, dropout=0.2, padding_idx=0, special_token_ids=None,
                 tokenizer_name=None, max_seq_length=512, num_labels=None, **kwargs):
        mapping = kwargs.get('id2label')
        if num_labels is None:
            num_labels = len(mapping) if mapping is not None else 2
        elif mapping is not None and len(mapping) != num_labels:
            raise ValueError('num_labels не соответствует id2label')
        kwargs.pop('pad_token_id', None)
        super().__init__(num_labels=num_labels, pad_token_id=padding_idx, **kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.padding_idx = padding_idx
        self.special_token_ids = list(special_token_ids or [])
        self.tokenizer_name = tokenizer_name
        self.max_seq_length = max_seq_length


@dataclass
class CRFClassifierOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    predictions: Optional[torch.Tensor] = None


class BiLSTMCRFModel(PreTrainedModel):
    config_class = BiLSTMCRFConfig
    base_model_prefix = 'network'
    main_input_name = 'input_ids'

    def __init__(self, config):
        super().__init__(config)
        self.network = BiLSTMCRF(config.vocab_size, config.num_labels,
            embedding_dim=config.embedding_dim, hidden_size=config.hidden_size,
            num_layers=config.num_layers, dropout=config.dropout,
            padding_idx=config.padding_idx, special_token_ids=config.special_token_ids,
            id2label=config.id2label)
        self.post_init()

    def forward(self, input_ids, attention_mask=None, labels=None, token_mask=None, decode=False):
        output = self.network(input_ids, attention_mask, labels, token_mask, decode=decode)
        return CRFClassifierOutput(loss=output.loss, logits=output.emissions,
                                   predictions=output.predictions)


def create_model(num_labels, vocab_size, label2id=None, id2label=None,
                 model_parameters=None, padding_idx=0, special_token_ids=(),
                 tokenizer_name=None, max_seq_length=512):
    """Создать архитектуру с обучаемыми эмбеддингами, без загрузки весов BERT."""
    if type(vocab_size) is not int or vocab_size < 1:
        raise ValueError('Для BiLSTM-CRF нужен размер словаря токенизатора')
    config = BiLSTMCRFConfig(vocab_size=vocab_size, num_labels=num_labels,
        label2id=label2id, id2label=id2label, padding_idx=padding_idx,
        special_token_ids=special_token_ids, tokenizer_name=tokenizer_name,
        max_seq_length=max_seq_length, **(model_parameters or {}))
    return BiLSTMCRFModel(config)


def load_model(path):
    """Прочитать локальный комплект, включая формат прежних checkpoint."""
    return BiLSTMCRFModel.from_pretrained(path, local_files_only=True)
