"""Маленькая локальная GLiNER для проверок без сети."""
from unittest.mock import patch
import pytest
from gliner import GLiNER, GLiNERConfig
from transformers import BertConfig, BertModel, BertTokenizerFast


@pytest.fixture
def tiny_model(tmp_path):
    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '[ENT]', '[SEP_TEXT]',
             'Москва', 'Россия', 'city', 'country']
    (tmp_path / 'vocab.txt').write_text('\n'.join(vocab), encoding='utf-8')
    tokenizer = BertTokenizerFast(vocab_file=str(tmp_path / 'vocab.txt'),
                                   do_lower_case=False, model_max_length=64)
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ENT]', '[SEP_TEXT]']})
    tokenizer.save_pretrained(tmp_path)
    encoder = BertConfig(vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1,
                         num_attention_heads=2, intermediate_size=32, max_position_embeddings=64)
    BertModel(encoder).save_pretrained(tmp_path)
    config = GLiNERConfig(model_name=str(tmp_path), hidden_size=16, dropout=0.,
                          max_width=4, max_len=32, vocab_size=len(tokenizer),
                          ent_token='[ENT]', sep_token='[SEP_TEXT]',
                          class_token_index=tokenizer.convert_tokens_to_ids('[ENT]'))
    with patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')):
        return GLiNER(config, tokenizer=tokenizer).to('cpu').eval()


