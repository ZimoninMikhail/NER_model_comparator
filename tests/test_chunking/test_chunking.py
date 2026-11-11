"""Разбиение и оценка длинных документов с реальным локальным токенизатором."""
from types import SimpleNamespace

import pytest
import torch
from transformers import BertTokenizerFast

from src.data_preparation.document_chunker import chunk_document
from src.data_preparation.bio_encoder import encode_batch, create_label_mapping
from src.data_preparation.bio_decoder import decode_bio
from src.inference.predictor import NERPredictor
from src.validation.document_predictions import collect_document_predictions
from src.validation.metrics import compute_ner_metrics


@pytest.fixture
def tokenizer(tmp_path):
    vocab = tmp_path / 'vocab.txt'
    vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nМосква\nРоссия\nОмск\nНовый\nгород\n.\n!\nг\nИ\nв\n', encoding='utf-8')
    return BertTokenizerFast(vocab_file=str(vocab), do_lower_case=False)


def entity(start, end):
    return {'type': 'CITY', 'spans': [{'start': start, 'end': end}]}


def encode(text, tokenizer, gold=None, limit=6):
    return encode_batch([{'doc_id': 'dev:0', 'split': 'dev', 'text': text,
                          'parsed_entities': gold or []}], tokenizer,
                         create_label_mapping(['CITY'])[0], limit)


def decoded(chunks):
    mapping = create_label_mapping(['CITY'])[1]
    return [(chunk, decode_bio(chunk['labels'], chunk['offset_mapping'], chunk['text'], mapping))
            for chunk in chunks]


def test_greedy_sentence_packing_then_token_fallback(tokenizer):
    text = 'Москва. Россия. Омск Новый город Москва Россия'
    chunks = chunk_document(text, tokenizer, max_length=6)
    assert chunks[0]['token_end'] == 4  # два предложения целиком
    assert chunks[0]['boundary'] == 'sentence'
    assert chunks[1]['boundary'] == 'token'
    assert chunks[-1]['boundary'] == 'document_end'


def test_boundary_moved_before_entity_without_duplicate_gold(tokenizer):
    text = 'Москва Россия Новый город Омск'
    start = text.index('Новый')
    gold = [entity(start, start + len('Новый город'))]
    chunks = encode(text, tokenizer, gold, limit=5)  # вместимость 3, сущность в токенах 2..4
    assert chunks[0]['token_end'] == 2
    assert sum(len(c['gold_entities']) for c in chunks) == 1
    docs = collect_document_predictions(decoded(chunks))
    metrics = compute_ner_metrics(docs, ['CITY'])
    assert len(docs) == 1
    assert docs[0]['true_entities'] == gold
    assert metrics['macro_f1'] == 1
    assert metrics['per_type']['CITY']['support'] == 1


def test_inference_processes_all_chunks_and_returns_original_coordinates(tokenizer):
    class Model:
        def __init__(self): self.calls = 0
        def to(self, device): return self
        def eval(self): return self
        def __call__(self, ids, attention_mask=None):
            self.calls += 1
            assert ids.shape[1] <= 5
            logits = torch.zeros((*ids.shape, 3)); logits[:, :, 1] = 1
            return SimpleNamespace(logits=logits)
    model = Model()
    predictor = NERPredictor(model, tokenizer, create_label_mapping(['CITY'])[1], device='cpu', max_length=5)
    text = 'Москва ' * 9
    results = predictor.predict_single(text)
    assert model.calls == 3
    assert [(e['start'], e['end']) for e in results] == [(i * 7, i * 7 + 6) for i in range(9)]
    assert all(e['text'] == text[e['start']:e['end']] for e in results)
    assert predictor.predict_single(text, return_spans=False) == [{'type': 'CITY', 'text': 'Москва'}] * 9


