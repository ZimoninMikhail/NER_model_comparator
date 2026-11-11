"""Создание BERT для NER и загрузка его локальных сохранённых весов.

Архитектура предоставляется transformers; параметры и формат весов прежние.
"""
from pathlib import Path
from transformers import AutoModelForTokenClassification


def create_model(model_name, num_labels, label2id=None, id2label=None):
    return AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=num_labels, label2id=label2id, id2label=id2label)


def load_model(path):
    return AutoModelForTokenClassification.from_pretrained(Path(path), local_files_only=True)
