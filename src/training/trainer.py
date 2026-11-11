"""Обучение с dev-оценкой и выбором checkpoint по entity-level macro-F1."""
import json
import shutil
from pathlib import Path
from datetime import datetime
from uuid import uuid4

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, set_seed

from src.device import resolve_device
from src.models.adapter import is_bilstm_crf, token_mask_from_samples
from src.validation.metrics import compute_ner_metrics
from src.data_preparation.bio_decoder import decode_bio
from src.validation.document_predictions import collect_document_predictions


class NERDataset(Dataset):
    def __init__(self, tensors, samples=None, dataset_info=None):
        self.input_ids = tensors['input_ids']
        self.attention_mask = tensors['attention_mask']
        self.labels = tensors['labels']
        self.samples = samples
        self.dataset_info = dataset_info or {}
        self.token_mask = None
        if self.dataset_info.get('architecture') == 'bilstm_crf':
            if samples is None:
                raise ValueError('Для CRF нужны координаты текстовых токенов')
            self.token_mask = token_mask_from_samples(samples, self.input_ids.shape[1]) & self.attention_mask.bool()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        inputs = {'input_ids': self.input_ids[idx], 'attention_mask': self.attention_mask[idx],
                  'labels': self.labels[idx]}
        if self.token_mask is not None:
            inputs['token_mask'] = self.token_mask[idx]
        return inputs


def load_batches(storage, paths, expected_split):
    if not paths:
        raise ValueError(f'Нет батчей {expected_split}')
    batches, samples, info = [], [], None
    for path in paths:
        data = storage.load_batch(path)
        if info is None:
            info = data['dataset_info']
        elif data['dataset_info'] != info:
            raise ValueError('Схемы батчей не совпадают')
        if data['metadata'].get('split') != expected_split:
            raise ValueError(f'Ожидался split {expected_split}: {path}')
        if not data.get('samples') or len(data['samples']) != len(data['tensors']['input_ids']):
            raise ValueError('Отсутствуют тексты, координаты и эталоны: пересоздайте prepared')
        for sample in data['samples']:
            if 'gold_entities' not in sample:
                raise ValueError('Нет исходных эталонов gold_entities')
            if sample.get('split') != expected_split:
                raise ValueError(f'Документ не принадлежит {expected_split}')
        samples.extend(data['samples'])
        batches.append(data['tensors'])
    max_length = max(b['input_ids'].shape[1] for b in batches)
    tensors = {key: torch.cat([F.pad(b[key], (0, max_length - b[key].shape[1]),
               value=-100 if key == 'labels' else 0) for b in batches])
               for key in ('input_ids', 'attention_mask', 'labels')}
    ids = [sample.get('sample_id', sample['doc_id']) for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError(f'Повторные документы в {expected_split}')
    return NERDataset(tensors, samples, info)


def build_dev_metrics(dataset, id2label, decoded=False):
    if not dataset.samples or not dataset.dataset_info.get('entity_types'):
        raise ValueError('Для dev нужны samples и фиксированная схема типов')
    entity_types = dataset.dataset_info['entity_types']
    # Не выбираем checkpoint по неопределённому macro-F1 без эталонных сущностей.
    gold_docs = collect_document_predictions([(s, []) for s in dataset.samples])
    if not compute_ner_metrics(gold_docs, entity_types)['macro_defined']:
        raise ValueError('В dev нет эталонных сущностей для выбора по macro-F1')

    def compute_metrics(evaluation):
        logits = evaluation.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        if decoded:
            if logits.ndim != 2:
                raise ValueError('Для CRF-оценки нужны декодированные Витерби BIO-ID')
            predictions = logits
        else:
            if logits.ndim != 3:
                raise ValueError('Для BERT-оценки нужны логиты по меткам')
            predictions = logits.argmax(axis=-1)
        if len(predictions) != len(dataset.samples):
            raise ValueError('Число dev-предсказаний не соответствует документам')
        docs = []
        for row, sample in enumerate(dataset.samples):
            offsets = sample['offset_mapping']
            if len(offsets) > predictions.shape[1]:
                raise ValueError('Недостаточно dev-предсказаний для координат')
            labels = [int(p) if int(mask) else -100 for p, mask in zip(
                predictions[row, :len(offsets)], dataset.attention_mask[row, :len(offsets)])]
            docs.append((sample, decode_bio(labels, offsets, sample['text'], id2label)))
        metrics = compute_ner_metrics(collect_document_predictions(docs), entity_types)
        return {key: metrics[key] for key in ('macro_f1','micro_f1','micro_precision','micro_recall')}
    return compute_metrics



class CRFTrainer(Trainer):
    """Общий Trainer с выдачей Витерби-ID вместо argmax при оценке CRF."""
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            output = model(**inputs, decode=True)
        loss = output.loss.mean().detach() if output.loss is not None else None
        if prediction_loss_only:
            return loss, None, None
        labels = inputs.get('labels')
        return loss, output.predictions.detach(), labels.detach() if labels is not None else None

def train_model(model, train_dataset, eval_dataset=None, output_dir='./data/models/temp',
                num_epochs=3, batch_size=8, learning_rate=2e-5, logging_steps=10,
                eval_steps=None, save_steps=None, warmup_steps=0, weight_decay=0.0,
                callbacks=None, seed=42, device="auto"):
    """Dev обязателен. Оценка и сохранение выполняются синхронно после эпохи.

    eval_steps/save_steps оставлены как аргументы совместимости; при epoch-стратегии
    они не управляют периодичностью. При равном F1 остаётся первый лучший checkpoint.
    """
    if eval_dataset is None or not len(eval_dataset):
        raise ValueError('Для выбора лучшей модели требуется непустой dev')
    if not len(train_dataset):
        raise ValueError('Train пуст')
    device = resolve_device(device)
    compute_metrics = build_dev_metrics(eval_dataset, model.config.id2label, decoded=is_bilstm_crf(model))
    args = TrainingArguments(output_dir=str(output_dir), num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate, logging_steps=logging_steps,
        warmup_steps=warmup_steps, weight_decay=weight_decay,
        evaluation_strategy='epoch', save_strategy='epoch',
        save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model='macro_f1', greater_is_better=True,
        report_to='none', fp16=False, seed=seed, data_seed=seed, no_cuda=(device == 'cpu'))
    trainer_class = CRFTrainer if is_bilstm_crf(model) else Trainer
    trainer = trainer_class(model=model, args=args, train_dataset=train_dataset,
        eval_dataset=eval_dataset, compute_metrics=compute_metrics, callbacks=callbacks)
    trainer.train()
    return trainer


def train_from_batches(prepared_storage, batch_paths,
                       model_name='DeepPavlov/rubert-base-cased', models_dir='./data/models',
                       num_epochs=1, batch_size=4, learning_rate=2e-5, logging_steps=10,
                       eval_steps=None, save_steps=None, warmup_steps=0, weight_decay=0.0,
                       eval_batch_paths=None, callbacks=None, seed=42, device="auto",
                       architecture="bert", model_parameters=None):
    from .model_factory import create_model, get_device
    train_dataset = load_batches(prepared_storage, batch_paths, 'train')
    dev_dataset = load_batches(prepared_storage, eval_batch_paths, 'dev')
    info = train_dataset.dataset_info
    if dev_dataset.dataset_info != info:
        raise ValueError('Train и dev используют разные схемы/токенизаторы')
    if set(s['doc_id'] for s in train_dataset.samples) & set(s['doc_id'] for s in dev_dataset.samples):
        raise ValueError('Train и dev содержат одинаковые doc_id')
    if info.get('architecture', 'bert') != architecture:
        raise ValueError('Архитектура не соответствует подготовленным модельным входам')
    device = get_device(device)
    set_seed(seed)
    extra = {}
    if architecture == 'bilstm_crf':
        extra = {'architecture': architecture, 'model_parameters': model_parameters,
                 'vocab_size': info.get('vocab_size'), 'padding_idx': info.get('padding_idx',0),
                 'special_token_ids': info.get('special_token_ids',[]),
                 'tokenizer_name': info.get('tokenizer_name'), 'max_seq_length': info.get('max_length',512)}
    model = create_model(model_name=model_name, num_labels=info['num_labels'],
        label2id=info.get('label2id'), id2label=info.get('id2label'), device=device, **extra)
    temp_dir = Path(models_dir) / ('.temp_checkpoints_' + uuid4().hex)
    trainer = train_model(model, train_dataset, dev_dataset, str(temp_dir),
        num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate,
        logging_steps=logging_steps, warmup_steps=warmup_steps, weight_decay=weight_decay,
        callbacks=callbacks, seed=seed, device=device)
    if not trainer.state.best_model_checkpoint:
        raise RuntimeError('Обучение не создало лучший checkpoint')
    history = trainer.state.log_history
    best_checkpoint_name = Path(trainer.state.best_model_checkpoint).name
    best_step = int(best_checkpoint_name.rsplit('-',1)[1])
    evaluations = [record for record in history if 'eval_macro_f1' in record]
    best_record = next(record for record in evaluations if record['step'] == best_step)
    selection = {'metric': 'macro_f1', 'split': 'dev', 'strategy': 'epoch',
        'tie_break': 'first_best', 'best_epoch': best_record['epoch'],
        'best_step': best_step, 'best_macro_f1': float(trainer.state.best_metric),
        'best_checkpoint': best_checkpoint_name, 'evaluation_history': evaluations}
    loss_history = [{'step': x.get('step',0), 'loss': x['loss'], 'epoch': x.get('epoch',0)}
                    for x in history if 'loss' in x]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    version = 'v' + timestamp
    final = Path(models_dir) / version
    # Trainer уже восстановил лучший checkpoint; сохраняем именно trainer.model.
    trainer.save_model(str(final))
    metadata = {'version': version, 'source_batches': [Path(p).name for p in batch_paths],
        'dev_batches': [str(p) for p in eval_batch_paths], 'num_epochs': num_epochs,
        'batch_size': batch_size, 'num_samples': len(train_dataset), 'model_name': model_name,
        'num_labels': info['num_labels'], 'timestamp': timestamp, 'selection': selection,
        'device': device, 'architecture': architecture, 'model_parameters': model_parameters or {},
        'training_parameters': {'learning_rate': learning_rate, 'warmup_steps': warmup_steps,
                                'weight_decay': weight_decay, 'logging_steps': logging_steps, 'seed': seed}}
    (final/'training_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    (final/'loss_history.json').write_text(json.dumps(loss_history,indent=2),encoding='utf-8')
    # Удаляем промежуточные checkpoint только после успешного сохранения результата.
    shutil.rmtree(temp_dir)
    print(f'Сохранена лучшая модель: {final}; эпоха {selection["best_epoch"]}; '
          f'macro-F1={selection["best_macro_f1"]:.4f}')
    return str(final), version, loss_history
