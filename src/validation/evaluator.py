"""Общая оценка сохранённых моделей на неизменном processed split без обучения."""
import hashlib
import json
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from src.config import validate_config
from src.data_analysis.processed_data_storage import ProcessedDataStorage
from src.data_preparation.annotation_flattener import PREPROCESSING_VERSION
from src.inference.inference_pipeline import InferencePipeline
from .metrics import compute_ner_metrics


def validate_documents(pipeline, processed_storage, batch_paths, entity_types, split):
    """Оценить все документы выбранных батчей, включая документы без сущностей.

    Никакие эталоны не передаются pipeline.predict(). Повреждённые/несогласованные
    processed-данные — ошибка, а не основание незаметно пропустить документ.
    """
    if split not in ('dev', 'test'):
        raise ValueError('Для оценки выберите dev или test')
    documents, seen = [], set()
    corpus_hash = hashlib.sha256()
    for path in batch_paths:
        batch = processed_storage.load_batch(Path(path))
        if not batch:
            raise ValueError(f'Пустой processed-батч: {path}')
        for doc in batch:
            doc_id, text = doc.get('doc_id'), doc.get('text')
            if not isinstance(doc_id, (str,int)) or isinstance(doc_id,bool) or doc_id in seen:
                raise ValueError('Идентификаторы оцениваемых документов должны быть уникальны')
            seen.add(doc_id)
            if not isinstance(text,str) or doc.get('split') != split:
                raise ValueError(f'Неверный текст или split документа: {doc_id}')
            if (doc.get('quality',{}).get('has_errors')
                    or doc.get('preprocessing_version') != PREPROCESSING_VERSION):
                raise ValueError(f'Повреждённый документ или неподдерживаемая обработка: {doc_id}')
            gold = doc.get('flat_entities')
            if not isinstance(gold,list):
                raise ValueError(f'Нет плоского эталона flat_entities: {doc_id}')
            if 'parsed_entities' in doc and doc['parsed_entities'] != gold:
                raise ValueError(f'Эталоны parsed_entities и flat_entities различаются: {doc_id}')
            record = {'doc_id':doc_id, 'text':text, 'true_entities':gold, 'pred_entities':[]}
            # Общий оценщик проверяет схему типов и непрерывные границы эталона.
            compute_ner_metrics([record], entity_types)
            intervals = sorted((e['spans'][0]['start'],e['spans'][0]['end'])
                               if 'spans' in e else (e['start'],e['end']) for e in gold)
            if any(left[1] > right[0] for left,right in zip(intervals,intervals[1:])):
                raise ValueError(f'Пересекающиеся эталонные сущности: {doc_id}')
            payload = json.dumps({'doc_id':doc_id,'text':text,'gold':gold},
                                 ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
            corpus_hash.update(len(payload).to_bytes(8,'big'))
            corpus_hash.update(payload)
            record['pred_entities'] = pipeline.predict(text)
            documents.append(record)
        print(f'Оценено документов {len(documents)}')
    if not documents:
        raise ValueError('Нет документов для оценки')
    metrics = compute_ner_metrics(documents, entity_types)
    metrics['entity_predictions'] = documents
    metrics['corpus_sha256'] = corpus_hash.hexdigest()
    return metrics


def _model_signature(path):
    """Идентичность фактически использованного комплекта, а не только его имени."""
    result = {}
    for item in sorted(Path(path).iterdir()):
        if item.is_file():
            digest = hashlib.sha256()
            with item.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):
                    digest.update(block)
            result[item.name] = digest.hexdigest()
    return result


def evaluate_model(config, dataset_version, split, model_version=None):
    """Оценить явно выбранную сохранённую модель без выбора checkpoint по test."""
    from src.pipeline import load_dataset_version
    config = validate_config(config)
    architecture = config['model']['architecture']
    if architecture != 'gliner' and not model_version:
        raise ValueError('Для оценки BERT/BiLSTM-CRF требуется явный --model-version')
    if architecture == 'gliner' and model_version is not None:
        raise ValueError('GLiNER задаётся конфигурацией, --model-version не используется')
    if split not in ('dev','test'):
        raise ValueError('Для оценки выберите dev или test')
    root, manifest = load_dataset_version(config, dataset_version)
    if architecture == 'gliner':
        pipeline = InferencePipeline(architecture='gliner', device=config['device'],
            gliner_model_path=config['model']['name'], type_mapping=config['model']['type_mapping'],
            threshold=config['model']['threshold'], max_length=config['data']['max_seq_length'])
    else:
        pipeline = InferencePipeline(model_version=model_version, device=config['device'],
            models_dir=config['paths']['models'],max_length=config['data']['max_seq_length'])
        if pipeline.architecture != architecture:
            raise ValueError('Архитектура выбранной модели не соответствует --config')
        experiment_path = pipeline.model_path/'experiment.json'
        experiment = json.loads(experiment_path.read_text(encoding='utf-8')) if experiment_path.is_file() else {}
        runtime_path = pipeline.model_path/'model_runtime.json'
        runtime = json.loads(runtime_path.read_text(encoding='utf-8')) if runtime_path.is_file() else experiment
        if runtime.get('dataset_version') != dataset_version:
            raise ValueError('Сохранённая модель обучена на другой или неизвестной версии данных')
        types = {label[2:] for label in pipeline.predictor.id2label.values() if label.startswith(('B-','I-'))}
        if types != set(config['data']['entity_types']):
            raise ValueError('Схема типов выбранной модели не соответствует датасету')
    model_signature = _model_signature(pipeline.model_path)
    storage = ProcessedDataStorage(str(root / split))
    paths = [root/split/name for name in manifest['splits'][split]]
    metrics = validate_documents(pipeline, storage, paths, config['data']['entity_types'], split)
    predictions = metrics.pop('entity_predictions')
    corpus_hash = metrics.pop('corpus_sha256')
    evaluation_id = architecture + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + uuid4().hex[:8]
    output = Path(config['paths']['reports'])/'evaluation'/evaluation_id
    output.mkdir(parents=True,exist_ok=False)
    prediction_path = output/'predictions.json'
    prediction_path.write_text(json.dumps(predictions,ensure_ascii=False,indent=2),encoding='utf-8')
    report = {'format_version':1,'status':'complete','kind':'pretrained_evaluation' if architecture == 'gliner' else 'model_evaluation',
        'evaluation_id':evaluation_id,'timestamp':datetime.now().isoformat(),
        'architecture':architecture,'fine_tuning':architecture == 'bert',
        'trained_on_dataset':architecture != 'gliner','model_version':model_version,'model_path':str(pipeline.model_path),
        'model_files_sha256':model_signature,'dataset_version':dataset_version,'split':split,
        'entity_types':config['data']['entity_types'],'threshold':config['model'].get('threshold'),
        'type_mapping':config['model'].get('type_mapping'),'device':pipeline.device,
        'preprocessing_version':manifest['preprocessing_version'],'source':manifest.get('source'),
        'priority_sha256':manifest['priority_sha256'],'annotation_corrections':manifest.get('annotation_corrections'),
        'corpus_sha256':corpus_hash,'batches':list(manifest['splits'][split]),
        'chunking':{'strategy':'sentences_then_words' if architecture == 'gliner' else 'sentences_then_tokens','overlap':0,'protect_gold_entities':False,
                    'max_seq_length':pipeline.max_length},
        'model_limits':{'max_words':getattr(pipeline.predictor.model.config,'max_len',None),
                        'max_width':getattr(pipeline.predictor.model.config,'max_width',None)},
        'environment':{name:version(name) for name in ('gliner','transformers','torch')},
        'config':config,'metrics':metrics,'predictions_path':str(prediction_path),
        'report_path':str(output/'report.json')}
    if architecture != 'gliner':
        report['training_config'] = experiment.get('config',{}).get('training')
        report['selection'] = experiment.get('selection')
    # report.json записывается последним: summary видит только законченные оценки.
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"{architecture} {split}: macro-F1={metrics['macro_f1']:.4f}; micro-F1={metrics['micro_f1']:.4f}")
    print(f"Отчёт: {report['report_path']}")
    return report
