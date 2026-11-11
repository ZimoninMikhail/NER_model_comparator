"""
Точка входа в MLOps пайплайн.

Поддерживаемые режимы:
    - type-statistics: Сохранение приоритета типов по train
    - prepare: Подготовка версии общих данных
    - train: Обучение на выбранной версии данных
    - update: Полный цикл подготовки и обучения
    - inference: Применение модели к новым данным
    - summary: Генерация отчёта о работе системы
    - benchmark: Измерение характеристик производительности
    - evaluate: Оценка сохранённых моделей на выбранном dev/test
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from uuid import uuid4
from src.config import load_config, validate_config

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.data_collection.storage import RawDataStorage
from src.data_analysis.processed_data_storage import ProcessedDataStorage
from src.data_preparation.prepared_data_storage import PreparedDataStorage
from src.validation.model_registry import ModelRegistry

from src.inference import InferencePipeline




def _make_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    return obj


def _config(config=None):
    return validate_config(config) if config is not None else load_config()


def _report_path(config, category, prefix, suffix='json'):
    folder = Path(config['paths']['reports']) / category
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid4().hex[:8]}.{suffix}"


def _read_texts(file_path):
    path = Path(file_path).expanduser()
    if not path.is_file():
        raise ValueError(f'Входной файл не найден: {path}')
    texts = [line for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not texts:
        raise ValueError(f'Входной файл пуст: {path}')
    return texts


def run_prepare(config=None):
    """Подготовить общую версию документов без обучения и токенизации."""
    from src.config import load_config
    from src.pipeline import prepare_data
    return prepare_data(config if config is not None else load_config())


def run_train(dataset_version, config=None, plot_loss=False, plot_metrics=False,
              plot_entities=False, open_result=False):
    from src.config import load_config
    from src.pipeline import train, load_dataset_version
    from src.visualization.visualizer import NERVisualizer
    config = _config(config)
    report = train(config, dataset_version)
    components = []
    if any((plot_loss, plot_metrics, plot_entities)):
        visualizer = NERVisualizer(config['paths']['visualization'])
        if plot_loss and report['loss_history']:
            history = report['loss_history']
            path = visualizer.plot_loss_curve([x['loss'] for x in history],
                                             [x['step'] for x in history])
            components.append({'type': 'plot', 'path': path, 'title': 'Кривая обучения'})
        if plot_metrics:
            path = visualizer.plot_metrics(report['metrics'])
            components.append({'type': 'plot', 'path': path, 'title': 'Метрики на dev'})
        if plot_entities:
            root, manifest = load_dataset_version(config, dataset_version)
            counts = {}
            for name in manifest['splits']['train']:
                for doc_path in (root / 'train' / name).glob('doc_*.json'):
                    doc = json.loads(doc_path.read_text(encoding='utf-8'))
                    for entity in doc.get('parsed_entities', []):
                        kind = entity['type']
                        counts[kind] = counts.get(kind, 0) + 1
            if counts:
                path = visualizer.plot_entity_distribution(counts)
                components.append({'type': 'plot', 'path': path, 'title': 'Сущности train'})
        if open_result and components:
            visualizer.create_dashboard(components, open_result=True)
    print(f"Обучение завершено: {report['version']}; данные: {dataset_version}")
    return report


def run_update(plot_loss=False, plot_metrics=False, plot_entities=False,
               open_result=False, config=None):
    """Полный цикл: подготовить версию и обучить на ней модель."""
    from src.config import load_config
    config = _config(config)
    if config['model']['architecture'] == 'gliner':
        raise ValueError('GLiNER используется без дообучения; режим update недоступен')
    dataset_version = run_prepare(config)
    return run_train(dataset_version, config=config, plot_loss=plot_loss,
                     plot_metrics=plot_metrics, plot_entities=plot_entities,
                     open_result=open_result)


def _create_inference_pipeline(config=None, model_version=None):
    from src.config import load_config, validate_config
    config = validate_config(config) if config is not None else load_config()
    if config['model']['architecture'] == 'gliner':
        return InferencePipeline(device=config['device'], architecture='gliner',
            gliner_model_path=config['model']['name'], model_version=model_version,
            type_mapping=config['model']['type_mapping'], threshold=config['model']['threshold'],
            max_length=config['data']['max_seq_length'])
    return InferencePipeline(device=config['device'],
        models_dir=config['paths']['models'], max_length=config['data']['max_seq_length'],
        model_version=model_version)


def run_inference(file_path=None, highlight: bool = False, plots: bool = False, open_result: bool = False, config=None, text=None, model_version=None):
    """
    Применение модели к данным из файла.
    
    Args:
        file_path: Путь к файлу с текстами (построчно)
        highlight: Подсветка сущностей в консоли
        plots: Построение графика распределения сущностей
        open_result: Открыть дашборд в браузере
    """
    print("\n" + "="*60)
    print("ЗАПУСК INFERENCE")
    print("="*60)
    
    config = _config(config)
    if (file_path is None) == (text is None):
        raise ValueError('Для inference укажите ровно один источник: файл или текст')
    if text is not None:
        if not text.strip():
            raise ValueError('Текст для inference пуст')
        texts = [text]
    else:
        texts = _read_texts(file_path)

    print(f"Загружено текстов: {len(texts)}")
    
    pipeline = _create_inference_pipeline(config, model_version=model_version)
    from src.visualization.visualizer import NERVisualizer
    visualizer = NERVisualizer(config['paths']['visualization'])
    
    results = []
    all_entities = []
    components = []
    
    for i, text in enumerate(texts):
        entities = pipeline.predict(text)
        results.append({
            'text': text,
            'entities': entities,
            'index': i
        })
        
        all_entities.extend(entities)
        
        if highlight and not open_result:
            colored_text = visualizer.highlight_entities_terminal(text, entities)
            print(f"\n{i+1}. {colored_text}")
        elif not open_result:
            print(f"\n{i+1}. {text[:100]}...")
            if entities:
                for e in entities:
                    print(f"     {e['type']}: '{e['text']}'")
            else:
                print("     Сущности не найдены")
    
    predictions_dir = Path(config['paths']['predictions'])
    predictions_dir.mkdir(parents=True, exist_ok=True)
    output_path = predictions_dir / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid4().hex[:8]}.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        results_serializable = _make_serializable(results)
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    
    print(f"\nРезультаты сохранены в: {output_path}")
    
    if plots and all_entities:
        plot_path = visualizer.plot_entity_distribution(
            all_entities, 
            title="Распределение сущностей в текстах",
            save=True,
            open_result=False
        )
        if plot_path:
            components.append({
                'type': 'plot',
                'path': plot_path,
                'title': 'Распределение сущностей по типам',
                'description': f'Всего найдено: {len(all_entities)} сущностей'
            })
    
    if open_result and highlight:
        for i, text in enumerate(texts):
            entities = results[i]['entities']
            if entities:
                html_content = visualizer.highlight_entities_html(text, entities)
                components.append({
                    'type': 'html',
                    'content': html_content,
                    'title': f'Текст {i+1}',
                    'description': f'Распознано {len(entities)} сущностей'
                })
    
    if open_result and components:
        visualizer.create_dashboard(components, open_result=True)
    
    return str(output_path)

def run_evaluate(dataset_version, split, config=None, plot_metrics=False, open_result=False, model_version=None):
    from src.validation.evaluator import evaluate_model
    report = evaluate_model(_config(config), dataset_version, split, model_version=model_version)
    label = {"bert":"BERT", "bilstm_crf":"BiLSTM-CRF", "gliner":"GLiNER"}[report["architecture"]]
    if plot_metrics:
        from src.visualization.visualizer import NERVisualizer
        visualizer = NERVisualizer(report['config']['paths']['visualization'])
        path = visualizer.plot_metrics(report['metrics'], title=f"{label}: {split}")
        if open_result:
            visualizer.create_dashboard([{'type':'plot','path':path,'title':f'{label}: {split}'}], open_result=True)
    return report


def run_summary(plots: bool = False, compare: bool = False, open_result: bool = False, plot_metrics: bool = False, config=None):
    print("\n" + "="*60)
    print("ГЕНЕРАЦИЯ ОТЧЁТА")
    print("="*60)
    
    config = _config(config)
    report = {
        'timestamp': datetime.now().isoformat(),
        'stages': {}
    }
    
    raw_storage = RawDataStorage(config['paths']['raw'])
    raw_batches = sorted(p.parent for p in raw_storage.base_path.rglob("metadata.json"))
    report['stages']['raw'] = {
        'num_batches': len(raw_batches),
        'batches': [str(b.relative_to(raw_storage.base_path)) for b in raw_batches]
    }
    if raw_batches:
        latest_raw = raw_storage.get_metadata(raw_batches[-1])
        report['stages']['raw']['latest_stats'] = {
            'num_documents': latest_raw.get('num_documents'),
            'total_entities': latest_raw.get('total_entities'),
            'entity_types': list(latest_raw.get('entity_type_distribution', {}).keys())[:10]
        }
    
    processed_storage = ProcessedDataStorage(config['paths']['processed'])
    processed_batches = sorted(p.parent for p in processed_storage.base_path.rglob("metadata.json"))
    report['stages']['processed'] = {
        'num_batches': len(processed_batches),
        'batches': [str(b.relative_to(processed_storage.base_path)) for b in processed_batches]
    }
    if processed_batches:
        latest_processed = processed_storage.get_metadata(processed_batches[-1])
        report['stages']['processed']['latest_stats'] = {
            'num_documents': latest_processed.get('num_documents'),
            'consistency_ratio': latest_processed.get('consistency_ratio'),
            'docs_removed': latest_processed.get('docs_removed')
        }
    
    prepared_storage = PreparedDataStorage(config['paths']['prepared'])
    prepared_batches = sorted(p.parent for p in prepared_storage.base_path.rglob("metadata.json"))
    report['stages']['prepared'] = {
        'num_batches': len(prepared_batches),
        'batches': [str(b.relative_to(prepared_storage.base_path)) for b in prepared_batches]
    }
    if prepared_batches:
        latest_prepared = {'metadata': json.loads((prepared_batches[-1] / 'metadata.json').read_text(encoding='utf-8'))}
        report['stages']['prepared']['latest_stats'] = {
            'num_samples': latest_prepared['metadata'].get('num_samples'),
            'max_length': latest_prepared['metadata'].get('max_length')
        }
    
    registry = ModelRegistry(config['paths']['models'])
    latest_model = registry.get_latest()
    all_versions = registry.list_versions()
    
    if latest_model:
        report['models'] = {
            'latest_version': latest_model['version'],
            'latest_metrics': latest_model.get('metrics', {}),
            'all_versions': all_versions
        }
    else:
        report['models'] = {'status': 'no_models_trained'}
    
    from src.validation.evaluation_reports import load_evaluation_reports, comparison_groups
    evaluations = load_evaluation_reports(config['paths']['reports'])
    report['evaluations'] = [{key:item[key] for key in (
        'evaluation_id','architecture','dataset_version','split','fine_tuning',
        'timestamp','metrics','report_path')} for item in evaluations]

    report_path = _report_path(config, 'summary', 'summary')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("\nОТЧЁТ:")
    print(f"   Время: {report['timestamp']}")
    print(f"   Raw батчей: {report['stages']['raw']['num_batches']}")
    print(f"   Processed батчей: {report['stages']['processed']['num_batches']}")
    print(f"   Prepared батчей: {report['stages']['prepared']['num_batches']}")
    
    if 'models' in report and report['models'].get('latest_version'):
        print(f"   Последняя модель: {report['models']['latest_version']}")
        metrics = report['models']['latest_metrics']
        if metrics:
            print(f"     Macro-F1: {metrics.get('macro_f1', 'нет в старом отчёте')}")
            print(f"     Micro-F1: {metrics.get('micro_f1', 'нет в старом отчёте')}")
            print(f"     Micro-P/R: {metrics.get('micro_precision', 'N/A')} / {metrics.get('micro_recall', 'N/A')}")

    if evaluations:
        latest_evaluation = evaluations[-1]
        print(f"   {latest_evaluation['architecture']} ({latest_evaluation['split']}): "
              f"macro-F1={latest_evaluation['metrics']['macro_f1']:.4f}")

    print(f"\nОтчёт сохранён в: {report_path}")
    
    from src.visualization.visualizer import NERVisualizer
    visualizer = NERVisualizer(config['paths']['visualization'])
    components = []
    
    if plot_metrics and latest_model:
        latest_metrics = latest_model.get('metrics', {})
        if 'macro_f1' in latest_metrics:
            path = visualizer.plot_metrics(latest_metrics)
            components.append({'type': 'plot', 'path': path, 'title': 'Macro- и micro-метрики'})
        else:
            print('[INFO] Старый отчёт без macro_f1: график метрик пропущен')

    if plot_metrics and evaluations:
        item = evaluations[-1]
        path = visualizer.plot_metrics(item['metrics'], title=f"{item['architecture']}: {item['split']}")
        components.append({'type':'plot','path':path,'title':f"{item['architecture']}: {item['split']}"})

    if compare:
        entries = [registry.get_version(version) for version in all_versions]
        groups = comparison_groups([entry for entry in entries if entry], evaluations)
        for group in groups:
            path = visualizer.plot_model_comparison(group['models'],save=True,open_result=False)
            if path:
                title = f"Сравнение: {group['dataset_version'] or 'неизвестная версия'}, {group['split']}"
                print(title)
                components.append({'type':'plot','path':path,'title':title,
                    'description':f"Результатов: {len(group['models'])}; одинаковые версия, split и схема типов"})

    if plots:
        all_entities = []
        for batch_dir in processed_batches:
            metadata = processed_storage.get_metadata(batch_dir)
            dist = metadata.get('entity_type_distribution', {})
            for ent_type, count in dist.items():
                for _ in range(count):
                    all_entities.append({'type': ent_type})
        
        if all_entities:
            path = visualizer.plot_entity_distribution(
                all_entities,
                title="Распределение сущностей в датасете",
                save=True,
                open_result=False
            )
            if path:
                components.append({
                    'type': 'plot',
                    'path': path,
                    'title': 'Распределение сущностей',
                    'description': f'Всего сущностей: {len(all_entities)}'
                })
    
    if open_result and components:
        visualizer.create_dashboard(components, open_result=True)
    elif open_result and not components:
        print("[INFO] Нет компонентов для отображения в дашборде")
    
    return str(report_path)

def run_benchmark(file_path=None, plots=False, open_result=False, config=None, warmup=None, repeats=None, model_version=None):
    from src.benchmark import measure_inference
    config = _config(config)
    if warmup is not None:
        config['benchmark']['warmup_runs'] = warmup
    if repeats is not None:
        config['benchmark']['repeats'] = repeats
    config = validate_config(config)
    texts = _read_texts(file_path) if file_path is not None else [
        'Москва является столицей России.', 'Париж — столица Франции.',
        'Озеро Байкал находится в Сибири.', 'Берлин расположен на реке Шпрее.']
    pipeline = _create_inference_pipeline(config, model_version=model_version)
    report = measure_inference(pipeline, texts, **config['benchmark'])
    report['timestamp'] = datetime.now().isoformat()
    path = _report_path(config, 'benchmark', 'benchmark')
    path.write_text(json.dumps(_make_serializable(report), ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Модель: {report['model_name']}; устройство: {report['device']}")
    print(f"Среднее: {report['inference']['avg_time_ms']:.2f} мс; "
          f"median: {report['inference']['median_time_ms']:.2f} мс; "
          f"p95: {report['inference']['p95_time_ms']:.2f} мс")
    print(f"Пропускная способность: {report['throughput']:.2f} текстов/сек")
    print(f"Сэмплированный пик RSS процесса: {report['memory']['cpu_rss_sampled_peak_mb']:.2f} МБ")
    print(f'Отчёт: {path}')
    if plots:
        from src.visualization.visualizer import NERVisualizer
        visualizer = NERVisualizer(config['paths']['visualization'])
        plot = visualizer.plot_benchmark(report)
        if open_result and plot:
            visualizer.create_dashboard([{'type':'plot','path':plot,'title':'Производительность'}], open_result=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description='NER experiment pipeline')
    parser.add_argument('-m', '--mode', required=True, choices=[
        'type-statistics','prepare','train','update','inference','summary','benchmark','evaluate'])
    parser.add_argument('--dataset', choices=['nerel','nerel-bio'], help='Профиль локального датасета')
    parser.add_argument('--config', help='Путь к YAML-конфигурации')
    parser.add_argument('--device', choices=['auto','cpu','cuda'])
    parser.add_argument('--model-version', help='ID зарегистрированной модели для inference/benchmark/evaluate')
    parser.add_argument('--dataset-version', help='ID версии данных для train/evaluate')
    parser.add_argument('--split', choices=['dev','test'], help='Обязательный split для evaluate')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('-f','--file', help='Входной файл, путь относительно текущей папки')
    source.add_argument('-t','--text', help='Один текст для inference, без временного файла')
    parser.add_argument('-l','--loss', action='store_true')
    parser.add_argument('-mt','--metrics', action='store_true')
    parser.add_argument('-p','--plots', action='store_true')
    parser.add_argument('-c','--compare', action='store_true')
    parser.add_argument('-b','--bench', action='store_true')
    parser.add_argument('-hi','--highlight', action='store_true')
    parser.add_argument('-o','--open', action='store_true')
    parser.add_argument('--warmup', type=int, help='Число прогревочных проходов benchmark')
    parser.add_argument('--repeats', type=int, help='Число измеряемых проходов benchmark')
    args = parser.parse_args(argv)
    allowed = {
        'model_version': {'inference','benchmark','evaluate'},
        'dataset': {'type-statistics','prepare','train','update','evaluate'},
        'loss': {'train','update'}, 'metrics': {'train','update','summary','evaluate'},
        'plots': {'train','update','inference','summary'}, 'compare': {'summary'},
        'bench': {'benchmark'}, 'highlight': {'inference'},
        'device': {'train','update','inference','benchmark','evaluate'},
        'file': {'inference','benchmark'}, 'text': {'inference'},
        'split': {'evaluate'}, 'dataset_version': {'train','evaluate'}, 'warmup': {'benchmark'}, 'repeats': {'benchmark'}}
    for flag, modes in allowed.items():
        value = getattr(args, flag)
        if value is not None and value is not False and args.mode not in modes:
            parser.error(f'--{flag.replace("_", "-")} не поддерживается в режиме {args.mode}')
    if args.mode in ('train','evaluate') and not args.dataset_version:
        parser.error(f'для {args.mode} требуется --dataset-version')
    if args.mode == 'evaluate' and args.split is None:
        parser.error('для evaluate требуется --split dev или --split test')
    if args.mode == 'inference' and args.file is None and args.text is None:
        parser.error('для inference требуется --file или --text')
    if args.open and not any([args.loss,args.metrics,args.plots,args.compare,args.bench,args.highlight]):
        parser.error('--open требует подходящего режиму флага визуализации')
    try:
        config = load_config(args.config, dataset=args.dataset)
        if config['model']['architecture'] == 'gliner' and args.mode in ('train','update'):
            raise ValueError('GLiNER используется без дообучения; train/update недоступны')
        if args.mode in ('train','evaluate') and args.dataset is None:
            from src.config import config_from_dataset_manifest
            config = config_from_dataset_manifest(config, args.dataset_version)
        if args.device is not None:
            config['device'] = args.device
        if args.mode == 'evaluate':
            return run_evaluate(args.dataset_version, args.split, config=config,
                plot_metrics=args.metrics, open_result=args.open, model_version=args.model_version)
        if args.mode == 'inference':
            return run_inference(file_path=args.file, text=args.text, config=config,
                highlight=args.highlight, plots=args.plots, open_result=args.open, model_version=args.model_version)
        if args.mode == 'type-statistics':
            from src.pipeline import calculate_type_statistics
            return calculate_type_statistics(config)
        if args.mode == 'prepare':
            return run_prepare(config)
        if args.mode == 'train':
            return run_train(args.dataset_version, config=config, plot_loss=args.loss,
                plot_metrics=args.metrics, plot_entities=args.plots, open_result=args.open)
        if args.mode == 'update':
            return run_update(config=config, plot_loss=args.loss, plot_metrics=args.metrics,
                plot_entities=args.plots, open_result=args.open)
        if args.mode == 'summary':
            return run_summary(config=config, plots=args.plots, compare=args.compare,
                plot_metrics=args.metrics, open_result=args.open)
        if args.mode == 'benchmark':
            return run_benchmark(config=config, file_path=args.file, plots=args.bench,
                open_result=args.open, warmup=args.warmup, repeats=args.repeats, model_version=args.model_version)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(2, f'Ошибка: {error}\n')


if __name__ == '__main__':
    main()
