"""
Точка входа в MLOps пайплайн.

Поддерживаемые режимы:
    - update: Полный цикл обработки данных и дообучение модели
    - inference: Применение модели к новым данным
    - summary: Генерация отчёта о работе системы
    - benchmark: Измерение характеристик производительности
"""

import argparse
import sys
import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.data_collection.loader import StreamingDataLoader
from src.data_collection.storage import RawDataStorage
from src.data_collection.metadata_calculator import calculate_batch_metadata

from src.data_analysis.parser import parse_batch_entities
from src.data_analysis.type_filter import filter_batch_geo, get_filter_statistics
from src.data_analysis.quality_checker import check_document_quality, calculate_batch_quality
from src.data_analysis.quality_filter import filter_batch_by_quality, get_quality_filter_statistics
from src.data_analysis.processed_data_storage import ProcessedDataStorage

from src.data_preparation.tokenizer_setup import load_tokenizer
from src.data_preparation.bio_encoder import create_label_mapping, encode_batch
from src.data_preparation.prepared_data_storage import PreparedDataStorage

from src.training.trainer import train_from_batches
from src.validation.model_registry import ModelRegistry
from src.validation.validator import validate_model

from src.inference import InferencePipeline


GEO_TYPES = ["COUNTRY", "CITY", "LOCATION", "STATE_OR_PROV", "DISTRICT"]
MAX_LENGTH = 512
MODEL_NAME = "DeepPavlov/rubert-base-cased"
MIN_CONSISTENCY = 0.95
MIN_ENTITIES = 1

#params
NUM_EPOCHS = 3
BATCH_SIZE = 2
TRAIN_SPLIT = 0.8

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


def _save_validation_report(metrics: dict, version: str):
    """Сохраняет отчёт о валидации в reports/validation/"""
    report_dir = Path("reports/validation")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = report_dir / f"val_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(_make_serializable(metrics), f, indent=2, ensure_ascii=False)
    
    print(f"   Отчёт валидации сохранён: {report_path}")
    return str(report_path)


def _save_training_history(history: dict, version: str):
    """Сохраняет историю обучения"""
    report_dir = Path("reports/training")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = report_dir / f"train_history_{version}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(_make_serializable(history), f, indent=2, ensure_ascii=False)
    
    print(f"   История обучения сохранена: {report_path}")
    return str(report_path)

def run_update(plot_loss: bool = False, plot_metrics: bool = False, 
               plot_entities: bool = False, open_result: bool = False):
    print("\n" + "="*60)
    print("ЗАПУСК UPDATE ПАЙПЛАЙНА")
    print("="*60)
    
    raw_storage = RawDataStorage("data/raw")
    processed_storage = ProcessedDataStorage("data/processed")
    prepared_storage = PreparedDataStorage("data/prepared")
    
    raw_storage.clear()
    processed_storage.clear()
    prepared_storage.clear()
    
    print("\n1. Загрузка данных...")
    loader = StreamingDataLoader(batch_size=32, shuffle=True)
    loader.load_train()
    print(f"   Всего документов: {len(loader)}")
    
    all_entities_for_plot = []
    
    for batch_num, batch in enumerate(loader):
        print(f"\n2. Обработка батча {batch_num + 1}...")
        
        raw_metadata = calculate_batch_metadata(batch)
        raw_path = raw_storage.save_batch(batch, raw_metadata)
        
        parsed_batch = parse_batch_entities(batch)
        geo_batch = filter_batch_geo(parsed_batch, remove_empty_docs=True)
        geo_stats = get_filter_statistics(geo_batch)
        
        for doc in geo_batch:
            for ent in doc.get('parsed_entities', []):
                all_entities_for_plot.append({'type': ent.get('type')})
        
        quality_batch = [check_document_quality(doc) for doc in geo_batch]
        quality_metrics = calculate_batch_quality(quality_batch)
        
        final_batch = filter_batch_by_quality(
            quality_batch,
            min_consistency=MIN_CONSISTENCY,
            min_entities=MIN_ENTITIES
        )
        quality_filter_stats = get_quality_filter_statistics(final_batch)
        
        processed_metadata = {
            "source_batch": raw_path.name,
            **geo_stats,
            **quality_metrics,
            **quality_filter_stats
        }
        processed_storage.save_batch(final_batch, processed_metadata)
        
        if final_batch:
            tokenizer = load_tokenizer(MODEL_NAME, MAX_LENGTH)
            label2id, id2label = create_label_mapping(GEO_TYPES)
            encoded_batch = encode_batch(final_batch, tokenizer, label2id, MAX_LENGTH)
            
            dataset_info = {
                'label2id': label2id,
                'id2label': id2label,
                'num_labels': len(label2id),
                'entity_types': GEO_TYPES,
                'tokenizer_name': MODEL_NAME,
                'max_length': MAX_LENGTH
            }
            
            prepared_metadata = {
                "source_batch": raw_path.name,
                "batch_id": final_batch[0].get('batch_id', 0),
                "num_samples": len(encoded_batch)
            }
            prepared_storage.save_batch(encoded_batch, prepared_metadata, dataset_info)
    
    print("\n3. Разделение на train/val...")
    
    all_batch_paths = prepared_storage.list_batches()
    print(f"   Найдено prepared батчей: {len(all_batch_paths)}")
    
    if not all_batch_paths:
        print("Ошибка: нет подготовленных данных для обучения")
        return False

    split_idx = int(len(all_batch_paths) * TRAIN_SPLIT)
    train_paths = all_batch_paths[:split_idx]
    val_paths = all_batch_paths[split_idx:]
    
    print(f"   Train батчей: {len(train_paths)}")
    print(f"   Val батчей: {len(val_paths)}")
    
    print("\n4. Обучение модели...")
    print(f"   Эпохи: {NUM_EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    
    model_path, version, loss_history = train_from_batches(
        prepared_storage=prepared_storage,
        batch_paths=train_paths,
        model_name=MODEL_NAME,
        models_dir="./data/models",
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    registry = ModelRegistry()
    registry.register_model(
        model_path=model_path,
        metrics=None,
        metadata={
            'status': 'trained',
            'train_batches': len(train_paths),
            'val_batches': len(val_paths),
            'num_epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE
        }
    )
    
    print("\n5. Валидация модели...")
    
    metrics = validate_model(
        model_path=Path(model_path),
        prepared_storage=prepared_storage,
        batch_paths=val_paths,
        device="cpu"
    )
    
    registry.update_metrics(version, metrics)
    
    _save_validation_report(metrics, version)
    
    first_batch_data = prepared_storage.load_batch(all_batch_paths[0])
    dataset_info = first_batch_data['dataset_info']
    
    training_history = {
        'version': version,
        'num_epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'train_batches': len(train_paths),
        'val_batches': len(val_paths),
        'metrics': metrics,
        'dataset_info': {
            'num_labels': dataset_info.get('num_labels'),
            'entity_types': dataset_info.get('entity_types'),
            'max_length': dataset_info.get('max_length')
        },
        'timestamp': datetime.now().isoformat()
    }
    _save_training_history(training_history, version)
    
    from src.visualization.visualizer import NERVisualizer
    visualizer = NERVisualizer()
    components = []
    
    if plot_entities and all_entities_for_plot:
        path = visualizer.plot_entity_distribution(
            all_entities_for_plot,
            title="Распределение сущностей в обучающих данных",
            save=True,
            open_result=False
        )
        if path:
            components.append({
                'type': 'plot',
                'path': path,
                'title': 'Распределение сущностей по типам',
                'description': f'Всего сущностей: {len(all_entities_for_plot)}'
            })
    
    if plot_metrics and metrics:
        path = visualizer.plot_metrics(
            metrics,
            title="Метрики модели на валидации",
            save=True,
            open_result=False
        )
        if path:
            components.append({
                'type': 'plot',
                'path': path,
                'title': 'Метрики качества модели',
                'description': f'F1: {metrics.get("f1", 0):.3f}'
            })
    
    if plot_loss and loss_history:
        losses = [item['loss'] for item in loss_history]
        steps = [item['step'] for item in loss_history]
        path = visualizer.plot_loss_curve(
            losses,
            steps=steps,
            title="Кривая обучения",
            save=True,
            open_result=False
        )
        if path:
            components.append({
                'type': 'plot',
                'path': path,
                'title': 'Кривая обучения',
                'description': f'Финальный loss: {losses[-1]:.4f}'
            })
    
    if open_result and components:
        visualizer.create_dashboard(components, open_result=True)
    
    print(f"\n" + "="*60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("="*60)
    print(f"   Версия модели: {version}")
    print(f"   Train батчей: {len(train_paths)}")
    print(f"   Val батчей: {len(val_paths)}")
    print(f"   F1: {metrics.get('f1', 0):.4f}")
    print(f"   Precision: {metrics.get('precision', 0):.4f}")
    print(f"   Recall: {metrics.get('recall', 0):.4f}")
    print("="*60)
    
    return True

def run_inference(file_path: str, highlight: bool = False, plots: bool = False, open_result: bool = False):
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
    
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"Ошибка: файл {file_path} не найден")
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    print(f"Загружено текстов: {len(texts)}")
    
    pipeline = InferencePipeline(
        model_version=None,
        tokenizer_name=MODEL_NAME,
        device="cpu",
        max_length=MAX_LENGTH
    )
    from src.visualization.visualizer import NERVisualizer
    visualizer = NERVisualizer()
    
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
            print(f"\n{i+1}. {colored_text[:100]}...")
        elif not open_result:
            print(f"\n{i+1}. {text[:100]}...")
            if entities:
                for e in entities:
                    print(f"     {e['type']}: '{e['text']}'")
            else:
                print("     Сущности не найдены")
    
    predictions_dir = Path("data/predictions")
    predictions_dir.mkdir(parents=True, exist_ok=True)
    output_path = predictions_dir / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
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

def run_summary(plots: bool = False, compare: bool = False, open_result: bool = False):
    print("\n" + "="*60)
    print("ГЕНЕРАЦИЯ ОТЧЁТА")
    print("="*60)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'stages': {}
    }
    
    raw_storage = RawDataStorage("data/raw")
    raw_batches = raw_storage.list_batches()
    report['stages']['raw'] = {
        'num_batches': len(raw_batches),
        'batches': [b.name for b in raw_batches]
    }
    if raw_batches:
        latest_raw = raw_storage.get_metadata(raw_batches[-1])
        report['stages']['raw']['latest_stats'] = {
            'num_documents': latest_raw.get('num_documents'),
            'total_entities': latest_raw.get('total_entities'),
            'entity_types': list(latest_raw.get('entity_type_distribution', {}).keys())[:10]
        }
    
    processed_storage = ProcessedDataStorage("data/processed")
    processed_batches = processed_storage.list_batches()
    report['stages']['processed'] = {
        'num_batches': len(processed_batches),
        'batches': [b.name for b in processed_batches]
    }
    if processed_batches:
        latest_processed = processed_storage.get_metadata(processed_batches[-1])
        report['stages']['processed']['latest_stats'] = {
            'num_documents': latest_processed.get('num_documents'),
            'consistency_ratio': latest_processed.get('consistency_ratio'),
            'docs_removed': latest_processed.get('docs_removed')
        }
    
    prepared_storage = PreparedDataStorage("data/prepared")
    prepared_batches = prepared_storage.list_batches()
    report['stages']['prepared'] = {
        'num_batches': len(prepared_batches),
        'batches': [b.name for b in prepared_batches]
    }
    if prepared_batches:
        latest_prepared = prepared_storage.load_batch(prepared_batches[-1])
        report['stages']['prepared']['latest_stats'] = {
            'num_samples': latest_prepared['metadata'].get('num_samples'),
            'max_length': latest_prepared['metadata'].get('max_length')
        }
    
    registry = ModelRegistry()
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
    
    report_dir = Path("reports/summary")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
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
            print(f"     F1: {metrics.get('f1', 'N/A')}")
            print(f"     Precision: {metrics.get('precision', 'N/A')}")
            print(f"     Recall: {metrics.get('recall', 'N/A')}")
    
    print(f"\nОтчёт сохранён в: {report_path}")
    
    from src.visualization.visualizer import NERVisualizer
    visualizer = NERVisualizer()
    components = []
    
    if compare and all_versions:
        models_data = []
        for version in all_versions:
            model_info = registry.get_version(version)
            if model_info:
                metrics = model_info.get('metrics', {})
                if metrics:
                    models_data.append({
                        'version': version,
                        'f1': metrics.get('f1', 0),
                        'precision': metrics.get('precision', 0),
                        'recall': metrics.get('recall', 0),
                        'timestamp': model_info.get('timestamp', '')
                    })
        
        if models_data:
            models_data.sort(key=lambda x: x.get('timestamp', ''))
            path = visualizer.plot_model_comparison(
                models_data,
                save=True,
                open_result=False
            )
            if path:
                components.append({
                    'type': 'plot',
                    'path': path,
                    'title': 'Сравнение версий моделей',
                    'description': f'Всего моделей: {len(models_data)}'
                })
    
    if plots:
        all_entities = []
        for batch_dir in processed_storage.list_batches():
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

def run_benchmark(file_path=None, plots=False, open_result=False):
    import time
    import psutil
    import tracemalloc
    
    print("\n" + "="*60)
    print("БЕНЧМАРКИНГ СИСТЕМЫ")
    print("="*60)
    
    if file_path and Path(file_path).exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"[INFO] Загружено {len(texts)} текстов из {file_path}")
    else:
        default_texts = [
            "Москва является столицей России.",
            "Президент России Владимир Путин посетил Санкт-Петербург.",
            "Компания Яндекс находится в Москве на улице Льва Толстого.",
            "Озеро Байкал находится в Сибири, недалеко от Иркутска.",
            "Париж - столица Франции, известная Эйфелевой башней.",
            "Берлин расположен на реке Шпрее.",
            "Лондон - столица Великобритании.",
            "Токио находится в Японии."
        ]
        texts = default_texts
        print(f"[INFO] Используются тестовые тексты по умолчанию ({len(texts)} шт.)")
    
    print("\n[INFO] Инициализация пайплайна инференса...")
    pipeline = InferencePipeline(
        model_version=None,
        tokenizer_name=MODEL_NAME,
        device="cpu",
        max_length=MAX_LENGTH
    )
    
    print("\n[INFO] Выполнение бенчмаркинга инференса...")
    times = []
    memory_samples = []
    results = []
    
    for i, text in enumerate(texts):
        tracemalloc.start()
        mem_before = tracemalloc.get_traced_memory()[1]
        
        start = time.time()
        entities = pipeline.predict(text)
        elapsed = time.time() - start
        
        mem_after = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        
        times.append(elapsed)
        memory_samples.append(mem_after - mem_before)
        results.append({'text': text, 'entities': entities, 'time_sec': elapsed})
        
        print(f"  Текст {i+1}: {elapsed:.4f} сек, память: {(mem_after - mem_before) / 1024:.2f} KB")
    
    avg_time = sum(times) / len(times)
    avg_memory = sum(memory_samples) / len(memory_samples) / 1024
    total_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    benchmark_results = {
        "timestamp": datetime.now().isoformat(),
        "model_name": MODEL_NAME,
        "num_texts": len(texts),
        "inference": {
            "avg_time_sec": avg_time,
            "avg_time_ms": avg_time * 1000,
            "min_time_sec": min(times),
            "max_time_sec": max(times),
            "std_time_sec": float(np.std(times)) if len(times) > 1 else 0,
            "times_per_text": times
        },
        "memory": {
            "avg_inference_memory_kb": avg_memory,
            "total_process_memory_mb": total_memory
        },
        "throughput": len(texts) / sum(times),
        "predictions": [
            {"text": t[:100], "num_entities": len(e)} 
            for t, e in zip(texts, [r['entities'] for r in results])
        ]
    }
    
    os.makedirs("reports/monitoring", exist_ok=True)
    report_path = f"reports/monitoring/benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(_make_serializable(benchmark_results), f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ БЕНЧМАРКИНГА")
    print("="*60)
    print(f"  Модель: {MODEL_NAME}")
    print(f"  Количество текстов: {len(texts)}")
    print(f"  Среднее время инференса: {avg_time*1000:.2f} мс")
    print(f"  Мин. время инференса: {min(times)*1000:.2f} мс")
    print(f"  Макс. время инференса: {max(times)*1000:.2f} мс")
    print(f"  Пропускная способность: {benchmark_results['throughput']:.2f} текстов/сек")
    print(f"  Средняя память на инференс: {avg_memory:.2f} KB")
    print(f"  Память процесса (RSS): {total_memory:.1f} MB")
    print(f"\n[INFO] Отчёт сохранён: {report_path}")
    
    # Визуализация
    if plots:
        from src.visualization.visualizer import NERVisualizer
        visualizer = NERVisualizer()
        components = []
        
        path = visualizer.plot_benchmark(
            benchmark_results,
            save=True,
            open_result=False
        )
        if path:
            components.append({
                'type': 'plot',
                'path': path,
                'title': 'Производительность инференса',
                'description': f'Среднее время: {avg_time*1000:.1f} мс, {benchmark_results["throughput"]:.1f} текстов/сек'
            })
        
        if open_result and components:
            visualizer.create_dashboard(components, open_result=True)
    
    return benchmark_results


def main():
    parser = argparse.ArgumentParser(description="NER MLOps Pipeline")
    
    parser.add_argument("-m", "--mode", required=True,
                        choices=["inference", "update", "summary", "benchmark"])
    parser.add_argument("-f", "--file", help="Путь к файлу")
    parser.add_argument("-t", "--text", help="Текст для обработки")
    
    parser.add_argument("-l", "--loss", action="store_true", help="Кривая обучения (update)")
    parser.add_argument("-mt", "--metrics", action="store_true", help="Метрики модели (update, summary)")
    parser.add_argument("-p", "--plots", action="store_true", help="Распределение сущностей")
    parser.add_argument("-c", "--compare", action="store_true", help="Сравнение моделей (summary)")
    parser.add_argument("-b", "--bench", action="store_true", help="График производительности (benchmark)")
    parser.add_argument("-hi", "--highlight", action="store_true", help="Подсветка сущностей (inference)")
    parser.add_argument("-o", "--open", action="store_true", help="Открыть результат в браузере")
    
    args = parser.parse_args()
    
    if args.open:
        has_viz = any([args.loss, args.metrics, args.plots, args.compare, 
                       args.bench, args.highlight])
        if not has_viz:
            print("Ошибка: -o/--open требует флаг визуализации (-l, -mt, -p, -c, -b, -hi)")
            sys.exit(1)
    
    if args.mode == "inference":
        if not args.file and not args.text:
            print("Ошибка: для inference укажите -f/--file или -t/--text")
            sys.exit(1)
        
        if args.text:
            temp_file = Path("inference_data/temp_text.txt")
            temp_file.parent.mkdir(parents=True, exist_ok=True)
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(args.text)
            file_path = str(temp_file)
        else:
            file_path = args.file
        
        run_inference(
            file_path=file_path,
            highlight=args.highlight,
            plots=args.plots,
            open_result=args.open
        )
    
    elif args.mode == "update":
        run_update(
            plot_loss=args.loss,
            plot_metrics=args.metrics,
            plot_entities=args.plots,
            open_result=args.open
        )
    
    elif args.mode == "summary":
        run_summary(
            plots=args.plots,
            compare=args.compare,
            open_result=args.open
        )
    
    elif args.mode == "benchmark":
        run_benchmark(
            file_path=args.file,
            plots=args.bench,
            open_result=args.open
        )

if __name__ == "__main__":
    main()
