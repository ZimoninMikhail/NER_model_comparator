import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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

# Конфигурация
GEO_TYPES = ["COUNTRY", "CITY", "LOCATION", "STATE_OR_PROV", "DISTRICT"]
MAX_LENGTH = 512
MODEL_NAME = "DeepPavlov/rubert-base-cased"

# Инициализация
loader = StreamingDataLoader(batch_size=32, shuffle=False)
raw_storage = RawDataStorage("data/raw")
processed_storage = ProcessedDataStorage("data/processed")
prepared_storage = PreparedDataStorage("data/prepared")

# Очистка
raw_storage.clear()
processed_storage.clear()
prepared_storage.clear()

# Загрузка токенизатора и маппинга меток
tokenizer = load_tokenizer(MODEL_NAME, MAX_LENGTH)
label2id, id2label = create_label_mapping(GEO_TYPES)

dataset_info = {
    'label2id': label2id,
    'id2label': id2label,
    'num_labels': len(label2id),
    'entity_types': GEO_TYPES,
    'tokenizer_name': MODEL_NAME,
    'max_length': MAX_LENGTH
}

# Загрузка данных
loader.load_train()

for batch in loader:
    # ========== ЭТАП 1: Сбор данных ==========
    raw_metadata = calculate_batch_metadata(batch)
    raw_path = raw_storage.save_batch(batch, raw_metadata)
    
    # ========== ЭТАП 2: Анализ данных ==========
    parsed_batch = parse_batch_entities(batch)
    geo_batch = filter_batch_geo(parsed_batch, remove_empty_docs=True)
    geo_stats = get_filter_statistics(geo_batch)
    
    quality_batch = [check_document_quality(doc) for doc in geo_batch]
    quality_metrics = calculate_batch_quality(quality_batch)
    
    final_batch = filter_batch_by_quality(quality_batch, min_consistency=0.95, min_entities=1)
    quality_filter_stats = get_quality_filter_statistics(final_batch)
    
    # Сохранение в processed
    processed_metadata = {
        "source_batch": raw_path.name,
        **geo_stats,
        **quality_metrics,
        **quality_filter_stats
    }
    processed_storage.save_batch(final_batch, processed_metadata)
    
    # ========== ЭТАП 3: Подготовка данных для BERT ==========
    if final_batch:
        encoded_batch = encode_batch(final_batch, tokenizer, label2id, MAX_LENGTH)
        
        prepared_metadata = {
            "source_batch": raw_path.name,
            "source_processed_batch": processed_storage.get_latest_batch().name,
            "batch_id": final_batch[0].get('batch_id', 0),
            "num_samples": len(encoded_batch)
        }
        
        prepared_storage.save_batch(encoded_batch, prepared_metadata, dataset_info)
        
        print(f"  Этап 3: {len(encoded_batch)} образцов подготовлено для BERT")

print("\n" + "="*50)
print("Все этапы завершены!")
print(f"Сырые данные: {raw_storage.base_path}")
print(f"Обработанные данные: {processed_storage.base_path}")
print(f"Подготовленные данные: {prepared_storage.base_path}")
print("="*50)