from src.data_collection.storage import RawDataStorage
from src.data_analysis.parser import parse_batch_entities
from src.data_analysis.processed_data_storage import ProcessedDataStorage

raw_storage = RawDataStorage("data/raw")
processed_storage = ProcessedDataStorage("data/processed")

batches = raw_storage.list_batches()
print(f"Найдено сырых батчей: {len(batches)}")

if not batches:
    print("Нет сырых данных. Сначала запустите loader_saver_test.")
    exit()

for i, batch_path in enumerate(batches):
    print(f"\n{'='*50}")
    print(f"Батч {i+1}/{len(batches)}: {batch_path.name}")
    print(f"{'='*50}")
    
    raw_batch = raw_storage.load_batch(batch_path)
    print(f"Загружено документов: {len(raw_batch)}")
    
    print("Парсинг сущностей...")
    parsed_batch = parse_batch_entities(raw_batch)
    
    print("Сохранение в processed...")
    processed_storage.save_batch(
        parsed_batch,
        {"stage": "parsed_only", "source_batch": batch_path.name}
    )

print(f"\n{'='*50}")
print("Готово! Все батчи обработаны и сохранены в data/processed/")