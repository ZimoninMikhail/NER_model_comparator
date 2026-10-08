from src.data_collection.loader import StreamingDataLoader
from src.data_collection.storage import RawDataStorage
from src.data_collection.metadata_calculator import calculate_batch_metadata

loader = StreamingDataLoader(batch_size=32, shuffle=True)
storage = RawDataStorage("data/raw")

print("Загрузка train сплита...")
loader.load_train()
print(f"Всего документов: {len(loader)}")

print("\nСохранение батчей...")
batch_count = 0

for batch in loader:
    batch_count += 1
    print(f"Батч #{batch_count}: {len(batch)} документов")
    
    metadata = calculate_batch_metadata(batch)
    
    batch_path = storage.save_batch(batch, metadata)
    print(f"  Сохранён: {batch_path}")
