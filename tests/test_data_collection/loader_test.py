from src.data_collection.loader import StreamingDataLoader

loader = StreamingDataLoader(batch_size=5, shuffle=False)
loader.load_train()
print(f"Всего документов: {len(loader)}")

batch = loader.get_next_batch()
print(f"Первый батч: {len(batch)} документов")
print(f"Первый документ: {batch[0]['text'][:100]}...")