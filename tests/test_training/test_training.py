"""
Тест обучения модели на подготовленных данных.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_preparation.prepared_data_storage import PreparedDataStorage
from src.training import train_from_prepared_batch
from src.validation.model_registry import ModelRegistry

prepared_storage = PreparedDataStorage("data/prepared")
registry = ModelRegistry()

latest_batch = prepared_storage.get_latest_batch()

if latest_batch:
    print(f"Используем батч: {latest_batch.name}")
    
    model_path, version = train_from_prepared_batch(
        prepared_storage=prepared_storage,
        batch_path=latest_batch,
        model_name="DeepPavlov/rubert-base-cased",
        models_dir="./data/models",
        num_epochs=1,
        batch_size=4
    )
    
    registry.register_model(
        model_path=model_path,
        metrics=None,
        metadata={'batch': latest_batch.name, 'epochs': 1, 'status': 'trained'}
    )
    
    print(f"\n✅ Обучение завершено!")
    print(f"   Версия: {version}")
    print(f"   Модель: {model_path}")
else:
    print("Нет подготовленных данных. Сначала запустите test_full_pipeline.py")