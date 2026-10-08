"""
Тест валидации модели.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_preparation.prepared_data_storage import PreparedDataStorage
from src.validation.validator import validate_model
from src.validation.model_registry import ModelRegistry


prepared_storage = PreparedDataStorage("data/prepared")
registry = ModelRegistry()


latest = registry.get_latest()
if not latest:
    print("Нет зарегистрированных моделей. Сначала обучите модель.")
    exit()

model_path = Path(latest['path'])
print(f"Модель: {latest['version']}")
print(f"Путь: {model_path}")

validation_batch = prepared_storage.get_latest_batch()

if validation_batch:
    print(f"\nВалидационный батч: {validation_batch.name}")
    
    metrics = validate_model(
        model_path=model_path,
        prepared_storage=prepared_storage,
        batch_path=validation_batch,
        device="cpu"
    )
    
    registry.update_metrics(latest['version'], metrics)
    
    print(f"\n✅ Валидация завершена!")
    print(f"   F1: {metrics['f1']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall: {metrics['recall']:.4f}")
else:
    print("Нет данных для валидации")