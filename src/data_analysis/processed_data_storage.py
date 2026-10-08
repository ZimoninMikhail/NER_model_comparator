"""
Модуль для сохранения обработанных (очищенных, отфильтрованных) данных.
"""

import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


class ProcessedDataStorage:
    """
    Сохраняет данные после Этапа 2 (анализ, фильтрация, очистка).
    """
    
    def __init__(self, base_path: str = "data/processed"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        print(f"ProcessedStorage инициализирован: {self.base_path}")
    
    def _get_timestamp(self) -> str:
        """Возвращает текущий timestamp в формате YYYYMMDD_HHMMSS_mmm."""
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    
    def save_batch(
        self,
        batch: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Сохраняет обработанный батч.
        
        Args:
            batch: Список документов
            metadata: Метаданные (качество, фильтрация)
            
        Returns:
            Path: Путь к папке сохранённого батча
        """
        if not batch:
            print("Предупреждение: пустой батч, ничего не сохранено")
            return None
        
        timestamp = self._get_timestamp()
        batch_name = f"batch_{timestamp}"
        batch_path = self.base_path / batch_name
        batch_path.mkdir(parents=True, exist_ok=True)
        
        # Загружаем исходные метаданные из raw (если указан source_batch)
        source_metadata = {}
        source_batch_name = metadata.get("source_batch") if metadata else None
        if source_batch_name:
            source_metadata_path = self.base_path.parent / "raw" / source_batch_name / "metadata.json"
            if source_metadata_path.exists():
                with open(source_metadata_path, 'r', encoding='utf-8') as f:
                    source_metadata = json.load(f)
        
        # Формируем метаданные согласно спецификации для processed
        full_metadata = {
            # Ссылка на исходный батч
            "source_batch": source_batch_name,
            "source_stage": "raw",
            
            # Текущие метаданные
            "batch_id": batch[0].get("batch_id", 0) if batch else 0,
            "timestamp": timestamp,
            "stage": "processed",
            "num_documents": len(batch),
            
            # Статистика из raw
            "total_entities_before": source_metadata.get("total_entities"),
            
            # Метаданные из переданных модулей (quality_checker, type_filter, quality_filter)
            **(metadata or {})
        }
        
        # Убираем source_batch из metadata, чтобы не дублировать
        full_metadata.pop("source_batch", None)
        
        # Добавляем статистику после фильтрации (если не передана)
        if "total_entities_after" not in full_metadata:
            total_entities = sum(doc.get("entities_count", 0) for doc in batch)
            full_metadata["total_entities_after"] = total_entities
        
        # Сохраняем метаданные
        metadata_path = batch_path / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(full_metadata, f, indent=2, ensure_ascii=False)
        
        # Сохраняем документы
        for i, doc in enumerate(batch):
            doc_path = batch_path / f"doc_{i:04d}.json"
            with open(doc_path, 'w', encoding='utf-8') as f:
                json.dump(doc, f, indent=2, ensure_ascii=False)
        
        print(f"Processed батч сохранён: {batch_path} ({len(batch)} документов)")
        return batch_path
    
    def load_batch(self, batch_path: Path) -> List[Dict[str, Any]]:
        """Загружает обработанный батч."""
        if not batch_path.exists():
            raise FileNotFoundError(f"Папка батча не найдена: {batch_path}")
        
        batch = []
        doc_files = sorted(batch_path.glob("doc_*.json"))
        
        for doc_file in doc_files:
            with open(doc_file, 'r', encoding='utf-8') as f:
                batch.append(json.load(f))
        
        print(f"Загружен processed батч: {batch_path} ({len(batch)} документов)")
        return batch
    
    def list_batches(self) -> List[Path]:
        """Возвращает список всех сохранённых батчей."""
        batches = [p for p in self.base_path.iterdir() if p.is_dir()]
        batches.sort(key=lambda x: x.name)
        return batches
    
    def get_metadata(self, batch_path: Path) -> Dict[str, Any]:
        """Загружает метаданные батча."""
        metadata_path = batch_path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Файл метаданных не найден: {metadata_path}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_latest_batch(self) -> Optional[Path]:
        """Возвращает путь к самому свежему батчу."""
        batches = self.list_batches()
        return batches[-1] if batches else None
    
    def clear(self) -> None:
        """Очищает папку со всеми обработанными данными."""
        if self.base_path.exists():
            shutil.rmtree(self.base_path)
            print(f"Очищена папка: {self.base_path}")
        self.base_path.mkdir(parents=True, exist_ok=True)