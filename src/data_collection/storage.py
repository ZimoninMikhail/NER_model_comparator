"""
Модуль для сохранения и загрузки сырых данных.
"""

import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


class RawDataStorage:
    """
    Сохраняет сырые данные в файловую систему.
    """
    
    def __init__(self, base_path: str = "data/raw"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        print(f"RawDataStorage инициализирован: {self.base_path}")
    
    def _get_timestamp(self) -> str:
        """Возвращает текущий timestamp в формате YYYYMMDD_HHMMSS_mmm."""
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    
    def save_batch(
        self, 
        batch: List[Dict[str, Any]], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Сохраняет батч документов.
        
        Args:
            batch: Список документов
            metadata: Метаданные (статистика и т.д.)
            
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
        
        # Формируем метаданные согласно спецификации для raw
        full_metadata = {
            "batch_id": batch[0].get("batch_id", 0) if batch else 0,
            "timestamp": timestamp,
            "stage": "raw",
            "num_documents": len(batch),
            **(metadata or {})
        }
        
        # Сохраняем метаданные
        metadata_path = batch_path / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(full_metadata, f, indent=2, ensure_ascii=False)
        
        # Сохраняем документы
        for i, doc in enumerate(batch):
            doc_path = batch_path / f"doc_{i:04d}.json"
            doc_to_save = {
                'text': doc.get('text', ''),
                'entities': doc.get('entities', ''),
                'doc_id': doc.get('doc_id', i),
                'batch_id': doc.get('batch_id', full_metadata["batch_id"])
            }
            with open(doc_path, 'w', encoding='utf-8') as f:
                json.dump(doc_to_save, f, indent=2, ensure_ascii=False)
        
        print(f"Raw батч сохранён: {batch_path} ({len(batch)} документов)")
        return batch_path
    
    def load_batch(self, batch_path: Path) -> List[Dict[str, Any]]:
        """Загружает батч из файловой системы."""
        if not batch_path.exists():
            raise FileNotFoundError(f"Папка батча не найдена: {batch_path}")
        
        batch = []
        doc_files = sorted(batch_path.glob("doc_*.json"))
        
        for doc_file in doc_files:
            with open(doc_file, 'r', encoding='utf-8') as f:
                batch.append(json.load(f))
        
        print(f"Загружен raw батч: {batch_path} ({len(batch)} документов)")
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
    
    def delete_batch(self, batch_path: Path) -> None:
        """Удаляет батч из файловой системы."""
        if not batch_path.exists():
            print(f"Предупреждение: батч не найден {batch_path}")
            return
        
        shutil.rmtree(batch_path)
        print(f"Батч удалён: {batch_path}")
    
    def get_latest_batch(self) -> Optional[Path]:
        """Возвращает путь к самому свежему батчу."""
        batches = self.list_batches()
        return batches[-1] if batches else None
    
    def clear(self) -> None:
        """Очищает папку со всеми сырыми данными."""
        if self.base_path.exists():
            shutil.rmtree(self.base_path)
            print(f"Очищена папка: {self.base_path}")
        self.base_path.mkdir(parents=True, exist_ok=True)
