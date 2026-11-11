"""
Модуль для сохранения подготовленных данных (тензоры для модели).
"""

import torch
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


class PreparedDataStorage:
    """
    Сохраняет подготовленные данные (тензоры) для модели.
    """

    def __init__(self, base_path: str = "data/prepared"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        print(f"PreparedStorage инициализирован: {self.base_path}")

    def _get_timestamp(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def clear(self) -> None:
        """Очищает папку со всеми подготовленными данными."""
        if self.base_path.exists():
            shutil.rmtree(self.base_path)
            print(f"Очищена папка: {self.base_path}")
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save_batch(
        self,
        encoded_batch: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        dataset_info: Dict[str, Any]
    ) -> Path:
        """
        Сохраняет подготовленный батч.

        Args:
            encoded_batch: Список документов с input_ids, attention_mask, labels
            metadata: Метаданные (batch_id, num_samples, source_batch)
            dataset_info: Информация о датасете (label2id, id2label, ...)
        """
        if not encoded_batch:
            print("Предупреждение: пустой батч, ничего не сохранено")
            return None

        # Координаты и исходный текст не поступают в модель, но нужны декодеру.
        samples = []
        for doc in encoded_batch:
            if 'text' not in doc or 'offset_mapping' not in doc or 'gold_entities' not in doc:
                raise ValueError('Для валидации нужны text, offset_mapping и gold_entities; пересоздайте тензоры')
            if not len(doc['input_ids']) == len(doc['labels']) == len(doc['attention_mask']) == len(doc['offset_mapping']):
                raise ValueError('Размеры полей подготовленного документа не совпадают')
            samples.append({key: doc[key] for key in ('doc_id', 'split', 'text', 'offset_mapping', 'gold_entities',
                'sample_id', 'chunk_index', 'chunk_count', 'chunking_version',
                'chunk_start', 'chunk_end', 'document_gold_entities') if key in doc})



        timestamp = self._get_timestamp()
        batch_path = self.base_path / f"batch_{timestamp}"
        batch_path.mkdir(parents=True, exist_ok=True)

        max_len = max(len(doc['input_ids']) for doc in encoded_batch)

        input_ids = []
        attention_mask = []
        labels = []

        for doc in encoded_batch:
            pad_len = max_len - len(doc['input_ids'])
            input_ids.append(doc['input_ids'] + [0] * pad_len)
            attention_mask.append(doc['attention_mask'] + [0] * pad_len)
            labels.append(doc['labels'] + [-100] * pad_len)

        tensors = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

        torch.save(tensors, batch_path / "tensors.pt")
        with open(batch_path / 'samples.json', 'w', encoding='utf-8') as stream:
            json.dump(samples, stream, ensure_ascii=False, indent=2)

        full_metadata = {
            "timestamp": timestamp,
            "stage": "prepared",
            "num_samples": len(encoded_batch),
            "max_length": max_len,
            **metadata
        }
        with open(batch_path / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(full_metadata, f, indent=2, ensure_ascii=False)

        with open(batch_path / "dataset_info.json", 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)

        print(f"Prepared батч сохранён: {batch_path} ({len(encoded_batch)} образцов)")
        return batch_path

    def load_batch(self, batch_path: Path) -> Dict[str, Any]:
        """Загружает подготовленный батч."""
        tensors = torch.load(batch_path / "tensors.pt")

        metadata = {}
        if (batch_path / "metadata.json").exists():
            with open(batch_path / "metadata.json", 'r', encoding='utf-8') as f:
                metadata = json.load(f)

        dataset_info = {}
        if (batch_path / "dataset_info.json").exists():
            with open(batch_path / "dataset_info.json", 'r', encoding='utf-8') as f:
                dataset_info = json.load(f)

        return {
            'samples': json.loads((batch_path / 'samples.json').read_text(encoding='utf-8'))
                if (batch_path / 'samples.json').is_file() else None,
            'tensors': tensors,
            'metadata': metadata,
            'dataset_info': dataset_info
        }

    def list_batches(self) -> List[Path]:
        """Возвращает список всех сохранённых батчей."""
        batches = [p for p in self.base_path.iterdir() if p.is_dir()]
        batches.sort(key=lambda x: x.name)
        return batches

    def get_latest_batch(self) -> Optional[Path]:
        """Возвращает путь к самому свежему батчу."""
        batches = self.list_batches()
        return batches[-1] if batches else None
