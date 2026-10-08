"""
Модуль для загрузки данных с Hugging Face и эмуляции потокового чтения.
"""

from datasets import load_dataset
from typing import List, Dict, Any, Optional
import random

class StreamingDataLoader:
    """
    Загрузчик данных с эмуляцией потокового чтения.
    
    Позволяет загрузить датасет с Hugging Face и получать из него данные
    батчами заданного размера, как если бы они приходили в реальном времени.
    
    Attributes:
        batch_size (int): Количество документов в одном батче
        full_dataset:  Весь кэшированный датасет
        dataset: Загруженный датасет Hugging Face
        current_idx (int): Текущая позиция в датасете
        batch_count (int): Количество выданных батчей
        total_docs (int): Общее количество документов в датасете
        current_split: Название текущей части датасета
        indices: Список индексов для перемешивания перед разделением на батчи
        shuffle: Флаг перемешивания
    """
    
    def __init__(
        self, 
        batch_size: int = 32, 
        dataset_name: str = "iluvvatar/NEREL", 
        shuffle: bool = True
    ):
        """
        Args:
            batch_size: Размер батча (количество документов)
            dataset_name: Название датасета на Hugging Face
            shuffle: перемешивать ли документы перед разделением на батчи
        """
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.full_dataset = None
        self.dataset = None
        self.current_idx = 0
        self.batch_count = 0
        self.total_docs = 0
        self.current_split = None
        self.indices = []
        self.shuffle = shuffle

    def _shuffle_indices(self):
        """Перемешивает индексы документов. (Для train и dev сплитов)"""
        if self.shuffle:
            random.shuffle(self.indices)
            print(f"Индексы перемешаны. Первые 10: {self.indices[:10]}")
        else:
            print("Индексы не перемешаны (сохранён порядок)")

    def load_train(self) -> None:
        """Загружает тренировочный сплит. Перемешивает при соответствующем флаге"""
        if self.full_dataset is None:
            self.full_dataset = load_dataset(self.dataset_name, trust_remote_code=True)

        self.dataset = self.full_dataset['train']
        self.total_docs = len(self.dataset)
        self.indices = list(range(self.total_docs))
        self._shuffle_indices() 

        self.current_idx = 0
        self.batch_count = 0
        self.current_split = 'train'
        print(f"Загружен train сплит: {self.total_docs} документов")
    
    def load_dev(self) -> None:
        """
        Загружает сплит для разработки/валидации (dev).
        Перемешивает при соответствующем флаге
        """
        if self.full_dataset is None:
            self.full_dataset = load_dataset(self.dataset_name, trust_remote_code=True)

        self.dataset = self.full_dataset['dev']
        self.total_docs = len(self.dataset)
        self.indices = list(range(self.total_docs))
        self._shuffle_indices() 

        self.current_idx = 0
        self.batch_count = 0
        self.current_split = 'dev'
        print(f"Загружен dev сплит: {self.total_docs} документов")
    
    def load_test(self) -> None:
        """Загружает тестовый сплит. НЕ перемешивает"""
        if self.full_dataset is None:
            self.full_dataset = load_dataset(self.dataset_name, trust_remote_code=True)
            
        self.dataset = self.full_dataset['test']
        self.total_docs = len(self.dataset)
        self.indices = list(range(self.total_docs))

        self.current_idx = 0
        self.batch_count = 0
        self.current_split = 'test'
        print(f"Загружен test сплит: {self.total_docs} документов")
    
    def has_next(self) -> bool:
        """
        Проверяет, остались ли ещё необработанные документы.
        
        Returns:
            True, если есть ещё документы, иначе False
        """
        if self.dataset is None:
            return False
        return self.current_idx < self.total_docs
    
    def get_next_batch(self) -> Optional[List[Dict[str, Any]]]:
        """
        Возвращает следующий батч документов.
        
        Returns:
            Список словарей с документами или None, если данные закончились.
            Каждый документ содержит как минимум поля 'text' и 'entities'.
        """
        if not self.has_next():
            return None

        start = self.current_idx
        end = min(start + self.batch_size, self.total_docs)
        
        batch = []
        for i in range(start, end):
            doc_idx = self.indices[i] 
            doc = self.dataset[doc_idx]
            batch.append({
                'text': doc['text'],
                'entities': doc['entities'],
                'doc_id': doc_idx,
                'batch_id': self.batch_count
            })
        
        self.current_idx = end
        self.batch_count += 1
        
        return batch
    
    def reset(self) -> None:
        """
        Сбрасывает указатель в начало датасета. Перемешивает при соответствующем флаге
        """
        self.current_idx = 0
        self.batch_count = 0

        if self.shuffle and self.current_split != 'test':
            self._shuffle_indices()
    
    def set_seed(self, seed: int) -> None:
        """
        Устанавливает seed для воспроизводимости перемешивания.
        
        Args:
            seed: Число для генератора случайных чисел
        """
        random.seed(seed)


    def __len__(self) -> int:
        """    
        Returns:
            Количество документов в датасете
        """
        return self.total_docs if self.dataset else 0
    
    def __iter__(self):
        return self
    
    def __next__(self) -> List[Dict[str, Any]]:
        """
        Returns:
            Следующий батч
        
        Raises:
            StopIteration: когда данные закончились
        """
        batch = self.get_next_batch()
        if batch is None:
            raise StopIteration
        return batch
