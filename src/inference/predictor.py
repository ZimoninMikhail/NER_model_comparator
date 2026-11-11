"""
Модуль для инференса модели на новых текстах.
"""

import torch
from typing import List, Dict, Any, Tuple
from src.device import resolve_device
from src.models.adapter import predict_token_labels, token_mask_from_samples
from src.data_preparation.bio_decoder import decode_bio
from src.data_preparation.document_chunker import chunk_document


class NERPredictor:
    """
    Предсказатель NER сущностей.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        id2label: Dict[int, str],
        device: str = "auto",
        max_length: int = 512
    ):
        """
        Args:
            model: Загруженная модель BERT
            tokenizer: Токенизатор
            id2label: Маппинг ID → метка
            device: Устройство ("cpu" или "cuda")
            max_length: Максимальная длина последовательности
        """
        self.model = model
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.device = resolve_device(device)
        self.max_length = max_length
        
        self.model.to(self.device)
        self.model.eval()
    
    def predict_single(
        self,
        text: str,
        return_spans: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Предсказывает сущности в одном тексте.
        
        Args:
            text: Входной текст
            return_spans: Возвращать ли позиции сущностей
            
        Returns:
            Список найденных сущностей
        """
        chunks = chunk_document(text, self.tokenizer, self.max_length)
        entities = []
        for chunk in chunks:
            input_ids = torch.tensor([chunk['input_ids']], dtype=torch.long, device=self.device)
            attention_mask = torch.tensor([chunk['attention_mask']], dtype=torch.long, device=self.device)
            with torch.no_grad():
                mask = token_mask_from_samples([{'offset_mapping':chunk['offset_mapping'], 'text':text}],
                                               len(chunk['input_ids']), self.device)
                predictions = predict_token_labels(self.model, input_ids, attention_mask, mask)[0].cpu().numpy()
            entities.extend(self._decode_predictions(predictions, chunk['offset_mapping'],
                                                      text, return_spans))

        return entities
    
    def predict_batch(
        self,
        texts: List[str],
        return_spans: bool = True
    ) -> List[List[Dict[str, Any]]]:
        """
        Предсказывает сущности для нескольких текстов.
        
        Args:
            texts: Список текстов
            return_spans: Возвращать ли позиции сущностей
            
        Returns:
            Список списков сущностей для каждого текста
        """
        results = []
        for text in texts:
            entities = self.predict_single(text, return_spans)
            results.append(entities)
        return results
    
    def _decode_predictions(
        self,
        predictions: List[int],
        offset_mapping: List[Tuple[int, int]],
        text: str,
        return_spans: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Декодирует предсказания в сущности.
        """
        return decode_bio(predictions, offset_mapping, text,
                          id2label=self.id2label, return_spans=return_spans)
