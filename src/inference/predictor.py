"""
Модуль для инференса модели на новых текстах.
"""

import torch
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer


class NERPredictor:
    """
    Предсказатель NER сущностей.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        id2label: Dict[int, str],
        device: str = "cpu",
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
        self.device = device
        self.max_length = max_length
        
        self.model.to(device)
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
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True
        )
        
        offset_mapping = encoding.pop('offset_mapping')[0].numpy()
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
        
        entities = self._decode_predictions(
            predictions, 
            offset_mapping, 
            text, 
            return_spans
        )
        
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
        entities = []
        current_entity = None
        
        for i, (pred_id, (start, end)) in enumerate(zip(predictions, offset_mapping)):
            if start == 0 and end == 0:
                continue
            
            label = self.id2label.get(pred_id, 'O')
            
            if label.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                entity_type = label[2:]
                current_entity = {
                    'type': entity_type,
                    'start': start,
                    'end': end,
                    'text': text[start:end]
                }
                
            elif label.startswith('I-') and current_entity:
                current_entity['end'] = end
                current_entity['text'] = text[current_entity['start']:end]
                
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            entities.append(current_entity)
        
        merged_entities = self._merge_entities(entities)
        
        if not return_spans:
            for e in merged_entities:
                del e['start']
                del e['end']
        
        return merged_entities
    
    def _merge_entities(self, entities: List[Dict]) -> List[Dict]:
        """Объединяет сущности из нескольких токенов."""
        if not entities:
            return []
        
        merged = []
        current = entities[0].copy()
        
        for i in range(1, len(entities)):
            if entities[i]['type'] == current['type'] and entities[i]['start'] == current['end']:
                current['end'] = entities[i]['end']
                current['text'] = current['text'] + entities[i]['text']
            else:
                merged.append(current)
                current = entities[i].copy()
        
        merged.append(current)
        return merged