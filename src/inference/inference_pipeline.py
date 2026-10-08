"""
Полный пайплайн инференса.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer

from .model_loader import ModelLoader
from .predictor import NERPredictor


class InferencePipeline:
    """
    Полный пайплайн для инференса.
    """
    
    def __init__(
        self,
        model_version: Optional[str] = None,
        tokenizer_name: str = "DeepPavlov/rubert-base-cased",
        device: str = "cpu",
        max_length: int = 512
    ):
        """
        Args:
            model_version: Версия модели (если None, берётся последняя)
            tokenizer_name: Название токенизатора
            device: Устройство
            max_length: Максимальная длина
        """
        self.device = device
        self.max_length = max_length
        
        loader = ModelLoader()
        model, id2label = loader.load_model(model_version)
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        self.predictor = NERPredictor(
            model=model,
            tokenizer=tokenizer,
            id2label=id2label,
            device=device,
            max_length=max_length
        )
        
        print(f"Инференс пайплайн готов")
        print(f"  device: {device}")
        print(f"  max_length: {max_length}")
    
    def predict(self, text: str) -> List[Dict[str, Any]]:
        """
        Предсказывает сущности в тексте.
        """
        return self.predictor.predict_single(text)
    
    def predict_file(self, file_path: Path) -> List[List[Dict[str, Any]]]:
        """
        Предсказывает сущности для всех строк в файле.
        
        Args:
            file_path: Путь к файлу (построчно)
            
        Returns:
            Список сущностей для каждой строки
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        results = self.predictor.predict_batch(texts)
        return results
    
    def predict_and_save(
        self,
        texts: List[str],
        output_path: Path
    ) -> None:
        """
        Предсказывает и сохраняет результаты в JSON.
        """
        import json
        results = self.predictor.predict_batch(texts)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Результаты сохранены: {output_path}")