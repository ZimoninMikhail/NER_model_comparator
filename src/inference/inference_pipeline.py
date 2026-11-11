"""
Полный пайплайн инференса.
"""

from pathlib import Path
import json
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer
from src.device import resolve_device
from src.models.adapter import is_bilstm_crf

from .model_loader import ModelLoader
from .predictor import NERPredictor


class InferencePipeline:
    """
    Полный пайплайн для инференса.
    """
    
    def __init__(
        self,
        model_version: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
        device: str = "auto",
        max_length: int = 512,
        models_dir: str = "data/models",
        architecture=None, gliner_model_path=None, type_mapping=None, threshold=0.5
    ):
        """
        Args:
            model_version: Версия модели (если None, берётся последняя)
            tokenizer_name: Старый аргумент совместимости, не переопределяет локальный токенизатор
            device: Устройство
            max_length: Максимальная длина
        """
        self.device = resolve_device(device)
        self.max_length = max_length
        
        if architecture == 'gliner':
            if model_version is not None:
                raise ValueError('--model-version не используется для GLiNER; путь задаётся в конфигурации')
            from src.models.gliner import load_model
            from .gliner_predictor import GLiNERPredictor
            if not gliner_model_path:
                raise ValueError('Укажите локальный комплект GLiNER в model.name')
            self.model_path = Path(gliner_model_path)
            model = load_model(self.model_path, device=self.device)
            self.architecture = 'gliner'
            self.tokenizer = model.data_processor.transformer_tokenizer
            self.predictor = GLiNERPredictor(model, type_mapping, threshold, self.max_length)
            print(f'Инференс GLiNER готов: device={self.device}, threshold={threshold}')
            return

        loader = ModelLoader(models_dir)
        self.model_path = loader.get_model_path(model_version)
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True, use_fast=True)
        except (OSError, ValueError) as error:
            raise ValueError(f'Не удалось загрузить локальный токенизатор из {self.model_path}. '
                             'Сохраните токенизатор вместе с этой моделью; загрузка из сети отключена') from error
        if not tokenizer.is_fast:
            raise ValueError('Для координат сущностей требуется локальный fast-токенизатор')
        model, id2label = loader.load_model(model_version)
        self.tokenizer = tokenizer
        self.architecture = 'bilstm_crf' if is_bilstm_crf(model) else 'bert'
        runtime_path = self.model_path / 'model_runtime.json'
        if runtime_path.is_file():
            runtime = json.loads(runtime_path.read_text(encoding='utf-8'))
            if runtime.get('architecture') != self.architecture:
                raise ValueError('Архитектура model_runtime.json не соответствует весам')
            trained_length = runtime.get('max_length')
            if type(trained_length) is not int or trained_length < 3:
                raise ValueError('Некорректная длина в model_runtime.json')
            self.max_length = min(self.max_length, trained_length)
        elif self.architecture == 'bilstm_crf':
            self.max_length = min(self.max_length, model.config.max_seq_length)

        self.predictor = NERPredictor(
            model=model,
            tokenizer=tokenizer,
            id2label=id2label,
            device=self.device,
            max_length=self.max_length
        )
        
        print(f"Инференс пайплайн готов")
        print(f"  device: {self.device}")
        print(f"  max_length: {self.max_length}")
    
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