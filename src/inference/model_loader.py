"""
Модуль для загрузки модели из реестра.
"""

from pathlib import Path
from typing import Optional, Dict, Any
from transformers import AutoModelForTokenClassification


class ModelLoader:
    """
    Загрузчик моделей из реестра версий.
    """
    
    def __init__(self, models_dir: str = "data/models"):
        self.models_dir = Path(models_dir)
        self.registry_path = self.models_dir / "registry.json"
        self._registry = None
        self._load_registry()
    
    def _load_registry(self):
        """Загружает реестр моделей."""
        import json
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                self._registry = json.load(f)
        else:
            self._registry = {'versions': [], 'latest': None}
    
    def get_latest_model_path(self) -> Optional[Path]:
        """Возвращает путь к последней модели."""
        latest_version = self._registry.get('latest')
        if latest_version:
            for v in self._registry['versions']:
                if v['version'] == latest_version:
                    return Path(v['path'])
        return None
    
    def get_model_by_version(self, version: str) -> Optional[Path]:
        """Возвращает путь к модели по версии."""
        for v in self._registry['versions']:
            if v['version'] == version:
                return Path(v['path'])
        return None
    
    def load_model(self, version: Optional[str] = None):
        """
        Загружает модель.
        
        Args:
            version: Версия модели (если None, загружается последняя)
            
        Returns:
            model, id2label
        """
        if version:
            model_path = self.get_model_by_version(version)
        else:
            model_path = self.get_latest_model_path()
        
        if not model_path:
            raise ValueError("Модель не найдена. Сначала обучите модель.")
        
        print(f"Загрузка модели: {model_path}")
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        id2label = model.config.id2label
        
        return model, id2label
    
    def list_versions(self) -> list:
        """Возвращает список всех версий."""
        return [v['version'] for v in self._registry['versions']]