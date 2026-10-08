"""
Модуль для регистрации и версионирования моделей.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np


def convert_to_serializable(obj: Any) -> Any:
    """
    Рекурсивно конвертирует numpy типы в стандартные Python типы.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


class ModelRegistry:
    """
    Хранилище версий моделей.
    """
    
    def __init__(self, base_path: str = "data/models"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.base_path / "registry.json"
        self._load_registry()
    
    def _load_registry(self):
        """Загружает реестр моделей."""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {'versions': [], 'latest': None}
    
    def _save_registry(self):
        """Сохраняет реестр моделей."""
        registry_serializable = convert_to_serializable(self.registry)
        with open(self.registry_path, 'w') as f:
            json.dump(registry_serializable, f, indent=2)
    
    def register_model(
        self,
        model_path: str,
        metrics: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Регистрирует обученную модель с метриками.
        
        Args:
            model_path: Путь к папке с модельой
            metrics: Метрики модели (precision, recall, f1)
            metadata: Дополнительные метаданные
            
        Returns:
            Версия модели
        """
        model_path = Path(model_path)
        version = model_path.name
        
        entry = {
            'version': version,
            'path': str(model_path),
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics or {},
            'metadata': metadata or {}
        }
        
        existing = None
        for i, v in enumerate(self.registry['versions']):
            if v['version'] == version:
                existing = i
                break
        
        if existing is not None:
            self.registry['versions'][existing] = entry
        else:
            self.registry['versions'].append(entry)
        
        self.registry['latest'] = version
        self._save_registry()
        
        print(f"Модель зарегистрирована: {version}")
        if metrics:
            f1 = metrics.get('f1', 'N/A')
            if isinstance(f1, (float, int)):
                print(f"  f1: {f1:.4f}")
            else:
                print(f"  f1: {f1}")
        
        return version
    
    def update_metrics(self, version: str, metrics: Dict[str, Any]) -> bool:
        """Обновляет метрики для существующей версии."""
        for i, v in enumerate(self.registry['versions']):
            if v['version'] == version:
                self.registry['versions'][i]['metrics'] = metrics
                self._save_registry()
                print(f"Метрики обновлены для версии: {version}")
                if metrics.get('f1'):
                    print(f"  f1: {metrics['f1']:.4f}")
                return True
        print(f"Версия {version} не найдена")
        return False
    
    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Возвращает последнюю зарегистрированную модель."""
        if self.registry['latest']:
            for v in self.registry['versions']:
                if v['version'] == self.registry['latest']:
                    return v
        return None
    
    def get_version(self, version: str) -> Optional[Dict[str, Any]]:
        """Возвращает модель по версии."""
        for v in self.registry['versions']:
            if v['version'] == version:
                return v
        return None
    
    def list_versions(self) -> List[str]:
        """Возвращает список всех версий."""
        return [v['version'] for v in self.registry['versions']]