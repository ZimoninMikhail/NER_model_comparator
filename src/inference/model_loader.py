"""Локальная загрузка зарегистрированных моделей, без обращения к Hub."""
import json
from pathlib import Path
from src.models.adapter import load_local_model


class ModelLoader:
    def __init__(self, models_dir='data/models'):
        self.models_dir = Path(models_dir).resolve()
        self.registry_path = self.models_dir / 'registry.json'
        self._registry = json.loads(self.registry_path.read_text(encoding='utf-8')) \
            if self.registry_path.is_file() else {'versions': [], 'latest': None}

    def get_model_path(self, version=None):
        version = version if version is not None else self._registry.get('latest')
        entry = next((entry for entry in self._registry['versions'] if entry['version'] == version), None)
        if entry is None:
            raise ValueError('Модель не найдена в реестре. Укажите версию или сначала обучите модель')
        # Папка рядом с реестром позволяет переносить весь комплект models.
        if Path(version).name != version or version in ('.','..'):
            raise ValueError('Некорректный ID версии модели')
        local = self.models_dir / version
        if local.is_dir():
            return local.resolve()
        registered = Path(entry['path']).expanduser()
        candidates = [registered] if registered.is_absolute() else [self.models_dir / registered, self.models_dir.parent.parent / registered]
        for path in candidates:
            if path.is_dir():
                return path.resolve()
        raise ValueError(f'Локальная папка версии {version} не найдена')

    def get_latest_model_path(self):
        return self.get_model_path()

    def get_model_by_version(self, version):
        return self.get_model_path(version)

    def load_model(self, version=None):
        path = self.get_model_path(version)
        print(f'Загрузка модели: {path}')
        model = load_local_model(path)
        return model, model.config.id2label

    def list_versions(self):
        return [entry['version'] for entry in self._registry['versions']]
