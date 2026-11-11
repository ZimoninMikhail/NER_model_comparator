"""Локальные BRAT-корпуса NEREL и NEREL-BIO. Сеть не используется."""
from pathlib import Path
import hashlib
import random


def sha256_bytes(content):
    return hashlib.sha256(content).hexdigest()


def read_entity_schema(path):
    active, types = False, []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        stripped = line.strip()
        if stripped.startswith('['):
            active = stripped == '[entities]'
        elif active and stripped and not stripped.startswith('#'):
            types.append(stripped.split()[0])
    if not types or len(types) != len(set(types)):
        raise ValueError(f'Некорректная схема entities в {path}')
    return types


def source_descriptor(config):
    """Идентичность источника не зависит от расположения локальной копии."""
    data = config['data']
    source = data.get('source')
    if not source:
        return None  # Совместимость с прежними конфигурациями Hugging Face.
    return {'format': source['format'], 'dataset_id': data['dataset_id'],
        'dataset_version': data['dataset_version'], 'repository': source['repository'],
        'commit': source['commit'], 'subdirectory': source['subdirectory']}


class BratDataLoader:
    def __init__(self, source_path, dataset_id, entity_types, commit, batch_size=32, shuffle=False):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size должен быть положительным целым числом')
        self.root = Path(source_path).resolve()
        if not self.root.is_dir():
            raise ValueError(f'Локальный датасет не найден: {self.root}. Скопируйте официальные исходники в эту папку')
        marker = self.root / 'SOURCE_COMMIT.txt'
        if not marker.is_file() or marker.read_text(encoding='utf-8').strip() != commit:
            raise ValueError(f'SOURCE_COMMIT.txt отсутствует или не соответствует профилю: {self.root}')
        schema = self.root / 'annotation.conf'
        if not schema.is_file() or set(read_entity_schema(schema)) != set(entity_types):
            raise ValueError(f'Схема типов профиля не соответствует annotation.conf: {self.root}')
        self.schema_sha256 = sha256_bytes(schema.read_bytes())
        self.commit = commit
        self.dataset_id = dataset_id
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random = random.Random(42)
        self.files = []
        self.current_split = None
        self.current_idx = self.batch_count = self.total_docs = 0
        # Проверка пар производится для выбранного split, чтобы statistics не читал dev/test.

    def _load(self, split):
        folder = self.root / split
        if not folder.is_dir():
            raise ValueError(f'Нет split {split}: {folder}')
        texts = {p.stem: p for p in folder.glob('*.txt')}
        annotations = {p.stem: p for p in folder.glob('*.ann')}
        if not texts or texts.keys() != annotations.keys():
            missing_ann = sorted(texts.keys() - annotations.keys())
            missing_text = sorted(annotations.keys() - texts.keys())
            raise ValueError(f'Неполные пары BRAT в {split}: без .ann={missing_ann}, без .txt={missing_text}')
        if any(p.is_symlink() for p in [*texts.values(), *annotations.values()]):
            raise ValueError('BRAT-файлы должны быть обычными файлами, не символическими ссылками')
        self.files = [(texts[key], annotations[key]) for key in sorted(texts)]
        self.current_split = split
        self.current_idx = self.batch_count = 0
        self.total_docs = len(self.files)
        if self.shuffle and split != 'test':
            self.random.shuffle(self.files)

    def load_train(self): self._load('train')
    def load_dev(self): self._load('dev')
    def load_test(self): self._load('test')

    def has_next(self): return self.current_idx < self.total_docs

    def get_next_batch(self):
        if not self.has_next(): return None
        pairs = self.files[self.current_idx:self.current_idx + self.batch_size]
        batch = []
        for text_path, annotation_path in pairs:
            text_bytes, ann_bytes = text_path.read_bytes(), annotation_path.read_bytes()
            try:
                text = text_bytes.decode('utf-8')
                lines = ann_bytes.decode('utf-8').splitlines()
            except UnicodeDecodeError as exc:
                raise ValueError(f'Исходники BRAT должны быть UTF-8: {text_path.stem}') from exc
            batch.append({'doc_id': f'{self.dataset_id}:{self.current_split}:{text_path.stem}',
                'source_doc_id': text_path.stem, 'split': self.current_split,
                'dataset_id': self.dataset_id, 'source_commit': self.commit,
                'batch_id': self.batch_count, 'text': text,
                # Нечитаемые T-строки тоже передаются парсеру для регистрации ошибки.
                'entities': [line for line in lines if line.startswith('T')],
                'source_annotations': lines,
                'source_files': {'text': str(text_path.relative_to(self.root)),
                    'annotations': str(annotation_path.relative_to(self.root)),
                    'text_sha256': sha256_bytes(text_bytes), 'annotations_sha256': sha256_bytes(ann_bytes)}})
        self.current_idx += len(pairs)
        self.batch_count += 1
        return batch

    def reset(self):
        self.current_idx = self.batch_count = 0
        if self.shuffle and self.current_split != 'test': self.random.shuffle(self.files)

    def set_seed(self, seed): self.random.seed(seed)
    def __len__(self): return self.total_docs
    def __iter__(self): return self
    def __next__(self):
        batch = self.get_next_batch()
        if batch is None: raise StopIteration
        return batch
