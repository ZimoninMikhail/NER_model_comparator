"""
Модуль для обучения BERT модели.
"""

import torch
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
from typing import Dict, Any, Optional, List
import json
from pathlib import Path
import shutil


class NERDataset(Dataset):
    """
    PyTorch Dataset для NER данных из prepared батча.
    """
    
    def __init__(self, tensors: Dict[str, torch.Tensor]):
        self.input_ids = tensors['input_ids']
        self.attention_mask = tensors['attention_mask']
        self.labels = tensors['labels']
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }


def train_model(
    model,
    train_dataset: NERDataset,
    eval_dataset: Optional[NERDataset] = None,
    output_dir: str = "./data/models/temp",
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    logging_steps: int = 10,
    eval_steps: int = 50,
    save_steps: int = 100
) -> Trainer:
    """
    Обучает модель NER.
    """
    print("\n" + "="*50)
    print("Настройка обучения")
    print("="*50)
    print(f"  epochs: {num_epochs}")
    print(f"  batch_size: {batch_size}")
    print(f"  learning_rate: {learning_rate}")
    print(f"  output_dir: {output_dir}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=eval_steps if eval_dataset else None,
        save_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=True if eval_dataset else False,
        report_to="none",
        fp16=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if eval_dataset else None,
    )
    
    print("\n" + "="*50)
    print("Начало обучения")
    print("="*50)
    
    trainer.train()
    
    print("\n" + "="*50)
    print("Обучение завершено!")
    print("="*50)
    
    return trainer


def train_from_batches(
    prepared_storage,
    batch_paths: List[Path],
    model_name: str = "DeepPavlov/rubert-base-cased",
    models_dir: str = "./data/models",
    num_epochs: int = 1,
    batch_size: int = 4
) -> tuple:
    """
    Обучает модель на нескольких prepared батчах
    
    Returns:
        (model_path, version, loss_history) - путь к модели, версия, история loss
    """
    from .model_factory import create_model, get_device
    from datetime import datetime
    import torch
    import torch.nn.functional as F
    
    print("\n" + "="*50)
    print("Загрузка подготовленных данных")
    print("="*50)
    print(f"  Батчей: {len(batch_paths)}")
    
    all_input_ids = []
    all_attention_masks = []
    all_labels = []
    dataset_info = None
    total_samples = 0
    
    target_length = 0
    for batch_path in batch_paths:
        batch_data = prepared_storage.load_batch(batch_path)
        if dataset_info is None:
            dataset_info = batch_data['dataset_info']
        tensors = batch_data['tensors']
        current_length = tensors['input_ids'].shape[1]
        if current_length > target_length:
            target_length = current_length
    
    print(f"  Целевая длина: {target_length}")
    
    for batch_path in batch_paths:
        batch_data = prepared_storage.load_batch(batch_path)
        tensors = batch_data['tensors']
        
        input_ids = tensors['input_ids']
        attention_mask = tensors['attention_mask']
        labels = tensors['labels']
        
        current_length = input_ids.shape[1]
        if current_length < target_length:
            pad_len = target_length - current_length
            input_ids = F.pad(input_ids, (0, pad_len), value=0)
            attention_mask = F.pad(attention_mask, (0, pad_len), value=0)
            labels = F.pad(labels, (0, pad_len), value=-100)
        
        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_labels.append(labels)
        total_samples += len(input_ids)
    
    combined_tensors = {
        'input_ids': torch.cat(all_input_ids, dim=0),
        'attention_mask': torch.cat(all_attention_masks, dim=0),
        'labels': torch.cat(all_labels, dim=0)
    }
    
    print(f"  total_samples: {total_samples}")
    print(f"  max_length: {combined_tensors['input_ids'].shape[1]}")
    print(f"  num_labels: {dataset_info['num_labels']}")
    
    train_dataset = NERDataset(combined_tensors)
    
    device = get_device()
    model = create_model(
        model_name=model_name,
        num_labels=dataset_info['num_labels'],
        label2id=dataset_info.get('label2id'),
        id2label=dataset_info.get('id2label'),
        device=device
    )
    
    temp_dir = Path("./data/models/.temp_checkpoints")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = train_model(
        model=model,
        train_dataset=train_dataset,
        num_epochs=num_epochs,
        batch_size=batch_size,
        output_dir=str(temp_dir)
    )
    
    loss_history = []
    if hasattr(trainer, 'state') and hasattr(trainer.state, 'log_history'):
        for log in trainer.state.log_history:
            if 'loss' in log:
                loss_history.append({
                    'step': log.get('step', len(loss_history)),
                    'loss': log['loss'],
                    'epoch': log.get('epoch', 0)
                })
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    version = f"v{timestamp}"
    final_model_path = Path(models_dir) / version
    final_model_path.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(final_model_path)
    
    metadata = {
        'version': version,
        'source_batches': [str(p.name) for p in batch_paths],
        'num_batches': len(batch_paths),
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'num_samples': total_samples,
        'model_name': model_name,
        'num_labels': dataset_info['num_labels'],
        'timestamp': timestamp
    }
    with open(final_model_path / "training_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    if loss_history:
        loss_path = final_model_path / "loss_history.json"
        with open(loss_path, 'w') as f:
            json.dump(loss_history, f, indent=2)
        print(f"  loss_history: {len(loss_history)} записей")
    
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    print(f"\n Модель сохранена: {final_model_path}")
    print(f"   Версия: {version}")
    print(f"   Источники: {len(batch_paths)} батчей, {total_samples} образцов")
    
    return str(final_model_path), version, loss_history