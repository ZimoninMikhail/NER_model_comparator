"""
Модуль для создания модели BERT для NER.
"""

from transformers import AutoModelForTokenClassification
import torch


def create_model(
    model_name: str,
    num_labels: int,
    label2id: dict = None,
    id2label: dict = None,
    device: str = "cpu"
) -> AutoModelForTokenClassification:
    """
    Создаёт модель BERT для токеновой классификации (NER).
    
    Args:
        model_name: Название модели на Hugging Face
        num_labels: Количество классов
        label2id: Маппинг метка → ID
        id2label: Маппинг ID → метка
        device: Устройство ("cpu" или "cuda")
        
    Returns:
        Модель BERT для NER
    """
    print(f"Создание модели: {model_name}")
    print(f"  num_labels: {num_labels}")
    
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label
    )
    
    model.to(device)
    print(f"  device: {device}")
    print(f"  параметров: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def get_device() -> str:
    """
    Определяет доступное устройство.
    
    Returns:
        "cuda" если GPU доступен, иначе "cpu"
    """
    if torch.cuda.is_available():
        device = "cuda"
        print(f"GPU доступен: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("GPU не найден, используем CPU")
    
    return device