"""
Модуль для загрузки и настройки токенизатора BERT.
"""

from transformers import AutoTokenizer


def load_tokenizer(
    model_name: str = "DeepPavlov/rubert-base-cased",
    max_length: int = 512
) -> AutoTokenizer:
    """
    Загружает токенизатор для BERT.
    
    Args:
        model_name: Название модели на Hugging Face
        max_length: Максимальная длина последовательности
        
    Returns:
        Токенизатор
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = max_length
    print(f"Токенизатор загружен: {model_name}")
    print(f"  max_length: {max_length}")
    print(f"  vocab_size: {tokenizer.vocab_size}")
    return tokenizer