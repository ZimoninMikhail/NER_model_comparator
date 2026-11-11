"""Выбор архитектуры и устройства. Реализации находятся в src.models."""
from torch.nn import Module
from src.device import resolve_device
from src.models import bert


def create_model(
    model_name: str,
    num_labels: int,
    label2id: dict = None,
    id2label: dict = None,
    device: str = "auto",
    architecture: str = 'bert',
    model_parameters=None,
    vocab_size=None,
    padding_idx=0,
    special_token_ids=(),
    tokenizer_name=None,
    max_seq_length=512
) -> Module:
    """Создать выбранную архитектуру через её модуль и перенести на устройство."""
    device = resolve_device(device)
    print(f"Создание модели: {model_name}")
    print(f"  num_labels: {num_labels}")
    
    if architecture == 'bilstm_crf':
        from src.models import bilstm_crf
        model = bilstm_crf.create_model(num_labels=num_labels, vocab_size=vocab_size,
            label2id=label2id, id2label=id2label, model_parameters=model_parameters,
            padding_idx=padding_idx, special_token_ids=special_token_ids,
            tokenizer_name=tokenizer_name, max_seq_length=max_seq_length)
    elif architecture == 'bert':
        model = bert.create_model(model_name, num_labels, label2id, id2label)
    else:
        raise ValueError(f'Неизвестная архитектура: {architecture}')

    model.to(device)
    print(f"  device: {device}")
    print(f"  параметров: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def get_device(requested='auto') -> str:
    """Совместимый интерфейс к общему выбору CPU/CUDA."""
    return resolve_device(requested)
