"""Единый выбор устройства для обучения и применения моделей."""
import torch


def resolve_device(requested='auto'):
    if requested not in ('auto', 'cpu', 'cuda'):
        raise ValueError('Устройство должно быть auto, cpu или cuda')
    if requested == 'cpu':
        return 'cpu'
    available = torch.cuda.is_available()
    if requested == 'cuda' and not available:
        raise ValueError('Запрошено cuda, но GPU недоступен. Проверьте драйвер и CUDA-сборку PyTorch или выберите cpu')
    return 'cuda' if available else 'cpu'
