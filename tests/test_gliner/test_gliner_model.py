"""Контракт адаптера GLiNER, локальный checkpoint и настоящие CPU-предсказания."""
from unittest.mock import patch

from src.models.gliner import load_model, predict_entities


def test_real_local_save_load_and_predictions_without_network(tiny_model, tmp_path):
    path = tmp_path / 'bundle'
    tiny_model.save_pretrained(path)
    text, mapping = 'Москва Россия', {'CITY': 'city', 'COUNTRY': 'country'}
    with patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')):
        expected = predict_entities(tiny_model, text, mapping, threshold=0.)
        # Предотвращаем случайную зависимость от исходной папки энкодера.
        (tmp_path / 'config.json').unlink()
        restored = load_model(path, device='cpu')
        actual = predict_entities(restored, text, mapping, threshold=0.)
    assert expected and actual == expected
    assert not restored.training
    assert next(restored.parameters()).device.type == 'cpu'
    assert all(e['type'] in mapping and e['text'] == text[e['start']:e['end']] for e in actual)


