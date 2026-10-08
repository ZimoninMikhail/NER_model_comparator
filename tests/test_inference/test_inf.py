"""
Тест инференса модели.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference import InferencePipeline

pipeline = InferencePipeline(
    model_version=None,  # последняя модель
    tokenizer_name="DeepPavlov/rubert-base-cased",
    device="cpu",
    max_length=512
)

test_texts = [
    "Я живу в Москве.",
    "Санкт-Петербург — красивый город.",
    "Он поехал в Новосибирск на конференцию.",
    "Владивосток находится на Дальнем Востоке.",
    "Этот город не имеет географических названий."
]

print("\n" + "="*50)
print("Тестирование инференса")
print("="*50)

for text in test_texts:
    entities = pipeline.predict(text)
    print(f"\nТекст: {text}")
    if entities:
        for ent in entities:
            print(f"  {ent['type']}: '{ent['text']}' (позиции {ent['start']}-{ent['end']})")
    else:
        print("  Сущности не найдены")