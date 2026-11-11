# Сравнение моделей NER для русского языка

ruBERT, BiLSTM-CRF и GLiNER без дообучения. Датасеты: NEREL и NEREL-BIO.
Основная метрика — macro-F1 по точному совпадению границ и типа сущности.

## Установка

Python 3.9. Зависимости закреплены в `requirements.txt`.

```bash
conda create -n Sci_env python=3.9
conda activate Sci_env
python -m pip install -r requirements.txt
```
## Данные

[NEREL](https://github.com/nerel-ds/NEREL) или
[NEREL-BIO](https://github.com/nerel-ds/NEREL-BIO).
Версия, commit и путь указаны в `config/datasets/`.
Локальная копия должна содержать `train/`, `dev/`, `test/` с парами `.txt/.ann`,
`annotation.conf` и `SOURCE_COMMIT.txt` с полным SHA commit.

```bash
python run.py -m type-statistics --dataset nerel
python run.py -m prepare --dataset nerel
```

## Обучение

```bash
# ruBERT
python run.py -m train --dataset nerel --dataset-version DATA_ID \
  --config config/config.yaml --device cpu -l -mt -p

# BiLSTM-CRF
python run.py -m train --dataset nerel --dataset-version DATA_ID \
  --config config/bilstm_crf.yaml --device cpu -l -mt -p
```

Для GPU доступна замена `--device cpu` на `--device cuda`.
Модели сохраняются в `data/models/`, лучший checkpoint выбирается по dev.

## Инференс

```bash
python run.py -m inference --model-version MODEL_ID --device cpu \
  -t "Иван живёт в Москве." -hi
```

Для GLiNER требуется локальный комплект готовой модели:

```python
from src.models.gliner import load_model

model = load_model("urchade/gliner_multi-v2.1", device="cpu",
                   local_files_only=False,
                   revision="443d26d654e0324125a96bebd8e796c14ff2efe6",
                   cache_dir="data/pretrained/cache")
model.save_pretrained("data/pretrained/gliner_multi-v2.1", safe_serialization=True)
```

```bash
python run.py -m inference --config config/gliner.yaml --device cpu \
  -t "Иван живёт в Москве." -hi
```

Типы и порог GLiNER задаются в `config/gliner.yaml`.

## Оценка и скорость

```bash
# BiLSTM-CRF: конфиг config/bilstm_crf.yaml и соответствующий MODEL_ID
python run.py -m evaluate --dataset nerel --config config/config.yaml \
  --dataset-version DATA_ID --model-version MODEL_ID --split test --device cpu -mt

python run.py -m evaluate --dataset nerel --config config/gliner.yaml \
  --dataset-version DATA_ID --split test --device cpu -mt

python run.py -m benchmark --model-version MODEL_ID --device cpu \
  -f inference_data/test_texts.txt --warmup 2 --repeats 5 -b

# GLiNER: параметр --config config/gliner.yaml вместо --model-version MODEL_ID
python run.py -m summary -mt -c
```

Сравнение скорости проводится на одинаковых текстах и устройстве.
Длинные документы разбиваются на неперекрывающиеся фрагменты;
сущности на границах могут распознаваться частично.
Данные и веса хранятся в `data/`, отчёты — в `reports/`. Оба каталога исключены из Git.

## Тесты

```bash
python -m pip install pytest
python -B -m pytest tests -q
```

Все параметры CLI: `python run.py --help`.
