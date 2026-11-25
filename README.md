# 2026_AITH_ML_Eng_Practices

## Данные
- **Источник:** [Kaggle - Humpback Whale Identification](https://www.kaggle.com/competitions/humpback-whale-identification)
- **Лицензия:** [Subject to Competition Rules](https://www.kaggle.com/competitions/humpback-whale-identification/rules)
- **Примечание:** соревнование в архиве с 01.03.2019

## Задача

По фотографии хвостовых плавников китов нужно определить конкретную особь
- **Домен:** Компьютерное зрение
- **Задача:** Мультиклассовая классификация
- **Метрика:** MAP@5

## Ограничения

- **Объём данных:** исходный датасет занимает порядка 5ГБ
- **GPU:** полное обучение модели на CUDA занимает около 10-15мин
- **Bake:** чтобы не упираться в CPU-Bound при декодировании и чтении, сначала данные запекаются в uint8 тензоры (7-10мин / 10ГБ)

Параметры препроцессинга и обучения лежат в `config.yaml`.

## Установка и запуск

```bash
git clone https://github.com/CqurFar/2026_AITH_ML_Eng_Practices.git
cd 2026_AITH_ML_Eng_Practices

uv sync --locked                                  # зависимости
uv run --locked pre-commit run --all-files        # проверка кода

uv run --locked kaggle auth login                 # авторизация на kaggle
uv run --locked python -m src.data.get_data       # загрузка датасета
uv run --locked python -m src.data.prepare_data   # проверка данных
uv run --locked python -m src.visualization.integrity   # eda и viz
uv run --locked python -m src.preprocessing.bake  # предобработка и сохранение тензоров

uv run --locked python -m src.models.model_building   # обучение
uv run --locked python -m src.models.model_evaluation # оценка
uv run --locked python -m src.models.model_prediction # предсказания
```

## Есть шимы, если важно, чтобы команды были как в ТЗ

```bash
uv run --locked python -m src.get_data
uv run --locked python -m src.train
```
