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

- **GPU:** обучение модели на CUDA занимает около 30-40мин
- **CPU:** обучение на CPU существенно увеличит время обучения
- **Объём данных:** исходный датасет занимает порядка 5ГБ, тензоры в uint8 еще 10ГБ

## Решение

Основные этапы:

1. **Data Ingestion** - загрузка с Kaggle в `data/raw/`
2. **Data Audit** - проверка данных, структуры и разметки 
3. **EDA** - исследование распределений, дубликатов и метаданных изображений
4. **Preprocessing** - ресайз, центрирование, нормализация по статистикам, аугментация
5. **Training** - модель `MobileNetV3-Small` с 3-fold CV, есть чекпоинты после каждой эпохи
8. **Evaluation** - оценка качества модели, OOF метрики по группам (`known / unseen_identity / new_whale`)
9. **Prediction** — `artifacts/submission.csv` (top-5 кандидатов на тесте)

Параметры препроцессинга и обучения лежат в `config.yaml`

## Установка и запуск

```bash
git clone https://github.com/CqurFar/2026_AITH_ML_Eng_Practices.git
cd 2026_AITH_ML_Eng_Practices

uv sync --locked                                  # зависимости
uv run pre-commit run --all-files                 # проверка кода

uv run kaggle auth login                          # авторизация на kaggle
uv run python -m src.data.get_data                # загрузка датасета
uv run python -m src.data.prepare_data            # проверка данных
uv run python -m src.visualization.integrity      # eda и viz
uv run python -m src.preprocessing.bake           # предобработка и сохранение тензоров

uv run python -m src.models.model_building        # обучение
uv run python -m src.models.model_evaluation      # оценка
uv run python -m src.models.model_prediction      # предсказания 
```

## Если важно как в ТЗ можно ввести

```bash
uv run python -m src.get_data
uv run python -m src.train
```

## Результаты

- `artifacts/model.pkl` — обученная модель
- `artifacts/metrics.json` — метрики
- `artifacts/oof.json` — out-of-fold предсказания
- `artifacts/submission.csv` — submission на тесте
- `artifacts/checkpoints/fit_model.pt` — чекпоинты
- `artifacts/logs/` — логи обучения
- `artifacts/eda/` — артефакты eda
