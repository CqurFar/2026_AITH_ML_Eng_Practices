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

## HW-1 baseline

- Commit: `2e619bb0a453933193c5439924548cd7364af8e0`
- Run ID: `20261126T044644889736Z`
- OOF MAP@5: `0.570978478591545`
- Артефакты: `artifacts/hw1-baseline/`
- Отчёт: [reports/hw-1.md](reports/hw-1.md)

## Установка и запуск

Требуется Python 3.11+.

```bash
git clone https://github.com/CqurFar/2026_AITH_ML_Eng_Practices.git
cd 2026_AITH_ML_Eng_Practices

uv sync --locked                                  # зависимости
uv run --locked pre-commit run --all-files        # проверка кода

uv run --locked kaggle auth login                 # авторизация на kaggle
uv run --locked python -m src.main --help         # список стадий
uv run --locked python -m src.main download       # загрузка датасета
uv run --locked python -m src.main prepare        # проверка данных
uv run --locked python -m src.main eda            # eda и viz
uv run --locked python -m src.main bake           # предобработка и сохранение тензоров
uv run --locked python -m src.main train          # обучение
uv run --locked python -m src.main evaluate       # оценка
uv run --locked python -m src.main predict        # предсказания
uv run --locked python -m src.main run-all        # весь пайплайн
```

## Есть шимы, если важно, чтобы команды были как в ТЗ

```bash
uv run --locked python -m src.get_data
uv run --locked python -m src.train
```
