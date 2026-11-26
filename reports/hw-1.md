# ДЗ 1. Воспроизводимый ML-проект

## Что сделано

1. **Data Ingestion** — загрузка с Kaggle в `data/raw/`
2. **Data Audit** — проверка данных, структуры и разметки
3. **EDA** — исследование распределений, дубликатов и метаданных изображений
4. **Preprocessing** — ресайз, центрирование, нормализация по статистикам, аугментация
5. **Training** — модель `MobileNetV3-Small` с 3-fold CV, есть чекпоинты после каждой эпохи
8. **Evaluation** — оценка качества модели, OOF метрики по группам (`known / unseen_identity / new_whale`)
9. **Prediction** — `artifacts/hw1-baseline/submission.csv` (top-5 кандидатов на тесте)

## Результаты

- Commit: `76a354f1bcd799c554a11a3bd2269342b14ff512`
- Run ID: `20261126T044644889736Z`
- `artifacts/hw1-baseline/model.pkl` — обученная модель
- `artifacts/hw1-baseline/metrics.json` — метрики
- `artifacts/hw1-baseline/oof.json` — out-of-fold предсказания
- `artifacts/hw1-baseline/submission.csv` — submission на тесте
- `artifacts/hw1-baseline/checkpoints/fit_<role>_<fold-or-final>_<hash>.pt` — чекпоинты
- `artifacts/hw1-baseline/logs/` — логи обучения

## Out-of-Fold

```bash
{
  "samples": 22133,
  "map_at_5": 0.570978478591545,
  "top_1_accuracy": 0.5114083043419329,
  "top_5_recall": 0.7545746170876068,
  "unknown_first_rate": 0.662043103058781,
  "known_samples": 8155,
  "known_map_at_5": 0.2343919885550787,
  "unseen_identity_samples": 8179,
  "unseen_identity_map_at_5": 0.7420711578432571,
  "new_whale_samples": 5799,
  "new_whale_map_at_5": 0.8030005173305743
}
```

## Скриншоты

![Console](https://raw.githubusercontent.com/CqurFar/2026_AITH_ML_Eng_Practices/main/assets/hw-1%20Console.png)

![Visualize](https://raw.githubusercontent.com/CqurFar/2026_AITH_ML_Eng_Practices/main/assets/hw-1%20Visualize.png)

## Основные библиотеки

- `Python 3.14`, `uv`, `Ruff 0.17.6`, `pre-commit 4.7.0`, `pytest 9.2.0`
- `PyTorch 2.14.1+cu130`, `torchvision 0.29.1+cu130`
- `polars 2.1.0`, `numpy 2.5.5`, `matplotlib 3.11.2`, `Pillow 13.0.0`
