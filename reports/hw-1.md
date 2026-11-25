# ДЗ 1. Воспроизводимый ML-проект

## Что сделано

1. **Data Ingestion** — загрузка с Kaggle в `data/raw/`
2. **Data Audit** — проверка данных, структуры и разметки 
3. **EDA** — исследование распределений, дубликатов и метаданных изображений
4. **Preprocessing** — ресайз, центрирование, нормализация по статистикам, аугментация
5. **Training** — модель `MobileNetV3-Small` с 3-fold CV, есть чекпоинты после каждой эпохи
8. **Evaluation** — оценка качества модели, OOF метрики по группам (`known / unseen_identity / new_whale`)
9. **Prediction** — `artifacts/submission.csv` (top-5 кандидатов на тесте)

## Результаты

- `artifacts/model.pkl` — обученная модель (совместимый output)
- `artifacts/metrics.json` — метрики (совместимый output)
- `artifacts/oof.json` — out-of-fold предсказания (совместимый output)
- `artifacts/runs/<run_id>/` — неизменяемый run bundle с `manifest.json` и хешами артефактов
- `artifacts/latest.json` — атомарный указатель на последний опубликованный run
- `artifacts/submission.csv` — submission на тесте
- `artifacts/checkpoints/` — контекстно-изолированные чекпоинты
- `artifacts/logs/` — логи обучения
- `artifacts/eda/` — артефакты eda

## Out-of-Fold

```bash
{
  "samples": 22175,
  "map_at_5": 0.49728297632469,
  "top_1_accuracy": 0.4487485907553551,
  "top_5_recall": 0.6733709131905299,
  "unknown_first_rate": 0.623630214205186,
  "known_samples": 8175,
  "known_map_at_5": 0.08054434250764526,
  "unseen_identity_samples": 8214,
  "unseen_identity_map_at_5": 0.6343803262722182,
  "new_whale_samples": 5786,
  "new_whale_map_at_5": 0.8914621500172831
}
```

## Скриншоты

![Console](https://raw.githubusercontent.com/CqurFar/2026_AITH_ML_Eng_Practices/main/assets/hw-1%20Console.png)

![Visualize](https://raw.githubusercontent.com/CqurFar/2026_AITH_ML_Eng_Practices/main/assets/hw-1%20Visualize.png)

## Основные библиотеки

- `Python 3.14`, `uv`, `Ruff 0.17.6`, `pre-commit 4.7.0`, `pytest 9.2.0`
- `PyTorch 2.14.1+cu130`, `torchvision 0.29.1+cu130`
- `polars 2.1.0`, `numpy 2.5.5`, `matplotlib 3.11.2`, `Pillow 13.0.0`