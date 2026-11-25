# ДЗ 1. Воспроизводимый ML-проект

## Что сделано

1. **Data Ingestion** — загрузка с Kaggle в `data/raw/`.
2. **Data Audit** — проверка CSV, изображений, SHA-256 и дубликатов.
3. **EDA** — распределения меток, дубликаты и метаданные изображений.
4. **Preprocessing** — ресайз с сохранением пропорций, паддинг, uint8 HWC-тензоры `(256,512,3)` и нормализация при загрузке.
5. **Training** — `MobileNetV3-Small`, 3-fold CV по 5 эпох, чекпоинты и финальный refit.
6. **Evaluation** — OOF-метрики по `known / unseen_identity / new_whale`, независимая калибровка каждой когорты и порог неизвестной особи.
7. **Prediction** — top-5 кандидатов для каждого изображения теста.

Финальные исправления review:

- Run bundle хранит `splits_manifest.json`, его хеш и dataset fingerprint в model/metrics.
- Калибровка known, unseen-identity и `new_whale` выполняется независимо; каждой когорте нужно минимум две независимые группы.
- Preprocessing path overrides отклоняют пересечение raw и output.
- `--processed-dir` задаёт `<root>/train`; `--artifacts-dir` задаёт `<root>/checkpoints` и `<root>/logs`.

## Final baseline

- Commit: `335fc0fe6cc5fce53b8f8400d8137fe11f0051c5`
- Run ID: `20261125T213244777337Z`
- GPU: NVIDIA GeForce RTX 3080 Ti 12 GB, CUDA 13.0
- Runtime обучения: `700.73` с (shell runtime 11m44.93s)
- Runtime prediction: `45.09` с
- Existing HWC-тензоры переиспользованы без rebake
- Финальный refit: 15 697 изображений, 5 004 классов
- Финальный порог: `0.2331177145242691`
- Train CSV SHA-256: `0fda63ed906d209ab6815c716fbd24c25b46f390a4ade6d24bac5c42fd491832`
- Dataset fingerprint: `0dd2894fc60c1449a3ef0e027790833a0de22bcd12121c5af63bfba63653973a`

## Результаты

Артефакты находятся в `artifacts/hw1-final/artifacts/`:

- `model.pkl` — модель; SHA-256 `737645ee5ec37b9a274412b48b4deaa9706b1c152ad790a3d5fb838c664b7243`
- `metrics.json` — метрики, provenance, environment и coverage; SHA-256 `ece11dacf377a9fd1afa9f0bc27202214bf6a467a5c2dc709ed46132bb6aaed9`
- `oof.json` — OOF-предсказания; SHA-256 `56620aa9550655d4ca2cc84bc8501309cb709478a80c1a91d02f08f31ab8081b`
- `submission.csv` — 7 960 строк; SHA-256 `777670abb657736dc4a7b1bacf198a72c76deba2dc11350f7326d6e3390e4ab7`
- `latest.json` — SHA-256 `b262e1ede15e0c8f4123421eb9b6774898bf08f6222c59cd158df2b8fe7ae13d`
- `runs/20261125T213244777337Z/manifest.json` — SHA-256 `099c2f78a472c9f96d4ac1d08391ff83b85ca59da0e4795daa39160d74a47bba`
- `runs/20261125T213244777337Z/splits_manifest.json` — SHA-256 `14d065fd05a3fbd3fb3be29fa3896c31247dfc362f43c2568de075ac662e735c`
- `runs/20261125T213244777337Z/` — immutable bundle с `model.pkl`, `metrics.json`, `oof.json`, `environment.json`, `splits_manifest.json` и `manifest.json`
- `latest.json` — атомарный указатель на run
- `checkpoints/` — fold/final чекпоинты
- `logs/` — логи обучения

Prepared-тензоры находятся в основном checkout:
`C:\Users\CqurFar\Desktop\Itmo\Инженерные практики в ML\03. Задания\humpback-whale\data\processed\humpback-whale-identification\train`.

Перед запуском проверены все 25 361 файл: имена соответствуют `train.csv`, dtype `uint8`, shape `(256,512,3)`, `preprocessing.json` соответствует конфигурации. Aggregate SHA-256 тензоров: `e89579436e7a43a791b45574db938653eec5a1764aaf6848645791d29ea61ca2`.

Предыдущий baseline `artifacts/hw1-baseline/` сохранён без изменений: 16 файлов, 339 288 747 байт.

## Out-of-Fold

```json
{
  "samples": 22133,
  "map_at_5": 0.5949818521363274,
  "top_1_accuracy": 0.5408665793159535,
  "top_5_recall": 0.7557041521709664,
  "unknown_first_rate": 0.7074955948131749,
  "known_samples": 8155,
  "known_map_at_5": 0.23489066012671161,
  "unseen_identity_samples": 8179,
  "unseen_identity_map_at_5": 0.7832497860374129,
  "new_whale_samples": 5799,
  "new_whale_map_at_5": 0.8358337644421452
}
```

### Coverage

- OOF-строк: `22 133`; уникальных изображений: `17 270`.
- `4 863` изображения встречаются в OOF дважды в разных протокольных ролях; ключи `(image, cohort, target)` уникальны.
- Calibration-строк: `14 779` — сумма по fold.
- Исключено из OOF: `9 664` уникальных изображения.
- Известных идентичностей только в train: `2 073`.
- Coverage-значения не суммируются в 25 361 из-за разных ролей изображения в разных fold.

Протокол: `custom open-set assignment: digest-grouped CV folds, identity-grouped calibration, digest-grouped new_whale`.

## Воспроизведение baseline

Из корня checkout:

```bash
RAW="C:/Users/CqurFar/Desktop/Itmo/Инженерные практики в ML/03. Задания/humpback-whale/data/raw/humpback-whale-identification"
PROCESSED="C:/Users/CqurFar/Desktop/Itmo/Инженерные практики в ML/03. Задания/humpback-whale/data/processed/humpback-whale-identification"

# Существующие HWC-тензоры уже проверены; bake не требуется.
uv run --locked python -m src.models.model_building --data-dir "$RAW" --processed-dir "$PROCESSED" --artifacts-dir "artifacts/hw1-final/artifacts"
uv run --locked python -m src.models.model_prediction --data-dir "$RAW" --artifacts-dir "artifacts/hw1-final/artifacts"
```

Полные команды, RED/GREEN, логи, runtime и хеши: `.opencode/progress/final-fix-report.md` (local-only).

## Проверки

- `uv run --locked pytest -q`: 113 passed.
- Pre-commit, Ruff lint и Ruff format: passed.
- Train/predict preprocessing parity: 7 passed.
- Run manifest, compatibility outputs, dataset fingerprint, splits manifest, OOF metrics и submission проверены отдельным verifier.
- Submission содержит ровно 7 960 ожидаемых изображений; test labels не использовались.

## Скриншоты

Для final baseline новые скриншоты не снимались. Результаты подтверждаются JSON/CSV, run manifest и сохранёнными логами.

## Основные библиотеки

- `Python 3.14.0`, `uv 0.12.5`, `Ruff 0.17.6`, `pre-commit 4.7.0`, `pytest 9.2.0`
- `PyTorch 2.14.1+cu130`, `torchvision 0.29.1+cu130`
- `polars 2.1.0`, `numpy 2.5.5`, `matplotlib 3.11.2`, `Pillow 13.0.0`

## Ограничения

- DVC не используется; raw, processed, artifacts и progress не входят в Git.
- Пути машино-зависимы; перенос требует явного указания корней.
- Test labels не использовались; leaderboard score не измерялся.
- Это один запуск с seed `42`; CUDA/cuDNN bitwise reproducibility не проверялась повторным обучением.
- Конфигурация `3 folds × 5 epochs` не тюнилась.
- OOF-строки являются протокольными оценками, а не уникальными изображениями.
- Финальный тег не создаётся до пользовательского review.
