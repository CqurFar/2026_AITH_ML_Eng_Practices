# ДЗ 1. Воспроизводимый ML-проект

## Что сделано

1. **Data Ingestion** — загрузка с Kaggle в `data/raw/`.
2. **Data Audit** — проверка структуры CSV, изображений, SHA-256 и дубликатов.
3. **EDA** — распределения меток, дубликаты и метаданные изображений.
4. **Preprocessing** — ресайз с сохранением пропорций, паддинг, uint8 HWC-тензоры `(256,512,3)` и нормализация при загрузке.
5. **Training** — `MobileNetV3-Small`, 3-fold CV по 5 эпох, чекпоинты после каждой эпохи и финальный refit.
6. **Evaluation** — OOF-метрики по когортам `known / unseen_identity / new_whale`, held-out calibration и порог неизвестной особи.
7. **Prediction** — top-5 кандидатов для каждого изображения теста.

## Corrected baseline

- Commit: `6fe7255f178a5737cbc35f133ff3c92f7b056b2c`
- Run ID: `20261125T205948944795Z`
- GPU: NVIDIA GeForce RTX 3080 Ti 12 GB, CUDA 13.0
- Runtime обучения: `731.88` с; runtime prediction: `44.90` с
- Финальный refit: 15 697 изображений, 5 004 классов
- Финальный порог: `0.2167438194155693`
- Train CSV SHA-256: `0fda63ed906d209ab6815c716fbd24c25b46f390a4ade6d24bac5c42fd491832`
- Dataset fingerprint: `0dd2894fc60c1449a3ef0e027790833a0de22bcd12121c5af63bfba63653973a`

## Результаты

Артефакты находятся в `artifacts/hw1-baseline/artifacts/`:

- `model.pkl` — обученная модель; SHA-256 `04baaf1394d0931d186bfa4cdd7c9631b8fb538ffdee251f0fa71f049f5460e9`
- `metrics.json` — метрики, provenance, environment и coverage; SHA-256 `95c1c6139e361aa1e808dda2a84bc49d0b797b344ddcc2283c14a94bdeffbd18`
- `oof.json` — out-of-fold предсказания; SHA-256 `a4c652c5c9de30b9c647fbea5f7e295a8ad0dd3a6881c4d2ee8ec94895cd9f90`
- `submission.csv` — 7 960 строк теста; SHA-256 `87764ce51e790eb61e781ea173c4d51629d4fc4e3a70e4e1ee57dd2eebba32ee`
- `runs/20261125T205948944795Z/` — неизменяемый run bundle с `manifest.json`, `environment.json` и хешами артефактов
- `latest.json` — атомарный указатель на опубликованный run
- `checkpoints/` — чекпоинты fold/final
- `logs/` — логи обучения

Prepared-тензоры находятся в основном checkout:
`C:\Users\CqurFar\Desktop\Itmo\Инженерные практики в ML\03. Задания\humpback-whale\data\processed\humpback-whale-identification\train`.

Проверены все 25 361 файл: имена соответствуют `train.csv`, dtype `uint8`, shape `(256,512,3)`, `preprocessing.json` соответствует конфигурации.

## Out-of-Fold

```json
{
  "samples": 22087,
  "map_at_5": 0.5915802659181117,
  "top_1_accuracy": 0.536831620410196,
  "top_5_recall": 0.7551953637886539,
  "unknown_first_rate": 0.7003214560601259,
  "known_samples": 8155,
  "known_map_at_5": 0.2357122419783364,
  "unseen_identity_samples": 8141,
  "unseen_identity_map_at_5": 0.7781107971993613,
  "new_whale_samples": 5791,
  "new_whale_map_at_5": 0.8304955966154377
}
```

### Coverage

- OOF-строк: `22 087`; уникальных изображений: `17 288`.
- `4 799` изображений встречаются в OOF дважды: в одном fold как `known`, в другом как `unseen_identity`. Это разные протокольные оценки, а не дубликаты одной строки.
- Calibration-строк: `14 825` — сумма по fold, не уникальные изображения.
- Исключено из OOF: `9 664` уникальных изображения.
- Известных идентичностей только в train: `2 073`.
- Итоговые coverage-значения не обязаны суммироваться в 25 361 из-за разных ролей одного изображения в разных fold.

Протокол: `custom open-set assignment: digest-grouped CV folds, identity-grouped calibration, digest-grouped new_whale`.

## Воспроизведение baseline

Из корня этого checkout:

```bash
RAW="C:/Users/CqurFar/Desktop/Itmo/Инженерные практики в ML/03. Задания/humpback-whale/data/raw/humpback-whale-identification"
PROCESSED="C:/Users/CqurFar/Desktop/Itmo/Инженерные практики в ML/03. Задания/humpback-whale/data/processed/humpback-whale-identification"

# Только если prepared-тензоры отсутствуют или не прошли валидацию:
uv run --locked python -m src.preprocessing.bake --data-dir "$RAW" --train-dir "$PROCESSED/train"

uv run --locked python -m src.models.model_building --data-dir "$RAW" --processed-dir "$PROCESSED" --artifacts-dir "artifacts/hw1-baseline/artifacts"
uv run --locked python -m src.models.model_prediction --data-dir "$RAW" --artifacts-dir "artifacts/hw1-baseline/artifacts"
```

Полные команды, логи, runtime, хеши и проверки: `.opencode/progress/task-8-report.md` и `.opencode/progress/hw1-baseline-run.md` (local-only).

## Скриншоты

Для corrected baseline новые скриншоты не снимались. Результаты подтверждаются сгенерированными `metrics.json`, `oof.json`, `submission.csv`, run manifest и сохранёнными логами.

## Основные библиотеки

- `Python 3.14.0`, `uv 0.12.5`, `Ruff 0.17.6`, `pre-commit 4.7.0`, `pytest 9.2.0`
- `PyTorch 2.14.1+cu130`, `torchvision 0.29.1+cu130`
- `polars 2.1.0`, `numpy 2.5.5`, `matplotlib 3.11.2`, `Pillow 13.0.0`

## Ограничения

- DVC не используется; raw, processed и artifacts не входят в Git.
- Пути к raw/processed машино-зависимы; перенос требует явного указания корней.
- Оценка на test недоступна: test labels не использовались, leaderboard score не измерялся.
- Это один запуск с seed `42`; CUDA/cuDNN bitwise reproducibility не проверялась повторным запуском.
- Конфигурация `3 folds × 5 epochs` не тюнилась.
- Старые CHW-тензоры `(3,256,512)` несовместимы с актуальным storage contract; они перезаписаны HWC в основном checkout.
- Финальный тег `hw-1.1` не создаётся до пользовательского review.
