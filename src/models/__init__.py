__all__ = [
    "fit_model",
    "main",
    "predict",
    "probabilities",
    "select_threshold",
    "summarize",
    "train",
    "validate_submission",
]

_LAZY = {
    "fit_model": "model_building",
    "main": "model_building",
    "probabilities": "model_building",
    "train": "model_building",
    "predict": "model_prediction",
    "validate_submission": "model_prediction",
    "summarize": "model_evaluation",
    "select_threshold": "model_evaluation",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(f"src.models.{_LAZY[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
