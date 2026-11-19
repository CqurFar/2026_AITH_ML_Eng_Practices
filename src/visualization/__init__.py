__all__ = [
    "consistency",
    "duplicates",
    "galleries",
    "integrity",
    "labels",
    "metadata",
]


def __getattr__(name: str):
    if name in __all__:
        import importlib

        return importlib.import_module(f"src.visualization.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
