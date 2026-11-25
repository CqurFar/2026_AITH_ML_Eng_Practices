import argparse
import sys

from utils import run_module

STAGE_MODULES = {
    "download": ("src.data.get_data",),
    "prepare": ("src.data.prepare_data",),
    "eda": (
        "src.visualization.integrity",
        "src.visualization.metadata",
        "src.visualization.labels",
        "src.visualization.galleries",
        "src.visualization.duplicates",
        "src.visualization.consistency",
    ),
    "bake": ("src.preprocessing.bake",),
    "train": ("src.models.model_building",),
    "evaluate": ("src.models.model_evaluation",),
    "predict": ("src.models.model_prediction",),
}
RUN_ALL_MODULES = tuple(module for modules in STAGE_MODULES.values() for module in modules)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=(*STAGE_MODULES, "run-all"))
    raw_argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(raw_argv[:1])
    modules = RUN_ALL_MODULES if args.command == "run-all" else STAGE_MODULES[args.command]
    for module in modules:
        run_module(module, raw_argv[1:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
