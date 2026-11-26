import sys

import pytest
from src import main as cli

SINGLE_STAGES = {
    "download": "src.data.get_data",
    "prepare": "src.data.prepare_data",
    "bake": "src.preprocessing.bake",
    "train": "src.models.model_building",
    "evaluate": "src.models.model_evaluation",
    "predict": "src.models.model_prediction",
}
EDA_MODULES = [
    "src.visualization.integrity",
    "src.visualization.metadata",
    "src.visualization.labels",
    "src.visualization.galleries",
    "src.visualization.duplicates",
    "src.visualization.consistency",
]
RUN_ALL_MODULES = [
    SINGLE_STAGES["download"],
    SINGLE_STAGES["prepare"],
    *EDA_MODULES,
    SINGLE_STAGES["bake"],
    SINGLE_STAGES["train"],
    SINGLE_STAGES["evaluate"],
    SINGLE_STAGES["predict"],
]


@pytest.mark.parametrize(("command", "module"), SINGLE_STAGES.items())
def test_stage_routes_to_module_and_passes_args_unchanged(
    command: str, module: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr(cli, "run_module", lambda name, args=None: calls.append((name, args)))
    stage_args = ["--help", "--data-dir", "custom data", "--flag=value", "--", "tail"]

    assert cli.main([command, *stage_args]) == 0

    assert calls == [(module, stage_args)]


def test_stage_preserves_leading_option_terminator(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(cli, "run_module", lambda name, args=None: calls.append((name, args)))
    stage_args = ["--", "--flag", "value"]

    assert cli.main(["predict", *stage_args]) == 0

    assert calls == [("src.models.model_prediction", stage_args)]


def test_eda_runs_visualization_modules_in_order_and_passes_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(cli, "run_module", lambda name, args=None: calls.append((name, args)))
    stage_args = ["--example", "value"]

    assert cli.main(["eda", *stage_args]) == 0

    assert calls == [(module, stage_args) for module in EDA_MODULES]


def test_run_all_runs_pipeline_in_order_and_passes_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(cli, "run_module", lambda name, args=None: calls.append((name, args)))
    stage_args = ["--example", "value"]

    assert cli.main(["run-all", *stage_args]) == 0

    assert calls == [(module, stage_args) for module in RUN_ALL_MODULES]


def test_help_lists_every_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    for command in (*SINGLE_STAGES, "eda", "run-all"):
        assert command in output


def test_main_defaults_to_process_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(sys, "argv", ["main.py", "predict", "--output", "submission.csv"])
    monkeypatch.setattr(cli, "run_module", lambda name, args=None: calls.append((name, args)))

    assert cli.main() == 0

    assert calls == [("src.models.model_prediction", ["--output", "submission.csv"])]
