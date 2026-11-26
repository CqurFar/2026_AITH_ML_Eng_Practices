import json
import logging
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from src import utils
from src.utils import (
    canonical_json,
    file_logger,
    fingerprint,
    read_json,
    run_module,
    sha256_file,
    utc_run_id,
    write_json_atomic,
)


def test_canonical_json_sorts_compacts_and_preserves_existing_escaping() -> None:
    assert canonical_json({"b": "é", "a": 1}) == '{"a":1,"b":"\\u00e9"}'


def test_canonical_json_rejects_non_finite_numbers() -> None:
    with pytest.raises(ValueError):
        canonical_json({"value": float("nan")})


def test_fingerprint_hashes_canonical_json() -> None:
    expected = (
        "43258cff783fe7036d8a43033f830adf"  # pragma: allowlist secret
        "c60ec037382473548ac742b888292777"  # pragma: allowlist secret
    )
    assert fingerprint({"b": 2, "a": 1}) == expected


def test_read_json_reads_utf8_value(tmp_path: Path) -> None:
    path = tmp_path / "value.json"
    path.write_text('{"name": "кот", "items": [1, 2]}', encoding="utf-8")

    assert read_json(path) == {"name": "кот", "items": [1, 2]}


def test_fsync_file_calls_fsync_for_real_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "value.bin"
    path.write_bytes(b"abc")
    descriptors: list[int] = []
    real_fsync = os.fsync

    def tracked_fsync(descriptor: int) -> None:
        descriptors.append(descriptor)
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", tracked_fsync)

    assert utils.fsync_file(path) is None

    assert len(descriptors) == 1
    assert path.read_bytes() == b"abc"


def test_write_json_atomic_replaces_target_from_same_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "nested" / "value.json"
    path.parent.mkdir()
    path.write_text('{"old": true}\n', encoding="utf-8")
    replacements: list[tuple[Path, Path]] = []
    real_replace = Path.replace

    def tracked_replace(source, target):
        replacements.append((Path(source), Path(target)))
        return real_replace(source, target)

    monkeypatch.setattr(Path, "replace", tracked_replace)

    assert write_json_atomic(path, {"new": True}) is None

    assert read_json(path) == {"new": True}
    assert len(replacements) == 1
    source, target = replacements[0]
    assert source.parent == target.parent == path.parent
    assert source != target
    assert not source.exists()


def test_write_json_atomic_fsyncs_before_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "value.json"
    events: list[str] = []
    real_fsync = os.fsync
    real_replace = Path.replace

    def tracked_fsync(descriptor: int) -> None:
        real_fsync(descriptor)
        events.append("fsync")

    def tracked_replace(source, target):
        events.append("replace")
        return real_replace(source, target)

    monkeypatch.setattr(os, "fsync", tracked_fsync)
    monkeypatch.setattr(Path, "replace", tracked_replace)

    write_json_atomic(path, {"new": True})

    assert events == ["fsync", "replace"]
    assert read_json(path) == {"new": True}


def test_write_json_atomic_failure_preserves_target_and_cleans_temporary(tmp_path: Path) -> None:
    path = tmp_path / "value.json"
    original = b'{"old": true}\n'
    path.write_bytes(original)

    with pytest.raises(ValueError):
        write_json_atomic(path, {"bad": float("nan")})

    assert path.read_bytes() == original
    assert list(tmp_path.rglob("*.tmp")) == []


def test_write_json_atomic_honors_indent_sorting_and_utf8(tmp_path: Path) -> None:
    path = tmp_path / "value.json"

    write_json_atomic(path, {"b": "é", "a": 1}, indent=4, sort_keys=True)

    assert path.read_bytes() == '{\n    "a": 1,\n    "b": "é"\n}\n'.encode("utf-8")


def test_sha256_file_matches_known_literal_with_any_chunk_size(tmp_path: Path) -> None:
    path = tmp_path / "value.bin"
    path.write_bytes(b"abc")
    expected = (
        "ba7816bf8f01cfea414140de5dae2223"  # pragma: allowlist secret
        "b00361a396177a9cb410ff61f20015ad"  # pragma: allowlist secret
    )

    assert sha256_file(path) == expected
    assert sha256_file(path, chunk_size=2) == expected


def test_file_logger_writes_and_reuses_one_file_handler(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "run.log"
    name = f"utils-test-{tmp_path.name}"

    try:
        logger = file_logger(path, name)
        again = file_logger(path, name)
        handlers = [
            handler for handler in logger.handlers if isinstance(handler, logging.FileHandler)
        ]

        assert logger is again
        assert len(handlers) == 1
        logger.info("logger works")
        handlers[0].flush()
        assert "logger works" in path.read_text(encoding="utf-8")
    finally:
        logger = logging.getLogger(name)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()


def test_utc_run_id_has_microsecond_format_and_current_utc_time() -> None:
    before = datetime.now(UTC)

    run_id = utc_run_id()

    after = datetime.now(UTC)
    assert re.fullmatch(r"\d{8}T\d{6}\d{6}Z", run_id)
    parsed = datetime.strptime(run_id, "%Y%m%dT%H%M%S%fZ").replace(tzinfo=UTC)
    assert before <= parsed <= after


def test_run_module_uses_current_python_and_passes_args_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = "r2_utils_probe"
    (tmp_path / f"{module}.py").write_text(
        "import json\n"
        "import sys\n"
        "from pathlib import Path\n"
        "Path(__file__).with_name('probe.json').write_text(\n"
        "    json.dumps({'argv': sys.argv[1:], 'executable': sys.executable}),\n"
        "    encoding='utf-8',\n"
        ")\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    stage_args = ["--flag", "value with spaces", "--", "tail"]

    assert run_module(module, stage_args) is None

    probe = json.loads((tmp_path / "probe.json").read_text(encoding="utf-8"))
    assert probe["argv"] == stage_args
    assert probe["executable"] == sys.executable
