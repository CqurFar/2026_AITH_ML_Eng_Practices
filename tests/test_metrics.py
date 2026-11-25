from pathlib import Path

import pytest
import torch
from src.data import Record
from src.models.model_evaluation import calibration_split, map_at_5, ranked_predictions


@pytest.mark.parametrize("position", [1, 2, 3, 4, 5])
def test_map_at_5_scores_each_of_the_first_five_positions(position: int) -> None:
    candidates = [f"label_{index}" for index in range(6)]

    assert map_at_5([candidates[position - 1]], [candidates]) == pytest.approx(1 / position)


def test_map_at_5_ignores_targets_outside_top_five() -> None:
    assert map_at_5(["label_5"], [[f"label_{index}" for index in range(6)]]) == 0.0


def test_map_at_5_scores_missing_target_as_zero() -> None:
    assert map_at_5(["missing"], [["label_0", "label_1"]]) == 0.0


def test_map_at_5_accepts_short_prediction_lists() -> None:
    assert map_at_5(["label_1"], [["label_0", "label_1"]]) == 0.5


def test_map_at_5_uses_first_occurrence_of_duplicate_candidates() -> None:
    assert map_at_5(["target"], [["wrong", "wrong", "target", "target"]]) == pytest.approx(1 / 3)


def test_map_at_5_treats_new_whale_as_an_ordinary_candidate() -> None:
    assert map_at_5(["new_whale"], [["label_0", "new_whale"]]) == 0.5


def test_map_at_5_averages_per_image_reciprocal_ranks() -> None:
    targets = ["first", "third", "missing"]
    predictions = [
        ["first", "other"],
        ["other", "other_2", "third"],
        ["other", "other_2", "other_3", "other_4", "other_5"],
    ]

    assert map_at_5(targets, predictions) == pytest.approx((1 + 1 / 3) / 3)


def test_calibration_keeps_known_identity_groups_together(tmp_path: Path) -> None:
    records = [
        Record(f"{identity}_{index}.png", identity, tmp_path / f"{identity}_{index}.png", digest)
        for identity in ("whale_a", "whale_b", "whale_c")
        for index, digest in enumerate((f"{identity}-1", f"{identity}-2"))
    ]
    unseen = [
        Record(f"{identity}.png", identity, tmp_path / f"{identity}.png", f"{identity}-digest")
        for identity in ("unseen_a", "unseen_b")
    ] + [
        Record(
            f"unknown_{index}.png",
            "new_whale",
            tmp_path / f"unknown_{index}.png",
            f"unknown-{index}",
        )
        for index in range(2)
    ]

    calibration, evaluation = calibration_split(records, unseen, seed=0, fraction=0.5)

    calibration_identities = {record.identity for record in calibration} & {
        record.identity for record in records
    }
    evaluation_identities = {record.identity for record in evaluation} & {
        record.identity for record in records
    }
    assert calibration_identities.isdisjoint(evaluation_identities)
    assert calibration_identities | evaluation_identities == {"whale_a", "whale_b", "whale_c"}


def test_calibration_groups_unseen_by_identity_and_new_whale_by_digest(tmp_path: Path) -> None:
    known = [
        Record(f"{identity}.png", identity, tmp_path / f"{identity}.png", f"{identity}-digest")
        for identity in ("known_a", "known_b")
    ]
    unseen = [
        Record(
            f"{identity}_{index}.png",
            identity,
            tmp_path / f"{identity}_{index}.png",
            f"{identity}-{index}",
        )
        for identity in ("unseen_a", "unseen_b")
        for index in (1, 2)
    ] + [
        Record(f"unknown_{index}.png", "new_whale", tmp_path / f"unknown_{index}.png", digest)
        for index, digest in enumerate(("unknown-1", "unknown-2", "unknown-3", "unknown-4"), 1)
    ]

    calibration, evaluation = calibration_split(known, unseen, seed=0, fraction=0.5)

    for identity in ("unseen_a", "unseen_b"):
        sides = [
            {record.image for record in cohort if record.identity == identity}
            for cohort in (calibration, evaluation)
        ]
        assert sorted(sides, key=len) == [set(), {f"{identity}_1.png", f"{identity}_2.png"}]
    unknown_sides = [
        {record.image for record in cohort if record.identity == "new_whale"}
        for cohort in (calibration, evaluation)
    ]
    assert all(unknown_sides)
    assert set.union(*unknown_sides) == {f"unknown_{index}.png" for index in range(1, 5)}


def test_calibration_reserves_groups_for_each_unknown_cohort(tmp_path: Path) -> None:
    known = [
        Record(
            f"known_{index}.png",
            f"known_{index}",
            tmp_path / f"known_{index}.png",
            f"known-{index}",
        )
        for index in range(2)
    ]
    unseen = [
        Record(
            f"unseen_{index}.png",
            f"unseen_{index}",
            tmp_path / f"unseen_{index}.png",
            f"unseen-{index}",
        )
        for index in range(20)
    ]
    unknown = [
        Record(
            f"unknown_{index}.png",
            "new_whale",
            tmp_path / f"unknown_{index}.png",
            f"unknown-{index}",
        )
        for index in range(2)
    ]

    calibration, evaluation = calibration_split(known, [*unseen, *unknown], seed=5, fraction=0.4)

    calibration_unknown = {record.image for record in calibration if record.identity == "new_whale"}
    evaluation_unknown = {record.image for record in evaluation if record.identity == "new_whale"}
    assert calibration_unknown
    assert evaluation_unknown
    assert calibration_unknown | evaluation_unknown == {record.image for record in unknown}
    for cohort in (calibration, evaluation):
        assert any(record.identity.startswith("known_") for record in cohort)
        assert any(record.identity.startswith("unseen_") for record in cohort)


@pytest.mark.parametrize("cohort", ["known", "unseen_identity", "new_whale"])
def test_calibration_requires_two_groups_in_each_cohort(tmp_path: Path, cohort: str) -> None:
    known = [
        Record(
            f"known_{index}.png",
            f"known_{index}",
            tmp_path / f"known_{index}.png",
            f"known-{index}",
        )
        for index in range(2)
    ]
    unseen = [
        Record(
            f"unseen_{index}.png",
            f"unseen_{index}",
            tmp_path / f"unseen_{index}.png",
            f"unseen-{index}",
        )
        for index in range(2)
    ]
    unknown = [
        Record(
            f"unknown_{index}.png",
            "new_whale",
            tmp_path / f"unknown_{index}.png",
            f"unknown-{index}",
        )
        for index in range(2)
    ]
    if cohort == "known":
        known = known[:1]
    elif cohort == "unseen_identity":
        unseen = unseen[:1]
    else:
        unknown = unknown[:1]

    with pytest.raises(ValueError, match=cohort):
        calibration_split(known, [*unseen, *unknown], seed=5, fraction=0.4)


def test_unknown_calibration_is_independent_of_known_cohort_size(tmp_path: Path) -> None:
    known = [
        Record(
            f"known_{index}.png",
            f"known_{index}",
            tmp_path / f"known_{index}.png",
            f"known-{index}",
        )
        for index in range(2)
    ]
    unseen = [
        Record(
            f"unseen_{index}.png",
            f"unseen_{index}",
            tmp_path / f"unseen_{index}.png",
            f"unseen-{index}",
        )
        for index in range(20)
    ]
    unknown = [
        Record(
            f"unknown_{index}.png",
            "new_whale",
            tmp_path / f"unknown_{index}.png",
            f"unknown-{index}",
        )
        for index in range(2)
    ]
    extra_known = [
        Record(
            f"known_{index}.png",
            f"known_{index}",
            tmp_path / f"known_{index}.png",
            f"known-{index}",
        )
        for index in range(2, 4)
    ]

    first_calibration, first_evaluation = calibration_split(known, [*unseen, *unknown], 5, 0.4)
    second_calibration, second_evaluation = calibration_split(
        [*known, *extra_known], [*unseen, *unknown], 5, 0.4
    )

    def unknown_sides(
        calibration: list[Record], evaluation: list[Record]
    ) -> tuple[set[str], set[str]]:
        return (
            {record.image for record in calibration if not record.identity.startswith("known_")},
            {record.image for record in evaluation if not record.identity.startswith("known_")},
        )

    assert unknown_sides(first_calibration, first_evaluation) == unknown_sides(
        second_calibration, second_evaluation
    )


def test_ranked_predictions_reserve_one_of_five_slots_for_new_whale() -> None:
    classes = [f"label_{index}" for index in range(6)]
    probabilities = torch.tensor([[0.40, 0.25, 0.15, 0.10, 0.06, 0.04]])

    assert ranked_predictions(probabilities, classes, threshold=0.5) == [
        ["new_whale", "label_0", "label_1", "label_2", "label_3"]
    ]
    assert ranked_predictions(probabilities, classes, threshold=0.0) == [
        ["label_0", "label_1", "label_2", "label_3", "new_whale"]
    ]
