import pytest
from src.models.model_prediction import validate_submission

CLASSES = ["whale_a", "whale_b", "whale_c"]


def test_submission_accepts_exact_image_set_and_canonical_candidates() -> None:
    rows = [
        {"Image": "test_1.png", "Id": "whale_b new_whale whale_a"},
        {"Image": "test_0.png", "Id": "new_whale"},
    ]

    validate_submission(rows, ["test_0.png", "test_1.png"], CLASSES)


@pytest.mark.parametrize(
    ("rows", "expected", "message"),
    [
        ([{"Image": "test_0.png", "Id": "whale_a"}], ["test_0.png", "test_1.png"], "images"),
        (
            [
                {"Image": "test_0.png", "Id": "whale_a"},
                {"Image": "test_1.png", "Id": "whale_b"},
                {"Image": "extra.png", "Id": "whale_a"},
            ],
            ["test_0.png", "test_1.png"],
            "images",
        ),
        (
            [
                {"Image": "test_0.png", "Id": "whale_a"},
                {"Image": "test_0.png", "Id": "whale_b"},
            ],
            ["test_0.png", "test_1.png"],
            "images",
        ),
        ([{"Image": "test_0.png", "Id": ""}], ["test_0.png"], "between one and five"),
        (
            [{"Image": "test_0.png", "Id": "a b c d e f"}],
            ["test_0.png"],
            "between one and five",
        ),
        (
            [{"Image": "test_0.png", "Id": "whale_a whale_a"}],
            ["test_0.png"],
            "duplicate or unknown",
        ),
        ([{"Image": "test_0.png", "Id": "unknown"}], ["test_0.png"], "duplicate or unknown"),
        ([{"Image": "test_0.png", "Id": "whale_a  whale_b"}], ["test_0.png"], "single spaces"),
        ([{"Image": "test_0.png", "Id": " whale_a"}], ["test_0.png"], "single spaces"),
        ([{"Image": "test_0.png", "Id": "whale_a "}], ["test_0.png"], "single spaces"),
        ([{"Image": "test_0.png", "Id": "whale_a\twhale_b"}], ["test_0.png"], "single spaces"),
        ([{"Image": "test_0.png", "Id": "whale_a\nwhale_b"}], ["test_0.png"], "single spaces"),
    ],
)
def test_submission_rejects_invalid_image_sets_or_candidate_formatting(
    rows: list[dict[str, str]], expected: list[str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_submission(rows, expected, CLASSES)
