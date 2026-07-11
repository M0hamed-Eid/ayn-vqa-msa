import pytest

from ayn_vqa.data.schema import Task1aRecord
from ayn_vqa.stages.select import ConstantSelector, RandomSelector, predict_split


def _record(record_id: str) -> Task1aRecord:
    return Task1aRecord.model_validate(
        {"id": record_id, "image": f"images/{record_id}.jpg", "audio": f"audio/msa/{record_id}.wav"}
    )


def test_random_selector_is_deterministic_given_a_seed() -> None:
    records = [_record(f"r{i}") for i in range(50)]

    # One selector instance per run, called across all 50 records -- exactly
    # how `predict_split` uses it. Constructing a fresh `RandomSelector` per
    # record would only ever draw the first value from a freshly-seeded RNG,
    # testing nothing about the sequence.
    selector_a = RandomSelector(seed=7)
    first_run = [selector_a.predict(r).pred for r in records]
    selector_b = RandomSelector(seed=7)
    second_run = [selector_b.predict(r).pred for r in records]

    assert first_run == second_run
    assert all(pred in (0, 1, 2) for pred in first_run)


def test_random_selector_different_seeds_usually_differ() -> None:
    records = [_record(f"r{i}") for i in range(50)]

    selector_1 = RandomSelector(seed=1)
    run_a = [selector_1.predict(r).pred for r in records]
    selector_2 = RandomSelector(seed=2)
    run_b = [selector_2.predict(r).pred for r in records]

    assert run_a != run_b


@pytest.mark.parametrize("value", [0, 1, 2])
def test_constant_selector_always_returns_its_value(value: int) -> None:
    selector = ConstantSelector(value=value)
    records = [_record(f"r{i}") for i in range(10)]

    preds = [selector.predict(r).pred for r in records]

    assert preds == [value] * 10


@pytest.mark.parametrize("bad_value", [-1, 3, 10])
def test_constant_selector_rejects_out_of_range_value(bad_value: int) -> None:
    with pytest.raises(ValueError, match="0, 1, or 2"):
        ConstantSelector(value=bad_value)


def test_predict_split_preserves_record_order() -> None:
    records = [_record("a"), _record("b"), _record("c")]
    predictions = predict_split(ConstantSelector(value=1), records)

    assert [p.record_id for p in predictions] == ["a", "b", "c"]
    assert all(p.pred == 1 for p in predictions)
