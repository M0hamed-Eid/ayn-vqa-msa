from __future__ import annotations

import json
from pathlib import Path

import httpx

from ayn_vqa.data.schema import Task1aRecord
from ayn_vqa.stages.retrieve import OllamaCategoryRetriever


def _record(id_: str, category: str, subcategory: str) -> Task1aRecord:
    return Task1aRecord(
        id=id_,
        image=f"images/{id_}.jpg",
        audio=f"audio/msa/{id_}.wav",
        label=0,
        country="Egypt",
        category=category,
        subcategory=subcategory,
    )


def _client_returning(category: str, subcategory: str) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = {"category": category, "subcategory": subcategory}
        return httpx.Response(200, json={"message": {"content": json.dumps(payload)}})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_retrieve_from_matching_subcategory_pool(tmp_path: Path) -> None:
    image_path = tmp_path / "a.jpg"
    image_path.write_bytes(b"fake")
    train = [
        _record("t1", "Food & Cooking", "Beverages"),
        _record("t2", "Food & Cooking", "Beverages"),
        _record("t3", "Food & Cooking", "Beverages"),
        _record("t4", "Food & Cooking", "Beverages"),
        _record("t5", "Food & Cooking", "Beverages"),
        _record("t6", "Sports & Recreation", "Outdoor Activities"),
    ]
    client = _client_returning("Food & Cooking", "Beverages")
    retriever = OllamaCategoryRetriever(train, k=2, seed=1, client=client)

    exemplar_ids = retriever.retrieve("query1", image_path)

    assert len(exemplar_ids) == 2
    assert all(eid.startswith("t") and eid != "t6" for eid in exemplar_ids)


def test_falls_back_to_category_when_subcategory_pool_too_small(tmp_path: Path) -> None:
    image_path = tmp_path / "a.jpg"
    image_path.write_bytes(b"fake")
    train = [
        _record("t1", "Food & Cooking", "Beverages"),  # only 1 in this subcategory
        _record("t2", "Food & Cooking", "Cooking & Eating Customs"),
        _record("t3", "Food & Cooking", "Cooking & Eating Customs"),
        _record("t4", "Food & Cooking", "Cooking & Eating Customs"),
    ]
    client = _client_returning("Food & Cooking", "Beverages")
    retriever = OllamaCategoryRetriever(train, k=2, seed=1, client=client, min_pool_size=5)

    exemplar_ids = retriever.retrieve("query1", image_path)

    assert len(exemplar_ids) == 2
    assert set(exemplar_ids).issubset({"t1", "t2", "t3", "t4"})


def test_returns_empty_on_classification_failure(tmp_path: Path) -> None:
    image_path = tmp_path / "a.jpg"
    image_path.write_bytes(b"fake")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    train = [_record("t1", "Food & Cooking", "Beverages")]
    retriever = OllamaCategoryRetriever(train, k=2, seed=1, client=client)

    assert retriever.retrieve("query1", image_path) == ()


def test_returns_empty_when_no_matching_pool(tmp_path: Path) -> None:
    image_path = tmp_path / "a.jpg"
    image_path.write_bytes(b"fake")
    train = [_record("t1", "Sports & Recreation", "Outdoor Activities")]
    client = _client_returning("Food & Cooking", "Beverages")
    retriever = OllamaCategoryRetriever(train, k=2, seed=1, client=client)

    assert retriever.retrieve("query1", image_path) == ()


def test_retrieval_is_deterministic_for_the_same_record_and_seed(tmp_path: Path) -> None:
    image_path = tmp_path / "a.jpg"
    image_path.write_bytes(b"fake")
    train = [_record(f"t{i}", "Food & Cooking", "Beverages") for i in range(10)]
    client = _client_returning("Food & Cooking", "Beverages")
    retriever = OllamaCategoryRetriever(train, k=3, seed=7, client=client)

    first = retriever.retrieve("query1", image_path)
    second = retriever.retrieve("query1", image_path)

    assert first == second


def test_k_larger_than_pool_returns_all_available(tmp_path: Path) -> None:
    image_path = tmp_path / "a.jpg"
    image_path.write_bytes(b"fake")
    train = [
        _record("t1", "Food & Cooking", "Beverages"),
        _record("t2", "Food & Cooking", "Beverages"),
    ]
    client = _client_returning("Food & Cooking", "Beverages")
    retriever = OllamaCategoryRetriever(train, k=10, seed=1, client=client)

    exemplar_ids = retriever.retrieve("query1", image_path)

    assert len(exemplar_ids) == 2
