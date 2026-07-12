"""Few-shot exemplar retrieval (M5): given a query image, find k training
records to show the select-stage VLM as worked examples ahead of the real
question.

Retrieves by VLM-predicted category/subcategory rather than image
embeddings -- this project has never needed a torch/CLIP dependency (only
faster-whisper's ctranslate2 backend), and predicted category is the only
signal legitimately available at inference time. Using the query's *true*
category/subcategory would be simpler, but dev/train are the only splits
that carry it; devtest/test don't (see `Task1aRecord`'s docstring in
`data/schema.py`), so retrieval can't depend on ground truth the eventual
scored splits won't have.

The category/subcategory taxonomy below is the exact set found by
enumerating all 3000 `train_msa.jsonl` records (see
docs/M5_FEWSHOT_RETRIEVAL.md) -- fixed and closed, not learned.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ayn_vqa.data.schema import Task1aRecord
from ayn_vqa.ollama_client import OllamaClient

CATEGORY_SUBCATEGORIES: dict[str, tuple[str, ...]] = {
    "Religion & Spirituality": (
        "Holy Texts & Scriptures",
        "Places of Worship",
        "Rituals & Ceremonies",
        "Spiritual Practices",
    ),
    "People, Society & Education": (
        "Community & Civic Life",
        "Healthcare & Well-being",
        "Occupations & Careers",
        "People & Everyday Life",
        "Social Interaction",
    ),
    "History, Geography & National Identity": (
        "Geography & Cultural Regions",
        "Historical Narratives",
        "National Symbols & Flags",
    ),
    "Food & Cooking": (
        "Beverages",
        "Cooking & Eating Customs",
        "Traditional & Regional Cuisines",
    ),
    "Culture, Arts & Entertainment": (
        "Festivals & Celebrations",
        "Film & Animation",
        "Heritage & History",
        "Modern Culture & Trends",
        "Music",
        "Traditional Arts & Crafts",
    ),
    "Vehicles & Transportation": ("Air Travel", "Public Transport", "Water Transport"),
    "Sports & Recreation": ("Outdoor Activities", "Traditional Sports", "eSports & Gaming"),
    "Objects, Materials & Clothing": ("Clothing & Fashion", "Everyday Objects"),
    "Geography, Buildings & Landmarks": ("Architecture & Design", "Famous Landmarks"),
}

_ALL_SUBCATEGORIES = tuple(
    sorted({sub for subs in CATEGORY_SUBCATEGORIES.values() for sub in subs})
)

_CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": list(CATEGORY_SUBCATEGORIES)},
        "subcategory": {"type": "string", "enum": list(_ALL_SUBCATEGORIES)},
    },
    "required": ["category", "subcategory"],
}

_CLASSIFY_PROMPT = """Look at this image and classify it into exactly one of these \
categories: {categories}

Then give your best-guess subcategory for it. Reply with just the category \
and subcategory."""


@dataclass(frozen=True)
class Exemplar:
    """One hydrated few-shot example: a training record's image, its
    question/options (from running it through the same ASR+parse stages as
    any other record), and its correct answer -- taken directly from the
    train label, no VLM call needed for that part."""

    record_id: str
    image_path: Path
    question: str
    options: tuple[str, str, str]
    correct_option_text: str


class ExemplarRetriever(Protocol):
    name: str

    def retrieve(self, record_id: str, image_path: Path) -> tuple[str, ...]:
        """Returns up to k training record ids to use as exemplars for
        this query -- ids only, not hydrated `Exemplar`s. Hydration needs
        ASR+parse, which belongs to `run_pipeline.py`'s stage-running
        machinery, not the retrieval concern itself."""
        ...


class OllamaCategoryRetriever:
    name = "ollama-category-retriever"

    def __init__(
        self,
        train_records: list[Task1aRecord],
        k: int,
        seed: int,
        model: str = "qwen2.5vl:7b",
        base_url: str = "http://localhost:11434",
        client: Any | None = None,
        min_pool_size: int = 5,
    ) -> None:
        self._k = k
        self._seed = seed
        self._min_pool_size = min_pool_size
        self._model = model
        self._ollama = OllamaClient(base_url=base_url, client=client)

        self._by_subcategory: dict[tuple[str, str], list[Task1aRecord]] = defaultdict(list)
        self._by_category: dict[str, list[Task1aRecord]] = defaultdict(list)
        for record in train_records:
            if record.category is None:
                continue
            self._by_category[record.category].append(record)
            if record.subcategory is not None:
                self._by_subcategory[(record.category, record.subcategory)].append(record)

    def retrieve(self, record_id: str, image_path: Path) -> tuple[str, ...]:
        classification = self._classify(image_path)
        if classification is None:
            return ()
        category, subcategory = classification

        pool = self._by_subcategory.get((category, subcategory), [])
        if len(pool) < self._min_pool_size:
            pool = self._by_category.get(category, [])
        if not pool:
            return ()

        rng = random.Random(f"{self._seed}:{record_id}")
        chosen = rng.sample(pool, k=min(self._k, len(pool)))
        return tuple(record.id for record in chosen)

    def _classify(self, image_path: Path) -> tuple[str, str] | None:
        try:
            content = self._ollama.chat(
                self._model,
                _CLASSIFY_PROMPT.format(categories=", ".join(CATEGORY_SUBCATEGORIES)),
                image_path=image_path,
                json_schema=_CLASSIFY_SCHEMA,
            )
            payload = json.loads(content)
            return str(payload["category"]), str(payload["subcategory"])
        except Exception:
            return None
