"""Test setup: replace the real embedding model with a deterministic fake, so the tests
need no model download, no API key and no internet."""

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

import models


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    monkeypatch.setattr(models, "get_embeddings", lambda: DeterministicFakeEmbedding(size=64))
