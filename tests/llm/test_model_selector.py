"""Tests for Task 4.2 — model_selector.get_model()."""

import os
import pytest

from backend.app.llm.model_selector import get_model, TierConfig


class TestGetModel:
    def test_default_tier_is_2(self, monkeypatch):
        monkeypatch.delenv("MODEL_TIER", raising=False)
        cfg = get_model()
        assert cfg.tier == 2

    def test_tier_1_returns_tier1_config(self, monkeypatch):
        monkeypatch.setenv("MODEL_TIER", "1")
        cfg = get_model()
        assert cfg.tier == 1

    def test_tier_2_returns_tier2_config(self, monkeypatch):
        monkeypatch.setenv("MODEL_TIER", "2")
        cfg = get_model()
        assert cfg.tier == 2

    def test_tier_3_returns_tier3_config(self, monkeypatch):
        monkeypatch.setenv("MODEL_TIER", "3")
        cfg = get_model()
        assert cfg.tier == 3

    def test_returns_tier_config_dataclass(self, monkeypatch):
        monkeypatch.setenv("MODEL_TIER", "2")
        cfg = get_model()
        assert isinstance(cfg, TierConfig)

    def test_config_has_required_fields(self, monkeypatch):
        monkeypatch.setenv("MODEL_TIER", "2")
        cfg = get_model()
        assert isinstance(cfg.embedding, str) and cfg.embedding
        assert isinstance(cfg.llm, str) and cfg.llm
        assert cfg.vector_store in ("qdrant", "pgvector")
        assert isinstance(cfg.description, str) and cfg.description

    def test_tier1_uses_qdrant(self, monkeypatch):
        monkeypatch.setenv("MODEL_TIER", "1")
        assert get_model().vector_store == "qdrant"

    def test_tier3_uses_pgvector(self, monkeypatch):
        monkeypatch.setenv("MODEL_TIER", "3")
        assert get_model().vector_store == "pgvector"

    def test_all_tiers_have_different_llms(self, monkeypatch):
        llms = set()
        for t in ("1", "2", "3"):
            monkeypatch.setenv("MODEL_TIER", t)
            llms.add(get_model().llm)
        assert len(llms) == 3  # all three must be distinct models

    def test_invalid_tier_raises_value_error(self, monkeypatch):
        monkeypatch.setenv("MODEL_TIER", "99")
        with pytest.raises(ValueError, match="not valid"):
            get_model()

    def test_non_numeric_tier_raises_value_error(self, monkeypatch):
        monkeypatch.setenv("MODEL_TIER", "two")
        with pytest.raises(ValueError, match="integer"):
            get_model()

    def test_config_is_frozen(self, monkeypatch):
        monkeypatch.setenv("MODEL_TIER", "2")
        cfg = get_model()
        with pytest.raises(Exception):  # dataclass frozen=True raises FrozenInstanceError
            cfg.llm = "hacked"
