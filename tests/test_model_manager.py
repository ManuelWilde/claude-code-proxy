"""Unit tests for model manager."""

import os
import sys
import pytest


class FakeProvider:
    def __init__(self, model):
        self.model = model


class FakeConfig:
    def __init__(self, big="gpt-4o", middle="gpt-4o", small="gpt-4o-mini"):
        self.big_model = big
        self.middle_model = middle
        self.small_model = small

    def get_tier(self, tier: str) -> FakeProvider:
        tiers = {
            "opus": FakeProvider(self.big_model),
            "sonnet": FakeProvider(self.middle_model),
            "haiku": FakeProvider(self.small_model),
        }
        return tiers[tier]


def _make_manager(fake_config):
    from src.core.model_manager import ModelManager
    return ModelManager(fake_config)


class TestModelMapping:
    def test_haiku_maps_to_small(self):
        mm = _make_manager(FakeConfig())
        assert mm.map_claude_model_to_openai("claude-3-5-haiku-20241022") == "gpt-4o-mini"

    def test_sonnet_maps_to_middle(self):
        mm = _make_manager(FakeConfig())
        assert mm.map_claude_model_to_openai("claude-3-5-sonnet-20241022") == "gpt-4o"

    def test_opus_maps_to_big(self):
        mm = _make_manager(FakeConfig(big="gpt-4o-max"))
        assert mm.map_claude_model_to_openai("claude-opus-4-20250514") == "gpt-4o-max"

    def test_unknown_claude_model_defaults_to_big(self):
        mm = _make_manager(FakeConfig(big="gpt-4o"))
        assert mm.map_claude_model_to_openai("claude-some-future-model") == "gpt-4o"

    def test_gpt_model_passthrough(self):
        mm = _make_manager(FakeConfig())
        assert mm.map_claude_model_to_openai("gpt-4o") == "gpt-4o"

    def test_o1_model_passthrough(self):
        mm = _make_manager(FakeConfig())
        assert mm.map_claude_model_to_openai("o1-preview") == "o1-preview"

    def test_deepseek_passthrough(self):
        mm = _make_manager(FakeConfig())
        assert mm.map_claude_model_to_openai("deepseek-chat") == "deepseek-chat"

    def test_doubao_passthrough(self):
        mm = _make_manager(FakeConfig())
        assert mm.map_claude_model_to_openai("doubao-pro-32k") == "doubao-pro-32k"

    def test_ark_passthrough(self):
        mm = _make_manager(FakeConfig())
        assert mm.map_claude_model_to_openai("ep-20240101-model") == "ep-20240101-model"

    def test_case_insensitive_matching(self):
        mm = _make_manager(FakeConfig())
        assert mm.map_claude_model_to_openai("Claude-3-5-Haiku") == "gpt-4o-mini"
        assert mm.map_claude_model_to_openai("CLAUDE-SONNET") == "gpt-4o"
