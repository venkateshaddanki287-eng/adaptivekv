"""Tests for adaptivekv.integration — Hugging Face model adapters."""

from __future__ import annotations

import pytest
import torch.nn as nn

from adaptivekv.exceptions import UnsupportedModelError
from adaptivekv.integration import HuggingFaceAdapter, apply_adaptive_kv


class MockModelConfig:
    def __init__(self, model_type: str) -> None:
        self.model_type = model_type


class MockHFModel(nn.Module):
    def __init__(self, model_type: str) -> None:
        super().__init__()
        self.config = MockModelConfig(model_type)


class TestHuggingFaceAdapter:
    """Test model validation and attachment."""

    @pytest.fixture
    def adapter(self) -> HuggingFaceAdapter:
        return HuggingFaceAdapter()

    @pytest.mark.parametrize("model_type", ["llama", "mistral", "qwen2", "gemma", "opt"])
    def test_supported_model_validation(
        self, adapter: HuggingFaceAdapter, model_type: str
    ) -> None:
        model = MockHFModel(model_type)
        detected = adapter.validate_model(model)
        assert detected == model_type

    def test_unsupported_model_raises_error(self, adapter: HuggingFaceAdapter) -> None:
        model = MockHFModel("unsupported_bert")
        with pytest.raises(UnsupportedModelError) as exc_info:
            adapter.validate_model(model)
        assert "unsupported_bert" in str(exc_info.value)

    def test_apply_adaptive_kv_helper(self) -> None:
        model = MockHFModel("llama")
        model_out, cache = apply_adaptive_kv(model, strategy="threshold")
        assert model_out is model
        assert cache is not None
