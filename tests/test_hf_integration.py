"""Integration test for AdaptiveKV with Hugging Face causal language models."""

from __future__ import annotations

import torch
import torch.nn as nn

from adaptivekv.cache import AdaptiveKVCache
from adaptivekv.config import AdaptiveKVConfig, TokenBudgetConfig
from adaptivekv.integration import apply_adaptive_kv


class MockConfig:
    """Mock Hugging Face model configuration."""

    def __init__(self, model_type: str = "opt", num_layers: int = 2) -> None:
        self.model_type = model_type
        self.num_hidden_layers = num_layers
        self.hidden_size = 64
        self.num_attention_heads = 4
        self.head_dim = 16


class MockAttentionLayer(nn.Module):
    """Mock transformer attention layer interacting with AdaptiveKVCache."""

    def __init__(self, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: AdaptiveKVCache | None = None,
    ) -> tuple[torch.Tensor, AdaptiveKVCache | None]:
        batch, q_len, _ = hidden_states.shape
        heads, head_dim = 4, 16

        key_states = hidden_states.view(batch, q_len, heads, head_dim).transpose(1, 2)
        value_states = hidden_states.view(batch, q_len, heads, head_dim).transpose(1, 2)

        if past_key_value is not None:
            past_key_value.update(key_states, value_states, layer_idx=self.layer_idx)

        return hidden_states, past_key_value


class MockCausalLM(nn.Module):
    """Mock decoder-only causal language model compatible with apply_adaptive_kv."""

    def __init__(self, model_type: str = "opt") -> None:
        super().__init__()
        self.config = MockConfig(model_type=model_type, num_layers=2)
        self.layers = nn.ModuleList([MockAttentionLayer(0), MockAttentionLayer(1)])

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: AdaptiveKVCache | None = None,
    ) -> tuple[torch.Tensor, AdaptiveKVCache | None]:
        batch, seq_len = input_ids.shape
        x = torch.randn(batch, seq_len, 64, device=input_ids.device)

        for layer in self.layers:
            x, past_key_values = layer(x, past_key_value=past_key_values)

        logits = torch.randn(batch, seq_len, 1000, device=input_ids.device)
        return logits, past_key_values

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 10,
        past_key_values: AdaptiveKVCache | None = None,
    ) -> torch.Tensor:
        kv_cache = past_key_values if past_key_values is not None else AdaptiveKVCache()
        cur_input = input_ids

        # Prompt phase
        logits, kv_cache = self.forward(cur_input, past_key_values=kv_cache)

        # Autoregressive generation steps
        for _ in range(max_new_tokens):
            next_token = torch.randint(0, 1000, (input_ids.shape[0], 1), device=input_ids.device)
            logits, kv_cache = self.forward(next_token, past_key_values=kv_cache)
            cur_input = torch.cat([cur_input, next_token], dim=-1)

        return cur_input


class TestHuggingFaceIntegration:
    """Test full integration with Hugging Face adapter and generation loop."""

    def test_apply_adaptive_kv_integration(self) -> None:
        model = MockCausalLM(model_type="opt")
        cfg = AdaptiveKVConfig(
            token_budget=TokenBudgetConfig(
                enable_token_eviction=True,
                max_cache_tokens=16,
                recent_window=4,
                sink_tokens=2,
            )
        )

        model, cache = apply_adaptive_kv(model, config=cfg)
        assert isinstance(cache, AdaptiveKVCache)

        prompt = torch.randint(0, 1000, (1, 20))
        output = model.generate(prompt, max_new_tokens=15, past_key_values=cache)

        assert output.shape == (1, 35)
        assert len(cache.layers) == 2

        for layer_idx in (0, 1):
            assert cache.get_seq_length(layer_idx) == 16

        assert cache.tokens_seen == 70
        assert cache.tokens_currently_cached == 32
        assert cache.tokens_evicted == 38
