"""Example script demonstrating AdaptiveKV V1 token eviction and mixed-bit quantization.

Run with:
    python examples/generate_with_adaptivekv.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

from adaptivekv import (
    AdaptiveKVCache,
    AdaptiveKVConfig,
    TokenBudgetConfig,
    apply_adaptive_kv,
    compute_cache_statistics,
)


class MockTransformerLayer(nn.Module):
    """Transformer layer simulating attention cache updates."""

    def __init__(self, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: AdaptiveKVCache | None = None,
    ) -> tuple[torch.Tensor, AdaptiveKVCache | None]:
        batch, seq_len, _ = hidden_states.shape
        heads, head_dim = 8, 64

        key_states = hidden_states.view(batch, seq_len, heads, head_dim).transpose(1, 2)
        value_states = hidden_states.view(batch, seq_len, heads, head_dim).transpose(1, 2)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states, value_states, layer_idx=self.layer_idx
            )

        out = key_states.transpose(1, 2).reshape(batch, -1, heads * head_dim)
        return out, past_key_value


class SimpleCausalLM(nn.Module):
    """Simple decoder-only causal model for demo purposes."""

    def __init__(self, num_layers: int = 4) -> None:
        super().__init__()

        class Config:
            model_type = "opt"

        self.config = Config()
        self.layers = nn.ModuleList([MockTransformerLayer(i) for i in range(num_layers)])

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: AdaptiveKVCache | None = None,
    ) -> tuple[torch.Tensor, AdaptiveKVCache | None]:
        batch, seq_len = input_ids.shape
        x = torch.randn(batch, seq_len, 512, device=input_ids.device)

        for layer in self.layers:
            x, past_key_values = layer(x, past_key_value=past_key_values)

        logits = torch.randn(batch, seq_len, 32000, device=input_ids.device)
        return logits, past_key_values

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        past_key_values: AdaptiveKVCache | None = None,
    ) -> torch.Tensor:
        cache = past_key_values or AdaptiveKVCache()
        cur_input = input_ids

        # Prompt prefill phase
        logits, cache = self.forward(cur_input, past_key_values=cache)

        # Autoregressive generation loop
        for _ in range(max_new_tokens):
            next_token = torch.randint(0, 32000, (input_ids.shape[0], 1), device=input_ids.device)
            logits, cache = self.forward(next_token, past_key_values=cache)
            cur_input = torch.cat([cur_input, next_token], dim=-1)

        return cur_input


def main() -> None:
    print("=" * 60)
    print("AdaptiveKV V1 — Token Eviction & Mixed-Bit Quantization Demo")
    print("=" * 60)

    # 1. Initialize model
    model = SimpleCausalLM(num_layers=4)

    # 2. Configure AdaptiveKV with token eviction + budget control
    config = AdaptiveKVConfig(
        token_budget=TokenBudgetConfig(
            enable_token_eviction=True,
            max_cache_tokens=256,
            keep_ratio=0.5,
            recent_window=64,
            sink_tokens=4,
        ),
        enable_quantization=True,
        enable_adaptive_bits=True,
    )

    # 3. Apply AdaptiveKV to model
    model, cache = apply_adaptive_kv(model, config=config)

    # 4. Generate tokens with long prompt
    prompt_len = 512
    gen_len = 256
    prompt = torch.randint(0, 32000, (1, prompt_len))

    print(f"\nGenerating {gen_len} tokens for prompt of length {prompt_len}...")
    output = model.generate(prompt, max_new_tokens=gen_len, past_key_values=cache)

    print(f"Generation completed successfully! Output sequence length: {output.shape[1]}")

    # 5. Compute & print execution statistics
    stats = compute_cache_statistics(cache)

    print("\n" + "=" * 60)
    print("AdaptiveKV Execution Performance Report")
    print("=" * 60)
    print(f"Total tokens processed:       {stats['tokens_seen']}")
    print(f"Active KV tokens retained:    {stats['tokens_currently_cached']}")
    print(f"Tokens evicted:               {stats['tokens_evicted']}")
    print(f"Token retention ratio:        {stats['token_retention_ratio'] * 100.0:.2f}%")
    print("-" * 60)
    print(f"Baseline FP16 KV memory:     {stats['original_estimated_kv_bytes'] / 1024:.2f} KB")
    print(f"AdaptiveKV resident memory:   {stats['current_kv_bytes'] / 1024:.2f} KB")
    print(f"Memory saved:                 {stats['memory_saved_bytes'] / 1024:.2f} KB")
    print(f"Memory reduction:             {stats['memory_reduction_percent']:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
