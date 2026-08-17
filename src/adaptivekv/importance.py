"""Importance scoring for KV-cache entries.

Provides abstractions and implementations for computing entry importance across
attention-based, magnitude-based, recency-based, layer-wise, and head-wise strategies.
Supports both group-level importance scoring and token-level importance scoring.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from adaptivekv.config import ImportanceConfig, ImportanceStrategy
from adaptivekv.exceptions import ImportanceError, InvalidStrategyError

# ── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class ImportanceScore:
    """Normalized importance scores for KV-cache blocks or tokens.

    Attributes:
        scores: 1D PyTorch float32 tensor of normalized importance values in [0.0, 1.0].
        strategy: Identifier of strategy used to compute scores.
        group_size: Number of cache elements per score entry.
    """

    scores: torch.Tensor
    strategy: str
    group_size: int

    @property
    def num_groups(self) -> int:
        """Number of score entries / groups."""
        return self.scores.numel()


# ── Base Abstract Analyzer ──────────────────────────────────────────────────

class BaseImportanceAnalyzer(ABC):
    """Abstract base class for KV-cache importance analyzers."""

    def __init__(self, config: ImportanceConfig | None = None) -> None:
        self.config = config or ImportanceConfig()

    @abstractmethod
    def compute_importance(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_weights: torch.Tensor | None = None,
        group_size: int = 128,
    ) -> ImportanceScore:
        """Compute normalized group-level importance scores for KV-cache states.

        Args:
            key_states: Key tensor of shape (batch, num_heads, seq_len, head_dim) or similar.
            value_states: Value tensor of same shape as key_states.
            attention_weights: Optional attention weight tensor (batch, num_heads, q_len, kv_seq_len).
            group_size: Granularity for importance score output.

        Returns:
            ImportanceScore dataclass containing 1D normalized scores in [0, 1].
        """

    @abstractmethod
    def compute_token_importance(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute token-level normalized importance scores in [0.0, 1.0].

        Args:
            key_states: Key tensor of shape (batch, num_heads, seq_len, head_dim) or similar.
            value_states: Value tensor of same shape as key_states.
            attention_weights: Optional attention weight tensor (batch, num_heads, q_len, kv_seq_len).

        Returns:
            1D float32 tensor of shape (seq_len,) containing normalized importance score per token.
        """

    def _normalize(self, scores: torch.Tensor) -> torch.Tensor:
        """Min-max normalize scores to [0.0, 1.0]."""
        if not self.config.normalize or scores.numel() <= 1:
            return scores

        min_val = torch.min(scores)
        max_val = torch.max(scores)
        denom = max_val - min_val

        if denom < 1e-8:
            # All scores identical: return uniform 0.5
            return torch.full_like(scores, 0.5)

        return (scores - min_val) / denom

    def _pool_to_groups(self, token_scores: torch.Tensor, group_size: int) -> torch.Tensor:
        """Pool token/element-level scores into group-level scores via mean pooling.

        Args:
            token_scores: 1D tensor of scores.
            group_size: Group size.

        Returns:
            1D tensor of group scores.
        """
        numel = token_scores.numel()
        if numel == 0:
            raise ImportanceError("Cannot compute importance for empty tensor.")

        if numel <= group_size:
            return torch.mean(token_scores).unsqueeze(0)

        remainder = numel % group_size
        if remainder != 0:
            padding = torch.full(
                (group_size - remainder,),
                fill_value=token_scores.mean().item(),
                dtype=token_scores.dtype,
                device=token_scores.device,
            )
            token_scores = torch.cat([token_scores, padding])

        grouped = token_scores.view(-1, group_size)
        return torch.mean(grouped, dim=-1)


# ── Attention Importance Analyzer ───────────────────────────────────────────

class AttentionImportanceAnalyzer(BaseImportanceAnalyzer):
    """Computes KV-cache importance based on attention weight accumulation.

    Tokens that receive high attention weights from query tokens are assigned
    higher importance scores. If attention weights are unavailable, falls back
    to key vector norm.
    """

    def compute_token_importance(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute token-level attention importance scores."""
        if key_states.numel() == 0:
            raise ImportanceError("Cannot compute importance for empty key_states.")

        dtype = torch.float32

        if attention_weights is not None and attention_weights.numel() > 0:
            attn = attention_weights.to(dtype)
            if attn.ndim >= 2:
                dim_indices = tuple(range(attn.ndim - 1))
                token_scores = torch.sum(attn, dim=dim_indices)
            else:
                token_scores = attn.reshape(-1)
        else:
            k = key_states.to(dtype)
            if k.ndim >= 2:
                norms = torch.norm(k, p=2, dim=-1)
                if norms.ndim > 1:
                    dim_indices = tuple(range(norms.ndim - 1))
                    token_scores = torch.mean(norms, dim=dim_indices)
                else:
                    token_scores = norms
            else:
                token_scores = torch.abs(k.reshape(-1))

        return self._normalize(token_scores)

    def compute_importance(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_weights: torch.Tensor | None = None,
        group_size: int = 128,
    ) -> ImportanceScore:
        """Compute attention-based group-level importance scores."""
        if key_states.numel() == 0:
            raise ImportanceError("Cannot compute importance for empty key_states.")

        device = key_states.device
        dtype = torch.float32

        token_scores = self.compute_token_importance(
            key_states, value_states, attention_weights=attention_weights
        )

        total_elements = key_states.numel()
        num_groups = max(1, math_ceil_div(total_elements, group_size))

        if token_scores.numel() != num_groups:
            if token_scores.numel() == 0:
                group_scores = torch.ones(num_groups, dtype=dtype, device=device)
            else:
                group_scores = torch.nn.functional.interpolate(
                    token_scores.unsqueeze(0).unsqueeze(0),
                    size=num_groups,
                    mode="nearest",
                ).reshape(-1)
        else:
            group_scores = token_scores

        normalized_scores = self._normalize(group_scores)
        return ImportanceScore(
            scores=normalized_scores,
            strategy=ImportanceStrategy.ATTENTION.value,
            group_size=group_size,
        )


# ── Magnitude Importance Analyzer ───────────────────────────────────────────

class MagnitudeImportanceAnalyzer(BaseImportanceAnalyzer):
    """Computes KV-cache importance based on key & value L2 vector magnitudes."""

    def compute_token_importance(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute token-level magnitude importance scores."""
        if key_states.numel() == 0:
            raise ImportanceError("Cannot compute importance for empty key_states.")

        k = key_states.to(torch.float32)
        v = value_states.to(torch.float32)

        if k.ndim >= 2 and v.ndim >= 2:
            k_norms = torch.norm(k, p=2, dim=-1)
            v_norms = torch.norm(v, p=2, dim=-1)
            combined = k_norms + v_norms
            if combined.ndim > 1:
                dim_indices = tuple(range(combined.ndim - 1))
                token_scores = torch.mean(combined, dim=dim_indices)
            else:
                token_scores = combined
        else:
            token_scores = torch.abs(k.reshape(-1)) + torch.abs(v.reshape(-1))

        return self._normalize(token_scores)

    def compute_importance(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_weights: torch.Tensor | None = None,
        group_size: int = 128,
    ) -> ImportanceScore:
        if key_states.numel() == 0:
            raise ImportanceError("Cannot compute importance for empty key_states.")

        k_flat = key_states.to(torch.float32).reshape(-1)
        v_flat = value_states.to(torch.float32).reshape(-1)

        combined = torch.abs(k_flat) + torch.abs(v_flat)
        group_scores = self._pool_to_groups(combined, group_size)
        normalized_scores = self._normalize(group_scores)

        return ImportanceScore(
            scores=normalized_scores,
            strategy=ImportanceStrategy.MAGNITUDE.value,
            group_size=group_size,
        )


# ── Recency Importance Analyzer ─────────────────────────────────────────────

class RecencyImportanceAnalyzer(BaseImportanceAnalyzer):
    """Computes KV-cache importance based on token recency (newer tokens = higher score)."""

    def compute_token_importance(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute token-level recency importance scores."""
        if key_states.numel() == 0:
            raise ImportanceError("Cannot compute importance for empty key_states.")

        seq_len = key_states.shape[-2] if key_states.ndim >= 2 else key_states.numel()
        if seq_len <= 1:
            token_scores = torch.ones(seq_len, dtype=torch.float32, device=key_states.device)
        else:
            token_scores = torch.linspace(
                0.0, 1.0, steps=seq_len, dtype=torch.float32, device=key_states.device
            )

        return self._normalize(token_scores)

    def compute_importance(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_weights: torch.Tensor | None = None,
        group_size: int = 128,
    ) -> ImportanceScore:
        if key_states.numel() == 0:
            raise ImportanceError("Cannot compute importance for empty key_states.")

        numel = key_states.numel()
        num_groups = max(1, math_ceil_div(numel, group_size))

        group_scores = torch.linspace(
            0.0, 1.0, steps=num_groups, dtype=torch.float32, device=key_states.device
        )
        normalized_scores = self._normalize(group_scores)

        return ImportanceScore(
            scores=normalized_scores,
            strategy=ImportanceStrategy.RECENCY.value,
            group_size=group_size,
        )


# ── Head-wise Importance Analyzer ───────────────────────────────────────────

class HeadImportanceAnalyzer(BaseImportanceAnalyzer):
    """Computes KV-cache importance per attention head."""

    def compute_token_importance(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute token-level head-wise importance scores."""
        if key_states.numel() == 0:
            raise ImportanceError("Cannot compute importance for empty key_states.")

        k = key_states.to(torch.float32)
        if k.ndim >= 2:
            norms = torch.norm(k, p=2, dim=-1)
            if norms.ndim > 1:
                dim_indices = tuple(range(norms.ndim - 1))
                token_scores = torch.mean(norms, dim=dim_indices)
            else:
                token_scores = norms
        else:
            token_scores = torch.norm(k, p=2, dim=-1).reshape(-1)

        return self._normalize(token_scores)

    def compute_importance(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_weights: torch.Tensor | None = None,
        group_size: int = 128,
    ) -> ImportanceScore:
        if key_states.numel() == 0:
            raise ImportanceError("Cannot compute importance for empty key_states.")

        k = key_states.to(torch.float32)
        if k.ndim >= 3:
            head_norms = torch.norm(k, p=2, dim=-1)
            token_scores = torch.mean(head_norms, dim=0).reshape(-1)
        else:
            token_scores = torch.norm(k, p=2, dim=-1).reshape(-1)

        group_scores = self._pool_to_groups(token_scores, group_size)
        normalized_scores = self._normalize(group_scores)

        return ImportanceScore(
            scores=normalized_scores,
            strategy="head",
            group_size=group_size,
        )


# ── Helper & Factory ────────────────────────────────────────────────────────

def math_ceil_div(a: int, b: int) -> int:
    """Ceiling division of integer a by b."""
    return (a + b - 1) // b


def create_importance_analyzer(
    config: ImportanceConfig | None = None,
) -> BaseImportanceAnalyzer:
    """Factory function for creating importance analyzer instances based on config."""
    cfg = config or ImportanceConfig()
    strat = cfg.strategy

    if strat == ImportanceStrategy.ATTENTION.value:
        return AttentionImportanceAnalyzer(cfg)
    elif strat == ImportanceStrategy.MAGNITUDE.value:
        return MagnitudeImportanceAnalyzer(cfg)
    elif strat == ImportanceStrategy.RECENCY.value:
        return RecencyImportanceAnalyzer(cfg)
    else:
        raise InvalidStrategyError(
            strat, tuple(s.value for s in ImportanceStrategy)
        )
