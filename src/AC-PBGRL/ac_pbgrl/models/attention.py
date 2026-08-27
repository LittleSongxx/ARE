from __future__ import annotations

import math

import torch
from torch import nn


class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, heads: int) -> None:
        super().__init__()
        if embedding_dim % heads:
            raise ValueError("embedding_dim must be divisible by heads")
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.head_dim = embedding_dim // heads
        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        query_mask: torch.Tensor,
        key_mask: torch.Tensor,
        allowed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, query_count, _ = query.shape
        key_count = key.shape[1]

        def split(tensor: torch.Tensor, count: int) -> torch.Tensor:
            return tensor.view(batch, count, self.heads, self.head_dim).transpose(1, 2)

        q = split(self.q_proj(query), query_count)
        k = split(self.k_proj(key), key_count)
        v = split(self.v_proj(value), key_count)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        valid = query_mask[:, None, :, None] & key_mask[:, None, None, :]
        if allowed is not None:
            valid = valid & allowed[:, None, :, :]
        scores = scores.masked_fill(~valid, -1.0e4)
        attention = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        attention = attention * valid.to(attention.dtype)
        normalizer = attention.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        attention = attention / normalizer
        result = torch.matmul(attention, v)
        result = result.transpose(1, 2).contiguous().view(batch, query_count, self.embedding_dim)
        result = self.out_proj(result)
        return result * query_mask.unsqueeze(-1).to(result.dtype)


class GraphEncoderLayer(nn.Module):
    def __init__(self, embedding_dim: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attention = MaskedMultiHeadAttention(embedding_dim, heads)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, node_mask: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(values)
        attended = self.attention(
            normalized,
            normalized,
            normalized,
            query_mask=node_mask,
            key_mask=node_mask,
            allowed=adjacency,
        )
        values = values + self.dropout(attended)
        values = values + self.dropout(self.feed_forward(self.norm2(values)))
        return values * node_mask.unsqueeze(-1).to(values.dtype)


class GraphEncoder(nn.Module):
    def __init__(self, embedding_dim: int, heads: int, layers: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [GraphEncoderLayer(embedding_dim, heads, dropout=dropout) for _ in range(layers)]
        )

    def forward(self, values: torch.Tensor, node_mask: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # ``torch.eye(..., dtype=bool)`` is lowered to ``EyeLike(bool)`` by the
        # PyTorch ONNX exporter.  ONNX Runtime 1.16 (the last Python 3.8 build
        # used by ROS Noetic) does not implement that combination.  Equality
        # of node indices has the same semantics and exports as portable
        # Range/Equal operators.
        node_indices = torch.arange(adjacency.shape[-1], device=adjacency.device)
        identity = node_indices[None, :, None] == node_indices[None, None, :]
        allowed = (adjacency.bool() | identity) & node_mask[:, :, None] & node_mask[:, None, :]
        for layer in self.layers:
            values = layer(values, node_mask, allowed)
        return values


class PointerAttention(nn.Module):
    def __init__(self, embedding_dim: int, tanh_clipping: float = 10.0) -> None:
        super().__init__()
        self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.scale = embedding_dim ** -0.5
        self.tanh_clipping = tanh_clipping

    def forward(self, query: torch.Tensor, candidates: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q = self.query(query).unsqueeze(1)
        k = self.key(candidates)
        score = (q * k).sum(dim=-1) * self.scale
        score = self.tanh_clipping * torch.tanh(score)
        return score.masked_fill(~mask, -1.0e4)
