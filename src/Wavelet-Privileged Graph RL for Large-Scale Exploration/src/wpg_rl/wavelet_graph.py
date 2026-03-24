from __future__ import annotations

import torch
import torch.nn.functional as F


def normalize_scales(scales) -> tuple[int, ...]:
    normalized = []
    for scale in scales:
        value = max(int(scale), 1)
        if value not in normalized:
            normalized.append(value)
    return tuple(normalized or (1, 2, 4))


def valid_node_mask(node_padding_mask: torch.Tensor) -> torch.Tensor:
    if node_padding_mask.dim() == 3:
        return ~node_padding_mask.squeeze(1).bool()
    if node_padding_mask.dim() == 2:
        return ~node_padding_mask.bool()
    raise ValueError(f"Unsupported node_padding_mask shape: {tuple(node_padding_mask.shape)}")


def build_adjacency(edge_mask: torch.Tensor, node_padding_mask: torch.Tensor, add_self_loops: bool = True) -> torch.Tensor:
    adjacency = (edge_mask == 0).to(dtype=torch.float32)
    valid = valid_node_mask(node_padding_mask).to(dtype=adjacency.dtype)
    pairwise_valid = valid.unsqueeze(1) * valid.unsqueeze(2)
    adjacency = adjacency * pairwise_valid

    if add_self_loops:
        eye = torch.eye(adjacency.size(-1), device=adjacency.device, dtype=adjacency.dtype).unsqueeze(0)
        adjacency = adjacency * (1.0 - eye) + eye * valid.unsqueeze(1)
    return adjacency


def build_random_walk(edge_mask: torch.Tensor, node_padding_mask: torch.Tensor, add_self_loops: bool = True) -> torch.Tensor:
    adjacency = build_adjacency(edge_mask, node_padding_mask, add_self_loops=add_self_loops)
    degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return adjacency / degree


def propagate(power_operator: torch.Tensor, hidden: torch.Tensor, steps: int) -> torch.Tensor:
    output = hidden
    for _ in range(max(int(steps), 1)):
        output = torch.matmul(power_operator, output)
    return output


def multiscale_wavelet_decompose(
    hidden: torch.Tensor,
    edge_mask: torch.Tensor,
    node_padding_mask: torch.Tensor,
    scales=(1, 2, 4),
):
    scales = normalize_scales(scales)
    random_walk = build_random_walk(edge_mask, node_padding_mask)

    low_features = []
    previous = hidden
    last_low = hidden
    for scale in scales:
        low = propagate(random_walk, hidden, scale)
        low_features.append(low)
        last_low = low

    high_features = []
    previous = hidden
    for low in low_features:
        high_features.append(previous - low)
        previous = low

    return low_features, high_features, random_walk, last_low


def decompose_graph_hidden(
    hidden: torch.Tensor,
    edge_mask: torch.Tensor,
    node_padding_mask: torch.Tensor,
    scales=(1, 2, 4),
    apply_layer_norm: bool = False,
):
    valid = valid_node_mask(node_padding_mask).unsqueeze(-1).to(dtype=hidden.dtype)
    masked_hidden = hidden * valid
    low_features, high_features, random_walk, last_low = multiscale_wavelet_decompose(
        masked_hidden,
        edge_mask,
        node_padding_mask,
        scales=scales,
    )
    low_raw = torch.cat(low_features, dim=-1)
    high_raw = torch.cat(high_features, dim=-1)
    if apply_layer_norm:
        low_raw = F.layer_norm(low_raw, (low_raw.size(-1),))
        high_raw = F.layer_norm(high_raw, (high_raw.size(-1),))
    return low_raw, high_raw, low_features, high_features, random_walk, last_low


def build_overlap_valid_mask(index_map: torch.Tensor, actor_node_padding_mask: torch.Tensor) -> torch.Tensor:
    if index_map.dim() != 2:
        raise ValueError(f"index_map must be [B, N_actor], got {tuple(index_map.shape)}")
    actor_valid = valid_node_mask(actor_node_padding_mask)
    return actor_valid & index_map.ge(0)


def gather_by_index(features: torch.Tensor, index_map: torch.Tensor, invalid_fill: float = 0.0) -> torch.Tensor:
    if features.dim() != 3:
        raise ValueError(f"features must be [B, N, D], got {tuple(features.shape)}")
    if index_map.dim() != 2:
        raise ValueError(f"index_map must be [B, M], got {tuple(index_map.shape)}")

    safe_index = index_map.clamp_min(0).long()
    gather_index = safe_index.unsqueeze(-1).expand(-1, -1, features.size(-1))
    gathered = torch.gather(features, 1, gather_index)

    invalid_mask = index_map.lt(0).unsqueeze(-1)
    if invalid_fill == 0.0:
        return gathered.masked_fill(invalid_mask, 0.0)
    fill_value = torch.as_tensor(invalid_fill, dtype=features.dtype, device=features.device)
    return torch.where(invalid_mask, fill_value, gathered)
