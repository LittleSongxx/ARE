from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .parameter import (
    ENABLE_WAVELET_HISTORY,
    ENABLE_WAVELET_UTILITY_LOSS,
    HISTORY_EMBED_DIM,
    HISTORY_ENCODER_MODE,
    HISTORY_INPUT_DIM,
    HISTORY_WAVELET_LEVELS,
    UTILITY_AUX_BASE_WEIGHT,
    UTILITY_AUX_LOSS_TYPE,
    UTILITY_AUX_WAVELET_WEIGHT,
    UTILITY_LOSS_MODE,
    UTILITY_LOSS_WEIGHT,
    UTILITY_PATCH_SIZE,
    UTILITY_PATCH_SIGMA,
    UTILITY_WAVELET_LEVELS,
    UTILITY_WAVELET_RHO,
)
from .wavelet import haar_decompose_2d, haar_decompose_time


class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = embedding_dim
        self.key_dim = self.value_dim
        self.tanh_clipping = 10
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, k, mask=None):
        n_batch, n_key, n_dim = k.size()
        n_query = q.size(1)

        k_flat = k.reshape(-1, n_dim)
        q_flat = q.reshape(-1, n_dim)

        shape_k = (n_batch, n_key, -1)
        shape_q = (n_batch, n_query, -1)

        q_proj = torch.matmul(q_flat, self.w_query).view(shape_q)
        k_proj = torch.matmul(k_flat, self.w_key).view(shape_k)

        logits = self.norm_factor * torch.matmul(q_proj, k_proj.transpose(1, 2))
        logits = self.tanh_clipping * torch.tanh(logits)

        if mask is not None:
            logits = logits.masked_fill(mask == 1, -1e8)
        attention = torch.log_softmax(logits, dim=-1)

        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = self.embedding_dim // self.n_heads
        self.key_dim = self.value_dim
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_value = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.value_dim))
        self.w_out = nn.Parameter(torch.Tensor(self.n_heads, self.value_dim, self.embedding_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, k=None, v=None, key_padding_mask=None, attn_mask=None):
        if k is None:
            k = q
        if v is None:
            v = q

        n_batch, n_key, n_dim = k.size()
        n_query = q.size(1)
        n_value = v.size(1)

        k_flat = k.contiguous().view(-1, n_dim)
        v_flat = v.contiguous().view(-1, n_dim)
        q_flat = q.contiguous().view(-1, n_dim)
        shape_v = (self.n_heads, n_batch, n_value, -1)
        shape_k = (self.n_heads, n_batch, n_key, -1)
        shape_q = (self.n_heads, n_batch, n_query, -1)

        q_proj = torch.matmul(q_flat, self.w_query).view(shape_q)
        k_proj = torch.matmul(k_flat, self.w_key).view(shape_k)
        v_proj = torch.matmul(v_flat, self.w_value).view(shape_v)

        logits = self.norm_factor * torch.matmul(q_proj, k_proj.transpose(2, 3))

        if attn_mask is not None:
            attn_mask = attn_mask.view(1, n_batch, n_query, n_key).expand_as(logits)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.repeat(1, n_query, 1)
            key_padding_mask = key_padding_mask.view(1, n_batch, n_query, n_key).expand_as(logits)

        if attn_mask is not None and key_padding_mask is not None:
            mask = attn_mask + key_padding_mask
        elif attn_mask is not None:
            mask = attn_mask
        elif key_padding_mask is not None:
            mask = key_padding_mask
        else:
            mask = None

        if mask is not None:
            logits = logits.masked_fill(mask > 0, -1e8)

        attention = torch.softmax(logits, dim=-1)
        heads = torch.matmul(attention, v_proj)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).reshape(-1, self.n_heads * self.value_dim),
            self.w_out.view(-1, self.embedding_dim),
        ).view(-1, n_query, self.embedding_dim)

        return out, attention


class Normalization(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.normalizer = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        return self.normalizer(x.view(-1, x.size(-1))).view(*x.size())


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim),
        )
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, src, key_padding_mask=None, attn_mask=None):
        h0 = src
        h = self.normalization1(src)
        h, _ = self.multi_head_attention(q=h, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        h = h + h0
        h1 = h
        h = self.normalization2(h)
        h = self.feed_forward(h)
        return h + h1


class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim),
        )
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, tgt, memory, key_padding_mask=None, attn_mask=None):
        h0 = tgt
        tgt = self.normalization1(tgt)
        memory = self.normalization1(memory)
        h, w = self.multi_head_attention(
            q=tgt,
            k=memory,
            v=memory,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )
        h = h + h0
        h1 = h
        h = self.normalization2(h)
        h = self.feed_forward(h)
        return h + h1, w


class Encoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=1):
        super().__init__()
        self.layers = nn.ModuleList(EncoderLayer(embedding_dim, n_head) for _ in range(n_layer))

    def forward(self, src, key_padding_mask=None, attn_mask=None):
        for layer in self.layers:
            src = layer(src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return src


class Decoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, n_head) for _ in range(n_layer)])

    def forward(self, tgt, memory, key_padding_mask=None, attn_mask=None):
        for layer in self.layers:
            tgt, w = layer(tgt, memory, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return tgt, w


class WaveletHistoryEncoder(nn.Module):
    """Encode history into low/high frequency contexts."""

    def __init__(self, history_input_dim, history_embed_dim, levels, mode=HISTORY_ENCODER_MODE):
        super().__init__()
        self.levels = max(int(levels), 1)
        self.mode = str(mode).strip().lower()
        if self.mode not in {"mlp_only", "wavelet_shared", "wavelet_split"}:
            self.mode = HISTORY_ENCODER_MODE

        self.input_proj = nn.Linear(history_input_dim, history_embed_dim)
        self.low_proj = nn.Sequential(
            nn.Linear(history_embed_dim, history_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(history_embed_dim, history_embed_dim),
        )
        if self.mode == "wavelet_shared":
            self.shared_proj = nn.Sequential(
                nn.Linear(history_embed_dim * (self.levels + 1), history_embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(history_embed_dim, history_embed_dim),
            )
        else:
            high_in_dim = history_embed_dim if self.mode == "mlp_only" else history_embed_dim * self.levels
            self.high_proj = nn.Sequential(
                nn.Linear(high_in_dim, history_embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(history_embed_dim, history_embed_dim),
            )

    def forward(self, history_inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(history_inputs)

        if self.mode == "mlp_only":
            summary = x.mean(dim=1)
            low_context = self.low_proj(summary)
            high_context = self.high_proj(summary)
            return low_context, high_context

        low, highs = haar_decompose_time(x, levels=self.levels)
        low_summary = low.mean(dim=1)
        high_summaries = [high.mean(dim=1) for high in highs]
        if len(high_summaries) == 0:
            high_summary = low_summary
        else:
            high_summary = torch.cat(high_summaries, dim=-1)

        if self.mode == "wavelet_shared":
            shared = self.shared_proj(torch.cat([low_summary, high_summary], dim=-1))
            return shared, shared

        low_context = self.low_proj(low_summary)
        high_context = self.high_proj(high_summary)
        return low_context, high_context


class UtilityAuxHead(nn.Module):
    """Predict candidate future-gain proxy for each action: [B, K, 1]."""

    def __init__(self, action_feature_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(action_feature_dim, action_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(action_feature_dim, 1),
        )

    def forward(self, action_features: torch.Tensor) -> torch.Tensor:
        return self.head(action_features)


class _HistoryStateFusion(nn.Module):
    """Residual-gate fusion for global/current-state modulation."""

    def __init__(self, embedding_dim, history_embed_dim):
        super().__init__()
        self.history_proj = nn.Linear(history_embed_dim, embedding_dim)
        self.gate = nn.Linear(embedding_dim + history_embed_dim, embedding_dim)

    def forward(self, state_feature: torch.Tensor, history_context: torch.Tensor) -> torch.Tensor:
        # state_feature: [B, 1, E], history_context: [B, H]
        history_expanded = history_context.unsqueeze(1)
        gate = torch.sigmoid(self.gate(torch.cat((state_feature, history_expanded), dim=-1)))
        delta = self.history_proj(history_expanded)
        return state_feature + gate * delta


class _HistoryNeighborFusion(nn.Module):
    """Residual-gate fusion for local neighbor/candidate correction."""

    def __init__(self, embedding_dim, history_embed_dim):
        super().__init__()
        self.history_proj = nn.Linear(history_embed_dim, embedding_dim)
        self.gate = nn.Linear(embedding_dim + history_embed_dim, embedding_dim)

    def forward(self, neighbor_feature: torch.Tensor, history_context: torch.Tensor) -> torch.Tensor:
        history_expanded = history_context.unsqueeze(1).expand(-1, neighbor_feature.size(1), -1)
        gate = torch.sigmoid(self.gate(torch.cat((neighbor_feature, history_expanded), dim=-1)))
        delta = self.history_proj(history_context).unsqueeze(1)
        return neighbor_feature + gate * delta


class PolicyNet(nn.Module):
    def __init__(
        self,
        node_dim,
        embedding_dim,
        enable_wavelet_history=ENABLE_WAVELET_HISTORY,
        history_input_dim=HISTORY_INPUT_DIM,
        history_embed_dim=HISTORY_EMBED_DIM,
        history_wavelet_levels=HISTORY_WAVELET_LEVELS,
        history_encoder_mode=HISTORY_ENCODER_MODE,
    ):
        super().__init__()

        self.enable_wavelet_history = bool(enable_wavelet_history)

        self.initial_embedding = nn.Linear(node_dim, embedding_dim)
        self.encoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=6)

        self.decoder = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)
        self.current_embedding = nn.Linear(embedding_dim * 2, embedding_dim)

        self.pointer = SingleHeadAttention(embedding_dim)

        if self.enable_wavelet_history:
            self.history_encoder = WaveletHistoryEncoder(
                history_input_dim=history_input_dim,
                history_embed_dim=history_embed_dim,
                levels=history_wavelet_levels,
                mode=history_encoder_mode,
            )
            self.history_state_fusion = _HistoryStateFusion(embedding_dim, history_embed_dim)
            self.history_neighbor_fusion = _HistoryNeighborFusion(embedding_dim, history_embed_dim)

    def encode_graph(self, node_inputs, node_padding_mask, edge_mask):
        node_feature = self.initial_embedding(node_inputs)
        return self.encoder(src=node_feature, key_padding_mask=node_padding_mask, attn_mask=edge_mask)

    def decode_state(self, enhanced_node_feature, current_index, node_padding_mask):
        embedding_dim = enhanced_node_feature.size()[2]
        current_node_feature = torch.gather(enhanced_node_feature, 1, current_index.repeat(1, 1, embedding_dim))
        enhanced_current_node_feature, _ = self.decoder(
            current_node_feature,
            enhanced_node_feature,
            node_padding_mask,
        )
        return current_node_feature, enhanced_current_node_feature

    def output_policy(
        self,
        current_node_feature,
        enhanced_current_node_feature,
        enhanced_node_feature,
        current_edge,
        edge_padding_mask,
        history_inputs=None,
    ):
        embedding_dim = enhanced_node_feature.size()[2]
        current_state_feature = self.current_embedding(
            torch.cat((enhanced_current_node_feature, current_node_feature), dim=-1)
        )

        low_context = None
        high_context = None
        if self.enable_wavelet_history and history_inputs is not None:
            low_context, high_context = self.history_encoder(history_inputs)
            current_state_feature = self.history_state_fusion(current_state_feature, low_context)

        neighboring_feature = torch.gather(enhanced_node_feature, 1, current_edge.repeat(1, 1, embedding_dim))
        if self.enable_wavelet_history and high_context is not None:
            neighboring_feature = self.history_neighbor_fusion(neighboring_feature, high_context)

        logp = self.pointer(current_state_feature, neighboring_feature, edge_padding_mask)
        return logp.squeeze(1)

    def forward(
        self,
        node_inputs,
        node_padding_mask,
        edge_mask,
        current_index,
        current_edge,
        edge_padding_mask,
        history_inputs=None,
    ):
        enhanced_node_feature = self.encode_graph(node_inputs, node_padding_mask, edge_mask)
        current_node_feature, enhanced_current_node_feature = self.decode_state(
            enhanced_node_feature,
            current_index,
            node_padding_mask,
        )
        return self.output_policy(
            current_node_feature,
            enhanced_current_node_feature,
            enhanced_node_feature,
            current_edge,
            edge_padding_mask,
            history_inputs=history_inputs,
        )


class QNet(nn.Module):
    def __init__(
        self,
        node_dim,
        embedding_dim,
        enable_wavelet_history=ENABLE_WAVELET_HISTORY,
        history_input_dim=HISTORY_INPUT_DIM,
        history_embed_dim=HISTORY_EMBED_DIM,
        history_wavelet_levels=HISTORY_WAVELET_LEVELS,
        history_encoder_mode=HISTORY_ENCODER_MODE,
        enable_wavelet_utility_loss=ENABLE_WAVELET_UTILITY_LOSS,
    ):
        super().__init__()

        self.enable_wavelet_history = bool(enable_wavelet_history)
        self.enable_wavelet_utility_loss = bool(enable_wavelet_utility_loss)

        self.initial_embedding = nn.Linear(node_dim, embedding_dim)
        self.encoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=6)

        self.decoder = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)
        self.current_embedding = nn.Linear(embedding_dim * 2, embedding_dim)

        self.q_values_layer = nn.Linear(embedding_dim * 2, 1)
        self.utility_aux_head = UtilityAuxHead(embedding_dim * 2)

        if self.enable_wavelet_history:
            self.history_encoder = WaveletHistoryEncoder(
                history_input_dim=history_input_dim,
                history_embed_dim=history_embed_dim,
                levels=history_wavelet_levels,
                mode=history_encoder_mode,
            )
            self.history_state_fusion = _HistoryStateFusion(embedding_dim, history_embed_dim)
            self.history_neighbor_fusion = _HistoryNeighborFusion(embedding_dim, history_embed_dim)

    def encode_graph(self, node_inputs, node_padding_mask, edge_mask):
        node_feature = self.initial_embedding(node_inputs)
        return self.encoder(src=node_feature, key_padding_mask=node_padding_mask, attn_mask=edge_mask)

    def decode_state(self, enhanced_node_feature, current_index, node_padding_mask):
        embedding_dim = enhanced_node_feature.size()[2]
        current_node_feature = torch.gather(enhanced_node_feature, 1, current_index.repeat(1, 1, embedding_dim))
        enhanced_current_node_feature, _ = self.decoder(
            current_node_feature,
            enhanced_node_feature,
            node_padding_mask,
        )
        return current_node_feature, enhanced_current_node_feature

    def _build_action_features(
        self,
        current_node_feature,
        enhanced_current_node_feature,
        enhanced_node_feature,
        current_edge,
        edge_padding_mask=None,
        history_inputs=None,
    ) -> torch.Tensor:
        del edge_padding_mask
        embedding_dim = enhanced_node_feature.size()[2]
        k_size = current_edge.size()[1]
        current_state_feature = self.current_embedding(
            torch.cat((enhanced_current_node_feature, current_node_feature), dim=-1)
        )

        low_context = None
        high_context = None
        if self.enable_wavelet_history and history_inputs is not None:
            low_context, high_context = self.history_encoder(history_inputs)
            current_state_feature = self.history_state_fusion(current_state_feature, low_context)

        neighboring_feature = torch.gather(enhanced_node_feature, 1, current_edge.repeat(1, 1, embedding_dim))
        if self.enable_wavelet_history and high_context is not None:
            neighboring_feature = self.history_neighbor_fusion(neighboring_feature, high_context)
        action_features = torch.cat((current_state_feature.repeat(1, k_size, 1), neighboring_feature), dim=-1)
        return action_features

    def output_q(
        self,
        current_node_feature,
        enhanced_current_node_feature,
        enhanced_node_feature,
        current_edge,
        edge_padding_mask,
        history_inputs=None,
    ):
        action_features = self._build_action_features(
            current_node_feature,
            enhanced_current_node_feature,
            enhanced_node_feature,
            current_edge,
            edge_padding_mask=edge_padding_mask,
            history_inputs=history_inputs,
        )
        return self.q_values_layer(action_features)

    def forward(
        self,
        node_inputs,
        node_padding_mask,
        edge_mask,
        current_index,
        current_edge,
        edge_padding_mask,
        history_inputs=None,
        return_aux=False,
    ):
        enhanced_node_feature = self.encode_graph(node_inputs, node_padding_mask, edge_mask)
        current_node_feature, enhanced_current_node_feature = self.decode_state(
            enhanced_node_feature,
            current_index,
            node_padding_mask,
        )
        action_features = self._build_action_features(
            current_node_feature,
            enhanced_current_node_feature,
            enhanced_node_feature,
            current_edge,
            edge_padding_mask=edge_padding_mask,
            history_inputs=history_inputs,
        )
        q_values = self.q_values_layer(action_features)

        if return_aux:
            utility_pred = self.utility_aux_head(action_features)
            return q_values, utility_pred
        return q_values


def _masked_regression_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, loss_type: str) -> torch.Tensor:
    mask = mask.bool()
    if mask.sum() <= 0:
        return pred.new_tensor(0.0)

    pred_valid = pred[mask]
    target_valid = target[mask]
    if loss_type.lower() == "mse":
        return F.mse_loss(pred_valid, target_valid)
    return F.smooth_l1_loss(pred_valid, target_valid)


def _rasterize_candidate_values(
    values: torch.Tensor,
    candidate_coords: torch.Tensor,
    valid_mask: torch.Tensor,
    patch_size: int,
    patch_sigma: float,
) -> torch.Tensor:
    patch = values.new_zeros((patch_size, patch_size))
    count = values.new_zeros((patch_size, patch_size))
    valid_mask = valid_mask.bool()
    if valid_mask.sum() <= 0:
        return patch

    coords = candidate_coords[valid_mask]
    vals = values[valid_mask]
    scale = torch.max(torch.abs(coords))
    if not torch.isfinite(scale) or float(scale.item()) <= 1e-6:
        scale = coords.new_tensor(1.0)

    coords = torch.clamp(coords / scale, min=-1.0, max=1.0)
    sigma = max(float(patch_sigma), 1e-4)
    snapped = torch.round(coords / sigma) * sigma
    x_idx = torch.clamp(torch.round((snapped[:, 0] + 1.0) * 0.5 * (patch_size - 1)), 0, patch_size - 1).long()
    y_idx = torch.clamp(torch.round((snapped[:, 1] + 1.0) * 0.5 * (patch_size - 1)), 0, patch_size - 1).long()

    for i in range(vals.shape[0]):
        yi = int(y_idx[i].item())
        xi = int(x_idx[i].item())
        patch[yi, xi] += vals[i]
        count[yi, xi] += 1.0

    patch = torch.where(count > 0, patch / torch.clamp(count, min=1.0), patch)
    return patch


def compute_wavelet_utility_aux_loss(
    pred_utility: torch.Tensor,
    target_utility: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
    candidate_coords: torch.Tensor | None = None,
    supervision_mask: torch.Tensor | None = None,
    loss_mode: str = UTILITY_LOSS_MODE,
    loss_weight: float = UTILITY_LOSS_WEIGHT,
    loss_type: str = UTILITY_AUX_LOSS_TYPE,
    base_weight: float = UTILITY_AUX_BASE_WEIGHT,
    wavelet_weight: float = UTILITY_AUX_WAVELET_WEIGHT,
    patch_size: int = UTILITY_PATCH_SIZE,
    patch_sigma: float = UTILITY_PATCH_SIGMA,
    wavelet_levels: int = UTILITY_WAVELET_LEVELS,
    wavelet_rho: float = UTILITY_WAVELET_RHO,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute utility auxiliary loss on valid candidates only."""
    loss_mode = str(loss_mode).strip().lower()
    if supervision_mask is None:
        supervision_mask = torch.isfinite(target_utility.squeeze(-1))
    target_utility = torch.nan_to_num(target_utility, nan=0.0, posinf=0.0, neginf=0.0)

    combined_mask = candidate_valid_mask.bool() & supervision_mask.bool()
    base_loss = _masked_regression_loss(pred_utility, target_utility, combined_mask.unsqueeze(-1), loss_type)

    wavelet_loss = pred_utility.new_tensor(0.0)
    wavelet_count = 0

    if loss_mode == "spatial2d" and candidate_coords is not None:
        patch_size = max(int(patch_size), 3)
        if patch_size % 2 == 0:
            patch_size += 1

        for b_idx in range(pred_utility.size(0)):
            valid_mask_b = combined_mask[b_idx]
            if valid_mask_b.sum() < 2:
                continue

            pred_patch = _rasterize_candidate_values(
                pred_utility[b_idx, :, 0],
                candidate_coords[b_idx],
                valid_mask_b,
                patch_size=patch_size,
                patch_sigma=patch_sigma,
            )
            target_patch = _rasterize_candidate_values(
                target_utility[b_idx, :, 0],
                candidate_coords[b_idx],
                valid_mask_b,
                patch_size=patch_size,
                patch_sigma=patch_sigma,
            )

            low_pred, high_pred = haar_decompose_2d(pred_patch.unsqueeze(0), levels=wavelet_levels)
            low_tgt, high_tgt = haar_decompose_2d(target_patch.unsqueeze(0), levels=wavelet_levels)

            sample_wavelet = F.l1_loss(low_pred, low_tgt)
            for (lh_p, hl_p, hh_p), (lh_t, hl_t, hh_t) in zip(high_pred, high_tgt):
                sample_wavelet = sample_wavelet + float(wavelet_rho) * (
                    F.l1_loss(lh_p, lh_t) + F.l1_loss(hl_p, hl_t) + F.l1_loss(hh_p, hh_t)
                )

            wavelet_loss = wavelet_loss + sample_wavelet
            wavelet_count += 1

        if wavelet_count > 0:
            wavelet_loss = wavelet_loss / wavelet_count

    total_loss = float(loss_weight) * (float(base_weight) * base_loss + float(wavelet_weight) * wavelet_loss)
    return total_loss, base_loss.detach(), wavelet_loss.detach()
