from __future__ import annotations

import math

import torch
import torch.nn as nn

from .parameter import (
    HPBG_CLUSTER_PRIOR_INDEX,
    HPBG_CLUSTER_RESOLUTION,
    HPBG_CRITIC_PRIVILEGED_DIM,
    UPDATING_MAP_SIZE,
    USE_HIERARCHICAL_GRAPH,
    USE_LF_ATTENTION_HF_RESIDUAL,
    USE_PRIVILEGED_WAVELET_DISTILLATION,
    WAVELET_FUSE_DIM,
    WAVELET_LF_QK,
    WAVELET_SCALES,
)
from .wavelet_graph import multiscale_wavelet_decompose, normalize_scales


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
        return torch.log_softmax(logits, dim=-1)


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
    def __init__(
        self,
        embedding_dim,
        n_head,
        use_lf_attention_hf_residual=False,
        wavelet_scales=WAVELET_SCALES,
        wavelet_fuse_dim=WAVELET_FUSE_DIM,
        wavelet_lf_qk=WAVELET_LF_QK,
    ):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim),
        )
        self.normalization2 = Normalization(embedding_dim)
        self.use_lf_attention_hf_residual = bool(use_lf_attention_hf_residual)
        self.wavelet_scales = normalize_scales(wavelet_scales)
        self.wavelet_lf_qk = bool(wavelet_lf_qk)
        self.n_wavelet = len(self.wavelet_scales)

        if self.use_lf_attention_hf_residual:
            low_input_dim = embedding_dim * self.n_wavelet
            high_input_dim = embedding_dim * self.n_wavelet
            if self.wavelet_lf_qk:
                self.q_low_fuse = nn.Sequential(
                    nn.Linear(low_input_dim, wavelet_fuse_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(wavelet_fuse_dim, embedding_dim),
                )
                self.k_low_fuse = nn.Sequential(
                    nn.Linear(low_input_dim, wavelet_fuse_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(wavelet_fuse_dim, embedding_dim),
                )
            else:
                self.lf_fuse = nn.Sequential(
                    nn.Linear(low_input_dim, wavelet_fuse_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(wavelet_fuse_dim, embedding_dim),
                )

            self.value_fuse = nn.Sequential(
                nn.Linear(embedding_dim + high_input_dim, wavelet_fuse_dim),
                nn.ReLU(inplace=True),
                nn.Linear(wavelet_fuse_dim, embedding_dim),
            )

    def forward(self, src, key_padding_mask=None, attn_mask=None):
        h0 = src
        h = self.normalization1(src)

        if self.use_lf_attention_hf_residual:
            low_features, high_features, _, _ = multiscale_wavelet_decompose(
                h,
                attn_mask,
                key_padding_mask,
                scales=self.wavelet_scales,
            )
            low_concat = torch.cat(low_features, dim=-1)
            high_concat = torch.cat(high_features, dim=-1)
            if self.wavelet_lf_qk:
                q = self.q_low_fuse(low_concat)
                k = self.k_low_fuse(low_concat)
            else:
                lf = self.lf_fuse(low_concat)
                q = lf
                k = lf
            v = self.value_fuse(torch.cat((h, high_concat), dim=-1))
            h, _ = self.multi_head_attention(
                q=q,
                k=k,
                v=v,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        else:
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
    def __init__(
        self,
        embedding_dim=128,
        n_head=8,
        n_layer=1,
        use_lf_attention_hf_residual=False,
        wavelet_scales=WAVELET_SCALES,
        wavelet_fuse_dim=WAVELET_FUSE_DIM,
        wavelet_lf_qk=WAVELET_LF_QK,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            EncoderLayer(
                embedding_dim,
                n_head,
                use_lf_attention_hf_residual=use_lf_attention_hf_residual,
                wavelet_scales=wavelet_scales,
                wavelet_fuse_dim=wavelet_fuse_dim,
                wavelet_lf_qk=wavelet_lf_qk,
            )
            for _ in range(n_layer)
        )

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


class CoarseFineContext(nn.Module):
    def __init__(
        self,
        embedding_dim,
        cluster_prior_index=HPBG_CLUSTER_PRIOR_INDEX,
        cluster_resolution=HPBG_CLUSTER_RESOLUTION,
        updating_map_size=UPDATING_MAP_SIZE,
        message_steps=1,
    ):
        super().__init__()
        self.cluster_prior_index = int(cluster_prior_index)
        self.message_steps = max(int(message_steps), 1)
        self.cluster_resolution_norm = max(float(cluster_resolution) / max(float(updating_map_size) * 2.0, 1.0), 1e-6)
        self.coarse_update = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.fuse = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.norm = Normalization(embedding_dim)

    def _bucket_coords(self, node_inputs):
        return torch.floor(node_inputs[..., :2] / self.cluster_resolution_norm).to(torch.long)

    def _coarse_context_for_batch(self, hidden, node_inputs, edge_mask, valid_mask):
        device = hidden.device
        dtype = hidden.dtype
        embedding_dim = hidden.size(-1)
        valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
        coarse_context = torch.zeros_like(hidden)
        if valid_indices.numel() == 0:
            return coarse_context

        hidden_valid = hidden.index_select(0, valid_indices)
        bucket_valid = self._bucket_coords(node_inputs).index_select(0, valid_indices)
        unique_buckets, inverse = torch.unique(bucket_valid, dim=0, return_inverse=True)
        n_cluster = int(unique_buckets.size(0))

        cluster_count = torch.zeros(n_cluster, 1, device=device, dtype=dtype)
        cluster_count.index_add_(0, inverse, torch.ones(hidden_valid.size(0), 1, device=device, dtype=dtype))
        cluster_hidden = torch.zeros(n_cluster, embedding_dim, device=device, dtype=dtype)
        cluster_hidden.index_add_(0, inverse, hidden_valid)
        cluster_hidden = cluster_hidden / cluster_count.clamp_min(1.0)

        if node_inputs.size(-1) > self.cluster_prior_index:
            prior = node_inputs[..., self.cluster_prior_index : self.cluster_prior_index + 1].index_select(0, valid_indices)
            prior = prior.clamp(0.0, 1.0).to(dtype=dtype)
            cluster_prior = torch.zeros(n_cluster, 1, device=device, dtype=dtype)
            cluster_prior.index_add_(0, inverse, prior)
            cluster_prior = cluster_prior / cluster_count.clamp_min(1.0)
            prior_context = (cluster_hidden * cluster_prior).sum(dim=0, keepdim=True) / cluster_prior.sum().clamp_min(1.0)
        else:
            prior_context = cluster_hidden.mean(dim=0, keepdim=True)

        cluster_adj = torch.eye(n_cluster, device=device, dtype=dtype)
        edge_sub = edge_mask.index_select(0, valid_indices).index_select(1, valid_indices)
        connected_rows, connected_cols = torch.nonzero(edge_sub <= 0, as_tuple=True)
        if connected_rows.numel() > 0:
            src_cluster = inverse.index_select(0, connected_rows)
            dst_cluster = inverse.index_select(0, connected_cols)
            cluster_adj[src_cluster, dst_cluster] = 1.0
            cluster_adj[dst_cluster, src_cluster] = 1.0

        degree = cluster_adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        cluster_transition = cluster_adj / degree
        propagated = cluster_hidden
        for _ in range(self.message_steps):
            neighbor_context = cluster_transition @ propagated
            propagated = self.coarse_update(torch.cat((propagated, neighbor_context), dim=-1))

        propagated = propagated + prior_context
        return coarse_context.index_copy(0, valid_indices, propagated.index_select(0, inverse))

    def forward(self, hidden, node_inputs, node_padding_mask, edge_mask):
        valid = (~node_padding_mask.squeeze(1).bool()).unsqueeze(-1).to(dtype=hidden.dtype)
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        global_context = (hidden * valid).sum(dim=1, keepdim=True) / denom

        coarse_context_batches = []
        batch_size = hidden.size(0)
        for batch_index in range(batch_size):
            coarse_context_batches.append(
                self._coarse_context_for_batch(
                    hidden[batch_index],
                    node_inputs[batch_index],
                    edge_mask[batch_index],
                    valid[batch_index].squeeze(-1).bool(),
                )
            )
        coarse_context = torch.stack(coarse_context_batches, dim=0)

        context = torch.cat(
            (
                hidden,
                global_context.expand_as(hidden),
                coarse_context,
            ),
            dim=-1,
        )
        return self.norm(hidden + self.fuse(context))


class PolicyNet(nn.Module):
    def __init__(
        self,
        node_dim,
        embedding_dim,
        use_lf_attention_hf_residual=USE_LF_ATTENTION_HF_RESIDUAL,
        use_privileged_wavelet_distillation=USE_PRIVILEGED_WAVELET_DISTILLATION,
        use_hierarchical_context=USE_HIERARCHICAL_GRAPH,
        wavelet_scales=WAVELET_SCALES,
        wavelet_fuse_dim=WAVELET_FUSE_DIM,
        wavelet_lf_qk=WAVELET_LF_QK,
    ):
        super().__init__()
        self.initial_embedding = nn.Linear(node_dim, embedding_dim)
        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            n_head=8,
            n_layer=6,
            use_lf_attention_hf_residual=use_lf_attention_hf_residual,
            wavelet_scales=wavelet_scales,
            wavelet_fuse_dim=wavelet_fuse_dim,
            wavelet_lf_qk=wavelet_lf_qk,
        )
        self.decoder = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)
        self.use_hierarchical_context = bool(use_hierarchical_context)
        self.hierarchical_context = CoarseFineContext(embedding_dim) if self.use_hierarchical_context else None
        self.current_embedding = nn.Linear(embedding_dim * 2, embedding_dim)
        self.pointer = SingleHeadAttention(embedding_dim)
        self.belief_target_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim // 2, HPBG_CRITIC_PRIVILEGED_DIM),
            nn.Sigmoid(),
        )
        del use_privileged_wavelet_distillation

    def encode_graph(self, node_inputs, node_padding_mask, edge_mask):
        node_feature = self.initial_embedding(node_inputs)
        hidden = self.encoder(src=node_feature, key_padding_mask=node_padding_mask, attn_mask=edge_mask)
        if self.use_hierarchical_context:
            hidden = self.hierarchical_context(hidden, node_inputs, node_padding_mask, edge_mask)
        return hidden

    def predict_belief_targets(self, enhanced_node_feature):
        return self.belief_target_head(enhanced_node_feature)

    def decode_state(self, enhanced_node_feature, current_index, node_padding_mask):
        embedding_dim = enhanced_node_feature.size()[2]
        current_node_feature = torch.gather(enhanced_node_feature, 1, current_index.repeat(1, 1, embedding_dim))
        enhanced_current_node_feature, _ = self.decoder(current_node_feature, enhanced_node_feature, node_padding_mask)
        return current_node_feature, enhanced_current_node_feature

    def output_policy(
        self,
        current_node_feature,
        enhanced_current_node_feature,
        enhanced_node_feature,
        current_edge,
        edge_padding_mask,
    ):
        embedding_dim = enhanced_node_feature.size()[2]
        current_state_feature = self.current_embedding(
            torch.cat((enhanced_current_node_feature, current_node_feature), dim=-1)
        )
        neighboring_feature = torch.gather(enhanced_node_feature, 1, current_edge.repeat(1, 1, embedding_dim))
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
        return_hidden=False,
    ):
        enhanced_node_feature = self.encode_graph(node_inputs, node_padding_mask, edge_mask)
        current_node_feature, enhanced_current_node_feature = self.decode_state(
            enhanced_node_feature,
            current_index,
            node_padding_mask,
        )
        logp = self.output_policy(
            current_node_feature,
            enhanced_current_node_feature,
            enhanced_node_feature,
            current_edge,
            edge_padding_mask,
        )
        if return_hidden:
            return logp, enhanced_node_feature
        return logp


class QNet(nn.Module):
    def __init__(
        self,
        node_dim,
        embedding_dim,
        use_lf_attention_hf_residual=USE_LF_ATTENTION_HF_RESIDUAL,
        wavelet_scales=WAVELET_SCALES,
        wavelet_fuse_dim=WAVELET_FUSE_DIM,
        wavelet_lf_qk=WAVELET_LF_QK,
    ):
        super().__init__()
        self.initial_embedding = nn.Linear(node_dim, embedding_dim)
        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            n_head=8,
            n_layer=6,
            use_lf_attention_hf_residual=use_lf_attention_hf_residual,
            wavelet_scales=wavelet_scales,
            wavelet_fuse_dim=wavelet_fuse_dim,
            wavelet_lf_qk=wavelet_lf_qk,
        )
        self.decoder = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)
        self.q_values_layer = nn.Linear(embedding_dim * 3, 1)

    def encode_graph(self, node_inputs, node_padding_mask, edge_mask):
        node_feature = self.initial_embedding(node_inputs)
        return self.encoder(src=node_feature, key_padding_mask=node_padding_mask, attn_mask=edge_mask)

    def decode_state(self, enhanced_node_feature, current_index, node_padding_mask):
        embedding_dim = enhanced_node_feature.size()[2]
        current_node_feature = torch.gather(enhanced_node_feature, 1, current_index.repeat(1, 1, embedding_dim))
        enhanced_current_node_feature, _ = self.decoder(current_node_feature, enhanced_node_feature, node_padding_mask)
        return current_node_feature, enhanced_current_node_feature

    def output_q(
        self,
        current_node_feature,
        enhanced_current_node_feature,
        enhanced_node_feature,
        current_edge,
        edge_padding_mask,
    ):
        del edge_padding_mask
        embedding_dim = enhanced_node_feature.size()[2]
        k_size = current_edge.size()[1]
        current_state_feature = torch.cat((enhanced_current_node_feature, current_node_feature), dim=-1)
        neighboring_feature = torch.gather(enhanced_node_feature, 1, current_edge.repeat(1, 1, embedding_dim))
        action_features = torch.cat((current_state_feature.repeat(1, k_size, 1), neighboring_feature), dim=-1)
        return self.q_values_layer(action_features)

    def forward(
        self,
        node_inputs,
        node_padding_mask,
        edge_mask,
        current_index,
        current_edge,
        edge_padding_mask,
        return_hidden=False,
    ):
        enhanced_node_feature = self.encode_graph(node_inputs, node_padding_mask, edge_mask)
        current_node_feature, enhanced_current_node_feature = self.decode_state(
            enhanced_node_feature,
            current_index,
            node_padding_mask,
        )
        q_values = self.output_q(
            current_node_feature,
            enhanced_current_node_feature,
            enhanced_node_feature,
            current_edge,
            edge_padding_mask,
        )
        if return_hidden:
            return q_values, enhanced_node_feature
        return q_values
