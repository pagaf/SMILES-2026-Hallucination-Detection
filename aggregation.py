from __future__ import annotations

import torch
import math


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    last_layer = hidden_states[-1]
    real_positions = attention_mask.nonzero(as_tuple=False)
    last_pos = int(real_positions[-1].item())
    return last_layer[last_pos]


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    n_layers, seq_len, hidden_dim = hidden_states.shape
    device = hidden_states.device

    real_positions = attention_mask.nonzero(as_tuple=False)
    last_pos = int(real_positions[-1].item())

    l2_norm_last = torch.norm(hidden_states[-1, last_pos], p=2).unsqueeze(0)
    l2_norm_penultimate = torch.norm(hidden_states[-2, last_pos], p=2).unsqueeze(0)

    cos = torch.nn.CosineSimilarity(dim=0)
    drift_1 = cos(hidden_states[-1, last_pos], hidden_states[-2, last_pos]).unsqueeze(0)

    if n_layers >= 4:
        drift_2 = cos(hidden_states[-2, last_pos], hidden_states[-4, last_pos]).unsqueeze(0)
    else:
        drift_2 = torch.tensor([1.0], device=device)

    mask_bool = attention_mask.bool()
    real_tokens = hidden_states[-1][mask_bool]
    n_real = real_tokens.shape[0]

    if n_real > 1:
        centered = real_tokens - real_tokens.mean(dim=0, keepdim=True)
        gram = torch.mm(centered, centered.T) / (n_real - 1)

        try:
            eigvals = torch.linalg.eigvalsh(gram).float()
            eigvals = torch.clamp(eigvals, min=1e-8)

            max_eigval = eigvals.max().unsqueeze(0)

            eig_sum = eigvals.sum()
            probs = eigvals / eig_sum
            spectral_entropy = -(probs * torch.log(probs)).sum().unsqueeze(0)
        except Exception:
            spectral_entropy = torch.tensor([0.0], device=device)
            max_eigval = torch.tensor([0.0], device=device)
    else:
        spectral_entropy = torch.tensor([0.0], device=device)
        max_eigval = torch.tensor([0.0], device=device)

    geometric_feats = torch.cat(
        [
            l2_norm_last,
            l2_norm_penultimate,
            drift_1,
            drift_2,
            spectral_entropy,
            max_eigval,
        ],
        dim=0,
    )

    return torch.log1p(geometric_feats)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    agg_features = aggregate(hidden_states, attention_mask)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features