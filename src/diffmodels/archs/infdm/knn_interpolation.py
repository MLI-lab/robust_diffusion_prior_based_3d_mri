"""Inverse-distance KNN interpolation onto the UNO grid, in plain torch."""

import torch

MAX_PAIRWISE_ELEMENTS = 2 ** 23


def gather_neighbours(x, idx):
    """Index (B, P, C) features by (B, Q, K) neighbour indices -> (B, Q, K, C)."""
    batch_size, num_queries, k = idx.shape
    channels = x.size(-1)
    flat_idx = idx.reshape(batch_size, num_queries * k, 1).expand(-1, -1, channels)
    return x.gather(1, flat_idx).reshape(batch_size, num_queries, k, channels)


def _search_keys(distances):
    """Pack (distance, point index) into one int64 key that sorts on both."""
    num_points = distances.size(-1)
    keys = distances.contiguous().view(torch.int32).to(torch.int64) << 32
    return keys | torch.arange(num_points, device=distances.device)


def knn_indices(query, points, k):
    """Indices of the k nearest ``points`` for each ``query`` -> (B, Q, K)."""
    batch_size, num_queries, _ = query.shape
    num_points = points.size(1)
    k = min(k, num_points)

    chunk = max(1, MAX_PAIRWISE_ELEMENTS // max(batch_size * num_points, 1))
    idx = torch.empty(batch_size, num_queries, k, dtype=torch.long, device=points.device)
    for start in range(0, num_queries, chunk):
        stop = min(start + chunk, num_queries)
        distances = torch.cdist(
            query[:, start:stop], points, compute_mode='donot_use_mm_for_euclid_dist')
        keys = _search_keys(distances.float())
        idx[:, start:stop] = keys.topk(k, dim=-1, largest=False).indices
    return idx


def knn_inverse_distance_weights(query, points, k):
    """Neighbour indices and their inverse-squared-distance weights."""
    with torch.no_grad():
        query = query.to(points.dtype)
        idx = knn_indices(query, points, k)
        neighbour_coords = gather_neighbours(points, idx)
        diff = neighbour_coords - query.unsqueeze(2)
        squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
        weights = 1.0 / torch.clamp(squared_distance, min=1e-16)
    return idx, weights
