"""Cluster-quality and reconstruction metrics.

The three cluster metrics mirror the TensorFlow eval metrics used by S3VDC
(Copyright (C) king.com Ltd 2019, MIT, https://github.com/king/s3vdc).

Each returns a plain float rather than TF's (value, update_op) pair: PyTorch has
no streaming-metric protocol, so the caller accumulates embeddings over the eval
set and calls these once on the concatenated result.
"""

import numpy as np
import torch
from sklearn.metrics import calinski_harabasz_score


def cluster_metrics(z, labels, centers):
    """Compute every cluster-quality metric for one eval pass.

    Args:
        z: (n, z_dim) latent means for the eval set.
        labels: (n,) hard cluster assignments.
        centers: (c, z_dim) the model's learned cluster centers (mu_c).

    Returns:
        dict of metric name -> float.
    """
    return {
        "separation": cluster_separation(centers),
        "silhouette": simplified_silhouette(centers, z, labels),
        "calinski_harabasz": calinski_harabasz(z, labels),
        "active_clusters": active_clusters(labels),
        "assignment_entropy": assignment_entropy(labels, centers.shape[0]),
    }


def reconstruction_metrics(recon, x, mask):
    """Masked reconstruction error, averaged over observed entries only.

    Mirrors S3VDC's mean_absolute_error / mean_squared_error /
    mean_cosine_distance eval metrics, but every reduction is restricted to
    entries where mask == 1 so missing measurements never count as error.

    Cosine distance is 1 - cos(x, recon) per row; both vectors are masked
    before the dot product so missing entries contribute to neither it nor the
    norms.

    Args:
        recon: (batch, i_dim) reconstruction. Raw decoder output has logits on
            binary columns, so pass model.reconstruct(x) for the probability
            scale.
        x: (batch, i_dim) targets.
        mask: (batch, i_dim) 1 where observed, 0 where missing.

    Returns:
        dict with mae, mse and cosine_distance, all floats.
    """
    observed = mask.sum().clamp_min(1.0)
    err = (recon - x) * mask

    cos = torch.nn.functional.cosine_similarity(recon * mask, x * mask, dim=1, eps=1e-8)

    return {
        "mae": (err.abs().sum() / observed).item(),
        "mse": (err.pow(2).sum() / observed).item(),
        "cosine_distance": (1.0 - cos).mean().item(),
    }


def cluster_separation(centers):
    """Average L2 distance between every pair of cluster centers.

    Formula: (2/(k^2-k)) * sum_{i=1}^{k} sum_{j=i+1}^{k} ||a_i - a_j||_2

    Implementation detail (following S3VDC): the pairwise squared distances D of
    a matrix A of row vectors can be written as a single matrix op. Since
        D[i,j] = (a[i]-a[j])(a[i]-a[j])' = r[i] - 2*a[i]a[j]' + r[j]
    where r[i] is the squared norm of row i, broadcasting r as a column vector
    gives D = r - 2*A*A' + r'.

    The relu/clamp guards against tiny negative values from floating-point
    cancellation, which would make sqrt produce NaN. Summing the full matrix
    counts each pair twice and adds a zero diagonal, hence dividing by k^2-k
    rather than by the number of unique pairs.

    Args:
        centers: (c, z_dim) cluster centers.

    Returns:
        Mean pairwise distance; higher is better. Returns 0.0 for c < 2.
    """
    k = centers.shape[0]
    if k < 2:
        return 0.0

    r = (centers * centers).sum(dim=1, keepdim=True)
    d_sq = r - 2.0 * centers @ centers.T + r.T

    return (d_sq.clamp_min(0.0).sqrt().sum() / (k**2 - k)).item()


def simplified_silhouette(centers, z, labels):
    """Silhouette score computed against cluster centers instead of all points.

    The standard silhouette needs every pairwise point distance, which is
    O(n^2). The simplified variant replaces "mean distance to points in a
    cluster" with "distance to that cluster's center", making it O(n*k):

        a(i) = distance from point i to its own cluster's center
        b(i) = distance from point i to the nearest other cluster's center
        s(i) = (b(i) - a(i)) / max(a(i), b(i))

    Args:
        centers: (c, z_dim) cluster centers, typically model.mu_c.
        z: (n, z_dim) latent points.
        labels: (n,) hard assignments indexing into centers.

    Returns:
        Mean s(i) in [-1, 1]; higher is better, negative means points sit closer
        to another cluster's center than their own. Returns 0.0 for c < 2.
    """
    c = centers.shape[0]
    if c < 2:
        return 0.0

    distances = torch.cdist(z, centers)
    own = distances.gather(1, labels.unsqueeze(1)).squeeze(1)

    other = distances.scatter(1, labels.unsqueeze(1), float("inf"))
    nearest_other = other.min(dim=1).values

    denom = torch.maximum(own, nearest_other).clamp_min(1e-12)
    return ((nearest_other - own) / denom).mean().item()


def calinski_harabasz(z, labels):
    """Ratio of between-cluster to within-cluster dispersion (variance ratio).

    CH = [BGSS / (k-1)] / [WGSS / (n-k)], where BGSS is the between-group and
    WGSS the within-group sum of squares. Unlike the other two metrics this one
    recomputes centroids from the assignments rather than using the model's
    learned mu_c, matching both sklearn and S3VDC's version.

    Args:
        z: (n, z_dim) latent points.
        labels: (n,) hard assignments.

    Returns:
        Variance ratio; higher is better. Returns 0.0 when fewer than two
        clusters are populated, rather than letting sklearn raise.
    """
    labels = to_numpy(labels)
    if len(np.unique(labels)) < 2:
        return 0.0

    return float(calinski_harabasz_score(to_numpy(z), labels))


def active_clusters(labels):
    """Number of clusters that won at least one point.

    Cluster collapse is the standard VaDE failure mode, so this is the cheapest
    early warning that a run has gone wrong.

    Args:
        labels: (n,) hard assignments.

    Returns:
        Int count of distinct labels present.
    """
    return int(np.unique(to_numpy(labels)).size)


def assignment_entropy(labels, c):
    """Normalized entropy of the cluster-size distribution, in [0, 1].

    1.0 means every cluster holds an equal share of the data; 0.0 means a single
    cluster absorbed everything. Complements active_clusters by catching the
    case where clusters are technically populated but wildly imbalanced.

    Args:
        labels: (n,) hard assignments.
        c: total number of clusters, including empty ones.

    Returns:
        Float in [0, 1]. Returns 0.0 for c < 2.
    """
    if c < 2:
        return 0.0

    counts = np.bincount(to_numpy(labels), minlength=c).astype(np.float64)
    p = counts[counts > 0] / counts.sum()

    return float(-(p * np.log(p)).sum() / np.log(c))


def to_numpy(tensor):
    """Detach a tensor to a numpy array, passing arrays through unchanged.

    Args:
        tensor: torch.Tensor on any device, or anything np.asarray accepts.

    Returns:
        numpy.ndarray.
    """
    return tensor.detach().cpu().numpy() if torch.is_tensor(tensor) else np.asarray(tensor)
