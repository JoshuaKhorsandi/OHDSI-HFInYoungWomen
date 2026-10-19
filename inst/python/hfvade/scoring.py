"""Trial scores for hyperparameter search.

Two numbers per trial, clustpy_vade_loss is a held-out
likelihood; cluster_quality is a structure score. Reconstruction keeps improving
long after cluster structure stops, and a model that has collapsed onto one live
component can still post a good likelihood, so neither is trustworthy alone.

The likelihood is ClustPy's, not this repo's. clustpy.deep.vade._get_gamma and
_compute_vade_loss are imported from the installed package and called on this
model's parameters, the score cannot be lowered by tuning w_numeric, responsibility_scale
or normalize_responsibilities.
"""

import torch
import torch.nn.functional as F
from clustpy.deep.vade import _compute_vade_loss, _get_gamma

from .losses import reconstruction_loss
from .train import unpack


@torch.no_grad()
def clustpy_vade_loss(model, loader, device="cpu"):
    """Held-out VaDE loss under ClustPy's formulation, per row.

    The loader is encoded batch by batch but `_compute_vade_loss` is called once
    on the concatenated result, so the evaluation set is a single batch as far as
    ClustPy's expression is concerned.

    The reconstruction term is this repo's masked mixed-type likelihood rather
    than ClustPy's `BCELoss(reduction='sum')`, which is undefined on the
    z-scored numeric block and would score neutral-filled entries as if they had
    been measured. It is evaluated at w_numeric = 1 so that a trial cannot
    improve its score by down-weighting a block.

    Encoding is deterministic (z = mu) where ClustPy samples z, so repeated
    calls on the same weights agree and Optuna is not fed sampling noise.

    Args:
        model: trained VaDE.
        loader: held-out DataLoader yielding (x,) or (x, mask).
        device: device to run on.

    Returns:
        Float, mean per-row loss. Lower is better.
    """
    model.to(device).eval()
    x, mask, recon, mu, logvar = encode_all(model, loader, device)

    pi = F.softmax(model.pi_logits, dim=0)
    gamma = _get_gamma(pi, model.mu_c, model.logvar_c, mu)

    latent = _compute_vade_loss(pi, model.mu_c, model.logvar_c, mu, logvar, x, gamma,
                                recon, zero_loss, 1.0, 0.0)
    rec = reconstruction_loss(model, recon, x, mask, w_numeric=1.0)

    return float(latent) / len(x) + float(rec)


def cluster_quality(metrics, c):
    """Silhouette discounted by how many of the c components stayed alive.

    A run that puts every row in one cluster scores a perfect silhouette, so the
    raw metric cannot be optimised directly. Scaling by active_clusters / c
    makes collapse cost proportionally what it saves.

    Args:
        metrics: dict from train.evaluate, with silhouette and active_clusters.
        c: number of mixture components the model was given.

    Returns:
        Float in [-1, 1]. Higher is better.
    """
    return metrics["silhouette"] * metrics["active_clusters"] / c


@torch.no_grad()
def encode_all(model, loader, device):
    """Run a whole loader through the model and concatenate every tensor.

    Args:
        model: VaDE in eval mode.
        loader: DataLoader yielding (x,) or (x, mask).
        device: device to run on.

    Returns:
        (x, mask, recon, mu, logvar), each an (n, ...) tensor on `device`, with
        mask all ones where the loader carried none.
    """
    parts = [[], [], [], [], []]

    for batch in loader:
        x, mask = unpack(batch)
        x = x.to(device)
        mask = torch.ones_like(x) if mask is None else mask.to(device)
        recon, mu, logvar, _ = model(x, sample=False)
        for part, value in zip(parts, (x, mask, recon, mu, logvar)):
            part.append(value)

    return tuple(torch.cat(part) for part in parts)


def zero_loss(recon, x):
    """Reconstruction stand-in for ClustPy's ssl_loss_fn slot.

    clustpy_vade_loss calls _compute_vade_loss with ssl_loss_weight = 0 to pull
    out the latent term alone, but the signature still requires a callable.

    Args:
        recon, x: ignored.

    Returns:
        Scalar zero tensor.
    """
    return torch.zeros((), device=recon.device)
