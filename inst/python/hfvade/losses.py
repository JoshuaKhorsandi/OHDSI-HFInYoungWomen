"""ELBO terms for VaDE.

The objective maximised is the evidence lower bound

    L = E_q[log p(x|z)] - E_q[log q(z,c|x) - log p(z,c)]

which train.py minimises as `reconstruction_loss + w * latent`, where w is the
S3VDC annealing weight from schedule.latent_weight. The latent term is
vanilla_kl during the warm-up phase (before the mixture prior exists) and
gmm_latent_loss afterwards.
"""

import torch
import torch.nn.functional as F

from .vade import LOG2PI


def reconstruction_loss(model, recon, x, mask, w_numeric):
    """Negative log-likelihood -E_q[log p(x|z)] over observed entries only.

    Binary columns use Bernoulli cross-entropy on logits (with model.pos_weight
    scaling the positive term); numeric columns use squared error, equivalent to
    a unit-variance Gaussian up to a constant. Each block is averaged over its
    observed entries per row, then rescaled by the block width, so rows with
    little observed data carry the same weight as complete ones while the total
    stays on a summed scale comparable to the latent term.

    Args:
        model: VaDE, supplying binary_idx, num_idx and pos_weight.
        recon: (batch, i_dim) raw decoder output.
        x: (batch, i_dim) targets, neutral-filled at missing positions.
        mask: (batch, i_dim) 1 where observed, 0 where missing.
        w_numeric: relative weight of the numeric block against the binary one.

    Returns:
        Scalar tensor, mean over the batch.
    """
    loss = x.new_zeros(x.shape[0])

    if model.binary_idx.numel() > 0:
        idx = model.binary_idx
        per_elem = F.binary_cross_entropy_with_logits(
            recon[:, idx], x[:, idx], pos_weight=model.pos_weight, reduction="none"
        )
        loss = loss + masked_mean(per_elem, mask[:, idx]) * idx.numel()

    if model.num_idx.numel() > 0:
        idx = model.num_idx
        per_elem = (recon[:, idx] - x[:, idx]).pow(2)
        loss = loss + w_numeric * masked_mean(per_elem, mask[:, idx]) * idx.numel()

    return loss.mean()


def vanilla_kl(mu, logvar):
    """KL(q(z|x) || N(0, I)), the standard VAE latent term.

    Used during the S3VDC gamma warm-up, when the mixture prior has not been
    initialised yet and a unit Gaussian is the only prior available.

    Args:
        mu, logvar: (batch, z_dim) posterior parameters.

    Returns:
        Scalar tensor, mean over the batch.
    """
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()


def gmm_latent_loss(model, mu, logvar):
    """Latent ELBO term under the Gaussian-mixture prior.

    Expands E_q[log q(z,c|x) - log p(z,c)] into four pieces, in order below:

        + sum_c gamma_c * 0.5 * sum_j [ log(2pi) + logvar_cj
                                        + ((mu_j - mu_cj)^2 + exp(logvar_j)) / exp(logvar_cj) ]
                                                        -E[log p(z|c)]
        - sum_c gamma_c log pi_c                        -E[log p(c)]
        + sum_c gamma_c log gamma_c                      E[log q(c|z)]
        - 0.5 * sum_j (1 + logvar_j)                     E[log q(z|x)], up to a constant

    Args:
        model: VaDE, supplying pi_logits, mu_c, logvar_c and responsibilities.
        mu, logvar: (batch, z_dim) posterior parameters.

    Returns:
        (loss, gamma) with loss a scalar mean over the batch and gamma the
        (batch, c) responsibilities, returned so callers can log cluster usage
        without recomputing them.
    """
    gamma = model.responsibilities(mu)
    log_pi = F.log_softmax(model.pi_logits, dim=0)

    diff = mu.unsqueeze(1) - model.mu_c.unsqueeze(0)
    logvar_c = model.logvar_c.unsqueeze(0)
    expected_nll = 0.5 * (
        LOG2PI
        + logvar_c
        + (diff.pow(2) + logvar.exp().unsqueeze(1)) / logvar_c.exp()
    ).sum(dim=2)

    loss = (gamma * expected_nll).sum(dim=1)
    loss = loss - (gamma * log_pi.unsqueeze(0)).sum(dim=1)
    loss = loss + (gamma * torch.log(gamma + 1e-10)).sum(dim=1)
    loss = loss - 0.5 * (1 + logvar).sum(dim=1)
    loss = loss.mean()

    return loss, gamma


def masked_mean(per_elem, mask):
    """Mean of per_elem over masked-in entries, per row.

    Uses torch.where rather than multiplication so that NaN at a masked-out
    position contributes nothing; NaN * 0 would still be NaN.

    Args:
        per_elem: (batch, k) per-element losses.
        mask: (batch, k) 1 where observed, 0 where missing.

    Returns:
        (batch,) tensor. Rows with nothing observed return 0.
    """
    kept = torch.where(mask.bool(), per_elem, torch.zeros_like(per_elem))
    return kept.sum(dim=1) / (mask.sum(dim=1) + 1e-8)
