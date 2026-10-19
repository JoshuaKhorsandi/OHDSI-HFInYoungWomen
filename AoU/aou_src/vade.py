"""Variational Deep Embedding (VaDE), with S3VDC improvements, for mixed binary/numeric tabular data.

Model definition and checkpointing. The losses that train it live in losses.py,
the S3VDC phase schedule in schedule.py.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import VaDEConfig


class VaDE(nn.Module):
    """Deep generative clustering model with a Gaussian-mixture latent prior, see Jiang et. al (2017)
    For scheduling, Mini-batch GMM and other VaDE improvements, see the implementation of the S3VDC paper from Cao et al (2020)

    Generative process, for c clusters and a z_dim latent space:

        c ~ Categorical(pi)
        z | c ~ N(mu_c, diag(sigma_c^2))
        x | z ~ Bernoulli(sigmoid(f(z)))   on binary features
                N(f(z), 1)                 on numeric features

    where f is the decoder. Inference uses a factorised posterior
    q(z, c | x) = q(z | x) q(c | z), with q(z | x) = N(mu_phi(x), sigma_phi(x)^2)
    produced by the encoder, and q(c | z) set to the exact posterior

        gamma_c = p(c | z) = pi_c N(z; mu_c, sigma_c^2) / sum_k pi_k N(z; mu_k, sigma_k^2)

    Attributes:
        cfg: VaDEConfig holding architecture and responsibility settings.
        binary_idx, num_idx: long buffers indexing the binary and numeric
            columns of x. Registered as buffers so they follow .to(device).
        pos_weight: optional (n_binary,) float buffer up-weighting positives in
            the binary reconstruction term, or None.
        pi_logits: (c,) learnable mixture logits.
        mu_c: (c, z_dim) learnable cluster means.
        logvar_c: (c, z_dim) learnable cluster log-variances.
    """

    def __init__(self, cfg, binary_idx, num_idx, pos_weight=None):
        """Build the model and initialise the mixture prior (ie pi, the parameter of the categorical distribution over c) to near-uniform.

        Args:
            cfg: VaDEConfig.
            binary_idx: sequence of column indices modelled as Bernoulli.
            num_idx: sequence of column indices modelled as Gaussian.
            pos_weight: optional per-binary-feature positive weight, shape
                (len(binary_idx),). See data.positive_weights.

        The mixture parameters are placeholders: train.init_gmm_params
        overwrites all three from a KMeans + GMM fit once the encoder has completed the S3VDC gamma 'warmup' phase. 
        """
        super().__init__()

        self.cfg = cfg
        self.register_buffer("binary_idx", torch.as_tensor(binary_idx, dtype=torch.long))
        self.register_buffer("num_idx", torch.as_tensor(num_idx, dtype=torch.long))
        self.register_buffer(
            "pos_weight",
            None if pos_weight is None else torch.as_tensor(pos_weight, dtype=torch.float32),
        )

        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

        self.pi_logits = nn.Parameter(torch.zeros(cfg.c))
        self.mu_c = nn.Parameter(torch.randn(cfg.c, cfg.z_dim) * 0.1)
        self.logvar_c = nn.Parameter(torch.zeros(cfg.c, cfg.z_dim))

    def forward(self, x, sample=True):
        """Encode, sample a latent, and decode.

        Args:
            x: (batch, i_dim) float tensor, already neutral-filled.
            sample: draw z from q(z|x) when True; use mu directly when False.

        Returns:
            (recon, mu, logvar, z) where recon is (batch, i_dim) raw decoder
            output -- logits on binary columns, values on numeric columns -- and
            mu, logvar, z are each (batch, z_dim).
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar, sample)
        return self.decoder(z), mu, logvar, z

    def reparameterize(self, mu, logvar, sample=True):
        """Draw z = mu + sigma * eps with eps ~ N(0, I), the so called reparameterization 'trick'

        Args:
            mu, logvar: (batch, z_dim) posterior parameters.
            sample: when False, or in eval mode, returns mu unchanged so
                encoding is deterministic.

        Returns:
            (batch, z_dim) latent tensor.
        """
        if not (sample and self.training):
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def responsibilities(self, mu):
        """Cluster posterior gamma_c = p(c | z), evaluated at z = mu.

        Computed in log space as log pi_c + log N(mu; mu_c, sigma_c^2).

        With cfg.normalize_responsibilities the S3VDC stabilisation is applied:
        each row's negative log joint is min-max rescaled onto
        [0, -responsibility_scale] before the softmax, which bounds how peaked
        gamma can become and keeps a cluster from being starved by a single
        dominant log-density. Otherwise a plain softmax is used. See Section 4.4: Avoid NaN Losses of Cao (2020)

        The rescaling is per row. Taking one min and max over the whole batch
        would make every row's responsibilities depend on the other rows it
        happened to be batched with, so a single extreme row would flatten the
        softmax for all of them.

        Args:
            mu: (batch, z_dim) latent means.

        Returns:
            (batch, c) responsibilities, each row summing to 1.
        """
        log_pi = F.log_softmax(self.pi_logits, dim=0)
        log_p = log_pi + gaussian_log_prob(mu, self.mu_c, self.logvar_c)

        if not self.cfg.normalize_responsibilities:
            return F.softmax(log_p, dim=1)

        neg = -log_p
        lo = neg.min(dim=1, keepdim=True).values
        hi = neg.max(dim=1, keepdim=True).values
        scaled = (neg - lo) / (hi - lo + 1e-12) * -self.cfg.responsibility_scale # lambda in orginal implementation
        return F.softmax(scaled, dim=1)

    @torch.no_grad()
    def predict(self, x):
        """Hard cluster assignment argmax_c gamma_c.

        Args:
            x: (batch, i_dim) float tensor on the same device as the model.

        Returns:
            (batch,) long tensor of cluster indices.
        """
        mu, _ = self.encoder(x)
        return self.responsibilities(mu).argmax(dim=1)

    @torch.no_grad()
    def reconstruct(self, x):
        """Decode x with binary columns mapped through a sigmoid.

        forward() returns logits on binary columns because the loss uses
        BCEWithLogits; this returns probabilities instead, so the output is
        directly comparable to the input.

        Args:
            x: (batch, i_dim) float tensor.

        Returns:
            (batch, i_dim) reconstruction, binary columns in [0, 1].
        """
        recon, _, _, _ = self.forward(x, sample=False)
        out = recon.clone()
        out[:, self.binary_idx] = torch.sigmoid(recon[:, self.binary_idx])
        return out


def save_checkpoint(model, path, **extra):
    """Write weights plus everything needed to rebuild the model.

    Args:
        model: trained VaDE.
        path: destination .pt file; parent directories are created.
        **extra: anything else to persist alongside (band_rate, feature_names,
            train ids, seed). Returned verbatim by load_checkpoint.

    Returns:
        The path written.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "cfg": model.cfg.to_dict(),
            "binary_idx": model.binary_idx.cpu().tolist(),
            "num_idx": model.num_idx.cpu().tolist(),
            "pos_weight": None if model.pos_weight is None else model.pos_weight.cpu(),
            **extra,
        },
        path,
    )
    return path


def load_checkpoint(path, device="cpu"):
    """Rebuild a VaDE saved by save_checkpoint.

    Args:
        path: .pt file to load.
        device: device to place the model on.

    Returns:
        (model, payload) with the model in eval mode and payload the full dict,
        including any extras passed to save_checkpoint.
    """
    payload = torch.load(path, map_location=device, weights_only=False)

    model = VaDE(
        VaDEConfig.from_dict(payload["cfg"]),
        payload["binary_idx"],
        payload["num_idx"],
        pos_weight=payload["pos_weight"],
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()

    return model, payload


def configure_optimizer(model, cfg):
    """Build the Adam optimizer over all parameters.

    Args:
        model: VaDE instance.
        cfg: TrainConfig supplying learning_rate and weight_decay.

    Returns:
        torch.optim.Adam.
    """
    return torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )


def gaussian_log_prob(mu, mu_c, logvar_c):
    """Log density of each point under each diagonal Gaussian component.

    Computes log N(mu_i; mu_c_k, diag(exp(logvar_c_k))) for every pair (i, k).

    Args:
        mu: (batch, z_dim) points to evaluate.
        mu_c: (c, z_dim) component means.
        logvar_c: (c, z_dim) component log-variances.

    Returns:
        (batch, c) log densities.
    """
    diff = mu.unsqueeze(1) - mu_c.unsqueeze(0)
    return -0.5 * (
        LOG2PI + logvar_c.unsqueeze(0) + diff.pow(2) / logvar_c.exp().unsqueeze(0)
    ).sum(dim=2)


class Encoder(nn.Module):
    """Inference network q(z | x) = N(mu_phi(x), diag(sigma_phi(x)^2)).

    An MLP trunk over cfg.hidden_dims feeding two linear heads, one for mu and
    one for log sigma^2. The log-variance head is clamped to
    +/-cfg.logvar_clip so exp() cannot overflow on rows with little observed
    data.
    """

    def __init__(self, cfg):
        """Args: cfg: VaDEConfig supplying i_dim, hidden_dims, z_dim, dropout,
        activation and logvar_clip."""
        super().__init__()

        layers = []
        x = cfg.i_dim

        for h in cfg.hidden_dims:
            layers += mlp_block(x, h, cfg.dropout_rate, cfg.activation)
            x = h

        self.encoder = nn.Sequential(*layers)
        self.mu = nn.Linear(x, cfg.z_dim)
        self.logvar = nn.Linear(x, cfg.z_dim)
        self.logvar_clip = cfg.logvar_clip

    def forward(self, x):
        """Args: x: (batch, i_dim). Returns: (mu, logvar), each (batch, z_dim)."""
        h = self.encoder(x)
        return self.mu(h), self.logvar(h).clamp(-self.logvar_clip, self.logvar_clip)


class Decoder(nn.Module):
    """Generative network f(z) parameterising p(x | z).

    An MLP over cfg.decoder_hidden_dims ending in a linear layer of width
    i_dim. The output is deliberately unactivated: binary columns are logits
    consumed by BCEWithLogits, numeric columns are means for a unit-variance
    Gaussian. A single output activation cannot serve both.
    """

    def __init__(self, cfg):
        """Args: cfg: VaDEConfig supplying z_dim, decoder_hidden_dims, i_dim,
        dropout and activation."""
        super().__init__()

        layers = []
        x = cfg.z_dim

        for h in cfg.decoder_hidden_dims:
            layers += mlp_block(x, h, cfg.dropout_rate, cfg.activation)
            x = h

        layers.append(nn.Linear(x, cfg.i_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        """Args: z: (batch, z_dim). Returns: (batch, i_dim) raw output."""
        return self.decoder(z)


def mlp_block(in_dim, out_dim, dropout_rate, activation):
    """One Linear -> activation -> optional Dropout block.

    Args:
        in_dim, out_dim: layer widths.
        dropout_rate: dropout probability; the layer is omitted when 0.
        activation: key into ACTIVATIONS.

    Returns:
        A list of nn.Modules, ready to extend a layer list.
    """
    layers = [nn.Linear(in_dim, out_dim), ACTIVATIONS[activation]()]
    if dropout_rate > 0:
        layers.append(nn.Dropout(dropout_rate))
    return layers


ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}

LOG2PI = 1.8378770664093453
