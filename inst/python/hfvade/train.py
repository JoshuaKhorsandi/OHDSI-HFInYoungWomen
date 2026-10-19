"""Step-based training loop implementing the S3VDC schedule.

One loop covers all four phases: warm-up on a vanilla VAE objective, a
gradient-free window that initialises the mixture prior from KMeans and a
diagonal GMM, then repeating anneal/static cycles on the full ELBO.
"""

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from .losses import gmm_latent_loss, reconstruction_loss, vanilla_kl
from .metrics import cluster_metrics, reconstruction_metrics
from .schedule import Phase, S3VDCSchedule
from .stopping import EarlyStopping
from .vade import configure_optimizer


def train(model, loader, train_cfg, sched_cfg, device="cpu", eval_loader=None):
    """Train a VaDE for sched_cfg.max_steps or until early stopping fires.

    Args:
        model: VaDE instance; moved to `device` and modified in place.
        loader: DataLoader yielding (x,) or (x, mask); cycled indefinitely, so
            its length does not bound training.
        train_cfg: TrainConfig for optimizer, logging cadence and stopping.
        sched_cfg: ScheduleConfig defining the phase boundaries.
        device: "cpu" or "cuda".
        eval_loader: DataLoader for periodic evaluation; defaults to `loader`.
            Should not shuffle or drop the last batch.

    Returns:
        List of per-step dicts with step, phase, loss, recon, latent,
        latent_weight and (once the mixture prior exists) active_clusters.
        GMM_INIT steps are absent since they take no gradient step.

    Side effects:
        On exit the model holds the best weights when early stopping is enabled
        with restore_best. An exception or KeyboardInterrupt bypasses that.
    """
    model.to(device)
    schedule = S3VDCSchedule(sched_cfg)
    optimizer = configure_optimizer(model, train_cfg)
    batches = cycle(loader)
    eval_loader = eval_loader if eval_loader is not None else loader
    stopper = configure_stopping(train_cfg)
    z_buffer = []
    history = []
    metrics = {}

    for step in range(sched_cfg.max_steps):
        phase = schedule.phase(step)
        x, mask = next(batches)
        x = x.to(device)
        mask = torch.ones_like(x) if mask is None else mask.to(device)
        model.train()

        if phase is Phase.GMM_INIT:
            model.eval()
            with torch.no_grad():
                mu, _ = model.encoder(x)
            z_buffer.append(mu.cpu())
            if schedule.is_gmm_fit_step(step):
                z = torch.cat(z_buffer)
                init_gmm_params(model, z)
                z_buffer = []
                log_line(step, phase, {"gmm_samples": z.shape[0]})
            continue

        recon, mu, logvar, _ = model(x, sample=schedule.samples_z(step))
        rec = reconstruction_loss(model, recon, x, mask, train_cfg.w_numeric)

        if phase is Phase.GAMMA:
            latent = vanilla_kl(mu, logvar)
            gamma = None
        else:
            latent, gamma = gmm_latent_loss(model, mu, logvar)

        weight = schedule.latent_weight(step)
        loss = rec + weight * latent

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        stats = {
            "loss": loss.item(),
            "recon": rec.item(),
            "latent": latent.item(),
            "latent_weight": weight,
        }
        if gamma is not None:
            stats["active_clusters"] = int(
                gamma.argmax(dim=1).unique().numel()
            )
        history.append({"step": step, "phase": phase.value, **stats})

        if step % train_cfg.log_every == 0:
            log_line(step, phase, stats)

        is_last = step == sched_cfg.max_steps - 1
        cycle_end = stopper is not None and schedule.is_cycle_end(step)

        if step % train_cfg.eval_every == 0 or is_last or cycle_end:
            metrics, _, _ = evaluate(model, eval_loader, device)
            log_line(step, phase, metrics)

        if cycle_end and stopper.update(metrics, model, step):
            log_line(step, phase, {
                "early_stop": 1,
                f"best_{stopper.metric}": stopper.best,
                "best_step": stopper.best_step,
            })
            break

    if stopper is not None and train_cfg.restore_best and stopper.best_state is not None:
        stopper.restore(model)
        model.to(device)

    return history


def configure_stopping(cfg):
    """Build an EarlyStopping from a TrainConfig, or None if disabled.

    Args:
        cfg: TrainConfig; early_stopping_metric of None disables stopping.

    Returns:
        EarlyStopping or None.
    """
    if cfg.early_stopping_metric is None:
        return None

    return EarlyStopping(
        cfg.early_stopping_metric,
        patience=cfg.early_stopping_patience,
        min_delta=cfg.early_stopping_min_delta,
        restore_best=cfg.restore_best,
    )


@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    """Encode a full loader and score reconstruction and cluster quality.

    Encoding is deterministic (z = mu) and the model is left in eval mode.

    Args:
        model: VaDE instance.
        loader: DataLoader yielding (x,) or (x, mask).
        device: device to run on.

    Returns:
        (metrics, z, labels) where metrics is a flat dict of floats combining
        reconstruction_metrics and cluster_metrics, z is an (n, z_dim) CPU
        tensor of latent means, and labels is an (n,) CPU tensor of hard
        assignments. z and labels stay on CPU because their consumers are
        sklearn and the verification metrics.
    """
    model.eval()
    latents, assignments, recon_stats = [], [], []

    for batch in loader:
        x, mask = unpack(batch)
        x = x.to(device)
        mask = torch.ones_like(x) if mask is None else mask.to(device)

        mu, _ = model.encoder(x)
        latents.append(mu.cpu())
        assignments.append(model.responsibilities(mu).argmax(dim=1).cpu())
        recon_stats.append(reconstruction_metrics(model.reconstruct(x), x, mask))

    z = torch.cat(latents)
    labels = torch.cat(assignments)

    metrics = {k: float(np.mean([s[k] for s in recon_stats])) for k in recon_stats[0]}
    metrics.update(cluster_metrics(z, labels, model.mu_c.detach().cpu()))

    return metrics, z, labels


def init_gmm_params(model, z):
    """Initialise the mixture prior from accumulated encoder outputs.

    Follows S3VDC: KMeans seeds a diagonal GMM, cluster means are taken from
    KMeans, and variances from the fitted GMM. Unlike S3VDC, pi is also
    initialised, from the GMM's mixture weights. The n_init=10 restarts match
    the sklearn default of the version S3VDC was written against.

    Args:
        model: VaDE whose mu_c, logvar_c and pi_logits are overwritten in place.
        z: (n, z_dim) CPU tensor of latent means, n >= model.cfg.c.

    Side effects:
        Mutates the model's mixture parameters under no_grad.
    """
    z = z.numpy()
    c = model.cfg.c

    kmeans = KMeans(n_clusters=c, n_init=10, random_state=0).fit(z)
    gmm = GaussianMixture(
        n_components=c,
        covariance_type="diag",
        max_iter=10000,
        means_init=kmeans.cluster_centers_,
        random_state=0,
    ).fit(z)

    with torch.no_grad():
        model.mu_c.copy_(torch.from_numpy(kmeans.cluster_centers_).float())
        model.logvar_c.copy_(torch.from_numpy(np.log(gmm.covariances_ + 1e-10)).float())
        model.pi_logits.copy_(torch.from_numpy(np.log(gmm.weights_ + 1e-10)).float())


def cycle(loader):
    """Yield (x, mask) batches from a loader forever, restarting at each epoch.

    Args:
        loader: any iterable of batches.

    Yields:
        (x, mask) tuples, mask None when the loader carries no mask.
    """
    while True:
        for batch in loader:
            yield unpack(batch)


def unpack(batch):
    """Normalise a batch into an (x, mask) pair.

    Args:
        batch: a tensor, a 1-tuple, or a 2-tuple from a DataLoader.

    Returns:
        (x, mask), with mask None when the batch carries only features.
    """
    if not isinstance(batch, (list, tuple)):
        return batch, None
    if len(batch) == 1:
        return batch[0], None
    return batch[0], batch[1]


def log_line(step, phase, stats):
    """Print one aligned progress line to stdout.

    Args:
        step: global step index.
        phase: Phase member.
        stats: dict of scalar values to render.
    """
    body = "  ".join(f"{k}={fmt(v)}" for k, v in stats.items())
    print(f"step {step:>7}  {phase.value:<8}  {body}")


def fmt(value):
    """Format a stat for the progress line.

    Args:
        value: any scalar.

    Returns:
        String; floats are shown to 4 decimals, non-floats via str.
    """
    return f"{value:.4f}" if isinstance(value, float) else str(value)
