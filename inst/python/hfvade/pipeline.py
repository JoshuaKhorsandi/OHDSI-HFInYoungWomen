import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .config import ScheduleConfig, TrainConfig, VaDEConfig
from .data import make_dataloader, preprocess_data
from .enrichment import cluster_enrichment
from .inference import predict_clusters
from .metrics import cluster_metrics
from .train import evaluate, train
from .vade import VaDE, save_checkpoint
from .verification import verify_clustering


def run_vade(features_path, metadata_path, output_folder, settings_path):
    settings = load_settings(settings_path)
    seed = settings["seed"]
    batch_size = settings["batch_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(output_folder)
    out.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)
    P = preprocess_data(features_path, metadata_path, dataset=settings["dataset"], seed=seed)
    vade_cfg, train_cfg, sched_cfg = build_configs(settings, P["X_in"].shape[1])
    check_schedule(vade_cfg, train_cfg, sched_cfg, batch_size)

    train_loader = make_dataloader(P["X_train"], P["M_train"], batch_size=batch_size,
                                   shuffle=True, drop_last=True)
    eval_loader = make_dataloader(P["X_test"], P["M_test"], batch_size=batch_size,
                                  shuffle=False, drop_last=False)

    seed_everything(seed)
    model = VaDE(vade_cfg, P["binary_idx"], P["num_idx"], pos_weight=P["pw"])
    history = pd.DataFrame(train(model, train_loader, train_cfg, sched_cfg, device=device,
                                 eval_loader=eval_loader))

    holdout, _, _ = evaluate(model, eval_loader, device=device)
    assignments = predict_clusters(model, P["X_in"], index=P["index"], device=device)
    assignments["split"] = np.where(P["is_train"], "train", "test")
    z_all = assignments[[c for c in assignments.columns if c.startswith("z_")]].to_numpy(copy=True)
    labels_all = assignments["cluster"].to_numpy(copy=True)

    metrics = {k: holdout[k] for k in ("mae", "mse", "cosine_distance")}
    metrics.update(cluster_metrics(torch.from_numpy(z_all),
                                   torch.from_numpy(labels_all),
                                   model.mu_c.detach().cpu()))

    report = {
        "device": str(device),
        "steps_run": len(history),
        "data": P["data_cfg"].to_dict(),
        "metrics": metrics,
        "metrics_holdout": holdout,
        "cluster_sizes": {int(k): int(v) for k, v in
                          assignments["cluster"].value_counts().sort_index().items()},
        "config": {**vade_cfg.to_dict(), **train_cfg.to_dict(),
                   **sched_cfg.to_dict(), "batch_size": batch_size, "seed": seed},
    }

    save_checkpoint(model, out / "model.pt",
                    band_rate=P["band_rate"], feature_names=list(P["feats"].columns),
                    is_train=P["is_train"], seed=seed)
    assignments.to_parquet(out / "assignments.parquet")
    history.to_csv(out / "training_history.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(report, indent=2, default=float))

    fig = plot_training_curves(history, sched_cfg, vade_cfg.c)
    fig.savefig(out / "training_curves.png", dpi=150)
    plt.close(fig)

    verification, feat_eta = verify_clustering(z_all, labels_all, P["feats"], vade_cfg.c, seed)
    (out / "verification.json").write_text(json.dumps(verification, indent=2))
    feat_eta.to_csv(out / "feature_eta2.tsv", sep="\t")

    enrichment = cluster_enrichment(assignments, P["feats"])
    enrichment = enrichment.sort_values(by=["cluster", "Effect"], ascending=False)
    enrichment.to_csv(out / "cluster_enrichment.tsv", sep="\t", index=False, na_rep="")

    return {"metrics": metrics, "cluster_sizes": report["cluster_sizes"],
            "verification": verification, "steps_run": len(history)}


def load_settings(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_configs(settings, i_dim):
    vade_cfg = VaDEConfig.from_dict({**settings, "i_dim": i_dim})
    train_cfg = TrainConfig.from_dict(settings)
    sched_cfg = ScheduleConfig.from_dict(settings)
    return vade_cfg, train_cfg, sched_cfg


def check_schedule(vade_cfg, train_cfg, sched_cfg, batch_size):
    warmup, cycle_len, n_cycles = schedule_shape(sched_cfg)
    if sched_cfg.gmm_steps * batch_size < vade_cfg.c:
        raise ValueError(f"gmm_steps * batch_size ({sched_cfg.gmm_steps * batch_size}) "
                         f"must be at least c ({vade_cfg.c})")
    if train_cfg.early_stopping_metric is not None and n_cycles <= train_cfg.early_stopping_patience:
        raise ValueError(f"{n_cycles} annealing cycles after step {warmup} is not more than "
                         f"early_stopping_patience ({train_cfg.early_stopping_patience})")


def plot_training_curves(hist, sched_cfg, c):
    warmup, cycle_len, n_cycles = schedule_shape(sched_cfg)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(hist["step"], hist["loss"], lw=0.6)
    axes[0].set(xlabel="step", ylabel="loss", title="total", yscale="log")

    axes[1].plot(hist["step"], hist["recon"], lw=0.6, color="tab:orange")
    axes[1].set(xlabel="step", ylabel="reconstruction", title="reconstruction")

    axes[2].plot(hist["step"], hist["latent"], lw=0.6, color="tab:green")
    axes[2].set(xlabel="step", ylabel="latent", title="latent and active clusters")
    if "active_clusters" in hist:
        twin = axes[2].twinx()
        twin.plot(hist["step"], hist["active_clusters"], lw=1.2, color="tab:red")
        twin.set(ylabel="active clusters", ylim=(0, c + 0.5))

    for ax in axes:
        ax.axvline(warmup, color="k", ls="--", lw=0.8)
        for k in range(1, n_cycles + 1):
            ax.axvline(warmup + k * cycle_len, color="grey", ls=":", lw=0.6)

    fig.tight_layout()
    return fig


def schedule_shape(sched_cfg):
    warmup = sched_cfg.gamma_steps + sched_cfg.gmm_steps
    cycle_len = sched_cfg.beta_steps + sched_cfg.static_steps
    return warmup, cycle_len, (sched_cfg.max_steps - warmup) // cycle_len
