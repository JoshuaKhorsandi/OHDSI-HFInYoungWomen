import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch

from .config import TrainConfig
from .data import make_dataloader, preprocess_data
from .pipeline import build_configs, load_settings, seed_everything
from .scoring import cluster_quality, clustpy_vade_loss
from .train import evaluate, train
from .vade import VaDE


def run_optimization(features_path, metadata_path, output_folder, settings_path):
    settings = load_settings(settings_path)
    options = settings["optimization"]
    seed = settings["seed"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(output_folder)
    out.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)
    P = preprocess_data(features_path, metadata_path, dataset=settings["dataset"], seed=seed)
    data = {
        "P": P,
        "device": device,
        "test_loader": make_dataloader(P["X_test"], P["M_test"], batch_size=256,
                                       shuffle=False, drop_last=False),
        "full_loader": make_dataloader(P["X_in"], P["M"], batch_size=256,
                                       shuffle=False, drop_last=False),
    }

    study = optuna.create_study(
        study_name=options["study_name"],
        storage=f"sqlite:///{(out / 'study.db').resolve().as_posix()}",
        directions=["minimize", "maximize"],
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
        load_if_exists=True,
    )
    study.optimize(lambda trial: objective(trial, settings, data), n_trials=options["n_trials"])

    results = trial_results(study)
    results.to_csv(out / "sweep_results.csv", index=False)
    pareto = pareto_front(study)
    pareto.to_csv(out / "pareto_front.csv", index=False)

    selected = select_trial(study.best_trials, options["selection"])
    optimized = {**settings, **params_from_trial(selected),
                 "optimization": {**options, "selected_trial": selected.number}}
    path = out / "optimizedSettings.json"
    path.write_text(json.dumps(optimized, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"selected trial {selected.number} ({options['selection']}): "
          f"val_loss={selected.values[0]:.2f} quality={selected.values[1]:+.4f}")
    return {"settings_path": str(path), "selected_trial": selected.number,
            "val_loss": selected.values[0], "quality": selected.values[1],
            "n_trials": len(results), "pareto_trials": pareto["trial"].tolist()}


def objective(trial, settings, data):
    """One trial: sample a configuration, train it, score both objectives.

    Args:
        trial: the Optuna trial.
        settings: the analysis settings the search space is merged onto.
        data: prepared matrix, held-out and full loaders, and device.

    Returns:
        (held-out VaDE loss, cluster quality). Optuna minimises the first and
        maximises the second.
    """
    params = suggest_params(trial, settings)
    model, steps = fit(params, data)

    metrics, _, labels = evaluate(model, data["full_loader"], device=data["device"])
    val_loss = clustpy_vade_loss(model, data["test_loader"], device=data["device"])
    quality = cluster_quality(metrics, params["c"])
    sizes = np.bincount(labels.numpy(), minlength=params["c"]).tolist()

    record = {**metrics, "val_loss": val_loss, "quality": quality,
              "steps_run": steps, "cluster_sizes": str(sizes)}
    for key, value in record.items():
        trial.set_user_attr(key, value)

    print(f"trial {trial.number:>3}  c={params['c']} z={params['z_dim']:>2} "
          f"lr={params['learning_rate']:.1e} -> val_loss={val_loss:9.2f} "
          f"quality={quality:+.4f} sil={metrics['silhouette']:+.4f} sizes={sizes}")

    return val_loss, quality


def suggest_params(trial, settings):
    """Sample one configuration from the search space.

    Args:
        trial: the Optuna trial.
        settings: analysis settings supplying the fixed schedule and seed.

    Returns:
        Flat dict of hyperparameters, with hidden_dims resolved to a list of
        widths and the fixed settings merged in, ready for the from_dict
        constructors on VaDEConfig, TrainConfig and ScheduleConfig.
    """
    return {
        **settings,
        "c": trial.suggest_int("c", 4, 6),
        "z_dim": trial.suggest_categorical("z_dim", [8, 10, 16, 20, 30]),
        "hidden_dims": HIDDEN_DIMS[trial.suggest_categorical("hidden_dims", list(HIDDEN_DIMS))],
        "batch_size": trial.suggest_categorical("batch_size", [128, 256]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 1e-6, 1e-4, 1e-3]),
        "w_numeric": trial.suggest_float("w_numeric", 0.2, 2.0),
        "dropout_rate": trial.suggest_categorical("dropout_rate", [0.0, 0.1, 0.2]),
        "responsibility_scale": trial.suggest_categorical("responsibility_scale", [20.0, 40.0, 50.0]),
    }


def fit(params, data):
    """Build and train one VaDE from a sampled configuration.

    Args:
        params: dict from suggest_params.
        data: prepared matrix, held-out loader and device.

    Returns:
        (model, steps_run) with the model on the device, holding the best
        weights early stopping saw.
    """
    seed_everything(params["seed"])
    P = data["P"]

    vade_cfg, _, sched_cfg = build_configs(params, P["X_in"].shape[1])
    train_cfg = TrainConfig.from_dict({**params, "log_every": 10**9, "eval_every": 1000})

    model = VaDE(vade_cfg, P["binary_idx"], P["num_idx"], pos_weight=P["pw"])
    history = train(model,
                    make_dataloader(P["X_train"], P["M_train"], batch_size=params["batch_size"]),
                    train_cfg, sched_cfg, device=data["device"], eval_loader=data["test_loader"])

    return model, history[-1]["step"] + 1


def trial_results(study):
    results = study.trials_dataframe(attrs=("number", "values", "params", "user_attrs", "state"))
    results = results[results["state"] == "COMPLETE"].sort_values("values_0").reset_index(drop=True)
    return results.rename(columns={"values_0": "val_loss", "values_1": "quality"})


def pareto_front(study):
    return pd.DataFrame([
        {"trial": t.number, "val_loss": t.values[0], "quality": t.values[1],
         **t.params, "cluster_sizes": t.user_attrs.get("cluster_sizes"),
         "silhouette": t.user_attrs.get("silhouette"),
         "active_clusters": t.user_attrs.get("active_clusters")}
        for t in study.best_trials
    ]).sort_values("val_loss").reset_index(drop=True)


def select_trial(trials, selection):
    if selection == "val_loss":
        return min(trials, key=lambda t: t.values[0])
    if selection == "quality":
        return max(trials, key=lambda t: t.values[1])
    if selection == "knee":
        loss = np.array([t.values[0] for t in trials])
        quality = np.array([t.values[1] for t in trials])
        distance = np.hypot(min_max(loss), 1.0 - min_max(quality))
        return trials[int(distance.argmin())]
    raise ValueError(f"selection must be 'knee', 'quality' or 'val_loss', got {selection!r}")


def params_from_trial(trial):
    return {**trial.params, "hidden_dims": HIDDEN_DIMS[trial.params["hidden_dims"]]}


def min_max(values):
    span = values.max() - values.min()
    return (values - values.min()) / span if span > 0 else np.zeros_like(values)


HIDDEN_DIMS = {
    "256x128x64x32": [256, 128, 64, 32],
    "256x128x64": [256, 128, 64],
    "128x64x32": [128, 64, 32],
}
