# Original trial export and checkpoint functions.

def export_trial(trial_number, out_dir=EXPORT_DIR):
    """Write model, metrics, assignments and enrichment for one trial.

    Args:
        trial_number: number of the trial to export.
        out_dir: parent directory; a subfolder named for the trial is created.

    Returns:
        Path to the trial's output folder.
    """
    trial = study.trials[trial_number]
    params = params_from_trial(trial)
    model, source = load_trial_model(trial, params)

    folder = Path(out_dir) / f"trial-{trial_number}"
    folder.mkdir(parents=True, exist_ok=True)

    metrics, _, _ = evaluate(model, FULL_LOADER, device=device)
    metrics["val_loss"] = clustpy_vade_loss(model, TEST_LOADER, device=device)
    metrics["quality"] = cluster_quality(metrics, params["c"])

    assignments = predict_clusters(model, P["X_in"], index=P["index"], device=device)
    assignments["split"] = np.where(P["is_train"], "train", "test")
    assignments.to_parquet(folder / "assignments.parquet")

    enrichment = cluster_enrichment(assignments, P["feats"])
    enrichment.to_csv(folder / "cluster_enrichment.tsv", sep="\t", index=False, na_rep="")

    save_checkpoint(model, folder / "model.pt",
                    band_rate=P["band_rate"], feature_names=list(P["feats"].columns),
                    study=STUDY_NAME, trial=trial_number, source=source, seed=seed)

    sizes = assignments["cluster"].value_counts().sort_index().to_dict()
    (folder / "metrics.json").write_text(json.dumps(
        {"study": STUDY_NAME, "trial": trial_number, "weights_source": source,
         "metrics": metrics, "cluster_sizes": {int(k): int(v) for k, v in sizes.items()},
         "params": params}, indent=2, default=str))

    print(f"trial-{trial_number}: {source}  clusters={sizes}  "
          f"val_loss={metrics['val_loss']:.2f}  silhouette={metrics['silhouette']:+.4f}")
    print(f"  -> {folder}/  " + ", ".join(sorted(f.name for f in folder.iterdir())))
    return folder


def params_from_trial(trial):
    """Rebuild a fit() parameter dict from a stored trial.

    Args:
        trial: an Optuna trial.

    Returns:
        dict matching what suggest_params returns, with hidden_dims resolved
        back to a list of widths and the fixed schedule merged in.
    """
    return {**SCHEDULE, "c": FIXED_C, **trial.params,
            "hidden_dims": HIDDEN_DIMS[trial.params["hidden_dims"]]}

def checkpoint_path(trial):
    """Locate a saved checkpoint for a trial, if SAVE_CHECKPOINTS was on when it ran.

    Args:
        trial: an Optuna trial.

    Returns:
        Path to the checkpoint if one exists on disk, else None.
    """
    candidate = EXPORT_DIR / f"trial-{trial.number}" / "model.pt"
    return candidate if candidate.exists() else None


def load_trial_model(trial, params):
    """Get a trained model for a trial, from its checkpoint or by retraining.

    Args:
        trial: an Optuna trial.
        params: its resolved parameters, from params_from_trial.

    Returns:
        (model, source) where source is "checkpoint" or "retrained".

    Raises:
        ValueError: if a checkpoint was trained on a different feature count
            than the prepared matrix holds.
    """
    path = checkpoint_path(trial)
    if path is None:
        model, _ = fit(params)
        return model, "retrained"

    model, payload = load_checkpoint(path, device=device)
    if len(payload.get("feature_names", [])) != I_DIM:
        raise ValueError(
            f"Checkpoint at {path} was trained on {len(payload.get('feature_names', []))} "
            f"features, but the current AoU matrix has {I_DIM}. Refusing to load a "
            "mismatched checkpoint -- retrain instead."
        )
    return model, "checkpoint"
