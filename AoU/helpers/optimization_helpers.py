# Original trial sampling, fitting, scoring, and logging functions.

def objective(trial):
    """One trial: sample a configuration, train it, score both objectives."""
    params = suggest_params(trial)
    model, steps = fit(params)

    metrics, _, labels = evaluate(model, FULL_LOADER, device=device)
    val_loss = clustpy_vade_loss(model, TEST_LOADER, device=device)
    quality = cluster_quality(metrics, params["c"])
    sizes = np.bincount(labels.numpy(), minlength=params["c"]).tolist()

    log_trial(trial, params, model,
              {**metrics, "val_loss": val_loss, "quality": quality,
               "steps_run": steps, "cluster_sizes": str(sizes)})

    print(f"trial {trial.number:>3}  c={params['c']} z={params['z_dim']:>2} "
          f"lr={params['learning_rate']:.1e} -> val_loss={val_loss:9.2f} "
          f"quality={quality:+.4f} sil={metrics['silhouette']:+.4f} sizes={sizes}")

    return val_loss, quality


def suggest_params(trial):
    """Sample one configuration from the search space."""
    return {
        **SCHEDULE,
        "c": FIXED_C,
        "z_dim": trial.suggest_categorical("z_dim", [8, 10, 16, 20, 30]),
        "hidden_dims": HIDDEN_DIMS[trial.suggest_categorical("hidden_dims", list(HIDDEN_DIMS))],
        "batch_size": trial.suggest_categorical("batch_size", [128, 256]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 1e-6, 1e-4, 1e-3]),
        "w_numeric": trial.suggest_float("w_numeric", 0.2, 2.0),
        "dropout_rate": trial.suggest_categorical("dropout_rate", [0.0, 0.1, 0.2]),
        "responsibility_scale": trial.suggest_categorical("responsibility_scale", [20.0, 40.0, 50.0]),
    }


def fit(params):
    """Build and train one VaDE from a sampled configuration."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = VaDEConfig.from_dict({**params, "i_dim": I_DIM})
    train_cfg = TrainConfig.from_dict({**params, "log_every": 10**9, "eval_every": 1000,
                                       "hist_every": 10**9, "embedding_every": 10**9})
    sched_cfg = ScheduleConfig.from_dict(params)

    model = VaDE(cfg, P["binary_idx"], P["num_idx"], pos_weight=P["pw"])
    history = train(model,
                    make_dataloader(P["X_train"], P["M_train"], batch_size=params["batch_size"]),
                    train_cfg, sched_cfg, device=device, eval_loader=TEST_LOADER)

    return model, history[-1]["step"] + 1

def log_trial(trial, params, model, record):
    """Attach a finished trial's results to the study (no wandb)."""
    for key, value in record.items():
        trial.set_user_attr(key, value)

    if SAVE_CHECKPOINTS:
        save_checkpoint(model, f"../checkpoints/study/{STUDY_NAME}-trial{trial.number}.pt",
                        band_rate=P["band_rate"],
                        feature_names=list(P["feats"].columns),
                        study=STUDY_NAME, trial=trial.number)