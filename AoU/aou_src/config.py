"""Configuration dataclasses.

Split by concern: VaDEConfig determines what weights exist, TrainConfig how they
are optimised, ScheduleConfig when each phase runs, DataConfig which matrix was
used. Keeping them separate means a checkpoint's architecture is described by
VaDEConfig alone.

All four share ConfigMixin, so a single flat dict -- a wandb sweep's config, a
YAML file -- can populate every one of them, each taking only the keys it owns.
"""

from dataclasses import dataclass, field, asdict, fields


class ConfigMixin:
    """Dict conversion for configs that must survive a round trip through wandb."""

    @classmethod
    def from_dict(cls, d):
        """Build from a dict, ignoring keys this config does not define.

        Args:
            d: any mapping, typically dict(wandb.config), which mixes fields
                from several configs plus unrelated run settings.

        Returns:
            An instance populated from the matching keys.
        """
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def to_dict(self):
        """Return the config as a plain JSON-serialisable dict."""
        return asdict(self)


@dataclass
class VaDEConfig(ConfigMixin):
    """Architecture and inference settings; determines the shape of the weights.

    Attributes:
        i_dim: number of input features.
        z_dim: latent dimensionality.
        c: number of mixture components (clusters).
        hidden_dims: encoder layer widths, outermost first.
        decoder_hidden_dims: decoder layer widths; defaults to hidden_dims
            reversed.
        dropout_rate: dropout after each hidden layer; 0 omits the layers.
        activation: key into vade.ACTIVATIONS.
        logvar_clip: symmetric clamp on the encoder's log-variance output,
            preventing exp() overflow on rows with little observed data.
        normalize_responsibilities: apply the S3VDC min-max rescaling to
            p(c|z) before the softmax instead of a plain softmax.
        responsibility_scale: width of that rescaled range; smaller values give
            softer, less peaked assignments.
    """

    i_dim: int
    z_dim: int = 20
    c: int = 10
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64, 32])
    decoder_hidden_dims: list[int] | None = None
    dropout_rate: float = 0.2
    activation: str = "relu"
    logvar_clip: float = 10.0
    normalize_responsibilities: bool = True
    responsibility_scale: float = 20.0

    def __post_init__(self):
        """Mirror the encoder widths onto the decoder when none were given."""
        if self.decoder_hidden_dims is None:
            self.decoder_hidden_dims = list(reversed(self.hidden_dims))


@dataclass
class DataConfig(ConfigMixin):
    """Identity and shape of the prepared matrix a run was trained on.

    Drives wandb's run grouping, and records enough about the matrix that two
    runs on the "same" dataset can be told apart when preprocessing changed.

    Attributes:
        dataset: short name, e.g. "heartfailure".
        version: bump whenever preprocessing changes, so old runs stay
            comparable among themselves but visibly separate from new ones.
        path: where the prepared matrix was read from.
        n_rows, n_features, n_binary, n_numeric: matrix shape and type split.
        observed_fraction: mean of the mask, i.e. how complete the data is.
    """

    dataset: str
    version: str = "v1"
    path: str | None = None
    n_rows: int | None = None
    n_features: int | None = None
    n_binary: int | None = None
    n_numeric: int | None = None
    observed_fraction: float | None = None

    @classmethod
    def describe(cls, dataset, x, mask, binary_idx, num_idx, version="v1", path=None):
        """Build a DataConfig by measuring the prepared arrays.

        Args:
            dataset: short dataset name.
            x: (n, i_dim) prepared feature matrix.
            mask: (n, i_dim) observation mask, or None.
            binary_idx, num_idx: column index sequences.
            version: preprocessing version label.
            path: source path to record.

        Returns:
            A populated DataConfig.
        """
        return cls(
            dataset=dataset,
            version=version,
            path=path,
            n_rows=int(x.shape[0]),
            n_features=int(x.shape[1]),
            n_binary=len(binary_idx),
            n_numeric=len(num_idx),
            observed_fraction=None if mask is None else round(float(mask.mean()), 4),
        )


@dataclass
class TrainConfig(ConfigMixin):
    """Optimisation, logging cadence and early stopping.

    Attributes:
        learning_rate, weight_decay: Adam settings.
        w_numeric: weight of the numeric reconstruction block against the
            binary one. Interacts with pos_weight, which also shifts the
            balance; tune the two together.
        log_every, eval_every, hist_every, embedding_every: step intervals for
            stdout lines, evaluation passes, histograms and latent embeddings.
            Evaluation runs the whole eval_loader, so eval_every is the
            expensive one.
        log_dir: TensorBoard root directory.
        early_stopping_metric: metric name from stopping.DIRECTIONS, or None to
            disable. Checked only at cycle boundaries.
        early_stopping_patience: non-improving cycles tolerated.
        early_stopping_min_delta: improvement required to reset patience.
        restore_best: reload the best weights when training ends.
    """

    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    w_numeric: float = 1.0
    log_every: int = 100
    eval_every: int = 500
    hist_every: int = 500
    embedding_every: int = 2000
    log_dir: str = "./tb"
    early_stopping_metric: str | None = None
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.0
    restore_best: bool = True


@dataclass
class ScheduleConfig(ConfigMixin):
    """Phase boundaries for the S3VDC schedule, all counted in steps.

    Attributes:
        gamma: constant weight on the vanilla KL during warm-up, and the floor
            added to the annealing weight afterwards.
        gamma_steps: length of the warm-up phase.
        gmm_steps: batches of encoder output accumulated before fitting the
            mixture prior. gmm_steps * batch_size must be at least c.
        beta_steps: length of each annealing ramp.
        static_steps: length of each full-weight phase after a ramp.
        max_steps: total steps. Ideally warmup plus a whole number of
            beta+static cycles, so early stopping's last checkpoint is not
            followed by untracked training.
    """

    gamma: float = 1e-3
    gamma_steps: int = 2000
    gmm_steps: int = 20
    beta_steps: int = 1000
    static_steps: int = 1000
    max_steps: int = 20000
