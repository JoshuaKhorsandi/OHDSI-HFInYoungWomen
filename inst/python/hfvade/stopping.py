"""Early stopping on cluster-quality metrics.

PyTorch has no built-in early stopping. This is the usual patience-and-restore
pattern, with one adjustment for the S3VDC schedule: the monitored value is only
comparable at cycle boundaries, where the latent weight is always 1.0. Sampling
mid-cycle would see the loss rise as the annealing weight ramps in and stop on a
schedule artefact rather than on a real plateau.
"""


class EarlyStopping:
    """Tracks the best value of a monitored metric and signals when to stop.

    Attributes:
        metric: the key watched in the metrics dict.
        direction: "max" or "min", looked up from DIRECTIONS.
        best: best value seen so far.
        best_step: step at which best was recorded.
        best_state: CPU copy of the state dict at best_step, or None.
        waited: consecutive non-improving checks since best.
    """

    def __init__(self, metric, patience=3, min_delta=0.0, restore_best=True):
        """Args:
            metric: key into the metrics dict; must appear in DIRECTIONS.
            patience: number of consecutive non-improving checks tolerated.
                Counted in cycles, not steps.
            min_delta: improvement over `best` required to reset the counter.
            restore_best: keep a weight snapshot at each improvement.

        Raises:
            ValueError: if metric is not a known metric name.
        """
        if metric not in DIRECTIONS:
            raise ValueError(
                f"unknown metric {metric!r}; expected one of {sorted(DIRECTIONS)}"
            )

        self.metric = metric
        self.direction = DIRECTIONS[metric]
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best

        self.best = float("-inf") if self.direction == "max" else float("inf")
        self.best_step = None
        self.best_state = None
        self.waited = 0

    def update(self, metrics, model, step):
        """Record one measurement and report whether training should stop.

        Args:
            metrics: dict from train.evaluate, containing self.metric.
            model: the model, snapshotted on improvement.
            step: global step index, stored as best_step on improvement.

        Returns:
            True when patience is exhausted, False otherwise.
        """
        value = metrics[self.metric]

        if self.improved(value):
            self.best = value
            self.best_step = step
            self.waited = 0
            if self.restore_best:
                self.best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            return False

        self.waited += 1
        return self.waited >= self.patience

    def improved(self, value):
        """Whether value beats the running best by at least min_delta.

        Args:
            value: the new measurement.

        Returns:
            Bool, respecting self.direction.
        """
        if self.direction == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def restore(self, model):
        """Load the best snapshot back into the model, in place.

        The snapshot lives on CPU, so callers should re-issue model.to(device)
        afterwards. No-op when no snapshot was taken.

        Args:
            model: the model to overwrite.
        """
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


DIRECTIONS = {
    "silhouette": "max",
    "calinski_harabasz": "max",
    "separation": "max",
    "assignment_entropy": "max",
    "active_clusters": "max",
    "mae": "min",
    "mse": "min",
    "cosine_distance": "min",
}
