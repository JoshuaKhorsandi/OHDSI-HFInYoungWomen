"""S3VDC training schedule (Cao et al. 2020).

Replaces VaDE's separate autoencoder pretraining with a four-phase, step-indexed
schedule that runs inside a single training loop.
"""

from enum import Enum


class Phase(Enum):
    """Training phases, in the order the schedule visits them.

    GAMMA: warm-up. Reconstruction plus a small constant weight (cfg.gamma) on
        the vanilla VAE KL. z is taken deterministically as mu.
    GMM_INIT: no gradient steps. Encoder outputs are accumulated so KMeans and a
        diagonal GMM can initialise the mixture prior.
    BETA: annealing. The mixture latent term is ramped in as (t/beta_steps)^3.
    STATIC: the full objective at weight 1.0.

    BETA and STATIC repeat cyclically until max_steps, so the latent term is
    periodically re-annealed rather than held fixed.
    """

    GAMMA = "gamma"
    GMM_INIT = "gmm_init"
    BETA = "beta"
    STATIC = "static"


class S3VDCSchedule:
    """Maps a global step index to the phase and loss weight for that step.

    Stateless: every method is a pure function of `step`, so the schedule stays
    correct even though GMM_INIT steps take no optimizer step.

    Timeline for warmup = gamma_steps + gmm_steps and cycle = beta_steps +
    static_steps:

        [0, gamma_steps)            GAMMA
        [gamma_steps, warmup)       GMM_INIT, fitted on the final step
        [warmup, max_steps)         BETA then STATIC, repeating every cycle

    Attributes:
        cfg: ScheduleConfig.
        warmup_steps: first step at which cyclic training begins.
        cycle_steps: length of one BETA + STATIC cycle.
    """

    def __init__(self, cfg):
        """Args: cfg: ScheduleConfig."""
        self.cfg = cfg
        self.warmup_steps = cfg.gamma_steps + cfg.gmm_steps
        self.cycle_steps = cfg.beta_steps + cfg.static_steps

    def phase(self, step):
        """Which Phase the given global step belongs to.

        Args:
            step: global step index, 0-based.

        Returns:
            A Phase member.
        """
        if step < self.cfg.gamma_steps:
            return Phase.GAMMA
        if step < self.warmup_steps:
            return Phase.GMM_INIT
        if self.cycle_steps == 0:
            return Phase.STATIC
        if self.cycle_step(step) < self.cfg.beta_steps:
            return Phase.BETA
        return Phase.STATIC

    def latent_weight(self, step):
        """Weight applied to the latent loss at this step.

        cfg.gamma during GAMMA, 0 during GMM_INIT, (t/beta_steps)^3 + gamma
        while annealing, and 1.0 during STATIC. The cubic ramp keeps the
        mixture term near zero early in each cycle, letting reconstruction
        recover before the clustering pressure returns.

        Args:
            step: global step index.

        Returns:
            Float weight.
        """
        phase = self.phase(step)
        if phase is Phase.GAMMA:
            return self.cfg.gamma
        if phase is Phase.GMM_INIT:
            return 0.0
        if phase is Phase.STATIC:
            return 1.0
        return (self.cycle_step(step) / self.cfg.beta_steps) ** 3 + self.cfg.gamma

    def is_gmm_fit_step(self, step):
        """True on the single step where the accumulated latents are fitted.

        Args:
            step: global step index.

        Returns:
            True on the last GMM_INIT step, False everywhere else.
        """
        return self.cfg.gmm_steps > 0 and step == self.warmup_steps - 1

    def is_cycle_end(self, step):
        """True on the final step of each BETA + STATIC cycle.

        These are the only steps at which model states are directly comparable,
        since latent_weight is 1.0 at each of them. Early stopping samples here
        for that reason.

        Args:
            step: global step index.

        Returns:
            True at a cycle boundary, False during warm-up or mid-cycle.
        """
        if step < self.warmup_steps or self.cycle_steps == 0:
            return False
        return self.cycle_step(step) == self.cycle_steps - 1

    def samples_z(self, step):
        """Whether z should be sampled rather than taken as mu.

        Sampling starts only once the mixture prior exists; before that S3VDC
        encodes deterministically.

        Args:
            step: global step index.

        Returns:
            True during BETA and STATIC.
        """
        return self.phase(step) in (Phase.BETA, Phase.STATIC)

    def cycle_step(self, step):
        """Position of `step` within its BETA + STATIC cycle.

        Args:
            step: global step index, expected to be at or past warmup_steps.

        Returns:
            Int in [0, cycle_steps).
        """
        return (step - self.warmup_steps) % self.cycle_steps
