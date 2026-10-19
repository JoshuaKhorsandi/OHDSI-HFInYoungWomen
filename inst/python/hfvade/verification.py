import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans as MBK
from sklearn.metrics import adjusted_rand_score


def verify_clustering(z, labels, feats, k, seed):
    feat_eta = pd.Series(
        {c: eta2(labels, feats[c].to_numpy()) for c in feats.columns}
    ).sort_values(ascending=False)

    verification = {
        "boot_ari": boot_ari(z, k, seed=seed),
        "matched_null_gap": matched_null_gap(z, k, seed=seed),
        **latent_health(z),
        "eta2_mean": float(feat_eta.mean()),
        "eta2_max": float(feat_eta.max()),
        "eta2_above_0.10": int((feat_eta > 0.10).sum()),
    }
    return verification, feat_eta.rename("eta2")


def boot_ari(z, k, B=10, frac=0.8, seed=42):
    rng = np.random.RandomState(seed); n = len(z); a = []
    for _ in range(B):
        i1 = rng.choice(n, int(frac*n), replace=False); i2 = rng.choice(n, int(frac*n), replace=False)
        l1 = MBK(k, n_init=3, random_state=0, batch_size=512).fit_predict(z[i1])
        l2 = MBK(k, n_init=3, random_state=0, batch_size=512).fit_predict(z[i2])
        p1 = {p: j for j, p in enumerate(i1)}; p2 = {p: j for j, p in enumerate(i2)}; it = np.intersect1d(i1, i2)
        if len(it) < 10: continue
        a.append(adjusted_rand_score(l1[[p1[p] for p in it]], l2[[p2[p] for p in it]]))
    return float(np.mean(a))


def matched_null_gap(z, k, B=10, seed=42):
    nz = np.random.RandomState(seed).multivariate_normal(z.mean(0), np.cov(z, rowvar=False), len(z))
    return boot_ari(z, k, B=B, seed=seed) - boot_ari(nz, k, B=B, seed=seed + 1)


def latent_health(z):
    s = z.std(0)
    _, sv, _ = np.linalg.svd(z - z.mean(0), full_matrices=False)
    return dict(min_std=float(s.min()), mean_std=float(s.mean()), pc1_frac=float(sv[0]**2 / (sv**2).sum()))


def eta2(labels, x):
    g = x.mean()
    ssb = sum((labels == c).sum() * (x[labels == c].mean() - g)**2 for c in np.unique(labels))
    return float(ssb / (((x - g)**2).sum() + 1e-12))
