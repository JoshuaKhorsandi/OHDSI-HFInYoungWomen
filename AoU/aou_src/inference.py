"""Final inference pass: run a whole cohort through a trained model once.

Produces one row per patient carrying the cluster assignment, the soft
responsibilities, and the latent coordinates, so downstream analysis can join
it back to the original features by index.
"""

import numpy as np
import pandas as pd
import torch


@torch.no_grad()
def predict_clusters(model, x, mask=None, index=None, batch_size=1024, device="cpu"):
    """Encode every row and return assignments, responsibilities and latents.

    x must already be the neutral-filled encoder input, not the raw matrix with
    missing values. mask is accepted for symmetry with training but is unused:
    the encoder never sees the mask, only the loss does.
    """
    model.to(device).eval()

    x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    latents, responsibilities = [], []

    for start in range(0, len(x), batch_size):
        batch = x[start : start + batch_size].to(device)
        mu, _ = model.encoder(batch)
        latents.append(mu.cpu())
        responsibilities.append(model.responsibilities(mu).cpu())

    z = torch.cat(latents).numpy()
    gamma = torch.cat(responsibilities).numpy()

    frame = pd.DataFrame(index=index if index is not None else pd.RangeIndex(len(z)))
    frame["cluster"] = gamma.argmax(axis=1)
    frame["confidence"] = gamma.max(axis=1)
    for k in range(gamma.shape[1]):
        frame[f"gamma_{k}"] = gamma[:, k]
    for d in range(z.shape[1]):
        frame[f"z_{d}"] = z[:, d]

    return frame


def save_assignments(frame, path, features=None, mask=None):
    """Write assignments to parquet, optionally joined to the feature matrix.

    Passing features writes one self-contained table -- rows, features, and
    cluster assignment together -- which is what downstream cluster
    characterisation needs. Column names are prefixed so they cannot collide
    with the gamma_/z_ columns.
    """
    output = frame

    if features is not None:
        # features = features.add_prefix("feat_")
        output = output.join(features, how="left")

    if mask is not None:
        output = output.join(mask.add_prefix("observed_"), how="left")

    path = str(path)
    output.to_parquet(path)

    return output