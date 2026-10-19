"""
Preparing sparse mixed-type matrices for training.
Contains utils for data preperation (using a 0.2 train/test split), creating a torch DataLoader object, masking, z-score normalizing numerical data, and renaming feature columns
for human readibility. 

The main exposed function is preprocess_data(), which wraps all data loading utils into a single function described below.
"""

import numpy as np
import torch
import pandas as pd

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from .config import DataConfig

def preprocess_data(features_path, metadata_path, dataset, seed):
    """
    Run the data preprocessing pipeline
    Args:
        Path to dataset and metadata
    Returns:
            dict with feats, X_train/M_train, X_test/M_test, X_in/M, binary_idx,
            num_idx, pw and data_cfg.
    """
    # Load
    features_df = load_features(features_path, impute_zero=True)
    patient_ids = features_df.index

    metadata = pd.read_parquet(metadata_path)
    covariate_map = metadata.set_index('covariateId')['covariateName'].to_dict()

    # Remap column names, check for duplicates
    features_df.columns = remap(features_df.columns, covariate_map)
    ids_train, ids_test = train_test_split(patient_ids, test_size=0.2, random_state=seed)
    features_df = features_df.loc[:, ~features_df.columns.duplicated()].copy()

    # Level encoding
    scaler = fit_lab_scaler(features_df.loc[ids_train])
    feats_df, mask_df = encode_measurements(features_df, scaler, clip=4.0)
    feats_df, mask_df = merge_unit_splits(feats_df, mask_df, metadata)

    # Filter binary features present in less than 10 rows
    binary_cols = [col for col in feats_df.columns if not col.startswith('prof_')]
    binary_present = feats_df[binary_cols].sum(axis=0) >= 10
    feats_df = feats_df.loc[:, list(binary_present[binary_present].index) + [col for col in feats_df.columns if col.startswith('prof_')]]
    mask_df  = mask_df.loc[:, feats_df.columns]

    # Drop all-zero feature columns; keep the mask aligned to the same columns.
    nonzero = (feats_df != 0).any(axis=0)
    feats_df = feats_df.loc[:, nonzero]
    mask_df  = mask_df.loc[:, feats_df.columns]

    X = feats_df.values.astype('float32')
    M = mask_df.values.astype('float32')

    # Neutral fill: where a measurement is missing, set the encoder input to the TRAIN
    # observed band-frequency so missingness leaves no discriminative tell in the input.
    # The mask still zeros these out of the reconstruction loss.
    fill_rate = band_rate(feats_df.loc[ids_train].values, mask_df.loc[ids_train].values)
    X_in = neutral_fill(X, M, fill_rate)

    is_train = patient_ids.isin(ids_train)   # Index.isin already returns an ndarray
    X_train, M_train = X_in[is_train],  M[is_train]
    X_test,  M_test  = X_in[~is_train], M[~is_train]

    binary_idx = np.where(np.all((X_train == 0) | (X_train == 1), axis=0))[0]
    num_idx = np.where(~np.all((X_train == 0) | (X_train == 1), axis=0))[0]

    return {
        "feats": feats_df, "mask": mask_df, "band_rate": fill_rate,
        "X_in": X_in, "M": M, "index": patient_ids, "is_train": is_train,
        "X_train": X_train, "M_train": M_train, "X_test": X_test, "M_test": M_test,
        "binary_idx": binary_idx.tolist(), "num_idx": num_idx.tolist(),
        "pw": positive_weights(X_train, M_train, binary_idx),
        "data_cfg": DataConfig.describe(dataset, X_in, M, binary_idx, num_idx,
                                        path=features_path),
    }


def load_features(path, impute_zero=True):
    """Pivot the long feature table and zero-fill constant columns.

    Args:
        path: parquet of (rowId, covariateId, covariateValue) triples.
        impute_zero: fill NaN with 0 in columns whose observed values are all
            identical, which are presence flags recorded only when present.

    Returns:
        DataFrame indexed by rowId, one column per covariate.
    """
    d = pd.read_parquet(path)
    p = d.pivot(index="rowId", columns="covariateId", values="covariateValue")
    for c in p.columns:
        if len(p[c].dropna().unique()) == 1 and impute_zero:
            p[c] = p[c].fillna(0)
    return p


def remap(cols, cmap):
    """Replace covariate ids with readable names.

    Args:
        cols: current column labels.
        cmap: dict of covariateId -> covariateName.

    Returns:
        List of names, stripped of any leading "prefix:" and space-free.
    """
    out = []
    for c in cols:
        n = cmap.get(int(c), str(c))
        if ":" in n:
            n = n.split(":", 1)[1]
        out.append(n.strip().replace(" ", "_"))
    return out


def fit_lab_scaler(train_df):
    """Per-lab robust location/scale (median / IQR) fit on TRAIN rows only.

    Args:
        train_df: training-row slice of the feature frame.

    Returns:
        dict with cols (the non-binary columns), med and iqr.
    """
    cols = [c for c in train_df.columns
            if not set(train_df[c].dropna()).issubset({0, 1, 0.0, 1.0})]
    med = train_df[cols].median()
    iqr = (train_df[cols].quantile(0.75) - train_df[cols].quantile(0.25)).replace(0, 1.0)
    return {"cols": cols, "med": med, "iqr": iqr}


def encode_measurements(df, scaler, clip=4.0):
    """Robust z-score the measurement columns, leaving binary columns untouched.

    Missing measurements become 0 with mask 0. Measurement columns are prefixed
    'prof_'. Binary columns always carry mask 1: absence there means zero, not
    unknown.

    Args:
        df: full feature frame.
        scaler: output of fit_lab_scaler.
        clip: symmetric clip applied after scaling.

    Returns:
        (feats, mask) aligned column-for-column.
    """
    cols, med, iqr = scaler["cols"], scaler["med"], scaler["iqr"]
    sub = df[cols]
    obs = sub.notna()
    z = ((sub - med) / iqr).clip(-clip, clip).where(obs, 0.0)
    z.columns = [f"prof_{c}" for c in cols]

    non_meas = df.drop(columns=cols)
    feats = pd.concat([non_meas, z], axis=1)
    mask = pd.concat([
        pd.DataFrame(1, index=df.index, columns=non_meas.columns),
        obs.astype(int).set_axis([f"prof_{c}" for c in cols], axis=1),
    ], axis=1)[feats.columns]
    return feats, mask


def merge_unit_splits(feats, mask, metadata, analysis_id=706):
    """Combine measurement columns recording one concept in several units.

    FeatureExtraction emits one covariate per (measurement concept, unit), so an
    analyte reported in both U/L and IU/L becomes two columns, each populated
    for a largely disjoint set of patients. That inflates the feature count and
    manufactures missingness.

    Merging runs after the per-column robust z-score in encode_measurements, so
    each column is already expressed in IQRs from its own median and differing
    unit scales cancel. No conversion factors are needed, and genuinely
    different scales such as ng/mL against ng/L merge correctly.

    Args:
        feats: z-scored feature frame from encode_measurements.
        mask: matching observation mask.
        metadata: covariate metadata with covariateId, covariateName, conceptId
            and analysisId.
        analysis_id: analysis holding measurement values.

    Returns:
        (feats, mask) with each duplicated concept reduced to a single column
        named without its unit suffix. Rows observed under several units take
        the mean of their z-scores; the merged mask is 1 where any source column
        was observed.
    """
    feats, mask = feats.copy(), mask.copy()

    for names in unit_groups(metadata, analysis_id):
        present = [n for n in names if n in feats.columns]
        if len(present) < 2:
            continue

        observed = mask[present] == 1
        count = observed.sum(axis=1)
        merged = feats[present].where(observed).sum(axis=1) / count.replace(0, 1)

        keep = present[0]
        feats[keep] = merged.where(count > 0, 0.0)
        mask[keep] = (count > 0).astype(int)
        feats = feats.drop(columns=present[1:])
        mask = mask.drop(columns=present[1:])

        stripped = keep.rsplit("_(", 1)[0]
        if stripped != keep and stripped not in feats.columns:
            feats = feats.rename(columns={keep: stripped})
            mask = mask.rename(columns={keep: stripped})

    return feats, mask


def unit_groups(metadata, analysis_id=706):
    """Column-name groups sharing a measurement conceptId.

    Args:
        metadata: covariate metadata frame.
        analysis_id: analysis whose covariates are measurement values.

    Returns:
        List of lists of prof_-prefixed column names, one per conceptId that
        appears under more than one unit. Single-unit concepts are omitted.
    """
    m = metadata[metadata["analysisId"] == analysis_id]
    cmap = dict(zip(m["covariateId"], m["covariateName"]))

    groups = []
    for _, g in m.groupby("conceptId"):
        if len(g) < 2:
            continue
        groups.append([f"prof_{n}" for n in remap(g["covariateId"], cmap)])
    return groups


def make_dataloader(x, mask=None, batch_size=256, shuffle=True, drop_last=True):
    """Wrap a feature matrix, and optionally its mask, in a DataLoader.

    Args:
        x: (n, i_dim) array or tensor of neutral-filled inputs.
        mask: (n, i_dim) 1/0 observation mask, or None to treat all entries as
            observed.
        batch_size, shuffle, drop_last: passed to DataLoader.

    Returns:
        DataLoader yielding (x,) or (x, mask) batches of float32 tensors.
    """
    tensors = [torch.as_tensor(x, dtype=torch.float32)]
    if mask is not None:
        tensors.append(torch.as_tensor(mask, dtype=torch.float32))
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )


def band_rate(x, mask):
    """Per-column mean over observed entries, used as the neutral fill value.

    For a binary column this is its prevalence; for a z-scored numeric column it
    is close to zero. Filling with this value means a missing entry carries the
    column's population-level information and nothing patient-specific.

    Args:
        x: (n, i_dim) array; may contain NaN at masked-out positions.
        mask: (n, i_dim) 1/0 observation mask.

    Returns:
        (i_dim,) float array. Columns with nothing observed return 0.
    """
    x, mask = observed_only(x, mask)
    observed = mask.sum(axis=0)
    return np.where(observed > 0, x.sum(axis=0) / np.maximum(observed, 1.0), 0.0)


def neutral_fill(x, mask, rate):
    """Replace masked-out entries with their column's band rate.

    Args:
        x: (n, i_dim) array, possibly containing NaN.
        mask: (n, i_dim) 1/0 observation mask.
        rate: (i_dim,) fill values, from band_rate on training rows.

    Returns:
        (n, i_dim) float32 array with no missing values.
    """
    return np.where(np.asarray(mask) == 1, np.asarray(x), rate).astype(np.float32)


def positive_weights(x, mask, binary_idx, cap=50.0):
    """Per-feature positive weights for the binary reconstruction term.

    Returns negatives/positives per column, computed over observed entries, so
    that the optimal constant prediction for each feature sits at 0.5 rather
    than at its prevalence. Without this, rare features are cheapest to
    reconstruct by always predicting zero and contribute nothing to the latent
    space.

    Args:
        x: (n, i_dim) array; may contain NaN at masked-out positions.
        mask: (n, i_dim) 1/0 observation mask.
        binary_idx: column indices to compute weights for.
        cap: clamp on the ratio, applied both ways. The cap binds below
            prevalence 1/(cap+1), keeping very rare features from dominating.

    Returns:
        (len(binary_idx),) float32 tensor, ready to pass to VaDE(pos_weight=).
    """
    x, mask = observed_only(np.asarray(x)[:, binary_idx], np.asarray(mask)[:, binary_idx])
    positives = x.sum(axis=0)
    observed = mask.sum(axis=0)
    negatives = observed - positives
    weights = np.where(positives > 0, negatives / np.maximum(positives, 1.0), 1.0)
    return torch.from_numpy(np.clip(weights, 1.0 / cap, cap).astype(np.float32))


def observed_only(x, mask):
    """Zero out masked-out entries in a NaN-safe way.

    Multiplying by the mask would leave NaN * 0 = NaN, so any column containing
    a missing value would poison every statistic computed over it.

    Args:
        x: (n, k) array, possibly containing NaN.
        mask: (n, k) 1/0 observation mask.

    Returns:
        (zeroed_x, mask) both float32 arrays.
    """
    x = np.asarray(x, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    return np.where(mask == 1, x, 0.0), mask
