# Helpers executed in the notebook namespace to preserve existing global dependencies.

def fit_lab_scaler(train_df):
    """Fit robust-scaling statistics (median/IQR) for non-binary columns.

    Args:
        train_df: feature DataFrame restricted to training patients.

    Returns:
        dict with "cols" (numeric column names), "median", "iqr" (Series),
        fit using training patients only.
    """
    numeric_cols = [
        col for col in train_df.columns
        if not set(train_df[col].dropna().unique()).issubset({0, 1, 0.0, 1.0})
    ]
    median = train_df[numeric_cols].median()
    iqr = (
        train_df[numeric_cols].quantile(0.75)
        - train_df[numeric_cols].quantile(0.25)
    ).replace(0, 1.0)
    return {"cols": numeric_cols, "median": median, "iqr": iqr}


def encode_measurements(df, scaler, clip=4.0):
    """Robustly z-score continuous measurements; keep binary features unchanged.

    Missing continuous values become 0, with mask=0 so they are excluded from
    reconstruction loss. Binary 0 means absent, not missing, so it stays mask=1.

    Args:
        df: full feature DataFrame (all patients), covariate-name columns.
        scaler: dict from fit_lab_scaler.
        clip: symmetric clip bound applied to z-scores.

    Returns:
        (feats, mask) DataFrames, same index as df.
    """
    numeric_cols = scaler["cols"]
    median = scaler["median"]
    iqr = scaler["iqr"]

    numeric = df[numeric_cols]
    observed_numeric = numeric.notna()

    numeric_scaled = (
        ((numeric - median) / iqr)
        .clip(-clip, clip)
        .where(observed_numeric, 0.0)
    )
    numeric_scaled.columns = [f"prof_{col}" for col in numeric_cols]

    binary = df.drop(columns=numeric_cols)
    feats = pd.concat([binary, numeric_scaled], axis=1)
    mask = pd.concat(
        [
            pd.DataFrame(1, index=df.index, columns=binary.columns),
            observed_numeric.astype(int).set_axis(numeric_scaled.columns, axis=1),
        ],
        axis=1,
    )[feats.columns]
    return feats, mask


def remap_covariate_names(col_names, covariate_map):
    """Map raw covariate_id columns to human-readable, underscore-safe names.

    Args:
        col_names: iterable of raw column labels (covariate_id-like).
        covariate_map: dict from int(covariateId) to covariateName.

    Returns:
        list of remapped, whitespace-safe names, same order as col_names.
    """
    remapped = []
    for col in col_names:
        try:
            name = covariate_map.get(int(float(col)), str(col))
        except (ValueError, TypeError):
            name = str(col)
        if ":" in name:
            name = name.split(":", 1)[1]
        name = name.strip().replace(" ", "_")
        remapped.append(name)
    return remapped


def build_aou_preprocessed_data(feature_matrix_path, metadata_path, seed):
    """Port of VADE_denovo_aug.ipynb's preprocessing cells (4, 6, 8, 10).

    Reproduces feats_df/mask_df/X_in/band_rate/binary_idx/num_idx/pos_weight
    exactly as the de-novo notebook does, so this optimization search tunes
    against the same feature matrix the reported de-novo model was trained on.

    Args:
        feature_matrix_path: path to feature_matrix_wide.csv.
        metadata_path: path to meta_HFCohort_deid.parquet (covariate_id -> name).
        seed: random seed for the train/test split (must match the reported
            de-novo run's seed to reproduce the same ids_train/ids_test).

    Returns:
        dict with keys: X_in, X_train, X_test, M, M_train, M_test, band_rate,
        binary_idx, num_idx, pw, feats, index, is_train.
    """
    # load + remap covariate names
    features_df = pd.read_csv(feature_matrix_path)
    features_df = features_df.set_index("person_id")
    features_df = features_df.loc[:, ~features_df.columns.duplicated()].copy()
    patient_ids = features_df.index

    metadata = pd.read_parquet(metadata_path)
    covariate_map = metadata.set_index("covariateId")["covariateName"].to_dict()
    remapped_names = remap_covariate_names(features_df.columns, covariate_map)
    pd.DataFrame({
        "covariate_id": features_df.columns,
        "remapped_name": remapped_names,
    }).to_csv("covariate_id_name_map.csv", index=False)
    features_df.columns = remapped_names

    # dedupe post-remap, train/test split, fit scaler 
    features_df = features_df.loc[:, ~features_df.columns.duplicated()].copy()
    patient_ids = features_df.index

    ids_train, ids_test = train_test_split(
        patient_ids, test_size=0.20, random_state=seed,
    )

    scaler = fit_lab_scaler(features_df.loc[ids_train])
    feats_df, mask_df = encode_measurements(features_df, scaler, clip=4.0)

    binary_cols = [col for col in feats_df.columns if not col.startswith("prof_")]
    keep_binary = feats_df[binary_cols].sum(axis=0) >= 10
    feats_df = feats_df.loc[
        :,
        list(keep_binary[keep_binary].index)
        + [col for col in feats_df.columns if col.startswith("prof_")]
    ]
    mask_df = mask_df.loc[:, feats_df.columns]

    nonzero = (feats_df != 0).any(axis=0)
    feats_df = feats_df.loc[:, nonzero]
    mask_df = mask_df.loc[:, feats_df.columns]

    X = feats_df.to_numpy(dtype="float32")
    M = mask_df.to_numpy(dtype="float32")

    band_rate = (
        feats_df.loc[ids_train]
        .where(mask_df.loc[ids_train] == 1)
        .mean()
        .fillna(0.0)
        .to_numpy(dtype="float32")
    )
    X_in = np.where(M == 1, X, band_rate).astype("float32")

    is_train = patient_ids.isin(ids_train)
    X_train, M_train = X_in[is_train], M[is_train]
    X_test, M_test = X_in[~is_train], M[~is_train]

    binary_idx = np.where(np.all((X_train == 0) | (X_train == 1), axis=0))[0].tolist()
    num_idx = np.where(~np.all((X_train == 0) | (X_train == 1), axis=0))[0].tolist()

    pw = positive_weights(X_train, M_train, binary_idx)

    return {
        "X_in": X_in, "X_train": X_train, "X_test": X_test,
        "M": M, "M_train": M_train, "M_test": M_test,
        "band_rate": band_rate, "binary_idx": binary_idx, "num_idx": num_idx,
        "pw": pw, "feats": feats_df, "index": patient_ids, "is_train": is_train,
    }


