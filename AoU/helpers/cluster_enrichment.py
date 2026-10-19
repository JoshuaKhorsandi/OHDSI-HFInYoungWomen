# Original cluster enrichment functions.

def is_binary_feature(x):
    """Whether every observed value of a feature is 0 or 1.

    Args:
        x: feature Series.

    Returns:
        True for binary features, False for continuous ones such as the
        z-scored prof_ labs.
    """
    u = pd.unique(x.dropna())
    return len(u) > 0 and set(np.asarray(u).tolist()) <= {0, 1, 0.0, 1.0}


def cluster_enrichment(assignments, features):
    """Per-cluster feature effects: log odds ratio for binary, SMD for continuous.

    Binary features use a continuity-corrected log odds ratio of membership
    against the rest of the cohort. Continuous features use a standardised mean
    difference against the whole-cohort standard deviation, because counting
    values equal to 1 is meaningless for a z-scored lab. The two are put on one
    scale as Cohen's d, converting log-OR by sqrt(3)/pi.

    Args:
        assignments: frame with a "cluster" column, indexed like features.
        features: entities x features DataFrame of encoded values.

    Returns:
        Long DataFrame with cluster, Feature, feature_type, Cohens_d, Effect,
        Log_OR, SMD and the cluster/cohort prevalence and mean columns, sorted
        by cluster then descending |Cohen's d|.
    """
    df = features.loc[assignments.index]
    cluster = assignments["cluster"]
    n_total = len(df)
    scale = np.sqrt(3) / np.pi

    rows = []
    for feature in df.columns:
        v = df[feature]
        binary = is_binary_feature(v)
        sd = v.std()
        ones = (v == 1).values
        zeros = (v == 0).values

        cohort_n = int(ones.sum())
        cohort_mean = float(v.mean())

        for k in sorted(cluster.unique()):
            incl = (cluster == k).values
            n_k = int(incl.sum())

            if binary:
                a = float((ones & incl).sum()) + 0.5
                b = float((zeros & incl).sum()) + 0.5
                c_out = float((ones & ~incl).sum()) + 0.5
                d_out = float((zeros & ~incl).sum()) + 0.5
                effect = float(np.log((a / b) / (c_out / d_out)))
                cohens_d, log_or, smd = effect * scale, effect, np.nan
            else:
                if pd.isna(sd) or sd == 0:
                    effect = 0.0
                else:
                    effect = float((v[incl].mean() - v[~incl].mean()) / sd)
                cohens_d, log_or, smd = effect, np.nan, effect

            rows.append({
                "cluster": k,
                "Feature": feature,
                "feature_type": "binary" if binary else "continuous",
                "Cohens_d": cohens_d,
                "Effect": effect,
                "Log_OR": log_or,
                "SMD": smd,
                "cluster_n": int((ones & incl).sum()),
                "cluster_pct": 100.0 * float((ones & incl).sum()) / n_k if n_k else np.nan,
                "cluster_mean": float(v[incl].mean()) if n_k else np.nan,
                "Cohort_n": cohort_n,
                "Cohort_pct": 100.0 * cohort_n / n_total,
                "Cohort_mean": cohort_mean,
            })

    out = pd.DataFrame(rows)
    out["abs_d"] = out["Cohens_d"].abs()
    return out.sort_values(["cluster", "abs_d"], ascending=[True, False]).drop(columns="abs_d")
