import json
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
from shap.plots._style import style_context
from sklearn.metrics import (auc, classification_report, confusion_matrix, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from .data import (encode_measurements, fit_lab_scaler, load_features, merge_unit_splits,
                   preprocess_data, remap)


def run_posthoc(features_path, metadata_path, assignments_path, output_folder, settings_path):
    settings = json.loads(Path(settings_path).read_text(encoding="utf-8"))
    options = settings["posthoc"]
    seed = settings["seed"]
    out = Path(output_folder)
    out.mkdir(parents=True, exist_ok=True)
    set_style()

    P = preprocess_data(features_path, metadata_path, dataset=settings["dataset"], seed=seed)
    assignments = pd.read_parquet(assignments_path)
    cluster = configure_clusters(assignments["cluster"], options.get("cluster_names"))
    df = P["feats"].join(cluster, how="inner")
    y = df["cluster"]

    X = df.drop(columns=["cluster"])
    X.columns = clean_names(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    d_train = xgboost.DMatrix(X_train, label=y_train)
    d_test = xgboost.DMatrix(X_test, label=y_test)

    measures, scales = unclipped_measurements(features_path, metadata_path, df,
                                              log=options["ridgeline_log"], seed=seed)
    ridge_features = [f for f in options["ridgeline_features"] if f in measures.columns]
    if ridge_features:
        plot_measurement_grid(ridge_features, measures, P["mask"], y, scales=scales)
        save_figure(out, "ridgeplots_measurements.svg", bbox_inches="tight")

    flags = class_flags(features_path, metadata_path, DRUG_CLASSES | CONDITION_CLASSES,
                        index=df.index)
    flags["BMI ≥ 30"] = bmi_flag(features_path, metadata_path, index=df.index)
    panel = available_conditions(options["radar_classes"], flags)
    if len(panel) >= 3:
        plot_combined_radar(panel, flags, y, scale="lift")
        save_figure(out, "radar_enrichment_combined.svg")
        plot_cluster_radars(panel, flags, y, scale="odds").subplots_adjust(
            top=0.86, bottom=0.04, hspace=0.34, wspace=0.46)
        save_figure(out, "radar_enrichment.svg")

    model = train_surrogate(d_train, d_test, seed)
    pred = plot_fidelity(model, d_test, y_test, seed)
    save_figure(out, "surrogate_fidelity.svg", bbox_inches="tight")
    report = classification_report(y_test, pred, digits=3)
    (out / "surrogate_classification_report.txt").write_text(report)

    aucs = plot_roc(model, d_test, y_test)
    save_figure(out, "roc_xgboost.svg", bbox_inches="tight")
    aucs.to_csv(out / "surrogate_auc.csv")

    explainer = shap.TreeExplainer(model)
    sv = explainer(X)
    sv.feature_names = [pretty_feature(f, 40) for f in X.columns]

    cols = [f"cluster {k}" for k in range(K)]
    mean_abs = pd.DataFrame(np.abs(sv.values).mean(0), index=X.columns, columns=cols)
    ranked = mean_abs.sum(axis=1).sort_values(ascending=False)
    mean_abs.assign(total=ranked).sort_values("total", ascending=False).to_csv(
        out / "shap_mean_abs.csv")

    plot_global_importance(mean_abs, ranked)
    save_figure(out, "feature_importance_xgboost.svg")

    plot_beeswarm_grid(sv)
    save_figure(out, "beeswarm_xgboost.svg")

    dependence = [ranked.index[0], *clean_names(options["dependence_features"])]
    for feature in dict.fromkeys(dependence):
        if feature in X.columns:
            plot_dependence(sv, X, y, feature)
            save_figure(out, f"{slug(pretty_feature(feature))}_dependence.svg")

    proba = model.predict(xgboost.DMatrix(X), iteration_range=(0, model.best_iteration + 1))
    plot_exemplars(sv, X, y, proba, out)

    return {"auc": aucs.to_dict(), "top_features": ranked.head(10).to_dict(),
            "figures": sorted(p.name for p in out.glob("*.svg"))}


def configure_clusters(cluster, names=None):
    global K, CLUSTER_LABELS
    names = {int(k): v for k, v in (names or {}).items()}
    observed = sorted(int(c) for c in cluster.unique())
    K = len(observed)
    CLUSTER_LABELS = [f"{c + 1} · {names[c + 1]}" if c + 1 in names else f"Cluster {c + 1}"
                      for c in observed]
    return cluster.map({c: i for i, c in enumerate(observed)}).rename("cluster")


def set_style(size=11):
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': False,
        'axes.spines.top': False, 'axes.spines.right': False, 'font.size': size,
        'axes.titlesize': 'large', 'axes.labelsize': 'medium',
        'xtick.labelsize': 'small', 'ytick.labelsize': 'small',
        'legend.fontsize': 'small', 'figure.titlesize': 'x-large',
        'font.sans-serif': FONT,
    })


def clean_names(columns):
    return [re.sub(r'[\[\]<>]', '', c) for c in columns]


def save_figure(out, name, **kwargs):
    plt.savefig(out / name, **kwargs)
    plt.close("all")


def train_surrogate(d_train, d_test, seed):
    params = {
        'objective': 'multi:softprob',
        'num_class': K,
        'eval_metric': ['mlogloss', 'merror'],
        'eta': 0.05,
        'max_depth': 3,
        'min_child_weight': 1,
        'colsample_bytree': 0.5,
        'subsample': 0.8,
        'lambda': 1,
        'seed': seed,
    }
    return xgboost.train(
        params,
        d_train,
        5000,
        evals=[(d_test, "test")],
        verbose_eval=50,
        early_stopping_rounds=20,
    )


def plot_measurement_grid(features, feats, mask, y, ncols=4, overlap=0.21, window=1.0,
                          scales=None):
    nrows = int(np.ceil(len(features) / ncols))
    height = 4.1 * nrows + 1.0
    fig = plt.figure(figsize=(4.8 * ncols, height))
    outer = fig.add_gridspec(nrows, ncols, hspace=0.52, wspace=0.14,
                             left=0.03, right=0.985, top=0.855, bottom=0.62 / height)
    for i, feature in enumerate(features):
        inner = outer[i // ncols, i % ncols].subgridspec(K, 1, hspace=-overlap)
        axes = [fig.add_subplot(inner[k]) for k in range(K)]
        draw_ridges(axes, feature, feats, mask, y, window=window, scales=scales)
        axes[0].set_title(pretty_feature(feature), loc='center', fontsize='large', pad=8,
                          fontweight='bold')

    cluster_legend(fig)
    band_legend(fig, MUTED, loc='upper right', bbox_to_anchor=(0.985, 1.0), ncol=4,
                compact=True)
    return fig


def unclipped_measurements(features_path, metadata_path, reference, log=(), seed=42):
    frame = log_transform(named_features(features_path, metadata_path), log)
    ids_train, _ = train_test_split(frame.index, test_size=0.2, random_state=seed)
    scaler = fit_lab_scaler(frame.loc[ids_train])
    feats, obs = encode_measurements(frame, scaler, clip=np.inf)
    feats, obs = merge_unit_splits(feats, obs, pd.read_parquet(metadata_path))
    columns = [c for c in reference.columns if c in feats.columns]
    return (feats.reindex(index=reference.index, columns=columns),
            measurement_scales(scaler, log).reindex(columns))


def plot_combined_radar(conditions, feats, y, scale='lift'):
    conditions = resolve_conditions(conditions, feats)
    table = cluster_value_table(conditions, feats, y, scale)
    angles = ring_angles(len(conditions))
    fig, ax = plt.subplots(figsize=(8.4, 8.2), subplot_kw={'projection': 'polar'})
    draw_radar(ax, None, conditions, extent=table.values)
    for k in range(K):
        draw_polygon(ax, angles, table[k].values, CLUSTER_COLORS[k], alpha=0.07,
                     label=cluster_label(k))
    fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.01), frameon=False,
               fontsize='medium', labelcolor=INK, ncol=2, handletextpad=0.6,
               columnspacing=1.8)
    fig.suptitle(f'Recorded Conditions and Drugs, All Clusters — {scale_title(scale)}',
                 x=0.5, y=0.985, fontsize='large', fontweight='bold')
    fig.subplots_adjust(top=0.90, bottom=0.16, left=0.10, right=0.90)
    return fig


def plot_cluster_radars(conditions, feats, y, scale='odds', ncols=2):
    conditions = resolve_conditions(conditions, feats)
    table = cluster_value_table(conditions, feats, y, scale)
    nrows = int(np.ceil(K / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 5.2 * nrows), squeeze=False,
                             subplot_kw={'projection': 'polar'})
    for k, ax in enumerate(axes.flat):
        if k >= K:
            ax.set_visible(False)
            continue
        draw_radar(ax, table[k].values, conditions, color=CLUSTER_COLORS[k],
                   extent=table.values)
        ax.set_title(f'{cluster_label(k)}   n = {(y.values == k).sum()}',
                     fontsize='medium', fontweight='bold', color='black', pad=30)
    fig.suptitle(f'Recorded Conditions and Drugs by Cluster — {scale_title(scale)}',
                 x=0.5, y=0.985, fontsize='large', fontweight='bold')
    fig.subplots_adjust(top=0.90, bottom=0.03, hspace=0.16, wspace=0.46)
    return fig


def class_flags(features_path, metadata_path, classes, index=None, analysis_ids=(201, 402, 501)):
    flags = concept_flags(features_path, metadata_path, analysis_ids)
    members = class_members(flags.columns, classes)
    grouped = pd.DataFrame({name: flags[cols].max(axis=1) for name, cols in members.items()
                            if cols}, index=flags.index)
    if index is not None:
        grouped = grouped.reindex(index).fillna(0)
    return grouped.astype(int)


def bmi_flag(features_path, metadata_path, index=None, threshold=30.0):
    meta = pd.read_parquet(metadata_path)
    values = pd.read_parquet(features_path)
    height = measurement_series(values, meta, 'Body height',
                                {'inch': 0.0254, 'centimeter': 0.01, 'meter': 1.0})
    weight = measurement_series(values, meta, 'Body weight',
                                {'ounce': 0.0283495, 'pound': 0.453592, 'gram': 0.001,
                                 'kilogram': 1.0})
    flag = (weight / height ** 2 >= threshold).astype(int)
    return flag if index is None else flag.reindex(index).fillna(0).astype(int)


def available_conditions(names, feats):
    columns = [c.lower().replace('_', ' ') for c in feats.columns]
    return [n for n in names
            if any(str(n).lower().replace('_', ' ').strip() in c for c in columns)]


def plot_fidelity(model, d_test, y_test, seed):
    proba = model.predict(d_test, iteration_range=(0, model.best_iteration + 1))
    pred = proba.argmax(1)
    cm = confusion_matrix(y_test, pred, labels=range(K), normalize='true')
    rng = np.random.default_rng(seed)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={'width_ratios': [1, 1.15]})
    ax = axes[0]
    im = ax.imshow(cm, cmap=SEQUENTIAL, vmin=0, vmax=1)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f'{cm[i, j]:.2f}', ha='center', va='center', fontsize='small',
                    color='#ffffff' if cm[i, j] > 0.55 else INK)
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xlabel('Predicted Cluster'); ax.set_ylabel('Assigned Cluster')
    ax.set_title('Surrogate Fidelity (Row-Normalised)', loc='left', fontsize='medium')
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.85).outline.set_visible(False)

    ax = axes[1]
    conf = proba.max(1)
    for k in range(K):
        m = y_test.values == k
        ax.scatter(k + rng.uniform(-.18, .18, m.sum()), conf[m], s=14,
                   color=CLUSTER_COLORS[k], alpha=.55, edgecolor=SURFACE, linewidth=.5)
    ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.set_xticks(range(K)); ax.set_xlabel('Assigned Cluster')
    ax.set_ylabel('Surrogate Confidence'); ax.set_ylim(0, 1)
    ax.set_title('Predicted Probability of the Assigned Cluster', loc='left', fontsize='medium')
    ax.yaxis.grid(True, color=GRID, lw=.8); ax.set_axisbelow(True)
    fig.tight_layout()
    return pred


def plot_roc(model, d_test, y_test):
    proba = model.predict(d_test, iteration_range=(0, model.best_iteration + 1))
    curves, scores = ovr_roc(y_test.values, proba)

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    for k in range(K):
        fpr, tpr = curves[k]
        ax.plot(fpr, tpr, color=CLUSTER_COLORS[k], lw=1.9, zorder=3,
                label=f'{cluster_label(k)} (AUC = {scores[k]:.3f})')
    fpr, tpr = curves['micro']
    ax.plot(fpr, tpr, color=INK, lw=1.5, ls=(0, (5, 2)), zorder=2,
            label=f"Micro-Average (AUC = {scores['micro']:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.005)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(f"Surrogate ROC, One-vs-Rest · Macro AUC = {scores['macro']:.3f}",
                 loc='left', fontsize='large', fontweight='bold')
    ax.grid(True, color=GRID, lw=.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='lower right', fontsize='small', labelcolor=INK)
    fig.tight_layout()

    index = [cluster_label(k) for k in range(K)] + ['Micro-Average', 'Macro-Average']
    values = [scores[k] for k in range(K)] + [scores['micro'], scores['macro']]
    return pd.Series(values, index=index, name='ROC-AUC')


def plot_global_importance(mean_abs, ranked, top=20):
    feats = ranked.head(top).index[::-1]
    vals = mean_abs.loc[feats]
    fig, ax = plt.subplots(figsize=(10, 0.34 * top + 1.2))
    left = np.zeros(len(feats))
    for k in range(K):
        w = vals[f'cluster {k}'].values
        ax.barh(range(len(feats)), w, left=left, height=.72, color=CLUSTER_COLORS[k],
                label=cluster_label(k), edgecolor=SURFACE, linewidth=1.4)
        left += w
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels([pretty_feature(f, 52) for f in feats], fontsize='small', color=INK)
    ax.set_xlabel('Mean |SHAP|')
    ax.set_title('Feature Importance by Cluster', loc='left', fontsize='large',
                 fontweight='bold')
    ax.xaxis.grid(True, color=GRID, lw=.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc='lower right', fontsize='small', labelcolor=INK)
    fig.tight_layout()


def plot_beeswarm_grid(sv, top=10):
    nrows = int(np.ceil(K / 2))
    fig = plt.figure(figsize=(13, 4.5 * nrows))
    for k in range(K):
        ax = fig.add_subplot(nrows, 2, k + 1)
        plt.sca(ax)
        shap.plots.beeswarm(sv[:, :, k], max_display=top, show=False,
                            color_bar=(k == 1), plot_size=None)
        ax.set_title(cluster_label(k), loc='left', fontsize='medium',
                     color='black', fontweight='bold')
        ax.tick_params(labelsize='x-small')
        ax.set_xlabel('SHAP (Log-Odds Toward This Cluster)', fontsize='small')
    fig.tight_layout()


def plot_dependence(sv, X, y, feature, highlight=True, ncols=None, panel=(3.4, 3.4)):
    ncols = ncols or K
    j = list(X.columns).index(feature)
    nrows = int(np.ceil(K / ncols))
    fig, axes = plt.subplots(nrows, ncols, sharey=True, squeeze=False,
                             figsize=(panel[0] * ncols, panel[1] * nrows))
    for k, ax in enumerate(axes.flat):
        if k >= K:
            ax.set_visible(False)
            continue
        draw_dependence(ax, X[feature].values, sv.values[:, j, k], y, k, highlight)
        ax.set_title(cluster_label(k), loc='left', fontsize='small',
                     color='black', fontweight='bold')
        ax.set_xlabel('Feature Value (Robust z-score)', fontsize='small', color=INK)
        ax.tick_params(labelsize='small', colors=INK)
        ax.yaxis.grid(True, color=GRID, lw=.8); ax.set_axisbelow(True)
    for row in axes:
        row[0].set_ylabel('SHAP (Log-Odds)', color=INK)
    if not highlight:
        axes[0, 0].legend(frameon=False, fontsize='x-small', labelcolor=INK, loc='upper left')
    fig.suptitle(f'{pretty_feature(feature, 70)} Per Cluster Distribution', x=.008, ha='left',
                 fontsize='large')
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.14 / nrows))
    return fig


def plot_exemplars(sv, X, y, proba, folder, top=10):
    for k in range(K):
        members = np.where(y.values == k)[0]
        i = members[proba[members, k].argmax()]
        with style_context(primary_color_positive=CLUSTER_COLORS[k],
                           primary_color_negative=MUTED,
                           secondary_color_positive=shade(CLUSTER_COLORS[k], .4),
                           secondary_color_negative=shade(MUTED, .4),
                           text_color=INK, hlines_color=GRID, vlines_color=MUTED,
                           tick_labels_color=MUTED):
            shap.plots.waterfall(sv[i, :, k], max_display=top, show=False)
        fig = plt.gcf()
        fig.set_size_inches(9, 4.2)
        fig.suptitle(f'{cluster_label(k)} — Patient (p={proba[i, k]:.2f})',
                     x=.01, ha='left', fontsize='medium', color='black', fontweight='bold')
        save_figure(folder, f'{slug(cluster_label(k))}_waterfall.svg', bbox_inches='tight')


def draw_ridges(axes, feature, feats, mask, y, labels=False, window=1.0, cut=3.0, scales=None):
    obs = [observed_values(feats, mask, y, feature, k) for k in range(K)]
    pooled = np.concatenate([v for v in obs if v.size])
    lo, hi = span(pooled, window)
    top = 0.0
    for k, (ax, v) in enumerate(zip(axes, obs)):
        if v.size > 1:
            draw_ridge(ax, v, CLUSTER_COLORS[k], lo, hi, cut)
            top = max(top, ax.get_ylim()[1])
    for k, (ax, v) in enumerate(zip(axes, obs)):
        ax.axvline(0.0, color=MUTED, lw=0.9, ls=(0, (4, 4)), zorder=2)
        ax.axhline(0, color=INK, lw=1.1, clip_on=False, zorder=5)
        draw_intervals(ax, v, CLUSTER_COLORS[k], -0.16 * top)
        ax.set_xlim(*RIDGE_XLIM)
        ax.set_ylim(-0.30 * top, top * 1.05)
        ax.set_yticks([]); ax.set_ylabel('')
        ax.patch.set_alpha(0)
        ax.spines[:].set_visible(False)
        ax.tick_params(bottom=False, labelbottom=False)
        if labels:
            ax.text(-0.02, 0.30, f'{cluster_label(k)}\nn = {v.size}', transform=ax.transAxes,
                    ha='right', va='bottom', fontsize='large', fontweight='bold',
                    color='black')
    if labels:
        axes[0].text(0.0, 1.0, ' Cohort median', transform=axes[0].get_xaxis_transform(),
                     fontsize='x-small', color=NOTE, va='top')
    ticks = np.arange(np.ceil(RIDGE_XLIM[0]), np.floor(RIDGE_XLIM[1]) + 0.5)
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels(tick_labels(ticks, feature, scales))
    axes[-1].set_xlabel(axis_label(feature, scales), fontsize='medium', color=INK)
    axes[-1].tick_params(bottom=True, labelbottom=True, labelsize='small', colors=INK)


def draw_ridge(ax, v, color, lo, hi, cut=3.0):
    inside = windowed(v, lo, hi)
    if inside.size < 2:
        return
    below, above = int(np.sum(v < lo)), int(np.sum(v > hi))
    x, density = ridge_density(inside, cut, tails=(below == 0, above == 0))
    ax.fill_between(x, density, color=color, alpha=0.85, linewidth=0, zorder=3)
    ax.plot(np.r_[x[0], x, x[-1]], np.r_[0.0, density, 0.0], color=SURFACE, lw=1.6, zorder=4)


def draw_intervals(ax, v, color, y):
    if v.size < 2:
        return
    for level, fade in BANDS:
        q = np.quantile(v, [(1 - level) / 2, (1 + level) / 2])
        ax.plot(q, [y, y], color=shade(color, fade), lw=5.5, solid_capstyle='butt', zorder=7)
    ax.scatter([np.median(v)], [y], s=16, color=INK, edgecolor=SURFACE, linewidth=0.8, zorder=8)


def band_legend(target, color, loc='upper left', compact=False, **kwargs):
    handles = [plt.Line2D([], [], color=shade(color, fade), lw=5.5,
                          label=f'{level:.0%}' + ('' if compact else ' of patients'))
               for level, fade in BANDS]
    handles.append(plt.Line2D([], [], ls='', marker='o', markersize=5, color=INK, label='Cluster Median'))
    target.legend(handles=handles, loc=loc, frameon=True, facecolor=SURFACE, edgecolor=GRID,
                  fontsize='x-small', labelcolor=INK, handlelength=1.6, borderpad=0.7,
                  labelspacing=0.5, **kwargs)


def cluster_legend(fig):
    handles = [plt.Line2D([], [], marker='s', ls='', markersize=9, color=CLUSTER_COLORS[k],
                          label=cluster_label(k)) for k in range(K)]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.03, 1.0), ncol=K,
               frameon=False, fontsize='small', labelcolor=INK, handletextpad=0.4,
               columnspacing=1.6)


def log_transform(frame, patterns):
    frame = frame.copy()
    for column in frame.columns:
        if any(str(p).lower() in column.lower() for p in patterns):
            frame[column] = np.log10(frame[column].where(frame[column] > 0))
    return frame


def measurement_scales(scaler, patterns):
    logged = [any(str(p).lower() in c.lower() for p in patterns) for c in scaler['cols']]
    scales = pd.DataFrame({'center': scaler['med'], 'spread': scaler['iqr'],
                           'log': pd.Series(logged, index=scaler['cols'])})
    return scales.rename(index=lambda c: f'prof_{c}')


def named_features(features_path, metadata_path):
    frame = load_features(features_path, impute_zero=True)
    metadata = pd.read_parquet(metadata_path)
    frame.columns = remap(frame.columns, metadata.set_index('covariateId')['covariateName'].to_dict())
    return frame.loc[:, ~frame.columns.duplicated()].copy()


def draw_radar(ax, v, conditions, color=None, extent=None):
    angles = ring_angles(len(conditions))
    spread = np.ravel(extent if extent is not None else v)
    floor, ceiling, ticks = radial_limits(spread)
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.set_ylim(floor, ceiling)
    ax.vlines(angles, floor, ceiling, color=GRID, ls=(0, (4, 4)), lw=0.8, zorder=1)
    if v is not None:
        draw_polygon(ax, angles, v, color)
    ax.plot(np.linspace(0, 2 * np.pi, 200), np.zeros(200), color=MUTED, lw=1.1, zorder=4)
    style_radar(ax, angles, conditions, ticks)


def draw_polygon(ax, angles, v, color, point_shape='o', alpha=0.16, lw=2.5, label=None):
    loop = np.append(angles, angles[0])
    values = np.append(v, v[0])
    ax.plot(loop, values, color=color, lw=lw, zorder=4, label=label, clip_on=False)
    ax.fill(loop, values, color=color, alpha=alpha, zorder=3)
    ax.scatter(angles, v, s=44, color=color, edgecolor=SURFACE, linewidth=0.8, zorder=5,
               marker=point_shape, clip_on=False)


def resolve_conditions(conditions, feats):
    columns = [c for c in feats.columns if not c.startswith('prof_')]
    lookup = {c.lower().replace('_', ' '): c for c in columns}
    resolved = []
    for name in conditions:
        key = str(name).lower().replace('_', ' ').strip()
        if key in lookup:
            resolved.append(lookup[key])
            continue
        hits = [c for k, c in lookup.items() if key in k]
        if len(hits) != 1:
            raise KeyError(f'{name!r} matches {len(hits)} categorical features: {hits[:6]}')
        resolved.append(hits[0])
    return resolved


def cluster_value_table(conditions, feats, y, scale):
    if scale == 'odds':
        x = feats[conditions]
        return pd.DataFrame({k: log_odds_ratio(x, y.values == k) for k in range(K)})
    if scale == 'lift':
        rates = cluster_prevalence(conditions, feats, y)
        return rates[list(range(K))].apply(log_lift, cohort=rates['cohort'])
    raise ValueError(f"scale must be 'odds' or 'lift', got {scale!r}")


def cluster_prevalence(conditions, feats, y):
    x = feats[conditions]
    columns = {k: x[y.values == k].mean() * 100 for k in range(K)}
    columns['cohort'] = x.mean() * 100
    return pd.DataFrame(columns)


def concept_flags(features_path, metadata_path, analysis_ids=(201, 402, 501)):
    meta = pd.read_parquet(metadata_path)
    meta = meta[meta['analysisId'].isin(analysis_ids)]
    names = {covariate_id: name.split(': ')[-1].strip()
             for covariate_id, name in zip(meta['covariateId'], meta['covariateName'])}

    long = pd.read_parquet(features_path)
    long = long[long['covariateId'].isin(set(names))]
    wide = long.pivot(index='rowId', columns='covariateId', values='covariateValue')
    wide.columns = [names[c] for c in wide.columns]
    return wide.notna().T.groupby(level=0).max().T.astype(int)


def measurement_series(values, meta, concept, units):
    factors = {}
    for covariate_id, name in zip(meta['covariateId'], meta['covariateName']):
        if concept.lower() in name.lower():
            factor = next((f for unit, f in units.items() if unit in name.lower()), None)
            if factor is not None:
                factors[covariate_id] = factor
    rows = values[values['covariateId'].isin(factors)]
    return (rows['covariateValue'] * rows['covariateId'].map(factors)).groupby(rows['rowId']).median()


def class_members(concepts, classes):
    return {name: [c for c in concepts if any(re.search(p, c, re.I) for p in patterns)]
            for name, patterns in classes.items()}


def draw_dependence(ax, values, shap_values, y, k, highlight):
    if highlight:
        others = y.values != k
        ax.scatter(values[others], shap_values[others], s=11, color=shade(MUTED, .62),
                   alpha=.85, edgecolor=SURFACE, linewidth=.3, zorder=2)
        members = y.values == k
        ax.scatter(values[members], shap_values[members], s=15, color=CLUSTER_COLORS[k],
                   alpha=.75, edgecolor=SURFACE, linewidth=.4, zorder=3)
    else:
        for c in range(K):
            m = y.values == c
            ax.scatter(values[m], shap_values[m], s=13, color=CLUSTER_COLORS[c], alpha=.6,
                       edgecolor=SURFACE, linewidth=.4,
                       label=cluster_label(c) if k == 0 else None)
    ax.axhline(0, color=MUTED, lw=.9, zorder=1)


def ovr_roc(y_true, proba):
    onehot = label_binarize(y_true, classes=range(K))
    if K == 2:
        onehot = np.column_stack([1 - onehot[:, 0], onehot[:, 0]])
    curves, scores = {}, {}
    for k in range(K):
        fpr, tpr, _ = roc_curve(onehot[:, k], proba[:, k])
        curves[k], scores[k] = (fpr, tpr), auc(fpr, tpr)
    fpr, tpr, _ = roc_curve(onehot.ravel(), proba.ravel())
    curves['micro'], scores['micro'] = (fpr, tpr), auc(fpr, tpr)
    if K == 2:
        scores['macro'] = roc_auc_score(y_true, proba[:, 1])
    else:
        scores['macro'] = roc_auc_score(y_true, proba, multi_class='ovr', average='macro',
                                        labels=range(K))
    return curves, scores


def cluster_label(k):
    return CLUSTER_LABELS[k]


def pretty_feature(name, width=44):
    label = re.sub(r'\s*\(.*\)$', '', pretty(name, 999))
    label = re.sub(r'\[[^\]]*\]', '', label)
    for prop in LOINC_PROPERTIES:
        label = label.replace(prop, '')
    label = re.split(r'\s+in\s+(?:Serum|Plasma|Blood)', label)[0]
    label = label.title()
    return label if len(label) <= width else label[:width - 1] + '…'


def wrap_label(name, width=26, per_line=14):
    label = re.sub(r'\s*\(.*\)$', '', re.sub(r'\[[^\]]*\]', '', pretty(name, 999)))
    label = re.split(r'\s+in\s+(?:Serum|Plasma|Blood)', ' '.join(label.split()))[0]
    if len(label) > width:
        label = label[:width - 1] + '…'
    return '\n'.join(textwrap.wrap(label, per_line, break_long_words=False))


def axis_label(feature, scales, base='Robust z-score'):
    unit = pretty_unit(feature) if has_scale(feature, scales) else ''
    return f'{base} ({unit})' if unit else base


def tick_labels(ticks, feature, scales):
    if not has_scale(feature, scales):
        return [f'{z:g}' for z in ticks]
    return [f'{z:g}\n{raw_value(z, feature, scales)}' for z in ticks]


def pretty(name, width=52):
    label = name[5:] if name.startswith('prof_') else name
    label = label.replace('_', ' ')
    return label if len(label) <= width else label[:width - 1] + '…'


def pretty_unit(name):
    match = re.search(r'\(([^()]*(?:\([^()]*\))?[^()]*)\)$', name)
    if not match:
        natural = re.search(r'_in_([A-Za-z]+)$', name)
        return natural.group(1) if natural else ''
    return '/'.join(UNIT_TOKENS.get(part, part.replace('_', ' '))
                    for part in match.group(1).split('_per_'))


def has_scale(feature, scales):
    return scales is not None and feature in scales.index


def raw_value(z, feature, scales):
    center, spread, log = scales.loc[feature, ['center', 'spread', 'log']]
    value = center + z * spread
    return compact_value(10 ** value if log else value)


def compact_value(value):
    if abs(value) >= 1e6:
        return f'{value / 1e6:,.3g}M'
    if abs(value) >= 1e4:
        return f'{value / 1e3:,.3g}k'
    return format_value(value)


def format_value(value):
    return f'{value:,.0f}' if abs(value) >= 100 else f'{value:,.3g}'


def slug(text):
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')


def shade(color, amount):
    rgb = np.array(mpl.colors.to_rgb(color))
    return tuple(rgb + (1 - rgb) * amount)


def ridge_density(v, cut=3.0, tails=(True, True), points=512, bw_adjust=0.9):
    kde = gaussian_kde(v)
    kde.set_bandwidth(kde.factor * bw_adjust)
    bandwidth = float(np.sqrt(kde.covariance.squeeze()))
    x = np.linspace(v.min() - cut * bandwidth * tails[0],
                    v.max() + cut * bandwidth * tails[1], points)
    return x, kde(x)


def observed_values(feats, mask, y, feature, k):
    rows = (y.values == k) & (mask[feature].reindex(feats.index).values == 1)
    return feats.loc[rows, feature].values


def span(v, coverage=1.0, pad=0.06, reach=0.25):
    tail = (1 - coverage) / 2
    lo, hi = np.quantile(v, [tail, 1 - tail])
    width = hi - lo
    lo = v.min() if lo - v.min() <= reach * width else lo
    hi = v.max() if v.max() - hi <= reach * width else hi
    return lo - pad * width, hi + pad * width


def windowed(v, lo, hi):
    return v[(v >= lo) & (v <= hi)]


def style_radar(ax, angles, conditions, ticks):
    ax.set_yticks(ticks)
    ax.set_yticklabels([])
    ax.set_xticks(angles)
    ax.set_xticklabels([title_label(wrap_label(c)) for c in conditions], fontsize='medium')
    ax.tick_params(axis='x', pad=7, colors=INK)
    ax.grid(color=GRID, lw=0.8)
    ax.spines['polar'].set_visible(False)
    label_rings(ax, np.pi / len(angles), ticks)


def label_rings(ax, angle, ticks):
    for tick in ticks:
        ax.text(angle, tick, f'{tick:g}', ha='center', va='center', fontsize='x-small',
                color='black', fontweight='bold', zorder=6,
                bbox=dict(facecolor=SURFACE, edgecolor='none', alpha=0.85, pad=1))


def ring_angles(n):
    return np.linspace(0, 2 * np.pi, n, endpoint=False)


def radial_limits(values, hole=0.30):
    limit = max(1.0, np.max(np.abs(values)) * 1.25)
    return -limit * (1 + hole), limit, ratio_rings(limit)


def ratio_rings(limit):
    return [t for t in np.arange(-4, 4.01, 1) if abs(t) <= limit]


def scale_title(scale):
    return 'Log2 Odds Ratio vs Rest' if scale == 'odds' else 'Log2 Enrichment vs Cohort Rate'


def title_label(text):
    return re.sub(r'[A-Za-z][\w\-]*',
                  lambda m: m.group(0) if any(c.isupper() for c in m.group(0))
                  else m.group(0).capitalize(), text)


def log_lift(rate, cohort):
    ratio = (rate / cohort.replace(0, np.nan)).fillna(1.0)
    return np.log2(ratio.clip(1 / LIFT_CLIP, LIFT_CLIP))


def log_odds_ratio(x, member, correction=0.5):
    carriers, others = x[member].sum(), x[~member].sum()
    odds = ((carriers + correction) / (member.sum() - carriers + correction)
            / ((others + correction) / ((~member).sum() - others + correction)))
    return np.log2(odds.clip(1 / ODDS_CLIP, ODDS_CLIP))


K = 4
CLUSTER_LABELS = [f"Cluster {k + 1}" for k in range(K)]
CLUSTER_COLORS = ["#568f8b", '#cd7e59', '#d15252', '#ddb247',
                  '#6f7fb8', '#8e6aa8', '#7a9a4f', '#b06f8e']
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#898781', '#e1e0d9', '#fcfcfb'
NOTE = '#454440'
SEQUENTIAL = LinearSegmentedColormap.from_list(
    'seq', ['#fcfcfb', '#cde2fb', '#86b6ef', '#3987e5', '#256abf', '#104281'])
FONT = ['Helvetica', 'Arial', 'DejaVu Sans']

BANDS = ((0.95, 0.62), (0.80, 0.34), (0.50, 0.0))
LIFT_CLIP = 4.0
ODDS_CLIP = 16.0
RIDGE_XLIM = (-2, 3)

LOINC_PROPERTIES = ('Mass/volume', 'Moles/volume', 'Units/volume', 'Number/volume',
                    'Enzymatic activity/volume', 'Volume Rate/Area', 'Mass Ratio', '#/volume')

UNIT_TOKENS = {
    'picogram': 'pg', 'nanogram': 'ng', 'microgram': 'µg', 'milligram': 'mg', 'gram': 'g',
    'milliliter': 'mL', 'deciliter': 'dL', 'microliter': 'µL', 'liter': 'L',
    'millimole': 'mmol', 'nanomole': 'nmol', 'milliequivalent': 'mEq',
    'international_unit': 'IU', 'micro-international_unit': 'µIU', 'microunit': 'µU',
    'unit': 'U', 'minute': 'min', 'percent': '%', 'thousand': '10³',
    'millimeter_mercury_column': 'mmHg', 'inch_(US)': 'in', 'ounce_(avoirdupois)': 'oz',
    'gram_of_creatinine': 'g creatinine', '1.73_square_meter': '1.73 m²',
}

DRUG_CLASSES = {
    'Loop diuretics': ['furosemide', 'bumetanide', 'torsemide', 'ethacrynic'],
    'Thiazide diuretics': ['hydrochlorothiazide', 'chlorothiazide', 'chlorthalidone',
                           'metolazone', 'indapamide'],
    'Aldosterone antagonists': ['spironolactone', 'eplerenone', 'finerenone'],
    'Beta blockers': ['metoprolol', 'carvedilol', 'atenolol', 'bisoprolol', 'propranolol',
                      'labetalol', 'nebivolol', 'sotalol', 'nadolol', 'betaxolol', 'esmolol',
                      'timolol'],
    'ACE inhibitors': ['lisinopril', 'enalapril', 'ramipril', 'captopril', 'benazepril',
                       'quinapril', 'fosinopril', 'perindopril'],
    'ARBs': ['losartan', 'valsartan', 'olmesartan', 'telmisartan', 'candesartan', 'irbesartan'],
    'ARNI': ['sacubitril'],
    'Statins': [r'statin\b'],
    'SGLT2 inhibitors': [r'gliflozin\b'],
    'GLP-1 agonists': [r'glutide\b'],
    'Metformin': ['metformin'],
    'Antiplatelets': ['aspirin', 'clopidogrel', 'ticagrelor', 'prasugrel'],
    'Antiarrhythmics': ['amiodarone', 'dofetilide', 'flecainide', 'digoxin'],
    'Nitrates and vasodilators': ['isosorbide', 'hydralazine', 'sildenafil'],
    'Estrogen or progestin': ['estradiol', 'estrogens', 'levonorgestrel', 'norethindrone',
                              'norgestimate', 'norgestrel', 'desogestrel', 'drospirenone',
                              'etonogestrel', 'norelgestromin', 'medroxyprogesterone',
                              'megestrol', 'progesterone'],
}

CONDITION_CLASSES = {
    'Lupus': [r'lupus(?! anticoagulant)'],
    'Sickle cell disease': ['sickle cell', 'hemoglobin ss'],
    'Autoimmune disease': [r'lupus(?! anticoagulant)', r'\bSLE\b', 'rheumatoid arthritis',
                           'systemic sclerosis', r'sj\w*gren', 'autoimmune',
                           'antiphospholipid'],
    'Anemia': ['anemia'],
    'Chronic kidney disease': ['chronic kidney disease', r'end.stage renal'],
    'Type 2 diabetes': ['type 2 diabetes'],
    'Type 1 diabetes': ['type 1 diabetes'],
    'Ischemic heart disease': [r'^(?!congenital).*coronary', 'myocardial infarction',
                              'angina', 'preinfarction', 'ischemic heart', r'\bstent',
                              'atherectomy'],
    'Prior MI': [r'^(?!percutaneous).*myocardial infarction'],
    'Prior PCI': [r'percutaneous.*coronary'],
    'Heart failure': ['heart failure', 'cardiomyopathy'],
    'Atrial fibrillation or flutter': ['atrial fibrillation', 'atrial flutter'],
    'Systemic hypertension': [r'^(?!.*pulmonary).*hypertension', 'hypertensive'],
    'Pulmonary hypertension': ['pulmonary hypertension'],
    'Hypertensive Pregnancy': ['eclampsia',
                                'maternal'],
    'Obesity': ['obesity', 'obese'],
    'Asthma or COPD': ['asthma', 'chronic obstructive'],
    'Thyroid disease': ['thyroid'],
}
