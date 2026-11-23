"""Saved scientific figures for the V1 feature-return study."""

import numpy as np
from build_friday_dataset import plt


def draw_figures(estimates, groups, weekly, ablations, figures, labels, features):
    count = len(features)
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 2 + count / 2), layout="constrained", sharey=True
    )
    for ax, horizon in zip(axes, ("daily", "weekly"), strict=True):
        for period, offset, color in (
            ("development", -0.1, "#71879b"),
            ("validation", 0.1, "#0c7c86"),
        ):
            rows = (
                estimates.loc[
                    (estimates.metric == "ic")
                    & (estimates.block == 4)
                    & (estimates.horizon == horizon)
                    & (estimates.period == period)
                ]
                .set_index("feature")
                .reindex(features)
            )
            y = np.arange(count) + offset
            ax.hlines(y, rows.ci_low, rows.ci_high, color=color, lw=2)
            ax.scatter(
                rows["mean"],
                y,
                color=color,
                label=f"{period.title()} ({'2017–2024' if period == 'development' else '2025'})",
            )
        ax.axvline(0, color="#a5a5a5", lw=1)
        ax.set(
            title=f"Next {'session' if horizon == 'daily' else 'Friday'} close return",
            xlabel="Mean weekly Spearman IC",
        )
        ax.set_yticks(np.arange(count), [labels[f] for f in features])
    axes[0].invert_yaxis()
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle("Which features relate to future returns?", fontsize=17)
    fig.savefig(figures / "information_coefficients.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(count, 2, figsize=(11, count * 2.6), layout="constrained")
    for i, feature in enumerate(features):
        for j, horizon in enumerate(("daily", "weekly")):
            ax = axes[i, j]
            for period, color in (
                ("development", "#71879b"),
                ("validation", "#0c7c86"),
            ):
                rows = groups.loc[
                    (groups.feature == feature)
                    & (groups.horizon == horizon)
                    & (groups.period == period)
                ]
                ax.errorbar(
                    rows.group,
                    rows["mean"] * 100,
                    yerr=[
                        (rows["mean"] - rows.ci_low) * 100,
                        (rows.ci_high - rows["mean"]) * 100,
                    ],
                    color=color,
                    marker="o",
                    ms=3,
                    capsize=2,
                    label=period.title(),
                )
            ax.axhline(0, color="#c0c0c0", lw=0.7)
            ax.set(
                title=f"{labels[feature]} · {horizon}",
                xticks=range(1, 6),
                xlabel="Feature percentile group (low → high)",
                ylabel="Mean future return (%)",
            )
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Feature groups and subsequent returns\nTies stay together; unavailable groups remain blank",
        fontsize=16,
    )
    fig.savefig(figures / "feature_groups.png", dpi=150)
    plt.close(fig)

    yearly = (
        weekly.assign(year=weekly.friday.dt.year)
        .groupby(["feature", "horizon", "year"])
        .ic.mean()
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 1.5 + count / 2), layout="constrained", sharey=True
    )
    for ax, horizon in zip(axes, ("daily", "weekly"), strict=True):
        table = yearly.xs(horizon, level="horizon").unstack("year").reindex(features)
        image = ax.imshow(table, vmin=-0.15, vmax=0.15, cmap="RdBu", aspect="auto")
        for i, j in np.ndindex(table.shape):
            ax.text(
                j, i, f"{table.iat[i, j]:+.2f}", ha="center", va="center", fontsize=7
            )
        ax.set_xticks(range(len(table.columns)), table.columns, rotation=45)
        ax.set_yticks(range(count), [labels[f] for f in features])
        ax.set_title(horizon.title())
    fig.colorbar(image, ax=axes, label="Mean weekly Spearman IC", shrink=0.75)
    fig.suptitle("Does the relationship persist across years?", fontsize=17)
    fig.savefig(figures / "yearly_ic.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 2 + count / 2), layout="constrained", sharey=True
    )
    for ax, metric in zip(axes, ("mae", "basket_return"), strict=True):
        for horizon, offset, color in (
            ("daily", -0.1, "#71879b"),
            ("weekly", 0.1, "#0c7c86"),
        ):
            rows = (
                ablations.loc[
                    (ablations.metric == metric)
                    & (ablations.block == 4)
                    & (ablations.horizon == horizon)
                ]
                .set_index("feature")
                .reindex(features)
            )
            y = np.arange(count) + offset
            ax.hlines(y, rows.ci_low * 100, rows.ci_high * 100, color=color, lw=2)
            ax.scatter(rows["mean"] * 100, y, color=color, label=horizon.title())
        ax.axvline(0, color="#999999", lw=1)
        ax.set_yticks(np.arange(count), [labels[f] for f in features])
        ax.set(
            title="Forecast MAE improvement"
            if metric == "mae"
            else "Top-15 gross return improvement",
            xlabel="Percentage points; positive favors keeping the feature",
        )
    axes[0].invert_yaxis()
    axes[1].legend()
    fig.suptitle(
        f"Does each feature help the {count}-input Ridge model?\n2025 monthly walk-forward, paired 95% intervals",
        fontsize=16,
    )
    fig.savefig(figures / "ablation.png", dpi=160)
    plt.close(fig)
