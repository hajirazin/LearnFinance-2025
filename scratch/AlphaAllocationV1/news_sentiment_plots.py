"""Distribution, technical quality, and conditional-return figures."""

import numpy as np
import pandas as pd
from build_friday_dataset import plt
from news_sentiment_analysis import THRESHOLD_KEYS, THRESHOLDS

COLORS = {"positive": "#22866c", "neutral": "#82909d", "negative": "#b8443c"}


def draw_distributions(articles, archived, distributions, figures):
    valid = archived.news_status.isin(["complete", "verified_empty"])
    with_news = archived.news_status.eq("complete")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    bins = np.linspace(-1, 1, 61)
    for ax, values, title, color in (
        (
            axes[0, 0],
            articles.sentiment_score,
            "Unique article scores · full archive",
            "#297c91",
        ),
        (
            axes[0, 1],
            archived.loc[with_news, "news_sentiment"],
            "Weekly stock sentiment · windows with articles",
            "#b77c24",
        ),
    ):
        ax.hist(
            values,
            bins=bins,
            weights=np.full(len(values), 100 / len(values)),
            color=color,
            alpha=0.85,
        )
        ax.axvline(0, color="#222222", lw=1)
        ax.axvline(-0.1, color="#b8443c", lw=1, ls="--", label="Score −0.10")
        ax.axvline(-0.5, color="#b8443c", lw=1, ls=":", label="Score −0.50")
        ax.set(
            title=title,
            xlabel="Sentiment score (−1 to +1)",
            ylabel="Share of observations in each bin (%)",
            xlim=(-1, 1),
        )
    axes[0, 0].legend(fontsize=8)
    share_empty = 100 * archived.loc[valid].news_status.eq("verified_empty").mean()
    axes[0, 1].text(
        0.02,
        0.96,
        f"{share_empty:.1f}% of valid stock-weeks have no articles.\nTheir neutral zeros are excluded from this histogram.",
        transform=axes[0, 1].transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    counts = articles.predicted_label.value_counts().reindex(
        ["positive", "neutral", "negative"], fill_value=0
    )
    ax = axes[1, 0]
    percentages = counts / counts.sum() * 100
    ax.bar(counts.index, percentages, color=[COLORS[x] for x in counts.index])
    for i, (n, pct) in enumerate(zip(counts, percentages, strict=True)):
        ax.text(i, pct + 1, f"{pct:.1f}%\n{n:,}", ha="center", va="bottom", fontsize=10)
    ax.set(
        title="FinBERT's winning article class",
        ylabel="Share of unique articles (%)",
        ylim=(0, percentages.max() + 10),
    )
    ax = axes[1, 1]
    displayed = distributions.iloc[[0, 2, 3]]
    x = np.arange(len(THRESHOLDS))
    for i, (_, row) in enumerate(displayed.iterrows()):
        ax.bar(
            x + (i - 1) * 0.25,
            [row[f"{key}_pct"] for key in THRESHOLD_KEYS],
            width=0.25,
            label=row.unit,
        )
    ax.set_xticks(x, ["< 0", "< −0.001\n(literal −0.1%)", "< −0.10", "< −0.50"])
    ax.set(
        title="How often are the scores negative?",
        ylabel="Share of each stated denominator (%)",
    )
    ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(
        "News sentiment distribution · 2016–2025\nA slightly negative score can still be a neutral-classified article",
        fontsize=16,
    )
    fig.savefig(figures / "news_distribution.png", dpi=160)
    plt.close(fig)


def draw_quality(articles, archived, quality, figures):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    years = archived.index.get_level_values("friday").year
    classes = archived.news_status.map(
        {
            "complete": "Articles",
            "verified_empty": "No articles",
            "missing_window": "Unknown coverage",
            "missing_revision": "Unknown coverage",
        }
    ).fillna("Unknown coverage")
    counts = pd.crosstab(years, classes).reindex(
        columns=["Articles", "No articles", "Unknown coverage"], fill_value=0
    )
    shares = counts.div(counts.sum(axis=1), axis=0) * 100
    ax = axes[0, 0]
    bottom = np.zeros(len(shares))
    for group, color in zip(
        shares.columns, ["#297c91", "#bfc7ce", "#b8443c"], strict=True
    ):
        ax.bar(shares.index, shares[group], bottom=bottom, label=group, color=color)
        bottom += shares[group].to_numpy()
    ax.set(
        title="Friday coverage · 356-stock cohort",
        ylabel="Stock-weeks (%)",
        xticks=shares.index,
        ylim=(0, 105),
    )
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8, loc="lower left")
    ax = axes[0, 1]
    annual = articles.groupby(articles.created_at.dt.year).size()
    ax.bar(annual.index, annual / 1000, color="#297c91")
    ax.set(
        title="Unique archived articles by publication year",
        ylabel="Articles (thousands)",
        xticks=annual.index,
    )
    ax.tick_params(axis="x", rotation=45)
    ax = axes[1, 0]
    for label, color in COLORS.items():
        values = articles.loc[articles.predicted_label.eq(label), "confidence"]
        ax.hist(
            values,
            bins=np.linspace(0.3, 1, 36),
            weights=np.full(len(values), 100 / len(values)),
            histtype="step",
            lw=1.5,
            label=label.title(),
            color=color,
        )
    ax.set(
        title="Model confidence · calibration unmeasured",
        xlabel="Largest predicted class probability",
        ylabel="Share within each class per bin (%)",
    )
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    n = quality["unique_articles_checked"]
    labels = [
        "Valid probability/score checks",
        "Blank source metadata",
        "Signed negative but neutral class",
    ]
    values = [
        100 * quality["valid_records"] / n,
        100 * quality["blank_source"] / n,
        100 * quality["negative_score_but_neutral_class"] / n,
    ]
    ax.barh(labels, values, color=["#22866c", "#b77c24", "#82909d"])
    ax.set(
        xlim=(0, 115),
        xlabel="Share of unique articles (%)",
        title="Technical evidence checks",
    )
    ax.invert_yaxis()
    for i, value in enumerate(values):
        ax.text(value + 1, i, f"{value:.1f}%", va="center")
    fig.suptitle(
        "News quality and coverage\nTechnical checks and model confidence; human-label accuracy is unmeasured",
        fontsize=16,
    )
    fig.savefig(figures / "news_quality.png", dpi=160)
    plt.close(fig)


def draw_effects(estimates, figures):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), layout="constrained", sharey=True)
    for ax, horizon, title in zip(
        axes,
        ("same_week", "daily", "weekly"),
        ("During the news week", "Next exchange-session close", "Next Friday close"),
        strict=True,
    ):
        for period, offset, color, name in (
            ("all", -0.1, "#71879b", "2016–2025"),
            ("validation", 0.1, "#0c7c86", "2025 exploratory"),
        ):
            rows = (
                estimates.loc[
                    (estimates.period == period)
                    & (estimates.horizon == horizon)
                    & (estimates.metric == "difference")
                    & (estimates.block == 4)
                ]
                .set_index("threshold")
                .reindex(THRESHOLDS)
            )
            y = np.arange(len(THRESHOLDS)) + offset
            ax.hlines(y, rows.ci_low * 100, rows.ci_high * 100, color=color, lw=2)
            ax.scatter(rows["mean"] * 100, y, color=color, label=name)
        ax.axvline(0, color="#999999", lw=1)
        ax.set(title=title, xlabel="Negative group minus neutral/positive group (pp)")
        ax.set_yticks(
            range(4),
            [
                "Score < 0",
                "Score < −0.001\n(literal −0.1%)",
                "Score < −0.10",
                "Score < −0.50",
            ],
        )
    axes[0].invert_yaxis()
    axes[2].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.suptitle(
        "Do negative-news stocks have different close returns?\nSame-Friday comparisons, neutral no-news included, paired 95% intervals",
        fontsize=16,
    )
    fig.savefig(figures / "negative_news_returns.png", dpi=160)
    plt.close(fig)
