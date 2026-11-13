"""Plot data coverage and Friday return diagnostics for the scratch datasets."""

import argparse
import json
from pathlib import Path
from statistics import NormalDist

import duckdb
import exchange_calendars as xcals
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.cbook import boxplot_stats
from matplotlib.dates import DateFormatter, YearLocator
from matplotlib.ticker import NullLocator, PercentFormatter

DATA = Path(__file__).resolve().parent / "data"
START = "2016-01-01"
END = "2025-12-31"
DEVELOPMENT_END = "2024-12-31"
COHORT_FRIDAY = "2016-01-08"
TOP_WHISKER_TICKERS = (
    "MSTR",
    "IMPUY",
    "SMCI",
    "TSLA",
    "VIPS",
    "AMD",
    "GFI",
    "COHR",
    "EDU",
    "MMYT",
)
MIN_WEEKS = 30
BOXPLOT_SYMBOLS = ["SPUS", "AAPL", "CVNA", "DHR", "KAOOY", "SXYAY"]
BLUE = "#267499"
RED = "#bd4f36"


def plot_coverage(close, volume, news, output):
    coverage = close.notna().groupby(close.index.year).mean()
    symbol_order = coverage.sum().sort_values(ascending=False, kind="stable").index
    fig = plt.figure(figsize=(12, 7), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=(1.15, 1))
    heatmap = fig.add_subplot(grid[:, 0])
    image = heatmap.imshow(
        coverage[symbol_order].T,
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap="Blues",
        interpolation="nearest",
    )
    heatmap.set_xticks(range(len(coverage)), coverage.index, rotation=40)
    heatmap.set_yticks(
        [0, 99, 199, 299, len(symbol_order) - 1], [1, 100, 200, 300, len(symbol_order)]
    )
    heatmap.set(
        title="Daily price coverage", ylabel="Roster securities, sorted by coverage"
    )
    colorbar = fig.colorbar(
        image, ax=heatmap, label="Share of exchange sessions with a bar"
    )
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1))

    zero_volume = volume.eq(0).sum(axis=1).groupby(volume.index.year).sum()
    bars = fig.add_subplot(grid[0, 1])
    bars.bar(zero_volume.index, zero_volume, color=RED)
    bars.set_xticks(zero_volume.index, zero_volume.index, rotation=40)
    bars.set(title="Zero-volume bars by year", ylabel="Daily bars")
    bars.grid(axis="y", alpha=0.25)

    histogram = fig.add_subplot(grid[1, 1])
    histogram.hist(
        news["weeks_with_articles"],
        bins=np.arange(0, 551, 25),
        color=BLUE,
        edgecolor="white",
    )
    histogram.set(
        title="News coverage across the selected securities",
        xlabel="Monday windows with articles (out of 521)",
        ylabel="Securities",
    )
    histogram.grid(axis="y", alpha=0.25)
    fig.suptitle(
        f"AlphaAllocationV2 | Data coverage for {len(symbol_order)} securities, 2016 to 2025",
        fontsize=16,
    )
    fig.savefig(output / "data_coverage.png", dpi=150)
    plt.close(fig)


def plot_outliers(weekly_returns, output):
    returns = weekly_returns.loc[weekly_returns.index <= DEVELOPMENT_END]
    eligible = returns.count().ge(MIN_WEEKS) & returns.std(ddof=1).gt(0)
    series = returns.loc[:, eligible]
    z = (series - series.mean()) / series.std(ddof=1)
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    three_sigma = z.abs().gt(3)
    boxplot_flag = series.lt(q1 - 1.5 * iqr) | series.gt(q3 + 1.5 * iqr)
    only_sigma = int((three_sigma & ~boxplot_flag).to_numpy().sum())
    only_iqr = int((boxplot_flag & ~three_sigma).to_numpy().sum())
    both = int((three_sigma & boxplot_flag).to_numpy().sum())
    z_values = z.stack().dropna().to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    histogram, qq, boxes, flags = axes.flat
    histogram.hist(z_values, bins=np.linspace(-10, 10, 101), log=True, color=BLUE)
    histogram.axvline(-3, color=RED, linestyle="--")
    histogram.axvline(3, color=RED, linestyle="--", label="±3 standard deviations")
    histogram.legend(frameon=False, loc="upper left", fontsize=9)
    histogram.set(
        title="Three-sigma check across securities",
        xlabel="Weekly return z-score within ticker",
        ylabel="Observations per bin (log scale)",
    )
    outside = int((np.abs(z_values) > 10).sum())
    histogram.text(
        0.99,
        0.95,
        f"{outside:,} beyond chart range",
        transform=histogram.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )

    edge = 0.5 / len(z_values)
    probabilities = np.r_[edge, np.linspace(0.001, 0.999, 399), 1 - edge]
    normal = NormalDist()
    theoretical = np.array([normal.inv_cdf(p) for p in probabilities])
    observed = np.quantile(z_values, probabilities)
    qq.plot(
        theoretical,
        theoretical,
        color="#7f8a92",
        linestyle="--",
        label="Normal reference",
    )
    qq.scatter(theoretical, observed, s=9, color=BLUE)
    qq.legend(frameon=False, loc="upper left", fontsize=9)
    qq.set(
        title="Normal Q-Q plot",
        xlabel="Expected quantile under a normal distribution",
        ylabel="Observed standardized return quantile",
    )

    sample = [symbol for symbol in BOXPLOT_SYMBOLS if symbol in series]
    boxes.boxplot(
        [100 * series[symbol].dropna() for symbol in sample],
        tick_labels=sample,
        orientation="horizontal",
        patch_artist=True,
        boxprops={"facecolor": "#b9daeb"},
        medianprops={"color": "#183e52"},
        flierprops={
            "marker": ".",
            "markersize": 4,
            "alpha": 0.65,
            "markeredgecolor": RED,
        },
    )
    boxes.set_xscale("symlog", linthresh=10)
    sample_returns = 100 * series[sample]
    ticks = [-50, -20, -10, 0, 10, 20, 50, 100, 200]
    low, high = sample_returns.min().min(), sample_returns.max().max()
    boxes.set_xticks([tick for tick in ticks if low <= tick <= high])
    boxes.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    boxes.xaxis.set_minor_locator(NullLocator())
    boxes.set(
        title=f"Weekly return box plots for {len(sample)} examples",
        xlabel="Return % (symmetric log scale)",
    )

    bars = flags.bar(
        ["Three sigma only", "1.5 IQR only", "Both"],
        [only_sigma, only_iqr, both],
        color=[BLUE, "#7967a9", RED],
    )
    flags.bar_label(bars, fmt=lambda count: f"{count:,.0f}", padding=4)
    flags.margins(y=0.15)
    flags.set(
        title="Outlier flags across all eligible histories",
        ylabel="Stock-weeks flagged",
    )
    for axis in axes.flat:
        axis.grid(alpha=0.18)
        axis.set_axisbelow(True)
    fig.suptitle(
        f"Friday close-to-close returns | {series.shape[1]} tickers, "
        f"{len(z_values):,} stock-weeks, 2016 to 2024",
        fontsize=15,
    )
    fig.savefig(output / "return_outliers.png", dpi=150)
    plt.close(fig)
    return {
        "all_development_labels": int(returns.count().sum()),
        "tickers_with_development_labels": int(returns.count().gt(0).sum()),
        "tickers_scored": series.shape[1],
        "stock_weeks_scored": len(z_values),
        "three_sigma_only": only_sigma,
        "iqr_only": only_iqr,
        "both": both,
    }


def summarize_return_flags(below, above):
    flags_by_symbol = (below | above).sum(axis=0)
    total = int(flags_by_symbol.sum())
    return {
        "below": int(below.to_numpy().sum()),
        "above": int(above.to_numpy().sum()),
        "total": total,
        "tickers_flagged": int(flags_by_symbol.gt(0).sum()),
        "tickers_flagged_more_than_once": int(flags_by_symbol.gt(1).sum()),
        "top_10_share_of_flags": float(flags_by_symbol.nlargest(10).sum() / total)
        if total
        else 0.0,
    }


def plot_pooled_friday_returns(weekly_returns, sessions, output):
    returns_pct = 100 * weekly_returns
    returns = returns_pct.stack().dropna().to_numpy()
    stats = boxplot_stats(returns, whis=1.5)[0]
    box_flags = summarize_return_flags(
        returns_pct.lt(stats["q1"]), returns_pct.gt(stats["q3"])
    )
    whisker_flags = summarize_return_flags(
        returns_pct.lt(stats["whislo"]), returns_pct.gt(stats["whishi"])
    )
    pooled_mean = float(returns.mean())
    pooled_std = float(returns.std(ddof=1))
    pooled_lower = pooled_mean - 3 * pooled_std
    pooled_upper = pooled_mean + 3 * pooled_std
    pooled_sigma_flags = summarize_return_flags(
        returns_pct.lt(pooled_lower), returns_pct.gt(pooled_upper)
    )
    ticker_stds = returns_pct.std(ddof=1)
    eligible = returns_pct.count().ge(MIN_WEEKS) & ticker_stds.gt(0)
    per_ticker_returns = returns_pct.loc[:, eligible]
    per_ticker_means = per_ticker_returns.mean()
    per_ticker_stds = ticker_stds.loc[eligible]
    per_ticker_sigma_flags = summarize_return_flags(
        per_ticker_returns.lt(per_ticker_means - 3 * per_ticker_stds),
        per_ticker_returns.gt(per_ticker_means + 3 * per_ticker_stds),
    )
    calendar_fridays = weekly_returns.index
    open_fridays = calendar_fridays.isin(sessions)
    friday_pairs = int((open_fridays[1:] & open_fridays[:-1]).sum())
    tickers = int(weekly_returns.notna().any(axis=0).sum())

    fig, ax = plt.subplots(figsize=(10, 4.3), layout="constrained")
    fig.suptitle("Friday close-to-Friday close returns", fontsize=20, fontweight="bold")
    ax.set_title(
        f"2016 to 2025  |  {friday_pairs} weeks  |  {tickers} tickers  |  "
        f"{len(returns):,} observed returns",
        fontsize=11,
        pad=14,
    )
    ax.bxp(
        [stats],
        orientation="horizontal",
        widths=0.32,
        patch_artist=True,
        boxprops={"facecolor": "#b9daeb", "edgecolor": BLUE},
        medianprops={"color": "#183e52", "linewidth": 1.7},
        flierprops={
            "marker": ".",
            "markersize": 4,
            "markeredgecolor": RED,
            "alpha": 0.35,
        },
    )
    ax.set_xscale("symlog", linthresh=10)
    ticks = [-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80, 200]
    low, high = returns.min(), returns.max()
    ax.set_xticks([tick for tick in ticks if low <= tick <= high])
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(0.62, 1.46)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", alpha=0.22)
    ax.set_axisbelow(True)
    ax.set_xlabel("Weekly adjusted-close return (linear from -10% to +10%)")
    outside_box = box_flags["total"]
    outside_whiskers = whisker_flags["total"]
    ax.text(
        0.02,
        0.93,
        f"Outside the box: {outside_box:,} ({outside_box / len(returns):.1%})\n"
        f"Beyond the whiskers: {outside_whiskers:,} ({outside_whiskers / len(returns):.1%})\n"
        f"Beyond pooled three sigma: {pooled_sigma_flags['total']:,} "
        f"({pooled_sigma_flags['total'] / len(returns):.1%})",
        transform=ax.transAxes,
        va="top",
        linespacing=1.5,
    )
    fig.savefig(output / "friday_return_boxplot_2016_2025.png", dpi=180)
    plt.close(fig)
    return {
        "calendar_fridays": len(calendar_fridays),
        "calendar_intervals": len(calendar_fridays) - 1,
        "exchange_fridays": int(open_fridays.sum()),
        "friday_pairs": friday_pairs,
        "tickers": tickers,
        "observed_returns": len(returns),
        "q1_pct": stats["q1"],
        "median_pct": stats["med"],
        "q3_pct": stats["q3"],
        "lower_whisker_pct": stats["whislo"],
        "upper_whisker_pct": stats["whishi"],
        "outside_box": box_flags,
        "beyond_whiskers": whisker_flags,
        "pooled_three_sigma": {
            **pooled_sigma_flags,
            "lower_pct": pooled_lower,
            "upper_pct": pooled_upper,
        },
        "per_ticker_three_sigma": {
            **per_ticker_sigma_flags,
            "tickers_scored": int(eligible.sum()),
            "returns_scored": int(per_ticker_returns.count().sum()),
            "minimum_weeks": MIN_WEEKS,
        },
    }


def usable_price(close, volume):
    return np.isfinite(close) & close.gt(0) & np.isfinite(volume) & volume.gt(0)


def plot_weekly_usable_universe(
    prices, calendar, sessions, output, *, audit_exclusions=True
):
    fridays = sessions[sessions.day_of_week == 4]
    friday_prices = prices.loc[prices["date"].isin(fridays)]
    price_ok = usable_price(friday_prices["close"], friday_prices["volume"])
    candidates = friday_prices.loc[price_ok, ["date", "symbol"]].copy()
    candidates["date"] = candidates["date"].dt.date
    close_times = pd.DatetimeIndex([calendar.session_close(day) for day in fridays])
    friday_windows = pd.DataFrame(
        {
            "date": fridays.date,
            "cutoff": close_times,
            "window_start": close_times.tz_convert("America/New_York")
            - pd.DateOffset(days=7),
        }
    )
    with duckdb.connect(str(DATA / "news/news_v1.duckdb"), read_only=True) as db:
        db.register("candidates", candidates)
        db.register("friday_windows", friday_windows)
        weekly = db.execute("""
            WITH coverage AS (
                SELECT f.date, c.symbol,
                    sum(epoch(least(f.cutoff, c.window_end_inclusive))
                      - epoch(greatest(f.window_start, c.window_start_exclusive)))
                        = epoch(f.cutoff) - epoch(f.window_start)
                    AND sum(c.future_revision_excluded_count) = 0 AS complete
                FROM friday_windows f
                JOIN news_coverage c ON c.window_start_exclusive < f.cutoff
                    AND c.window_end_inclusive > f.window_start
                WHERE c.status IN ('complete', 'verified_empty')
                GROUP BY f.date, c.symbol, f.cutoff, f.window_start
            ), article_versions AS (
                SELECT f.date, e.symbol, e.provider, e.provider_article_id,
                    bool_or(e.updated_at <= f.cutoff) AS available_at_close
                FROM friday_windows f
                JOIN news_events e ON e.created_at > f.window_start
                    AND e.created_at <= f.cutoff
                GROUP BY f.date, e.symbol, e.provider, e.provider_article_id
            ), article_availability AS (
                SELECT date, symbol, bool_and(available_at_close) AS all_versions_available
                FROM article_versions
                GROUP BY date, symbol
            )
            SELECT f.date, count(p.symbol) AS price_usable,
                count(p.symbol) FILTER (WHERE c.complete
                    AND coalesce(a.all_versions_available, true)) AS news_usable
            FROM friday_windows f
            LEFT JOIN candidates p USING (date)
            LEFT JOIN coverage c ON c.date = f.date AND c.symbol = p.symbol
            LEFT JOIN article_availability a ON a.date = f.date AND a.symbol = p.symbol
            GROUP BY f.date ORDER BY f.date
        """).df()
    weekly["date"] = pd.to_datetime(weekly["date"])
    weekly = weekly.set_index("date")
    if not weekly["news_usable"].le(weekly["price_usable"]).all():
        raise ValueError("Weekly counts must be nested")

    minimum_date = weekly["price_usable"].idxmin()
    symbols_with_prices = set(prices["symbol"])
    usable_on_minimum = set(
        candidates.loc[candidates["date"].eq(minimum_date.date()), "symbol"]
    )
    excluded = sorted(symbols_with_prices - usable_on_minimum)
    if audit_exclusions:
        write_friday_price_exclusions(friday_prices, fridays, excluded, output)

    plot_dates = pd.date_range(START, END, freq="W-FRI")
    chart = weekly.reindex(plot_dates)
    news_present = chart["news_usable"].gt(0)
    fig, ax = plt.subplots(figsize=(12, 5.1), layout="constrained")
    fig.suptitle(
        "Usable stock universe by trading Friday", fontsize=18, fontweight="bold"
    )
    ax.plot(
        chart.index,
        chart["price_usable"],
        color=BLUE,
        lw=2,
        label="Usable Friday price",
    )
    ax.plot(
        chart.index,
        chart["news_usable"].where(news_present),
        color="#f4a540",
        lw=1.5,
        linestyle="--",
        label="Also has usable news coverage",
    )
    ax.set(
        title="2016 to 2025  |  Gaps mark Friday holidays or missing news coverage",
        ylabel="Stocks",
        ylim=(0, 455),
        xlim=(plot_dates[0], plot_dates[-1]),
    )
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    ax.grid(axis="y", alpha=0.24)
    ax.legend(loc="lower center", ncols=2, frameon=False)
    fig.savefig(output / "usable_universe_by_week.png", dpi=180)
    plt.close(fig)
    return {
        "trading_fridays": len(weekly),
        "fridays_with_no_full_news_window": int(weekly["news_usable"].eq(0).sum()),
        "price_usable_min_median_max": [
            int(weekly["price_usable"].min()),
            float(weekly["price_usable"].median()),
            int(weekly["price_usable"].max()),
        ],
        "median_news_usable": float(weekly["news_usable"].median()),
        "minimum_price_universe_friday": minimum_date.date().isoformat(),
        "symbols_excluded_on_minimum_friday": excluded,
        "latest_friday": weekly.index[-1].date().isoformat(),
        "latest_counts": {name: int(count) for name, count in weekly.iloc[-1].items()},
    }


def write_friday_price_exclusions(friday_prices, fridays, symbols, output):
    dates_and_symbols = pd.MultiIndex.from_product(
        [fridays, symbols], names=["date", "symbol"]
    )
    bars = friday_prices.set_index(["date", "symbol"])[["close", "volume"]]
    audit = bars.reindex(dates_and_symbols)
    observed = audit.index.isin(bars.index)
    reason = np.select(
        [
            ~observed,
            ~np.isfinite(audit["close"]) | audit["close"].le(0),
            audit["volume"].eq(0),
            ~np.isfinite(audit["volume"]) | audit["volume"].lt(0),
        ],
        ["no price bar", "invalid close", "zero reported volume", "invalid volume"],
        default="usable",
    )
    audit["reason"] = reason
    audit.loc[audit["reason"].ne("usable")].reset_index().to_csv(
        output / "friday_price_exclusions.csv", index=False, date_format="%Y-%m-%d"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    plots = parser.add_mutually_exclusive_group()
    plots.add_argument("--full-period-boxplot", action="store_true")
    plots.add_argument("--weekly-universe", action="store_true")
    plots.add_argument(
        "--first-friday-cohort",
        action="store_true",
        help="Redraw diagnostics for securities with usable prices on January 8, 2016",
    )
    parser.add_argument(
        "--exclude-top-whisker-tickers",
        action="store_true",
        help="Remove the ten previously identified tickers from the first-Friday cohort",
    )
    args = parser.parse_args()
    if args.exclude_top_whisker_tickers and not args.first_friday_cohort:
        parser.error("--exclude-top-whisker-tickers requires --first-friday-cohort")
    output = DATA / "eda"
    output.mkdir(parents=True, exist_ok=True)
    roster = json.loads((DATA / "ohlcv_manifest.json").read_text())["universe"][
        "symbols"
    ]
    prices = pd.read_csv(DATA / "daily_ohlcv_2016_2025.csv", parse_dates=["date"])
    calendar = xcals.get_calendar("XNYS")
    sessions = calendar.sessions_in_range(START, END)
    if args.weekly_universe:
        summary = plot_weekly_usable_universe(prices, calendar, sessions, output)
        print(json.dumps(summary, indent=2))
        return
    close = prices.pivot(index="date", columns="symbol", values="close").reindex(
        index=sessions, columns=roster
    )
    volume = prices.pivot(index="date", columns="symbol", values="volume").reindex_like(
        close
    )
    if args.first_friday_cohort:
        price_ok = usable_price(close, volume)
        symbols = price_ok.columns[price_ok.loc[COHORT_FRIDAY]]
        output /= "first_friday_cohort"
        if args.exclude_top_whisker_tickers:
            excluded = pd.Index(TOP_WHISKER_TICKERS)
            missing = excluded.difference(symbols)
            if len(missing):
                raise ValueError(
                    f"Expected excluded tickers in the cohort: {list(missing)}"
                )
            symbols = symbols.difference(excluded)
            output /= "without_top_10"
        close = close.loc[:, symbols]
        volume = volume.loc[:, symbols]
        output.mkdir(parents=True, exist_ok=True)
        analysis_close = close.where(price_ok.loc[:, symbols])
    else:
        analysis_close = close
    friday_closes = analysis_close.reindex(pd.date_range(START, END, freq="W-FRI"))
    weekly_returns = friday_closes / friday_closes.shift(1) - 1
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
        }
    ):
        if args.full_period_boxplot:
            summary = plot_pooled_friday_returns(weekly_returns, sessions, output)
        else:
            with duckdb.connect(
                str(DATA / "news/news_v1.duckdb"), read_only=True
            ) as db:
                news = db.execute("""select symbol, count(*) filter(where article_count>0) weeks_with_articles
                                     from weekly_sentiment group by symbol""").df()
            assert set(news["symbol"]) == set(roster)
            news = news.loc[news["symbol"].isin(close.columns)]
            plot_coverage(close, volume, news, output)
            summary = plot_outliers(weekly_returns, output)
            if args.first_friday_cohort:
                summary = {
                    "cohort_friday": COHORT_FRIDAY,
                    "cohort_size": len(close.columns),
                    "development": summary,
                    "full_period": plot_pooled_friday_returns(
                        weekly_returns, sessions, output
                    ),
                }
                if args.exclude_top_whisker_tickers:
                    summary["excluded_tickers"] = list(TOP_WHISKER_TICKERS)
                    summary["weekly_universe"] = plot_weekly_usable_universe(
                        prices.loc[prices["symbol"].isin(symbols)],
                        calendar,
                        sessions,
                        output,
                        audit_exclusions=False,
                    )
                (output / "summary.json").write_text(
                    json.dumps(summary, indent=2) + "\n"
                )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
