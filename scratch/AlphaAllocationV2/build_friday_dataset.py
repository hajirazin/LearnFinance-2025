"""Build Friday features, verified return labels, and a development heatmap."""

import argparse
import json
from itertools import combinations_with_replacement

import duckdb
import exchange_calendars as xcals
import numpy as np
import pandas as pd
from audit_data import (
    COHORT_FRIDAY,
    DATA,
    DEVELOPMENT_END,
    END,
    START,
    plt,
    usable_price,
)
from brain_api.news.models import (
    CONFIDENCE_RECENCY_TAU_HOURS,
    NEWS_PROVIDER,
    NEWS_SCHEMA_VERSION,
    NEWS_SENTIMENT_MODEL,
    NEWS_SENTIMENT_REVISION,
)
from verify_friday_prices import EVIDENCE, fetch_evidence

OUTPUT = DATA / "friday"
MIN_RANK_STOCKS = 30
FEATURES = {
    "momentum_5d": "Momentum 5d",
    "momentum_20d": "Momentum 20d",
    "momentum_12_1": "Momentum 12-1",
    "volatility_20d": "Volatility 20d",
    "volatility_252d": "Volatility 252d",
    "relative_volume_20d": "Relative volume 20d",
    "news_sentiment": "News sentiment",
    "news_article_count": "News article count",
}
OHLCV_FEATURES = {
    "open": "Friday open",
    "high": "Friday high",
    "low": "Friday low",
    "close": "Friday close",
    "volume": "Friday volume",
}
PRICE_TOLERANCE = 0.01
LARGE_DAILY_MOVE = 0.40
FORTIVE_WEEK = (pd.Timestamp("2016-07-01"), "DHR")
ACTION_DATES = [
    ("KAOOY", "2018-05-03"),
    ("HTHIY", "2025-02-19"),
]


def load_prices():
    prices = pd.read_csv(
        DATA / "daily_ohlcv_2016_2025.csv",
        usecols=["date", "symbol", "close", "volume"],
        parse_dates=["date"],
    )
    calendar = xcals.get_calendar("XNYS")
    sessions = calendar.sessions_in_range(START, END)
    close = prices.pivot(index="date", columns="symbol", values="close").reindex(
        sessions
    )
    volume = prices.pivot(index="date", columns="symbol", values="volume").reindex_like(
        close
    )
    ok = usable_price(close, volume)
    symbols = ok.columns[ok.loc[COHORT_FRIDAY]]
    return calendar, close.where(ok).loc[:, symbols], volume.where(ok).loc[:, symbols]


def stack(frame):
    return frame.rename_axis(index="friday", columns="symbol").stack().sort_index()


def audit_candidates(close, friday_grid):
    friday_close = close.reindex(friday_grid)
    raw = stack(friday_close.shift(-1) / friday_close - 1).dropna()
    pooled = raw.sub(raw.mean()).abs().gt(3 * raw.std())
    groups = raw.groupby(level="symbol")
    per_stock = raw.sub(groups.transform("mean")).abs().gt(3 * groups.transform("std"))
    q1, q3 = raw.quantile([0.25, 0.75])
    iqr = q3 - q1
    whiskers = raw.lt(q1 - 1.5 * iqr) | raw.gt(q3 + 1.5 * iqr)
    audit = raw.loc[pooled | per_stock | whiskers].to_frame("yahoo_return")
    audit["pooled_three_sigma"] = pooled
    audit["ticker_three_sigma"] = per_stock
    audit["pooled_iqr"] = whiskers
    audit["end_friday"] = audit.index.get_level_values("friday") + pd.Timedelta(days=7)
    daily = stack(close / close.shift() - 1).dropna()
    daily = (
        daily.loc[daily.abs().ge(LARGE_DAILY_MOVE)].rename("yahoo_return").to_frame()
    )
    daily.index = daily.index.set_names(["session", "symbol"])
    daily["previous_session"] = (
        close.index.to_series()
        .shift()
        .reindex(daily.index.get_level_values("session"))
        .to_numpy()
    )
    return raw, audit, daily


def requested_bars(audit, daily):
    weeks = audit.reset_index()
    days = daily.reset_index()
    targets = pd.concat(
        [
            weeks[["symbol", "friday"]].rename(columns={"friday": "date"}),
            weeks[["symbol", "end_friday"]].rename(columns={"end_friday": "date"}),
            days[["symbol", "session"]].rename(columns={"session": "date"}),
            days[["symbol", "previous_session"]].rename(
                columns={"previous_session": "date"}
            ),
        ]
    )
    return targets.drop_duplicates()


def reference_returns(evidence, index, starts, ends):
    def reference_close(dates):
        keys = pd.MultiIndex.from_arrays([dates, index.get_level_values("symbol")])
        return evidence.reindex(keys).to_numpy()

    return reference_close(ends) / reference_close(starts) - 1


def verify_prices(audit, daily, close):
    payload = json.loads(EVIDENCE.read_text())
    ref = pd.DataFrame(
        payload["adjusted_closes"], columns=["symbol", "date", "close", "volume"]
    )
    ref["date"] = pd.to_datetime(ref["date"])
    reference = ref.loc[usable_price(ref["close"], ref["volume"])].set_index(
        ["date", "symbol"]
    )["close"]
    for frame, starts, ends in (
        (audit, audit.index.get_level_values("friday"), audit["end_friday"]),
        (daily, daily["previous_session"], daily.index.get_level_values("session")),
    ):
        frame["sip_adjusted_return"] = reference_returns(
            reference, frame.index, starts, ends
        )
        frame["status"] = np.select(
            [
                frame["sip_adjusted_return"].isna(),
                frame["yahoo_return"]
                .sub(frame["sip_adjusted_return"])
                .abs()
                .le(PRICE_TOLERANCE),
            ],
            ["unresolved_no_sip_close", "verified_against_sip"],
            default="unresolved_source_disagreement",
        )
    raw = {
        (symbol, day): value
        for symbol, day, value, volume in payload["fortive_raw_closes"]
        if value > 0 and volume > 0
    }
    dhr_start = raw["DHR", "2016-07-01"]
    corrected_week = (
        raw["DHR", "2016-07-08"] + 0.5 * raw["FTV", "2016-07-08"]
    ) / dhr_start - 1
    corrected_step = (
        raw["DHR", "2016-07-05"] + 0.5 * raw["FTV", "2016-07-05"]
    ) / dhr_start
    dhr_action = pd.Timestamp("2016-07-05")
    scale = close.loc[dhr_action, "DHR"] / (
        close.loc["2016-07-01", "DHR"] * corrected_step
    )
    # A constant factor preserves all return features strictly before the spinoff.
    adjusted = close.copy()
    adjusted.loc[adjusted.index < dhr_action, "DHR"] *= scale
    audit.loc[FORTIVE_WEEK, "status"] = "corrected_fortive_distribution"
    daily.loc[(dhr_action, "DHR"), "status"] = "corrected_fortive_distribution"
    for symbol, day in ACTION_DATES:
        affected = (
            (audit.index.get_level_values("symbol") == symbol)
            & (audit.index.get_level_values("friday") < pd.Timestamp(day))
            & audit["end_friday"].ge(day)
        )
        audit.loc[affected, "status"] = "unresolved_adr_ratio_change"
    return adjusted, corrected_week


def friday_news(fridays, calendar, symbols, *, db_path=DATA / "news/news_v1.duckdb"):
    cutoff = pd.DatetimeIndex([calendar.session_close(day) for day in fridays])
    windows = pd.DataFrame(
        {
            "friday": fridays.date,
            "cutoff": cutoff,
            "window_start": cutoff.tz_convert("America/New_York")
            - pd.DateOffset(days=7),
        }
    )
    with duckdb.connect(str(db_path), read_only=True) as db:
        db.register("friday_windows", windows)
        db.register("roster", pd.DataFrame({"symbol": symbols}))
        news = db.execute(
            """
            WITH coverage AS (
                SELECT f.friday, c.symbol,
                    sum(epoch(least(f.cutoff, c.window_end_inclusive))
                        - epoch(greatest(f.window_start, c.window_start_exclusive)))
                        = epoch(f.cutoff) - epoch(f.window_start)
                    AND sum(c.future_revision_excluded_count) = 0 AS complete
                FROM friday_windows f JOIN news_coverage c
                    ON c.window_start_exclusive < f.cutoff AND c.window_end_inclusive > f.window_start
                JOIN roster r USING (symbol)
                WHERE c.status IN ('complete', 'verified_empty')
                    AND c.provider = $provider AND c.schema_version = $schema
                    AND c.sentiment_model = $model AND c.sentiment_model_revision = $revision
                GROUP BY f.friday, c.symbol, f.cutoff, f.window_start
            ), versions AS (
                SELECT f.friday, e.symbol, e.provider_article_id, e.created_at, e.updated_at,
                    e.sentiment_score, e.updated_at <= f.cutoff AS available,
                    e.confidence * exp((epoch(e.created_at) - epoch(f.cutoff)) / (3600 * $tau)) AS weight
                FROM friday_windows f JOIN news_events e
                    ON e.created_at > f.window_start AND e.created_at <= f.cutoff
                JOIN roster r USING (symbol)
                WHERE e.provider = $provider AND e.schema_version = $schema
                    AND e.sentiment_model = $model AND e.sentiment_model_revision = $revision
                QUALIFY row_number() OVER (
                    PARTITION BY f.friday, e.symbol, e.provider_article_id
                    ORDER BY available DESC, e.updated_at DESC
                ) = 1
            ), aggregates AS (
                SELECT friday, symbol, bool_and(available) AS all_available, count(*) AS articles,
                    sum(sentiment_score * weight) / nullif(sum(weight), 0) AS sentiment,
                    max(updated_at) AS latest_revision
                FROM versions GROUP BY friday, symbol
            )
            SELECT f.friday, r.symbol, epoch(f.cutoff) AS cutoff_epoch,
                CASE WHEN NOT coalesce(c.complete, false) THEN 'missing_window'
                     WHEN a.articles IS NULL THEN 'verified_empty'
                     WHEN NOT a.all_available THEN 'missing_revision'
                     WHEN a.sentiment IS NULL THEN 'invalid_sentiment'
                     ELSE 'complete' END AS news_status,
                a.articles AS news_article_count, a.sentiment AS news_sentiment,
                epoch(a.latest_revision) AS latest_news_revision_epoch
            FROM friday_windows f CROSS JOIN roster r
            LEFT JOIN coverage c ON c.friday = f.friday AND c.symbol = r.symbol
            LEFT JOIN aggregates a ON a.friday = f.friday AND a.symbol = r.symbol
        """,
            {
                "provider": NEWS_PROVIDER,
                "schema": NEWS_SCHEMA_VERSION,
                "model": NEWS_SENTIMENT_MODEL,
                "revision": NEWS_SENTIMENT_REVISION,
                "tau": CONFIDENCE_RECENCY_TAU_HOURS,
            },
        ).df()
    news["friday"] = pd.to_datetime(news["friday"])
    for field in ("news_article_count", "news_sentiment"):
        news.loc[news["news_status"].eq("verified_empty"), field] = 0
        news.loc[~news["news_status"].isin(["complete", "verified_empty"]), field] = (
            np.nan
        )
    news["cutoff_utc"] = pd.to_datetime(news.pop("cutoff_epoch"), unit="s", utc=True)
    news["latest_news_revision_utc"] = pd.to_datetime(
        news.pop("latest_news_revision_epoch").where(
            news["news_status"].eq("complete")
        ),
        unit="s",
        utc=True,
    )
    return news.set_index(["friday", "symbol"]).sort_index()


def build_panel(close, volume, adjusted, fridays, raw, audit, daily, fortive_return):
    daily_logs = np.log(adjusted).diff()
    breaks = daily.loc[daily["status"].str.startswith("unresolved")].index
    for session, symbol in breaks:
        daily_logs.loc[session, symbol] = np.nan
    for symbol, day in ACTION_DATES:
        daily_logs.loc[day, symbol] = np.nan

    panel = pd.DataFrame(
        {
            "adjusted_close_yahoo": stack(close.loc[fridays]),
            "price_usable_at_cutoff": stack(close.loc[fridays].notna()),
        }
    )
    for name, lookback, skip in (
        ("momentum_5d", 5, 0),
        ("momentum_20d", 20, 0),
        ("momentum_12_1", 231, 21),
    ):
        trailing = daily_logs.rolling(lookback).sum().shift(skip)
        panel[name] = stack(np.expm1(trailing).loc[fridays])
    for lookback in (20, 252):
        volatility = daily_logs.rolling(lookback).std(ddof=1) * np.sqrt(252)
        panel[f"volatility_{lookback}d"] = stack(volatility.loc[fridays])
    relative_volume = volume / volume.shift().rolling(20).median()
    panel["relative_volume_20d"] = stack(relative_volume.loc[fridays])
    price_features = list(FEATURES)[:-2]
    panel.loc[~panel["price_usable_at_cutoff"], price_features] = np.nan

    next_friday = panel.index.get_level_values("friday") + pd.Timedelta(days=7)
    panel["next_friday"] = next_friday
    panel["raw_yahoo_forward_return"] = raw
    panel["forward_return"] = raw
    panel["label_status"] = np.select(
        [
            next_friday > pd.Timestamp(END),
            ~next_friday.isin(fridays),
            raw.reindex(panel.index).isna(),
        ],
        ["outside_archive", "friday_exchange_holiday", "unusable_price_endpoint"],
        default="observed_yahoo",
    )
    panel.loc[audit.index, "label_status"] = audit["status"]
    unresolved = panel["label_status"].str.startswith("unresolved")
    panel.loc[unresolved, "forward_return"] = np.nan
    # A weekly return can hide a discontinuity if another move offsets it.
    for session, symbol in breaks:
        affected = (
            (panel.index.get_level_values("symbol") == symbol)
            & (panel.index.get_level_values("friday") < session)
            & panel["next_friday"].ge(session)
            & panel["forward_return"].notna()
        )
        panel.loc[affected, "label_status"] = "unresolved_daily_break"
        panel.loc[affected, "forward_return"] = np.nan
    panel.loc[FORTIVE_WEEK, "forward_return"] = fortive_return
    panel["forward_return_pct"] = 100 * panel["forward_return"]
    returns_by_friday = panel.groupby(level="friday")["forward_return"]
    panel["realized_return_rank"] = returns_by_friday.rank(
        method="average", ascending=False, na_option="keep"
    )
    panel["realized_rank_universe_size"] = returns_by_friday.transform("count")
    return panel


def weekly_spearman(data, left, right):
    paired = data[[left, right]].dropna()
    paired.columns = ["left", "right"]
    ranks = paired.groupby(level="friday").rank(method="average")
    centered = ranks - ranks.groupby(level="friday").transform("mean")
    products = (
        pd.DataFrame(
            {
                "xy": centered["left"] * centered["right"],
                "xx": centered["left"] ** 2,
                "yy": centered["right"] ** 2,
            }
        )
        .groupby(level="friday")
        .sum()
    )
    n = paired.groupby(level="friday").size()
    denom = np.sqrt(products["xx"] * products["yy"])
    rho = products["xy"].div(denom).where(n.ge(MIN_RANK_STOCKS) & denom.gt(0)).dropna()
    return pd.DataFrame({"spearman": rho, "paired_stocks": n.reindex(rho.index)})


def correlation_heatmap(panel, features=FEATURES, *, filename="feature_spearman"):
    keys = list(features)
    dates = panel.index.get_level_values("friday")
    data = panel.loc[panel["price_usable_at_cutoff"] & (dates <= DEVELOPMENT_END), keys]
    rows = []
    for left, right in combinations_with_replacement(keys, 2):
        weekly = weekly_spearman(data, left, right)
        rows.append(
            {
                "left": left,
                "right": right,
                "mean_weekly_spearman": weekly["spearman"].mean(),
                "weeks": len(weekly),
                "median_paired_stocks": weekly["paired_stocks"].median(),
            }
        )
    results = pd.DataFrame(rows)
    matrix = pd.DataFrame(np.nan, index=keys, columns=keys)
    for row in results.itertuples(index=False):
        matrix.loc[row.left, row.right] = matrix.loc[row.right, row.left] = (
            row.mean_weekly_spearman
        )

    extra = len(keys) - len(FEATURES)
    fig, ax = plt.subplots(
        figsize=(11 + 0.45 * extra, 8 + 0.45 * extra), layout="constrained"
    )
    fig.suptitle(
        "How much do the Friday features overlap?", fontsize=19, fontweight="bold"
    )
    image = ax.imshow(matrix, cmap="RdBu", vmin=-1, vmax=1)
    ax.set_title(
        "Mean of weekly cross-sectional Spearman correlations | 2016 to 2024", pad=13
    )
    labels = list(features.values())
    ax.set_xticks(range(len(keys)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(keys)), labels)
    for i in range(len(keys)):
        for j in range(len(keys)):
            value = matrix.iloc[i, j]
            ax.text(
                j,
                i,
                f"{value:+.2f}",
                ha="center",
                va="center",
                color="white" if abs(value) >= 0.60 else "#18283b",
                fontsize=11,
            )
    fig.colorbar(image, ax=ax, label="Spearman rank correlation", shrink=0.85)
    fig.savefig(OUTPUT / f"{filename}_heatmap.png", dpi=180)
    plt.close(fig)
    results.to_csv(OUTPUT / f"{filename}.csv", index=False)


def join_friday_ohlcv(panel):
    bars = pd.read_csv(
        DATA / "daily_ohlcv_2016_2025.csv",
        usecols=["date", "symbol", *OHLCV_FEATURES],
        parse_dates=["date"],
    ).rename(columns={"date": "friday"})
    fridays = panel.index.get_level_values("friday").unique()
    bars = bars.loc[bars["friday"].isin(fridays)].set_index(["friday", "symbol"])
    panel = panel.join(bars, validate="one_to_one")
    eligible = panel.loc[panel["price_usable_at_cutoff"]]
    if not np.allclose(
        eligible["close"], eligible["adjusted_close_yahoo"], rtol=1e-12, atol=0
    ):
        raise ValueError("Friday OHLCV closes do not match the saved panel")
    return panel


def plot_ohlcv_heatmap():
    with duckdb.connect() as db:
        panel = db.read_parquet(str(OUTPUT / "friday_dataset.parquet")).df()
    panel = join_friday_ohlcv(panel.set_index(["friday", "symbol"]).sort_index())
    correlation_heatmap(
        panel, FEATURES | OHLCV_FEATURES, filename="feature_spearman_with_ohlcv"
    )


def feature_vs_realized_rank(panel):
    data = panel.loc[
        panel["next_friday"].le(DEVELOPMENT_END), [*FEATURES, "realized_return_rank"]
    ].copy()
    # Rank 1 is best. Reverse the sign so positive IC indicates better returns.
    data["rank_score"] = -data.pop("realized_return_rank")
    weekly = pd.concat(
        {feature: weekly_spearman(data, feature, "rank_score") for feature in FEATURES},
        names=["feature"],
    )
    weekly.to_csv(OUTPUT / "weekly_feature_rank_ic.csv")
    summary = (
        weekly.groupby(level="feature")
        .agg(
            mean_rank_ic=("spearman", "mean"),
            weeks=("spearman", "size"),
            median_paired_stocks=("paired_stocks", "median"),
        )
        .sort_values("mean_rank_ic")
    )
    summary.to_csv(OUTPUT / "feature_rank_ic.csv")

    fig, ax = plt.subplots(figsize=(9, 5.1), layout="constrained")
    values = summary["mean_rank_ic"]
    bars = ax.barh(
        [FEATURES[feature] for feature in summary.index],
        values,
        color=np.where(values.ge(0), "#267499", "#bd4f36"),
    )
    ax.axvline(0, color="#52616e", linewidth=0.9)
    ax.bar_label(bars, labels=[f"{value:+.3f}" for value in values], padding=5)
    ax.margins(x=0.26)
    ax.grid(axis="x", alpha=0.22)
    ax.set_axisbelow(True)
    ax.set_xlabel(
        "Mean weekly Spearman rank correlation (positive favors higher feature values)"
    )
    fig.suptitle(
        "Friday features vs. realized next-Friday rank", fontsize=17, fontweight="bold"
    )
    ax.set_title("356-stock cohort | 2016 to 2024", fontsize=11)
    fig.savefig(OUTPUT / "feature_vs_realized_rank.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fetch-verification", action="store_true")
    mode.add_argument("--ohlcv-heatmap", action="store_true")
    args = parser.parse_args()
    if args.ohlcv_heatmap:
        plot_ohlcv_heatmap()
        return
    calendar, close, volume = load_prices()
    grid = pd.date_range(START, END, freq="W-FRI")
    fridays = grid.intersection(close.index)
    raw, audit, daily = audit_candidates(close, grid)
    targets = requested_bars(audit, daily)
    if args.fetch_verification:
        fetch_evidence(targets)
    adjusted, corrected_week = verify_prices(audit, daily, close)
    panel = build_panel(
        close, volume, adjusted, fridays, raw, audit, daily, corrected_week
    )
    panel = panel.join(
        friday_news(fridays, calendar, close.columns), validate="one_to_one"
    )
    if (
        len(panel) != len(fridays) * len(close.columns)
        or panel["news_status"].isna().any()
    ):
        raise ValueError("Incomplete Friday panel")
    if panel["latest_news_revision_utc"].gt(panel["cutoff_utc"]).any():
        raise ValueError("A news revision is newer than its Friday cutoff")
    OUTPUT.mkdir(exist_ok=True)
    staged = OUTPUT / "friday_dataset.tmp.parquet"
    with duckdb.connect() as db:
        db.register("friday_panel", panel.reset_index())
        db.execute(
            "COPY (SELECT * FROM friday_panel ORDER BY friday, symbol) TO ? (FORMAT PARQUET)",
            [str(staged)],
        )
    staged.replace(OUTPUT / "friday_dataset.parquet")
    audit["forward_return"] = panel["forward_return"]
    audit.to_csv(OUTPUT / "flagged_weekly_returns.csv")
    daily.to_csv(OUTPUT / "flagged_daily_returns.csv")
    correlation_heatmap(panel)
    feature_vs_realized_rank(panel)
    summary = {
        "rows": len(panel),
        "fridays": len(fridays),
        "stocks": len(close.columns),
        "label_status": panel["label_status"].value_counts().to_dict(),
        "news_status": panel["news_status"].value_counts().to_dict(),
        "feature_nonmissing": panel[list(FEATURES)].count().to_dict(),
        "verified_fortive_week_return": corrected_week,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
