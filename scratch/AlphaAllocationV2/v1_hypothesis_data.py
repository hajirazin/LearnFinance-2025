"""Build daily windows for the V1 factor comparisons."""

import duckdb
import exchange_calendars as xcals
import numpy as np
import pandas as pd
from build_dataset import DATA, ROSTER, load_roster
from build_friday_dataset import friday_news
from build_v1_holdout_data import OUTPUT as PRICE_OUTPUT
from compare_feature_sets import SEVEN
from run_v1_holdout import load_prices

OUTPUT = DATA / "v1_hypotheses"
ARMS = {
    "current": (False, False, False),
    "h1_features": (True, False, False),
    "h2_year": (False, True, False),
    "h3_training_filter": (False, False, True),
    "h4_combined": (True, True, True),
    "h5_features_training_filter": (True, False, True),
}
START = pd.Timestamp("2021-01-01")
VALIDATION = pd.Timestamp("2025-01-01")
TEST = pd.Timestamp("2026-01-09")
WIDTH = 255


def session_data():
    symbols = load_roster(ROSTER)["symbols"]
    with duckdb.connect() as db:
        smaller = set(
            db.execute(
                "SELECT DISTINCT symbol FROM read_parquet(?)",
                [str(DATA / "friday/friday_dataset.parquet")],
            ).fetchnumpy()["symbol"]
        )
    calendar = xcals.get_calendar("XNYS")
    prices = load_prices(symbols, calendar)
    close, volume = prices["close"], prices["volume"]
    usable = np.isfinite(close) & close.gt(0) & np.isfinite(volume) & volume.gt(0)
    close, volume = close.where(usable), volume.where(usable)
    logs = np.log(close).diff()
    breaks = pd.read_csv(
        DATA / "friday/flagged_daily_returns.csv", parse_dates=["session"]
    )
    for row in breaks.loc[breaks["status"].str.startswith("unresolved")].itertuples():
        if row.session in logs.index and row.symbol in logs.columns:
            logs.loc[row.session, row.symbol] = np.nan
    recent = logs.index.year == 2026
    logs.loc[recent] = logs.loc[recent].mask(np.expm1(logs.loc[recent]).abs().ge(0.4))
    return calendar, prices, close, volume, logs, smaller


def daily_features(calendar, close, volume, logs, *, evaluation):
    sessions = close.index
    dates = sessions[(sessions >= "2020-01-01") & (sessions <= "2025-12-26")]
    if evaluation:
        dates = dates[dates.year == 2025]
    history = friday_news(dates, calendar, close.columns)
    if evaluation:
        recent = sessions[(sessions >= "2025-12-29") & (sessions <= "2026-10-30")]
        current = friday_news(
            recent,
            calendar,
            close.columns,
            db_path=PRICE_OUTPUT / "news/news_v1.duckdb",
        )
        history = pd.concat([history, current])
    news = history["news_sentiment"].unstack("symbol")
    feature_frames = {
        "momentum_20d": np.expm1(logs.rolling(20).sum()),
        "momentum_12_1": np.expm1(logs.rolling(231).sum().shift(21)),
        "volatility_20d": logs.rolling(20).std(ddof=1) * np.sqrt(252),
        "relative_volume_20d": volume / volume.shift().rolling(20).median(),
        "news_sentiment": news,
        "close": close,
        "volume": volume,
    }
    return np.stack(
        [
            feature_frames[name]
            .reindex(index=sessions, columns=close.columns)
            .to_numpy(dtype=np.float32)
            for name in SEVEN
        ],
        axis=-1,
    )


def context_start(sessions, end, year):
    if year:
        return sessions.searchsorted(
            sessions[end] - pd.Timedelta(days=365), side="right"
        )
    return end - 59


def windows(logs, features, dates, *, use_features, year):
    sessions, symbols = logs.index, logs.columns
    width = WIDTH if year else 60
    frames, targets, masks, keys = [], [], [], []
    log_values = logs.to_numpy(dtype=np.float32)
    for date, end in zip(dates, sessions.get_indexer(dates), strict=True):
        start = context_start(sessions, end, year)
        count = end - start + 1
        if start < 0 or not 0 < count <= width:
            raise ValueError(f"Invalid context for {date}")
        raw_close = log_values[start : end + 1].T
        eligible = np.isfinite(raw_close).all(axis=1)
        if use_features:
            raw_features = features[start : end + 1].swapaxes(0, 1)
            # One observed value per channel at the decision and at least
            # 80% coverage across the actual context; gaps stay masked.
            observed = np.isfinite(raw_features)
            eligible &= observed[:, -1].all(axis=1) & (
                observed.mean(axis=1) >= 0.8
            ).all(axis=1)
            raw = np.concatenate([raw_features, raw_close[..., None]], axis=-1)[
                eligible
            ]
        else:
            raw = raw_close[eligible, :, None]
        observed = np.isfinite(raw)
        means = np.nansum(
            np.where(observed, raw, np.nan), axis=1, keepdims=True
        ) / observed.sum(axis=1, keepdims=True)
        padded = (
            np.broadcast_to(means, (len(raw), width, raw.shape[-1]))
            .copy()
            .astype(np.float32)
        )
        padded[:, -count:] = np.where(observed, raw, means)
        mask = np.zeros(padded.shape, dtype=bool)
        mask[:, -count:] = observed
        y = np.full((len(raw), 5), np.nan, dtype=np.float32)
        if end + 5 < len(sessions):
            y[:] = log_values[end + 1 : end + 6, eligible].T
        frames.append(padded)
        masks.append(mask)
        targets.append(y)
        keys.extend((date, symbol) for symbol in symbols[eligible])
    return (
        pd.MultiIndex.from_tuples(keys, names=["friday", "symbol"]),
        np.concatenate(frames),
        np.concatenate(masks),
        np.concatenate(targets),
    )


def decision_dates(sessions, *, evaluation):
    ends = sessions.to_series().groupby(sessions.to_period("W-FRI")).max()
    if evaluation:
        dates = ends[(ends >= TEST) & (ends <= "2026-10-30")]
        return pd.DatetimeIndex(dates[dates.dt.day_of_week.eq(4)])
    dates = pd.DatetimeIndex(ends[(ends >= START) & (ends <= "2025-12-31")])
    # These are production-style last-session anchors. Five daily outcomes
    # can cross a holiday, so the caller purges by the actual target date.
    return dates
