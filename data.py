import os
import time

import pandas as pd
from fredapi import Fred

from config import FRED_API_KEY

_fred = Fred(api_key=FRED_API_KEY)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
CACHE_TTL_SECONDS = 24 * 60 * 60  # 1 day


def _cache_path(series_id, start_date, end_date):
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = f"{series_id}_{start_date or 'na'}_{end_date or 'na'}.csv"
    return os.path.join(CACHE_DIR, key)


def _read_cache(path):
    if not os.path.exists(path):
        return None
    age_seconds = time.time() - os.path.getmtime(path)
    if age_seconds > CACHE_TTL_SECONDS:
        return None
    try:
        return pd.read_csv(path, parse_dates=["date"])
    except Exception:
        return None


def get_series(series_id, start_date=None, end_date=None, use_cache=True):
    """
    Pull a FRED series and return a DataFrame with columns [date, value].

    start_date / end_date accept anything fredapi/pandas can parse
    (e.g. "2020-01-01") or None for the full available history.
    """
    cache_path = _cache_path(series_id, start_date, end_date)

    if use_cache:
        cached = _read_cache(cache_path)
        if cached is not None:
            return cached

    try:
        raw = _fred.get_series(
            series_id, observation_start=start_date, observation_end=end_date
        )
    except Exception as e:
        raise ValueError(
            f"Failed to fetch series '{series_id}' from FRED. "
            f"Check that the series ID is valid. Details: {e}"
        ) from e

    if raw is None or raw.empty:
        raise ValueError(f"No data returned for series '{series_id}'.")

    df = raw.reset_index()
    df.columns = ["date", "value"]
    df = df.dropna(subset=["value"]).reset_index(drop=True)

    df.to_csv(cache_path, index=False)

    return df


def get_regime_inputs(start_date=None, end_date=None, use_cache=True):
    """
    Pull T10Y2Y, DTWEXBGS, FEDFUNDS and SP500 and align them on SP500's
    trading-day calendar. Series that update less often than daily
    (FEDFUNDS is monthly) are forward-filled onto that calendar, i.e. each
    trading day carries the most recently published value as of that date.

    Returns a DataFrame with columns [date, SP500, T10Y2Y, DTWEXBGS, FEDFUNDS].
    """
    sp500 = get_series("SP500", start_date, end_date, use_cache=use_cache)
    sp500 = sp500.rename(columns={"value": "SP500"}).sort_values("date")

    aligned = sp500.reset_index(drop=True)
    for series_id in ("T10Y2Y", "DTWEXBGS", "FEDFUNDS"):
        driver = get_series(series_id, start_date, end_date, use_cache=use_cache)
        driver = driver.rename(columns={"value": series_id}).sort_values("date")
        aligned = pd.merge_asof(aligned, driver, on="date", direction="backward")

    aligned = aligned.dropna().reset_index(drop=True)
    return aligned
