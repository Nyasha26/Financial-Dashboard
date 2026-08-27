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


def get_dollar_ma_series(window=200, start_date=None, end_date=None, use_cache=True):
    """
    DTWEXBGS with an N-day moving average computed on DTWEXBGS's own
    (business-day) calendar, before any alignment onto a target index's
    calendar. Computing the MA first and aligning it afterwards keeps it a
    true N-trading-day dollar-market average even when the target
    calendar is coarser than daily (e.g. a monthly index) - aligning the
    raw value alone and taking an N-row rolling mean post-alignment would
    silently turn "200-day" into "200-month" for a monthly target.
    """
    dxy = get_series("DTWEXBGS", start_date, end_date, use_cache=use_cache)
    dxy = dxy.rename(columns={"value": "DTWEXBGS"}).sort_values("date").reset_index(drop=True)
    dxy["DTWEXBGS_MA"] = dxy["DTWEXBGS"].rolling(window, min_periods=window).mean()
    return dxy


def get_regime_drivers(dollar_ma_window=200, start_date=None, end_date=None, use_cache=True):
    """
    T10Y2Y, DTWEXBGS (+ its moving average), and THREEFYTP10 (NY Fed ACM
    10-year term premium) as one frame, ready to be aligned onto any
    target index's calendar via merge_asof. This is the single place all
    regime drivers are assembled, so adding a driver here propagates it to
    every tab's regime frame automatically.
    """
    t10y2y = get_series("T10Y2Y", start_date, end_date, use_cache=use_cache)
    t10y2y = t10y2y.rename(columns={"value": "T10Y2Y"}).sort_values("date")

    dxy = get_dollar_ma_series(dollar_ma_window, start_date, end_date, use_cache=use_cache)

    term_premium = get_series("THREEFYTP10", start_date, end_date, use_cache=use_cache)
    term_premium = term_premium.rename(columns={"value": "THREEFYTP10"}).sort_values("date")

    drivers = pd.merge_asof(t10y2y, dxy, on="date", direction="backward")
    drivers = pd.merge_asof(drivers, term_premium, on="date", direction="backward")
    return drivers.dropna().reset_index(drop=True)


def get_aligned_series(
    target_ids,
    start_date=None,
    end_date=None,
    use_cache=True,
    include_fed=False,
    dollar_ma_window=200,
):
    """
    Pull one or more target series - if more than one, each after the
    first is aligned (forward-filled via merge_asof) onto the first
    target's calendar - and align T10Y2Y + DTWEXBGS (+ its moving
    average), and optionally FEDFUNDS, onto that same calendar. Drivers
    that update less often than the target (FEDFUNDS is monthly) are
    forward-filled the same way, i.e. each row carries the most recently
    published driver value as of that date.

    Returns a DataFrame with columns
    [date, <target_ids...>, T10Y2Y, DTWEXBGS, DTWEXBGS_MA, THREEFYTP10]
    (+ FEDFUNDS if include_fed=True). dropna() at the end means the panel
    is naturally bounded by whichever target has the shortest history.
    """
    if isinstance(target_ids, str):
        target_ids = [target_ids]

    base = get_series(target_ids[0], start_date, end_date, use_cache=use_cache)
    aligned = base.rename(columns={"value": target_ids[0]}).sort_values("date").reset_index(drop=True)

    for target_id in target_ids[1:]:
        other = get_series(target_id, start_date, end_date, use_cache=use_cache)
        other = other.rename(columns={"value": target_id}).sort_values("date")
        aligned = pd.merge_asof(aligned, other, on="date", direction="backward")

    drivers = get_regime_drivers(dollar_ma_window, start_date, end_date, use_cache=use_cache)
    aligned = pd.merge_asof(aligned, drivers, on="date", direction="backward")

    if include_fed:
        fedfunds = get_series("FEDFUNDS", start_date, end_date, use_cache=use_cache)
        fedfunds = fedfunds.rename(columns={"value": "FEDFUNDS"}).sort_values("date")
        aligned = pd.merge_asof(aligned, fedfunds, on="date", direction="backward")

    return aligned.dropna().reset_index(drop=True)


def get_regime_inputs(start_date=None, end_date=None, use_cache=True):
    """
    Pull T10Y2Y, DTWEXBGS, THREEFYTP10, FEDFUNDS and SP500 and align them
    on SP500's trading-day calendar.

    Returns a DataFrame with columns
    [date, SP500, T10Y2Y, DTWEXBGS, DTWEXBGS_MA, THREEFYTP10, FEDFUNDS].
    """
    return get_aligned_series(
        "SP500", start_date, end_date, use_cache=use_cache, include_fed=True
    )


RELATIVE_VALUE_INSTRUMENTS = (
    "SP500",
    "BAMLHYH0A0HYM2TRIV",
    "BAMLCC0A0CMTRIV",
    "BAMLEMCBPITRIV",
)


def get_relative_value_inputs(start_date=None, end_date=None, use_cache=True):
    """
    S&P 500 plus the three ICE BofA credit total-return indices, all
    aligned onto SP500's trading-day calendar and bounded by the credit
    indices' shorter FRED history (2023-08-28+), so every instrument is
    compared over the exact same dates and regimes.

    Returns a DataFrame with columns
    [date, SP500, BAMLHYH0A0HYM2TRIV, BAMLCC0A0CMTRIV, BAMLEMCBPITRIV,
    T10Y2Y, DTWEXBGS, DTWEXBGS_MA, THREEFYTP10].
    """
    return get_aligned_series(
        list(RELATIVE_VALUE_INSTRUMENTS), start_date, end_date, use_cache=use_cache
    )
