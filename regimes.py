import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
MONTHS_PER_YEAR = 12

FED_LOOKBACK_DAYS = 182  # ~6 calendar months
FED_HOLD_THRESHOLD = 0.10  # pp change over the lookback treated as "roughly flat"

YIELD_CURVE_ORDER = ["Inverted", "Flattening", "Steep"]
DOLLAR_ORDER = ["Strong Dollar", "Weak Dollar"]
FED_ORDER = ["Cutting", "Hold", "Hiking"]


def _as_of_n_days_ago(df: pd.DataFrame, column: str, days: int) -> np.ndarray:
    """
    For each row's date, look up the value of `column` as of the most
    recent date at least `days` calendar days earlier (i.e. the same
    "value N days ago" semantics as a monthly series would have if it were
    checked on a daily calendar). Rows with no eligible earlier date get NaN.
    """
    target = df[["date"]].reset_index().rename(columns={"index": "_orig_idx"})
    target["_asof"] = target["date"] - pd.Timedelta(days=days)
    target = target.sort_values("_asof")

    lookup = df[["date", column]].sort_values("date").rename(columns={"date": "_asof"})

    merged = pd.merge_asof(target, lookup, on="_asof", direction="backward")
    merged = merged.sort_values("_orig_idx")
    return merged[column].to_numpy()


def classify_yield_curve_regime(t10y2y: pd.Series) -> pd.Series:
    values = np.select(
        [t10y2y < 0, t10y2y <= 0.5],
        ["Inverted", "Flattening"],
        default="Steep",
    )
    return pd.Categorical(values, categories=YIELD_CURVE_ORDER, ordered=True)


def classify_dollar_regime(dtwexbgs: pd.Series, dtwexbgs_ma: pd.Series) -> pd.Categorical:
    """
    dtwexbgs_ma must already be the moving average computed on DTWEXBGS's
    own daily calendar (see data.get_dollar_ma_series) - computing it here
    on whatever calendar `dtwexbgs` happens to be aligned to would silently
    change the window length for non-daily targets.
    """
    values = np.where(dtwexbgs > dtwexbgs_ma, "Strong Dollar", "Weak Dollar")
    values = np.where(dtwexbgs_ma.isna(), None, values)
    return pd.Categorical(values, categories=DOLLAR_ORDER, ordered=True)


def classify_fed_regime(
    df: pd.DataFrame,
    lookback_days: int = FED_LOOKBACK_DAYS,
    hold_threshold: float = FED_HOLD_THRESHOLD,
) -> pd.Categorical:
    change = df["FEDFUNDS"].to_numpy() - _as_of_n_days_ago(df, "FEDFUNDS", lookback_days)
    values = np.select(
        [change > hold_threshold, change < -hold_threshold],
        ["Hiking", "Cutting"],
        default="Hold",
    )
    values = np.where(np.isnan(change), None, values)
    return pd.Categorical(values, categories=FED_ORDER, ordered=True)


def build_regime_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the aligned [date, SP500, T10Y2Y, DTWEXBGS, DTWEXBGS_MA, FEDFUNDS]
    frame from data.get_regime_inputs and adds regime classification
    columns (yield curve, dollar, Fed, and the combined label). Rows that
    fall in the classifiers' burn-in window are dropped.
    """
    out = df.copy()
    out["yield_curve_regime"] = classify_yield_curve_regime(out["T10Y2Y"])
    out["dollar_regime"] = classify_dollar_regime(out["DTWEXBGS"], out["DTWEXBGS_MA"])
    out["fed_regime"] = classify_fed_regime(out)
    out = out.dropna(
        subset=["yield_curve_regime", "dollar_regime", "fed_regime"]
    ).reset_index(drop=True)

    out["regime_label"] = (
        out["yield_curve_regime"].astype(str)
        + " / "
        + out["dollar_regime"].astype(str)
        + " / "
        + out["fed_regime"].astype(str)
    )
    return out


def build_two_factor_regime_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Like build_regime_frame but classifies only the yield-curve and dollar
    regimes (no Fed regime, no combined label) - for indices where the
    Fed-funds factor isn't part of the analysis.
    """
    out = df.copy()
    out["yield_curve_regime"] = classify_yield_curve_regime(out["T10Y2Y"])
    out["dollar_regime"] = classify_dollar_regime(out["DTWEXBGS"], out["DTWEXBGS_MA"])
    out = out.dropna(subset=["yield_curve_regime", "dollar_regime"]).reset_index(drop=True)
    return out


def compute_regime_stats(
    df: pd.DataFrame,
    group_cols,
    price_col: str = "SP500",
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """
    Per-regime annualized return, annualized vol, a Sharpe-like return/vol
    ratio, and % of the sample's periods (trading days, or months for a
    monthly price series) in that regime, for `price_col`.

    group_cols can be a single column name (e.g. "yield_curve_regime") or a
    list of column names (e.g. ["yield_curve_regime", "dollar_regime"] for
    a 2-way cross). periods_per_year should match price_col's frequency
    (252 for a daily series, 12 for a monthly one).
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    d = df.copy()
    d["_return"] = d[price_col].pct_change()
    d = d.dropna(subset=["_return"])
    total_periods = len(d)

    stats = (
        d.groupby(group_cols, observed=True)["_return"]
        .agg(mean="mean", std="std", n_days="count")
        .reset_index()
    )
    stats["annualized_return"] = stats["mean"] * periods_per_year
    stats["annualized_vol"] = stats["std"] * np.sqrt(periods_per_year)
    stats["sharpe"] = stats["annualized_return"] / stats["annualized_vol"]
    stats["pct_of_days"] = stats["n_days"] / total_periods * 100
    stats = stats.drop(columns=["mean", "std"])

    return stats


def regime_periods(df: pd.DataFrame, label_col: str = "regime_label") -> pd.DataFrame:
    """
    Collapses the day-by-day regime frame into contiguous regime periods:
    start date, end date, duration, and the S&P 500 total return over that
    stretch (close on the last day vs. close on the first day).
    """
    d = df[["date", label_col, "SP500"]].reset_index(drop=True)
    group_id = (d[label_col] != d[label_col].shift()).cumsum()

    rows = []
    for _, g in d.groupby(group_id):
        start = g["date"].iloc[0]
        end = g["date"].iloc[-1]
        rows.append(
            {
                "regime": g[label_col].iloc[0],
                "start_date": start,
                "end_date": end,
                "duration_days": (end - start).days + 1,
                "trading_days": len(g),
                "sp500_return_pct": (g["SP500"].iloc[-1] / g["SP500"].iloc[0] - 1) * 100,
            }
        )

    periods = pd.DataFrame(rows)
    return periods.sort_values("start_date", ascending=False).reset_index(drop=True)
