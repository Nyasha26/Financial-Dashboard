import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data import get_aligned_series, get_regime_inputs, get_series
from regimes import (
    DOLLAR_ORDER,
    FED_ORDER,
    MONTHS_PER_YEAR,
    TRADING_DAYS_PER_YEAR,
    YIELD_CURVE_ORDER,
    build_regime_frame,
    build_two_factor_regime_frame,
    compute_regime_stats,
    regime_periods,
)

st.set_page_config(page_title="Nyasha Mugabe Dashboard", layout="wide")

# Preset series. Dict order fixes each series' color/style identity so it
# doesn't shift when the selection changes.
PRESETS = {
    "DGS10": "10Y Treasury",
    "DGS2": "2Y Treasury",
    "T10Y2Y": "2s10s Spread",
    "BAMLH0A0HYM2": "HY OAS",
    "FEDFUNDS": "Fed Funds Rate",
    "CPIAUCSL": "CPI",
}

# Validated categorical palette (see dataviz skill), fixed order.
CATEGORICAL_COLORS = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
# First three slots are color-distinguishable from each other in all
# combinations; the last three of the original six get a dashed line so they
# stay distinguishable from the first three even for viewers who can't rely
# on hue alone.
SERIES_STYLE = {
    sid: {
        "color": CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)],
        "dash": "solid" if i < 3 else "dash",
    }
    for i, sid in enumerate(PRESETS)
}


@st.cache_data(show_spinner=False, ttl=3600)
def load_series(series_id: str, bypass_cache: bool = False) -> pd.DataFrame:
    return get_series(series_id, use_cache=not bypass_cache)


@st.cache_data(show_spinner=False, ttl=3600)
def load_regime_frame(bypass_cache: bool = False) -> pd.DataFrame:
    raw = get_regime_inputs(use_cache=not bypass_cache)
    return build_regime_frame(raw)


# Global equity indices, classified on the same yield-curve and dollar
# regimes as S&P 500 (no Fed regime for these). FRED has no native Russell
# 2000 price index (only a volatility index), so it's left out entirely
# rather than faked with an unrelated proxy.
GLOBAL_INDICES = {
    "Nikkei 225": dict(
        series_id="NIKKEI225",
        periods_per_year=TRADING_DAYS_PER_YEAR,
        period_noun="trading days",
        is_proxy=False,
        note="Native FRED series (NIKKEI225), daily close.",
    ),
    "FTSE 100 (UK proxy)": dict(
        series_id="SPASTT01GBM661N",
        periods_per_year=MONTHS_PER_YEAR,
        period_noun="months",
        is_proxy=True,
        note="FRED has no FTSE 100 series. This is OECD's monthly broad "
        "UK share-price index (SPASTT01GBM661N) - a directional proxy, "
        "not the literal FTSE 100.",
    ),
    "SSE Composite (China proxy)": dict(
        series_id="SPASTT01CNM661N",
        periods_per_year=MONTHS_PER_YEAR,
        period_noun="months",
        is_proxy=True,
        note="FRED has no Shanghai Composite series. This is OECD's "
        "monthly broad China share-price index (SPASTT01CNM661N) - a "
        "directional proxy, not the literal SSE Composite.",
    ),
}


@st.cache_data(show_spinner=False, ttl=3600)
def load_global_index_frame(index_key: str, bypass_cache: bool = False) -> pd.DataFrame:
    series_id = GLOBAL_INDICES[index_key]["series_id"]
    raw = get_aligned_series(series_id, use_cache=not bypass_cache, include_fed=False)
    return build_two_factor_regime_frame(raw)


st.title("Nyasha Mugabe Dashboard")

# ---------------------------------------------------------------------------
# Sidebar (controls for the Single Series tab)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Series")
    selected_ids = st.multiselect(
        "Select series",
        options=list(PRESETS.keys()),
        default=["DGS10", "DGS2"],
        format_func=lambda sid: f"{PRESETS[sid]} ({sid})",
    )

    st.header("Date range")
    default_end = dt.date.today()
    default_start = default_end - dt.timedelta(days=365 * 5)
    date_range = st.date_input(
        "Range",
        value=(default_start, default_end),
        max_value=default_end,
    )

    secondary_ids = []
    if len(selected_ids) >= 2:
        st.header("Axes")
        secondary_ids = st.multiselect(
            "Plot on secondary (right) y-axis",
            options=selected_ids,
            format_func=lambda sid: f"{PRESETS[sid]} ({sid})",
            help="Useful when comparing series on very different scales, "
            "e.g. HY OAS vs. the 2s10s spread.",
        )

    refresh_clicked = st.button("Refresh Data", width="stretch")

if refresh_clicked:
    load_series.clear()
    load_regime_frame.clear()
    load_global_index_frame.clear()


def render_single_series_tab() -> None:
    if not selected_ids:
        st.info("Select one or more series from the sidebar to begin.")
        return

    if not isinstance(date_range, tuple) or len(date_range) != 2:
        st.info("Pick both a start and end date.")
        return

    start_date, end_date = date_range
    if start_date > end_date:
        st.error("Start date must be before end date.")
        return

    series_data = {}
    errors = {}

    with st.spinner("Fetching data from FRED..."):
        for sid in selected_ids:
            try:
                df_full = load_series(sid, bypass_cache=refresh_clicked)
            except ValueError as e:
                errors[sid] = str(e)
                continue

            mask = (df_full["date"] >= pd.Timestamp(start_date)) & (
                df_full["date"] <= pd.Timestamp(end_date)
            )
            df = df_full.loc[mask].reset_index(drop=True)
            series_data[sid] = df

    for sid, msg in errors.items():
        st.warning(f"{PRESETS[sid]} ({sid}): {msg}")

    empty_ids = [sid for sid, df in series_data.items() if df.empty]
    for sid in empty_ids:
        st.info(f"{PRESETS[sid]} ({sid}): no data in the selected date range.")
        del series_data[sid]

    if not series_data:
        return

    # -- Summary stat cards --
    for sid, df in series_data.items():
        latest = df["value"].iloc[-1]
        first = df["value"].iloc[0]
        change = latest - first
        pct = (change / first * 100) if first else None

        st.markdown(f"**{PRESETS[sid]}** ({sid})")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latest", f"{latest:,.2f}")
        c2.metric(
            "Change over period",
            f"{change:+,.2f}",
            delta=f"{pct:+.1f}%" if pct is not None else None,
        )
        c3.metric("Min", f"{df['value'].min():,.2f}")
        c4.metric("Max", f"{df['value'].max():,.2f}")

    # -- Chart --
    fig = go.Figure()
    for sid, df in series_data.items():
        style = SERIES_STYLE[sid]
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["value"],
                name=f"{PRESETS[sid]} ({sid})",
                mode="lines",
                line=dict(color=style["color"], width=2, dash=style["dash"]),
                yaxis="y2" if sid in secondary_ids else "y1",
            )
        )

    layout = dict(
        height=520,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        if len(series_data) > 1
        else dict(visible=False),
        xaxis=dict(title="Date"),
        yaxis=dict(title="Value" if not secondary_ids else "Left axis"),
        hovermode="x unified",
    )
    if secondary_ids:
        layout["yaxis2"] = dict(title="Right axis", overlaying="y", side="right")
    fig.update_layout(**layout)

    st.plotly_chart(fig, width="stretch", theme="streamlit")

    if secondary_ids:
        st.caption(
            "Dual-axis chart: series on the right axis use an independent scale "
            "from series on the left, so visual co-movement doesn't imply the "
            "series are the same magnitude."
        )

    # -- Raw data table --
    with st.expander("Show raw data"):
        wide = None
        for sid, df in series_data.items():
            col = df[["date", "value"]].rename(
                columns={"value": f"{PRESETS[sid]} ({sid})"}
            )
            wide = col if wide is None else wide.merge(col, on="date", how="outer")
        wide = wide.sort_values("date", ascending=False).reset_index(drop=True)
        st.dataframe(wide, width="stretch")


# ---------------------------------------------------------------------------
# Regime Analysis tab
# ---------------------------------------------------------------------------
REGIME_METRICS = {
    "Annualized return": ("annualized_return", "%", "diverging"),
    "Annualized volatility": ("annualized_vol", "%", "sequential"),
    "Sharpe-like ratio": ("sharpe", "", "diverging"),
}


def _ordered_stats(
    rf: pd.DataFrame,
    group_col: str,
    order: list[str],
    price_col: str = "SP500",
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    stats = compute_regime_stats(
        rf, group_col, price_col=price_col, periods_per_year=periods_per_year
    ).set_index(group_col)
    return stats.reindex(order).reset_index()


def _factor_bar(
    stats: pd.DataFrame,
    group_col: str,
    order: list[str],
    title: str,
    period_noun: str = "trading days",
) -> go.Figure:
    colors = [CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)] for i in range(len(order))]
    text = [
        f"{p:.0f}% of {period_noun}" if pd.notna(p) else "no data"
        for p in stats["pct_of_days"]
    ]
    fig = go.Figure(
        go.Bar(
            x=stats[group_col].astype(str),
            y=stats["annualized_return"] * 100,
            marker_color=colors,
            text=text,
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title,
        yaxis_title="Annualized return (%)",
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )
    return fig


def _render_cross_heatmap(
    rf: pd.DataFrame,
    price_col: str,
    periods_per_year: float,
    period_noun: str,
    key_prefix: str,
) -> None:
    metric_label = st.radio(
        "Cell metric",
        list(REGIME_METRICS.keys()),
        horizontal=True,
        key=f"{key_prefix}_metric",
    )
    metric_col, unit, scale_kind = REGIME_METRICS[metric_label]

    cross = compute_regime_stats(
        rf,
        ["yield_curve_regime", "dollar_regime"],
        price_col=price_col,
        periods_per_year=periods_per_year,
    )
    value_pivot = cross.pivot(
        index="yield_curve_regime", columns="dollar_regime", values=metric_col
    ).reindex(index=YIELD_CURVE_ORDER, columns=DOLLAR_ORDER)
    pct_pivot = cross.pivot(
        index="yield_curve_regime", columns="dollar_regime", values="pct_of_days"
    ).reindex(index=YIELD_CURVE_ORDER, columns=DOLLAR_ORDER)
    n_pivot = cross.pivot(
        index="yield_curve_regime", columns="dollar_regime", values="n_days"
    ).reindex(index=YIELD_CURVE_ORDER, columns=DOLLAR_ORDER)

    display_vals = value_pivot * 100 if unit == "%" else value_pivot

    cell_text = [
        [
            (f"{v:.1f}{unit}" if pd.notna(v) else "n/a")
            + (
                "*"
                if pd.notna(pct_pivot.iloc[r, c]) and pct_pivot.iloc[r, c] < 5
                else ""
            )
            for c, v in enumerate(row)
        ]
        for r, row in enumerate(display_vals.values)
    ]
    customdata = [
        [
            [pct_pivot.iloc[r, c], n_pivot.iloc[r, c]]
            for c in range(len(DOLLAR_ORDER))
        ]
        for r in range(len(YIELD_CURVE_ORDER))
    ]

    if scale_kind == "diverging":
        zmax = float(pd.Series(display_vals.values.flatten()).abs().max())
        zmax = zmax if zmax and pd.notna(zmax) else 1.0
        heat_kwargs = dict(colorscale="RdBu", zmid=0, zmin=-zmax, zmax=zmax)
    else:
        heat_kwargs = dict(colorscale="Blues")

    fig = go.Figure(
        go.Heatmap(
            z=display_vals.values,
            x=[str(c) for c in display_vals.columns],
            y=[str(i) for i in display_vals.index],
            text=cell_text,
            texttemplate="%{text}",
            customdata=customdata,
            hovertemplate=(
                "%{y} / %{x}<br>"
                + metric_label
                + ": %{z:.2f}"
                + unit
                + "<br>%{customdata[1]} "
                + period_noun
                + " (%{customdata[0]:.1f}% of sample)"
                + "<extra></extra>"
            ),
            colorbar=dict(title=metric_label),
            **heat_kwargs,
        )
    )
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Dollar regime",
        yaxis_title="Yield curve regime",
    )
    st.plotly_chart(fig, width="stretch", theme="streamlit")
    st.caption(
        f"* fewer than 5% of {period_noun} in this bucket — interpret with caution."
    )


def render_regime_tab() -> None:
    with st.spinner("Loading regime data from FRED..."):
        try:
            rf = load_regime_frame(bypass_cache=refresh_clicked)
        except ValueError as e:
            st.error(str(e))
            return

    if rf.empty:
        st.info(
            "Not enough overlapping history yet to classify regimes (the "
            "dollar regime needs 200 trading days of DTWEXBGS history and "
            "the Fed regime needs ~6 months of FEDFUNDS history before the "
            "first day can be classified)."
        )
        return

    st.caption(
        f"S&P 500 sets the trading-day calendar; other series are forward-filled "
        f"onto it. After the classifier burn-in (200-day dollar moving average, "
        f"6-month Fed-funds lookback), regimes are classified for "
        f"{rf['date'].min().date()} - {rf['date'].max().date()} "
        f"({len(rf):,} trading days)."
    )

    st.subheader("S&P 500 return by single-factor regime")
    col1, col2, col3 = st.columns(3)
    with col1:
        stats = _ordered_stats(rf, "yield_curve_regime", YIELD_CURVE_ORDER)
        st.plotly_chart(
            _factor_bar(stats, "yield_curve_regime", YIELD_CURVE_ORDER, "Yield curve"),
            width="stretch",
            theme="streamlit",
        )
    with col2:
        stats = _ordered_stats(rf, "dollar_regime", DOLLAR_ORDER)
        st.plotly_chart(
            _factor_bar(stats, "dollar_regime", DOLLAR_ORDER, "Dollar"),
            width="stretch",
            theme="streamlit",
        )
    with col3:
        stats = _ordered_stats(rf, "fed_regime", FED_ORDER)
        st.plotly_chart(
            _factor_bar(stats, "fed_regime", FED_ORDER, "Fed funds"),
            width="stretch",
            theme="streamlit",
        )
    st.caption(
        "Bar labels show the share of trading days in that regime — "
        "treat thin bars' return figures cautiously; small samples are noisy."
    )

    st.subheader("Yield curve x Dollar regime")
    _render_cross_heatmap(
        rf,
        price_col="SP500",
        periods_per_year=TRADING_DAYS_PER_YEAR,
        period_noun="trading days",
        key_prefix="sp500",
    )

    with st.expander("All regime combinations (yield curve / dollar / Fed), aggregated"):
        combo = compute_regime_stats(rf, "regime_label").sort_values(
            "pct_of_days", ascending=False
        )
        combo = combo.rename(
            columns={
                "regime_label": "Regime",
                "n_days": "Trading days",
                "annualized_return": "Ann. return",
                "annualized_vol": "Ann. vol",
                "sharpe": "Sharpe-like",
                "pct_of_days": "% of days",
            }
        )
        st.dataframe(
            combo.style.format(
                {
                    "Ann. return": "{:+.1%}",
                    "Ann. vol": "{:.1%}",
                    "Sharpe-like": "{:+.2f}",
                    "% of days": "{:.1f}%",
                }
            ),
            width="stretch",
        )

    st.subheader("Regime periods")
    with st.expander("Show all regime periods", expanded=False):
        periods = regime_periods(rf)
        periods = periods.rename(
            columns={
                "regime": "Regime",
                "start_date": "Start",
                "end_date": "End",
                "duration_days": "Duration (calendar days)",
                "trading_days": "Trading days",
                "sp500_return_pct": "S&P 500 return",
            }
        )
        st.dataframe(
            periods.style.format({"S&P 500 return": "{:+.1f}%"}),
            width="stretch",
        )


# ---------------------------------------------------------------------------
# Global Indices tab
# ---------------------------------------------------------------------------
def render_global_indices_tab() -> None:
    st.info(
        "Russell 2000 isn't included: FRED has no Russell 2000 price index "
        "(only a volatility index, RVXCLS, which isn't usable for returns)."
    )

    index_key = st.selectbox("Index", list(GLOBAL_INDICES.keys()))
    meta = GLOBAL_INDICES[index_key]

    with st.spinner(f"Loading {index_key} data from FRED..."):
        try:
            gf = load_global_index_frame(index_key, bypass_cache=refresh_clicked)
        except ValueError as e:
            st.error(str(e))
            return

    if gf.empty:
        st.info("Not enough overlapping history yet to classify regimes for this index.")
        return

    if meta["is_proxy"]:
        st.warning(meta["note"])

    st.caption(
        f"{len(gf):,} {meta['period_noun']} classified, "
        f"{gf['date'].min().date()} - {gf['date'].max().date()}."
    )

    price_col = meta["series_id"]
    periods_per_year = meta["periods_per_year"]
    period_noun = meta["period_noun"]

    st.subheader(f"{index_key} return by regime")
    col1, col2 = st.columns(2)
    with col1:
        stats = _ordered_stats(
            gf, "yield_curve_regime", YIELD_CURVE_ORDER, price_col, periods_per_year
        )
        st.plotly_chart(
            _factor_bar(
                stats, "yield_curve_regime", YIELD_CURVE_ORDER, "Yield curve", period_noun
            ),
            width="stretch",
            theme="streamlit",
        )
    with col2:
        stats = _ordered_stats(
            gf, "dollar_regime", DOLLAR_ORDER, price_col, periods_per_year
        )
        st.plotly_chart(
            _factor_bar(stats, "dollar_regime", DOLLAR_ORDER, "Dollar", period_noun),
            width="stretch",
            theme="streamlit",
        )
    st.caption(
        f"Bar labels show the share of {period_noun} in that regime — "
        "treat thin bars' return figures cautiously; small samples are noisy."
    )

    st.subheader("Yield curve x Dollar regime")
    _render_cross_heatmap(
        gf,
        price_col=price_col,
        periods_per_year=periods_per_year,
        period_noun=period_noun,
        key_prefix=f"global_{price_col}",
    )


tab1, tab2, tab3 = st.tabs(["Single Series", "Regime Analysis", "Global Indices"])
with tab1:
    render_single_series_tab()
with tab2:
    render_regime_tab()
with tab3:
    render_global_indices_tab()
