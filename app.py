import datetime as dt
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data import (
    get_aligned_series,
    get_regime_inputs,
    get_relative_value_inputs,
    get_series,
)
from regimes import (
    DOLLAR_ORDER,
    FED_ORDER,
    MONTHS_PER_YEAR,
    TERM_PREMIUM_ORDER,
    TRADING_DAYS_PER_YEAR,
    YIELD_CURVE_ORDER,
    build_regime_frame,
    build_three_factor_regime_frame,
    compute_beta,
    compute_regime_stats,
    regime_periods,
)

st.set_page_config(page_title="Nyasha Mugabe Macro Regime Dashboard", layout="wide")

# Preset series. Dict order fixes each series' color/style identity so it
# doesn't shift when the selection changes.
PRESETS = {
    "DGS10": "10Y Treasury",
    "DGS2": "2Y Treasury",
    "T10Y2Y": "2s10s Spread",
    "BAMLH0A0HYM2": "HY OAS",
    "FEDFUNDS": "Fed Funds Rate",
    "CPIAUCSL": "CPI",
    "DTWEXBGS": "Nominal Dollar Index",
    "THREEFYTP10": "10Y Term Premium",
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


# Global equity indices, classified on the same yield-curve, dollar, and
# term-premium regimes as S&P 500 (no Fed regime for these). FRED has no
# native Russell 2000 price index (only a volatility index), so it's left
# out entirely rather than faked with an unrelated proxy.
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
    return build_three_factor_regime_frame(raw)


# ICE BofA total-return indices, classified on the same yield-curve,
# dollar, and term-premium regimes. All three only have history on FRED
# back to 2023-08-28, so the analysis window is short - the date range
# shown in the tab makes that visible rather than hiding it.
CREDIT_INDICES = {
    "US High Yield (ICE BofA)": dict(
        series_id="BAMLHYH0A0HYM2TRIV",
        periods_per_year=TRADING_DAYS_PER_YEAR,
        period_noun="trading days",
        is_proxy=False,
        note="ICE BofA US High Yield Index, Total Return Index Value (BAMLHYH0A0HYM2TRIV).",
    ),
    "US Corporate (ICE BofA)": dict(
        series_id="BAMLCC0A0CMTRIV",
        periods_per_year=TRADING_DAYS_PER_YEAR,
        period_noun="trading days",
        is_proxy=False,
        note="ICE BofA US Corporate Index, Total Return Index Value (BAMLCC0A0CMTRIV).",
    ),
    "EM Corporate (ICE BofA)": dict(
        series_id="BAMLEMCBPITRIV",
        periods_per_year=TRADING_DAYS_PER_YEAR,
        period_noun="trading days",
        is_proxy=False,
        note="ICE BofA Emerging Markets Corporate Plus Index, Total Return "
        "Index Value (BAMLEMCBPITRIV).",
    ),
}


@st.cache_data(show_spinner=False, ttl=3600)
def load_credit_index_frame(index_key: str, bypass_cache: bool = False) -> pd.DataFrame:
    series_id = CREDIT_INDICES[index_key]["series_id"]
    raw = get_aligned_series(series_id, use_cache=not bypass_cache, include_fed=False)
    return build_three_factor_regime_frame(raw)


# S&P 500 vs. the three ICE BofA credit indices, all aligned onto SP500's
# calendar and bounded by the credit indices' shorter history, so the
# comparison is over identical dates. Colors deliberately skip the
# palette's 4th slot (yellow) - it's a documented failure pair with the
# 2nd slot (orange) when both are visible at once, which a grouped bar
# chart with all 4 instruments always does.
RV_INSTRUMENTS = {
    "S&P 500": "SP500",
    "US High Yield": "BAMLHYH0A0HYM2TRIV",
    "US Corporate": "BAMLCC0A0CMTRIV",
    "EM Corporate": "BAMLEMCBPITRIV",
}
RV_COLORS = {
    "S&P 500": CATEGORICAL_COLORS[0],  # blue
    "US High Yield": CATEGORICAL_COLORS[1],  # orange
    "US Corporate": CATEGORICAL_COLORS[2],  # aqua
    "EM Corporate": CATEGORICAL_COLORS[7],  # red
}


@st.cache_data(show_spinner=False, ttl=3600)
def load_relative_value_frame(bypass_cache: bool = False) -> pd.DataFrame:
    raw = get_relative_value_inputs(use_cache=not bypass_cache)
    return build_three_factor_regime_frame(raw)


# Regime drivers, sized to the move requested for the Beta tab. T10Y2Y,
# THREEFYTP10, and FEDFUNDS are percentage-point spreads/rates (1bp =
# 0.01); DTWEXBGS is an index level, so "$0.01" is read as a 0.01-point
# move.
BETA_DRIVERS = {
    "T10Y2Y": dict(name="Yield curve", move_size=0.01, move_label="1bp"),
    "DTWEXBGS": dict(name="Dollar index", move_size=0.01, move_label="$0.01"),
    "THREEFYTP10": dict(name="Term premium", move_size=0.01, move_label="1bp"),
    "FEDFUNDS": dict(name="Fed funds", move_size=0.01, move_label="1bp"),
}


@st.cache_data(show_spinner=False, ttl=3600)
def load_beta_frame(bypass_cache: bool = False) -> pd.DataFrame:
    return get_aligned_series(
        list(RV_INSTRUMENTS.values()), use_cache=not bypass_cache, include_fed=True
    )


st.title("Nyasha Mugabe Macro Regime Dashboard")
st.markdown("*Analysis of Equity and Credit returns in different macroeconomic regimes*")

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
    load_credit_index_frame.clear()
    load_relative_value_frame.clear()
    load_beta_frame.clear()


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
    col1, col2, col3, col4 = st.columns(4)
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
    with col4:
        stats = _ordered_stats(rf, "term_premium_regime", TERM_PREMIUM_ORDER)
        st.plotly_chart(
            _factor_bar(stats, "term_premium_regime", TERM_PREMIUM_ORDER, "Term premium"),
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

    with st.expander("All regime combinations (yield curve / dollar / Fed / term premium), aggregated"):
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
# Instrument regime tabs (Global Indices, Credit Indices) - both pick one
# instrument from a dict and show the same yield-curve/dollar regime
# treatment as the S&P 500 tab, minus the Fed regime.
# ---------------------------------------------------------------------------
def render_instrument_regime_tab(
    indices: dict, loader, key_prefix: str, banner: str = None
) -> None:
    if banner:
        st.info(banner)

    index_key = st.selectbox("Index", list(indices.keys()), key=f"{key_prefix}_select")
    meta = indices[index_key]

    with st.spinner(f"Loading {index_key} data from FRED..."):
        try:
            gf = loader(index_key, bypass_cache=refresh_clicked)
        except ValueError as e:
            st.error(str(e))
            return

    if gf.empty:
        st.info("Not enough overlapping history yet to classify regimes for this index.")
        return

    if meta["is_proxy"]:
        st.warning(meta["note"])
    else:
        st.caption(meta["note"])

    st.caption(
        f"{len(gf):,} {meta['period_noun']} classified, "
        f"{gf['date'].min().date()} - {gf['date'].max().date()}."
    )

    price_col = meta["series_id"]
    periods_per_year = meta["periods_per_year"]
    period_noun = meta["period_noun"]

    st.subheader(f"{index_key} return by regime")
    col1, col2, col3 = st.columns(3)
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
    with col3:
        stats = _ordered_stats(
            gf, "term_premium_regime", TERM_PREMIUM_ORDER, price_col, periods_per_year
        )
        st.plotly_chart(
            _factor_bar(
                stats, "term_premium_regime", TERM_PREMIUM_ORDER, "Term premium", period_noun
            ),
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
        # Stable per-tab key (not derived from price_col) so the metric
        # choice persists when the user switches instruments within a tab,
        # rather than spawning a new radio widget (and losing the choice)
        # every time.
        key_prefix=key_prefix,
    )


def render_global_indices_tab() -> None:
    render_instrument_regime_tab(
        GLOBAL_INDICES,
        load_global_index_frame,
        key_prefix="global",
        banner="Russell 2000 isn't included: FRED has no Russell 2000 price "
        "index (only a volatility index, RVXCLS, which isn't usable for "
        "returns).",
    )


def render_credit_indices_tab() -> None:
    render_instrument_regime_tab(
        CREDIT_INDICES, load_credit_index_frame, key_prefix="credit"
    )


# ---------------------------------------------------------------------------
# Relative Value tab: S&P 500 vs. credit indices, by regime
# ---------------------------------------------------------------------------
def _rv_grouped_bar(rv: pd.DataFrame, group_col: str, order: list[str]) -> go.Figure:
    # No in-figure title here on purpose: the panel label is rendered via
    # st.markdown above the chart instead, so it doesn't compete with the
    # legend for the same top strip of the figure (that collision is what
    # made the title and legend render crammed onto one overlapping line).
    fig = go.Figure()
    for label, sid in RV_INSTRUMENTS.items():
        stats = _ordered_stats(rv, group_col, order, price_col=sid, periods_per_year=TRADING_DAYS_PER_YEAR)
        fig.add_trace(
            go.Bar(
                name=label,
                x=stats[group_col].astype(str),
                y=stats["annualized_return"] * 100,
                marker_color=RV_COLORS[label],
            )
        )
    fig.update_layout(
        barmode="group",
        yaxis_title="Annualized return (%)",
        yaxis=dict(rangemode="tozero"),
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def _rv_spread_table(rv: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["yield_curve_regime", "dollar_regime"]
    sp_stats = compute_regime_stats(
        rv, group_cols, price_col="SP500", periods_per_year=TRADING_DAYS_PER_YEAR
    )
    table = sp_stats[group_cols + ["pct_of_days", "n_days", "annualized_return"]].rename(
        columns={"annualized_return": "S&P 500"}
    )

    credit_cols = []
    for label, sid in RV_INSTRUMENTS.items():
        if sid == "SP500":
            continue
        credit_stats = compute_regime_stats(
            rv, group_cols, price_col=sid, periods_per_year=TRADING_DAYS_PER_YEAR
        )
        table = table.merge(
            credit_stats[group_cols + ["annualized_return"]].rename(columns={"annualized_return": label}),
            on=group_cols,
        )
        table[f"{label} spread"] = table["S&P 500"] - table[label]
        credit_cols += [label, f"{label} spread"]

    table = table[group_cols + ["pct_of_days", "n_days", "S&P 500"] + credit_cols]

    order_index = {
        (y, d): i
        for i, (y, d) in enumerate((y, d) for y in YIELD_CURVE_ORDER for d in DOLLAR_ORDER)
    }
    table = (
        table.assign(
            _order=table.apply(
                lambda r: order_index.get(
                    (r["yield_curve_regime"], r["dollar_regime"]), 999
                ),
                axis=1,
            )
        )
        .sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )
    return table


def render_relative_value_tab() -> None:
    with st.spinner("Loading relative value data from FRED..."):
        try:
            rv = load_relative_value_frame(bypass_cache=refresh_clicked)
        except ValueError as e:
            st.error(str(e))
            return

    if rv.empty:
        st.info(
            "Not enough overlapping history yet across S&P 500 and the "
            "credit indices to run this comparison."
        )
        return

    st.caption(
        "S&P 500 vs. the three ICE BofA credit total-return indices, "
        "compared over the exact same dates and regimes - the window is "
        "bounded by the credit indices' shorter FRED history (back to "
        f"2023-08-28). Currently {len(rv):,} trading days, "
        f"{rv['date'].min().date()} - {rv['date'].max().date()}."
    )

    st.subheader("Annualized return by regime: S&P 500 vs. credit")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Yield curve regime**")
        st.plotly_chart(
            _rv_grouped_bar(rv, "yield_curve_regime", YIELD_CURVE_ORDER),
            width="stretch",
            theme="streamlit",
        )
    with col2:
        st.markdown("**Dollar regime**")
        st.plotly_chart(
            _rv_grouped_bar(rv, "dollar_regime", DOLLAR_ORDER),
            width="stretch",
            theme="streamlit",
        )
    with col3:
        st.markdown("**Term premium regime**")
        st.plotly_chart(
            _rv_grouped_bar(rv, "term_premium_regime", TERM_PREMIUM_ORDER),
            width="stretch",
            theme="streamlit",
        )
    st.caption(
        "Same yield-curve, dollar, and term-premium regime definitions "
        "used across the dashboard. Thin buckets are noisy - check the "
        "sample-size columns in the table below before reading too much "
        "into any one figure."
    )

    st.subheader("Relative value: S&P 500 minus credit, by regime")
    table = _rv_spread_table(rv)
    display = table.rename(
        columns={
            "yield_curve_regime": "Yield curve",
            "dollar_regime": "Dollar",
            "pct_of_days": "% of days",
            "n_days": "Trading days",
        }
    )
    pct_cols = [c for c in display.columns if c not in ("Yield curve", "Dollar", "% of days", "Trading days")]
    fmt = {c: "{:+.1%}" for c in pct_cols}
    fmt["% of days"] = "{:.1f}%"
    st.dataframe(display.style.format(fmt), width="stretch")
    st.caption(
        "\"Spread\" = S&P 500's annualized return minus that credit "
        "index's annualized return in the same regime bucket - positive "
        "means equity outperformed credit in that regime, negative means "
        "credit outperformed equity."
    )


# ---------------------------------------------------------------------------
# Beta tab: sensitivity of S&P 500 and credit indices to a small move in
# each regime driver
# ---------------------------------------------------------------------------
def _beta_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for inst_label, price_col in RV_INSTRUMENTS.items():
        for driver_col, meta in BETA_DRIVERS.items():
            result = compute_beta(df, price_col, driver_col)
            beta = result["beta"]
            impact_bps = beta * meta["move_size"] * 100 if pd.notna(beta) else float("nan")
            rows.append(
                {
                    "instrument": inst_label,
                    "driver": f"{meta['name']} ({meta['move_label']})",
                    "impact_bps": impact_bps,
                    "r_squared": result["r_squared"],
                    "n": result["n"],
                }
            )
    return pd.DataFrame(rows)


def render_beta_tab() -> None:
    with st.spinner("Loading beta data from FRED..."):
        try:
            df = load_beta_frame(bypass_cache=refresh_clicked)
        except ValueError as e:
            st.error(str(e))
            return

    if df.empty:
        st.info(
            "Not enough overlapping history yet across S&P 500 and the "
            "credit indices to compute sensitivities."
        )
        return

    st.caption(
        "S&P 500 and the three ICE BofA credit indices, aligned onto the "
        "same calendar and bounded by the credit indices' shorter FRED "
        "history (back to 2023-08-28), so every instrument's beta is "
        f"measured over the exact same window. Currently {len(df):,} "
        f"trading days, {df['date'].min().date()} - {df['date'].max().date()}."
    )
    st.markdown(
        "Each cell is a simple linear beta: the instrument's daily % "
        "return per the driver's move (1bp for a rate/spread, $0.01 for "
        "the dollar index), expressed in basis points of return. "
        "Computed only on days the driver actually moved, so Fed funds - "
        "which only changes a handful of times a year - isn't diluted by "
        "the many flat days forward-filled in between."
    )

    beta_df = _beta_table(df)
    driver_order = [f"{meta['name']} ({meta['move_label']})" for meta in BETA_DRIVERS.values()]
    instrument_order = list(RV_INSTRUMENTS.keys())

    impact_pivot = beta_df.pivot(index="instrument", columns="driver", values="impact_bps")
    impact_pivot = impact_pivot.reindex(index=instrument_order, columns=driver_order)
    n_pivot = beta_df.pivot(index="instrument", columns="driver", values="n")
    n_pivot = n_pivot.reindex(index=instrument_order, columns=driver_order)
    r2_pivot = beta_df.pivot(index="instrument", columns="driver", values="r_squared")
    r2_pivot = r2_pivot.reindex(index=instrument_order, columns=driver_order)

    zmax = float(pd.Series(impact_pivot.values.flatten()).abs().max())
    zmax = zmax if zmax and pd.notna(zmax) else 1.0

    cell_text = [
        [f"{v:+.1f}" if pd.notna(v) else "n/a" for v in row]
        for row in impact_pivot.values
    ]
    customdata = [
        [[n_pivot.iloc[r, c], r2_pivot.iloc[r, c]] for c in range(len(driver_order))]
        for r in range(len(instrument_order))
    ]

    fig = go.Figure(
        go.Heatmap(
            z=impact_pivot.values,
            x=impact_pivot.columns.tolist(),
            y=impact_pivot.index.tolist(),
            text=cell_text,
            texttemplate="%{text}",
            customdata=customdata,
            hovertemplate=(
                "%{y} / %{x}<br>Impact: %{z:+.2f} bps"
                "<br>n=%{customdata[0]}, R²=%{customdata[1]:.2f}"
                "<extra></extra>"
            ),
            colorscale="RdBu",
            zmid=0,
            zmin=-zmax,
            zmax=zmax,
            colorbar=dict(title="bps"),
        )
    )
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Driver (move size shown in header)",
        yaxis_title="Instrument",
    )
    st.plotly_chart(fig, width="stretch", theme="streamlit")
    st.caption(
        "Hover a cell for sample size (n) and R². Fed funds only "
        "moves a handful of times within this window (n is often in the "
        "single or low double digits) - treat that column as much less "
        "statistically reliable than the daily-moving drivers."
    )

    with st.expander("Full detail: beta, R², sample size"):
        detail = beta_df.rename(
            columns={
                "instrument": "Instrument",
                "driver": "Driver",
                "impact_bps": "Impact (bps)",
                "r_squared": "R²",
                "n": "N",
            }
        )
        st.dataframe(
            detail.style.format({"Impact (bps)": "{:+.2f}", "R²": "{:.3f}"}),
            width="stretch",
            hide_index=True,
        )


# ---------------------------------------------------------------------------
# Formulas tab (formulas, data sources, and the live source code)
# ---------------------------------------------------------------------------
# Every FRED series referenced anywhere in the app, kept as one explicit
# list rather than derived from the tab-specific dicts above - those only
# capture what's *selectable* per tab, not the regime-driver series (T10Y2Y,
# DTWEXBGS, FEDFUNDS) or SP500, which is used as the regime target but never
# appears in the Single Series dropdown.
DATA_SOURCES = [
    dict(series_id="SP500", label="S&P 500", freq="Daily",
         used_in="Regime Analysis, Relative Value (as the regime target - not in the Single Series dropdown)"),
    dict(series_id="DGS10", label="10Y Treasury Yield", freq="Daily", used_in="Single Series"),
    dict(series_id="DGS2", label="2Y Treasury Yield", freq="Daily", used_in="Single Series"),
    dict(series_id="T10Y2Y", label="10Y-2Y Treasury Spread", freq="Daily",
         used_in="Single Series; yield-curve regime driver (every regime tab); Beta"),
    dict(series_id="DTWEXBGS", label="Trade-Weighted US Dollar Index, Broad, Nominal", freq="Daily",
         used_in="Single Series; dollar regime driver, via its 200-day moving average (every regime tab); Beta"),
    dict(series_id="FEDFUNDS", label="Effective Fed Funds Rate", freq="Monthly",
         used_in="Single Series; Fed regime driver (Regime Analysis tab only); Beta"),
    dict(series_id="THREEFYTP10", label="NY Fed ACM 10-Year Treasury Term Premium", freq="Daily",
         used_in="Single Series; term-premium regime driver (every regime tab); Beta"),
    dict(series_id="CPIAUCSL", label="CPI, All Urban Consumers", freq="Monthly", used_in="Single Series"),
    dict(series_id="BAMLH0A0HYM2", label="ICE BofA US High Yield Index OAS (spread)", freq="Daily", used_in="Single Series"),
    dict(series_id="NIKKEI225", label="Nikkei 225", freq="Daily", used_in="Global Indices"),
    dict(series_id="SPASTT01GBM661N", label="OECD UK broad share-price index (FTSE 100 proxy)", freq="Monthly", used_in="Global Indices"),
    dict(series_id="SPASTT01CNM661N", label="OECD China broad share-price index (SSE Composite proxy)", freq="Monthly", used_in="Global Indices"),
    dict(series_id="BAMLHYH0A0HYM2TRIV", label="ICE BofA US High Yield Total Return Index", freq="Daily", used_in="Credit Indices, Relative Value"),
    dict(series_id="BAMLCC0A0CMTRIV", label="ICE BofA US Corporate Total Return Index", freq="Daily", used_in="Credit Indices, Relative Value"),
    dict(series_id="BAMLEMCBPITRIV", label="ICE BofA EM Corporate Total Return Index", freq="Daily", used_in="Credit Indices, Relative Value"),
]


def _read_source_file(filename: str) -> str:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def render_formulas_tab() -> None:
    st.markdown(
        "The exact math behind every annualized return, annualized "
        "volatility, and Sharpe-like ratio shown across this dashboard "
        "(implemented once, in `regimes.compute_regime_stats`, and reused "
        "by every regime-analysis tab)."
    )

    st.subheader("1. Period return")
    st.latex(r"r_t = \frac{P_t}{P_{t-1}} - 1")
    st.markdown(
        "The simple period-over-period % change of the price series "
        "(`price_col.pct_change()`) - one value per trading day for a "
        "daily instrument (S&P 500, Nikkei, the ICE BofA credit indices), "
        "or per month for a monthly one (the FTSE/SSE OECD proxies)."
    )

    st.subheader("2. Annualized return")
    st.latex(r"\text{Annualized Return} = \bar{r} \times N")
    st.markdown(
        "The **mean** period return within a regime bucket, scaled to a "
        "yearly figure by $N$, the number of periods per year - **252** "
        "for a daily series, **12** for a monthly one. This is a simple "
        "(arithmetic) scaling, not compounded/geometric - that's what "
        "lets a bucket made of scattered, non-contiguous days (a regime "
        "can start and stop many times across the sample) still be "
        "compared on equal footing, since there's no single continuous "
        "holding period to compound over."
    )

    st.subheader("3. Annualized volatility")
    st.latex(r"\text{Annualized Volatility} = \sigma_r \times \sqrt{N}")
    st.markdown(
        "The **sample standard deviation** of period returns within the "
        "bucket (pandas' default, $n-1$ denominator), scaled by "
        r"$\sqrt{N}$ - the standard square-root-of-time rule, which "
        "assumes returns are roughly independent from one period to the "
        "next."
    )

    st.subheader("4. Sharpe-like ratio")
    st.latex(r"\text{Sharpe-like Ratio} = \frac{\text{Annualized Return}}{\text{Annualized Volatility}}")
    st.warning(
        "This is **return divided by volatility**, not a true Sharpe "
        "ratio - no risk-free rate is subtracted from the return first. "
        "It's a measure of return earned per unit of volatility within a "
        "regime, not risk-adjusted excess return in the textbook sense."
    )

    st.subheader("5. Share of the sample (\"% of days\")")
    st.latex(r"\%\ \text{of days} = \frac{n_{\text{bucket}}}{n_{\text{total}}} \times 100")
    st.markdown(
        "The fraction of the sample's periods that fall into a given "
        "regime bucket. Shown next to every return figure across the "
        "dashboard so a thin bucket (small $n_{\\text{bucket}}$, e.g. a "
        "handful of days) doesn't get read with the same confidence as a "
        "well-populated one."
    )

    st.divider()
    st.subheader("Worked example, live")
    example_regime = "Steep"
    with st.spinner("Loading S&P 500 regime data..."):
        try:
            rf = load_regime_frame(bypass_cache=refresh_clicked)
        except ValueError as e:
            st.info(f"Couldn't load live data for the worked example: {e}")
            rf = None

    if rf is not None and not rf.empty:
        stats = compute_regime_stats(
            rf, "yield_curve_regime", price_col="SP500", periods_per_year=TRADING_DAYS_PER_YEAR
        )
        row = stats[stats["yield_curve_regime"] == example_regime].iloc[0]
        n = int(row["n_days"])
        ann_return = row["annualized_return"]
        ann_vol = row["annualized_vol"]
        sharpe = row["sharpe"]
        mean_r = ann_return / TRADING_DAYS_PER_YEAR
        std_r = ann_vol / (TRADING_DAYS_PER_YEAR**0.5)

        st.markdown(
            f"S&P 500, **{example_regime}** yield-curve regime, "
            f"{n:,} trading days in the current sample ($N = 252$):"
        )
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Mean daily return", f"{mean_r:+.3%}")
        c2.metric("Daily std dev", f"{std_r:.3%}")
        c3.metric(
            "Annualized return", f"{ann_return:+.1%}", help=f"{mean_r:+.3%} x 252"
        )
        c4.metric(
            "Annualized vol", f"{ann_vol:.1%}", help=f"{std_r:.3%} x sqrt(252)"
        )
        c5.metric(
            "Sharpe-like ratio", f"{sharpe:+.2f}", help=f"{ann_return:+.1%} / {ann_vol:.1%}"
        )

    st.divider()
    st.subheader("Data sources")
    st.markdown("Every FRED series this dashboard pulls, and which tab(s) use it.")
    st.dataframe(
        pd.DataFrame(DATA_SOURCES).rename(
            columns={
                "series_id": "FRED series ID",
                "label": "What it is",
                "freq": "Native frequency",
                "used_in": "Used in",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "There is no database - each series is pulled directly from the "
        "FRED API (via `fredapi`) and cached to a local CSV per series/date "
        "range for 24 hours (`data.py`'s `get_series`); nothing else is "
        "stored or queried."
    )

    st.divider()
    st.subheader("Reference database schema")
    st.markdown(
        "The app doesn't run a database today - the CSV cache above is "
        "all there is. This is the schema **if** the data model here were "
        "backed by one: how the raw series, regime classifications, and "
        "computed stats would normalize into tables. SQLite-flavored DDL "
        "(runs as-is in SQLite; swap `INTEGER PRIMARY KEY AUTOINCREMENT` "
        "for `SERIAL PRIMARY KEY` and `BOOLEAN`/`TIMESTAMP` are native "
        "types for Postgres)."
    )
    st.caption(
        "Also worth knowing: this app runs on Streamlit Community Cloud, "
        "whose filesystem is ephemeral - a SQLite file written there would "
        "be wiped on every redeploy/restart, so a *real* persistent "
        "version of this would need Postgres (or similar) with its own "
        "hosted instance, not SQLite."
    )
    st.code(
        '''-- Raw FRED series observations (one row per series per date;
-- mirrors data.get_series's per-series CSV cache)
CREATE TABLE fred_series_value (
    series_id   TEXT    NOT NULL,   -- FRED series ID, e.g. 'SP500', 'T10Y2Y'
    date        DATE    NOT NULL,
    value       REAL    NOT NULL,
    PRIMARY KEY (series_id, date)
);

-- Static metadata about each series (mirrors app.py's DATA_SOURCES table)
CREATE TABLE fred_series_meta (
    series_id    TEXT PRIMARY KEY,
    label        TEXT NOT NULL,
    native_freq  TEXT NOT NULL CHECK (native_freq IN ('Daily', 'Monthly', 'Quarterly')),
    is_proxy     BOOLEAN NOT NULL DEFAULT 0,
    proxy_note   TEXT
);

-- Day-level (or month-level, for monthly instruments) regime classification
-- per target instrument (mirrors regimes.build_regime_frame /
-- build_three_factor_regime_frame)
CREATE TABLE regime_day (
    target_series_id    TEXT NOT NULL,  -- the instrument being classified, e.g. 'SP500'
    date                 DATE NOT NULL,
    yield_curve_regime    TEXT NOT NULL CHECK (yield_curve_regime IN ('Inverted','Flattening','Steep')),
    dollar_regime          TEXT NOT NULL CHECK (dollar_regime IN ('Strong Dollar','Weak Dollar')),
    term_premium_regime    TEXT NOT NULL CHECK (term_premium_regime IN ('Negative','Low','Elevated')),
    fed_regime              TEXT CHECK (fed_regime IN ('Cutting','Hold','Hiking')),  -- NULL outside the S&P 500 tab
    regime_label             TEXT NOT NULL,  -- e.g. 'Steep / Weak Dollar / Hold / Low'
    PRIMARY KEY (target_series_id, date),
    FOREIGN KEY (target_series_id) REFERENCES fred_series_meta(series_id)
);

-- Precomputed regime-bucket statistics (mirrors compute_regime_stats)
CREATE TABLE regime_stats (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    target_series_id    TEXT NOT NULL,
    group_by             TEXT NOT NULL,  -- e.g. 'yield_curve_regime' or 'yield_curve_regime,dollar_regime'
    regime_bucket         TEXT NOT NULL,  -- e.g. 'Steep' or 'Steep|Weak Dollar'
    n_periods              INTEGER NOT NULL,
    pct_of_periods          REAL NOT NULL,
    annualized_return       REAL NOT NULL,
    annualized_vol           REAL NOT NULL,
    sharpe_like_ratio         REAL,
    computed_at                TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (target_series_id) REFERENCES fred_series_meta(series_id)
);

-- Contiguous regime periods (mirrors regimes.regime_periods)
CREATE TABLE regime_period (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    target_series_id    TEXT NOT NULL,
    regime_label          TEXT NOT NULL,
    start_date             DATE NOT NULL,
    end_date                 DATE NOT NULL,
    trading_days              INTEGER NOT NULL,
    return_pct                 REAL NOT NULL,
    FOREIGN KEY (target_series_id) REFERENCES fred_series_meta(series_id)
);

CREATE INDEX idx_fred_series_value_date ON fred_series_value(date);
CREATE INDEX idx_regime_day_label ON regime_day(regime_label);
CREATE INDEX idx_regime_period_dates ON regime_period(start_date, end_date);''',
        language="sql",
    )

    st.divider()
    st.subheader("Source code")
    st.markdown(
        "Read live from the app's own files at render time, so this is "
        "always exactly what's running - not a snapshot that can drift out "
        "of sync with the code above."
    )
    with st.expander(
        "regimes.py - regime classification, annualized return/vol/Sharpe, regime periods"
    ):
        st.code(_read_source_file("regimes.py"), language="python")
    with st.expander(
        "data.py - FRED fetch, disk caching, calendar alignment"
    ):
        st.code(_read_source_file("data.py"), language="python")


tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "Single Series",
        "Regime Analysis",
        "Global Indices",
        "Credit Indices",
        "Relative Value",
        "Beta",
        "Methodology",
    ]
)
with tab1:
    render_single_series_tab()
with tab2:
    render_regime_tab()
with tab3:
    render_global_indices_tab()
with tab4:
    render_credit_indices_tab()
with tab5:
    render_relative_value_tab()
with tab6:
    render_beta_tab()
with tab7:
    render_formulas_tab()
