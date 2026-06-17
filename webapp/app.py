from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webapp.data.normalization import grouped_roi, race_summary, summarize_bets
from webapp.data.repository import load_all_normalized, load_config, write_inventory


CONFIG_PATH = ROOT / "config/current_model_webapp_mvp_v1.yaml"


@st.cache_data(show_spinner="保存済み予測を読み込み中...")
def cached_data(config_path: str) -> tuple[dict, pd.DataFrame, str]:
    config = load_config(config_path)
    inventory_path = write_inventory(config)
    df = load_all_normalized(config)
    return config, df, str(inventory_path)


def fmt_yen(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{int(round(float(value))):,}円"


def fmt_pct(value: float | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}%"


def sidebar_filters(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    if df.empty:
        return df
    st.sidebar.header("Filters")
    default_strategy = config["app"].get("default_strategy", "ROLLING_10Y")
    strategies = sorted(df["strategy"].dropna().unique().tolist())
    strategy = st.sidebar.selectbox("Strategy", strategies, index=strategies.index(default_strategy) if default_strategy in strategies else 0)
    source_values = sorted(df["source_type"].dropna().unique().tolist())
    default_sources = [s for s in source_values if s != "FIXTURE"]
    selected_sources = st.sidebar.multiselect("Source type", source_values, default=default_sources)
    exclude_fixture = st.sidebar.checkbox("Exclude fixture", value=bool(config["app"].get("exclude_fixture_by_default", True)))
    tier_values = ["ALL", "CORE", "MARGIN", "HIGH", "VERY_HIGH", "NONE"]
    selected_tier = st.sidebar.selectbox("Tier", tier_values, index=1)
    racecourses = ["ALL"] + sorted(df["racecourse"].dropna().unique().tolist())
    racecourse = st.sidebar.selectbox("Racecourse", racecourses)
    dates = pd.to_datetime(df["race_date"], errors="coerce")
    min_date = dates.min().date()
    max_date = dates.max().date()
    date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    filtered = df[df["strategy"].eq(strategy)].copy()
    if selected_sources:
        filtered = filtered[filtered["source_type"].isin(selected_sources)]
    if exclude_fixture:
        filtered = filtered[~filtered["fixture"]]
    if selected_tier != "ALL":
        if selected_tier == "CORE":
            race_ids = filtered.loc[filtered["expected_value"].ge(1.0), "race_id"].unique()
        else:
            race_ids = filtered.loc[filtered["tier"].eq(selected_tier), "race_id"].unique()
        filtered = filtered[filtered["race_id"].isin(race_ids)]
    if racecourse != "ALL":
        filtered = filtered[filtered["racecourse"].eq(racecourse)]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = [pd.Timestamp(d).date().isoformat() for d in date_range]
        filtered = filtered[(filtered["race_date"] >= start) & (filtered["race_date"] <= end)]
    return filtered


def dashboard(df: pd.DataFrame) -> None:
    st.header("Dashboard")
    summary = summarize_bets(df)
    cols = st.columns(4)
    cols[0].metric("全体ROI", fmt_pct(summary["roi"]))
    cols[1].metric("総収支", fmt_yen(summary["total_profit_yen"]))
    cols[2].metric("総投資額", fmt_yen(summary["total_stake_yen"]))
    cols[3].metric("総払戻額", fmt_yen(summary["total_payout_yen"]))
    cols = st.columns(5)
    cols[0].metric("購入数", f"{summary['bets']:,}")
    cols[1].metric("的中数", f"{summary['hits']:,}")
    cols[2].metric("的中率", fmt_pct(summary["hit_rate"] * 100 if not pd.isna(summary["hit_rate"]) else np.nan))
    cols[3].metric("対象レース数", f"{summary['races']:,}")
    cols[4].metric("対象期間", f"{summary['date_min']} - {summary['date_max']}")
    if df.empty:
        st.warning("No rows under current filters.")
        return
    bets = df[df["selected_for_bet"]].sort_values("race_date").copy()
    if bets.empty:
        st.warning("No bets under current filters.")
        return
    bets["cumulative_profit_yen"] = bets["profit_yen"].cumsum()
    bets["cumulative_stake_yen"] = bets["stake_yen"].cumsum()
    bets["cumulative_payout_yen"] = bets["payout_yen"].cumsum()
    bets["cumulative_roi"] = np.where(bets["cumulative_stake_yen"] > 0, bets["cumulative_payout_yen"] / bets["cumulative_stake_yen"] * 100, np.nan)
    st.plotly_chart(px.line(bets, x="race_date", y="cumulative_profit_yen", title="Cumulative Profit"), use_container_width=True)
    st.plotly_chart(px.line(bets, x="race_date", y="cumulative_roi", title="Cumulative ROI"), use_container_width=True)
    monthly = df.copy()
    monthly["month"] = pd.to_datetime(monthly["race_date"]).dt.to_period("M").astype(str)
    st.plotly_chart(px.bar(grouped_roi(monthly, "month"), x="month", y="roi", hover_data=["bets", "profit_yen"], title="Monthly ROI"), use_container_width=True)


def race_calendar(df: pd.DataFrame) -> None:
    st.header("Race Calendar")
    if df.empty:
        st.warning("No race data under current filters.")
        return
    available = sorted(df["race_date"].unique().tolist())
    selected_date = st.date_input("開催日", value=pd.to_datetime(available[-1]).date(), min_value=pd.to_datetime(available[0]).date(), max_value=pd.to_datetime(available[-1]).date())
    selected_str = selected_date.isoformat()
    st.caption("Available dates: " + ", ".join(available[-20:]))
    day_df = df[df["race_date"].eq(selected_str)]
    if day_df.empty:
        st.warning("Selected date has no rows under current filters.")
        return
    races = race_summary(day_df).sort_values(["racecourse", "RaceNum"])
    races["label"] = races["racecourse"] + races["RaceNum"].astype(str) + "R / " + races["Kyori"].fillna(0).astype(int).astype(str) + "m / 予想" + races["selected_horses"].astype(int).astype(str) + "頭 / 的中" + races["hits"].astype(int).astype(str) + "頭 / 収支" + races["profit_yen"].astype(int).astype(str) + "円"
    st.dataframe(races[["label", "runners", "selected_horses", "actual_place_horses", "stake_yen", "payout_yen", "profit_yen"]], use_container_width=True, hide_index=True)
    label = st.selectbox("Race", races["label"].tolist())
    race_id = races.loc[races["label"].eq(label), "race_id"].iloc[0]
    detail = day_df[day_df["race_id"].eq(race_id)].sort_values("Umaban")
    show_race_detail(detail)


def show_race_detail(detail: pd.DataFrame) -> None:
    summary = summarize_bets(detail)
    st.subheader(f"{detail['racecourse'].iloc[0]} {int(detail['RaceNum'].iloc[0])}R")
    cols = st.columns(5)
    cols[0].metric("予想頭数", int(detail["selected_for_bet"].sum()))
    cols[1].metric("複勝圏内頭数", int(detail["target_place_paid"].sum()))
    cols[2].metric("的中", summary["hits"])
    cols[3].metric("レース収支", fmt_yen(summary["total_profit_yen"]))
    cols[4].metric("レースROI", fmt_pct(summary["roi"]))

    selected = detail[detail["selected_for_bet"]]
    actual = detail[detail["target_place_paid"].eq(1)]
    left, right = st.columns(2)
    with left:
        st.markdown("**予想した馬**")
        if selected.empty:
            st.write("なし")
        for _, row in selected.iterrows():
            st.write(f"- {int(row['Umaban'])}番 {row['horse_name']}: EV {row['expected_value']:.3f} / 複勝 {row['fuku_odds_low']} - {row['fuku_odds_high']}")
    with right:
        st.markdown("**実際の複勝圏内馬**")
        if actual.empty:
            st.write("なし")
        for _, row in actual.iterrows():
            st.write(f"- {int(row['Umaban'])}番 {row['horse_name']}: 払戻 {int(row['fuku_pay'])}円")

    display = detail.rename(columns={
        "Umaban": "馬番",
        "horse_name": "馬名",
        "probability_market": "市場確率",
        "probability_raw": "Raw確率",
        "probability_calibrated": "補正済み確率",
        "fuku_odds_low": "複勝オッズ下限",
        "fuku_odds_high": "複勝オッズ上限",
        "expected_value": "EV",
        "tier": "Tier",
        "selected_for_bet": "購入対象か",
        "actual_finish_position": "実着順",
        "target_place_paid": "実際に複勝圏内か",
        "fuku_pay": "複勝払戻",
        "stake_yen": "購入額",
        "profit_yen": "収支",
    })
    cols = ["馬番", "馬名", "市場確率", "Raw確率", "補正済み確率", "複勝オッズ下限", "複勝オッズ上限", "EV", "Tier", "購入対象か", "実着順", "実際に複勝圏内か", "複勝払戻", "購入額", "収支"]
    st.dataframe(display[cols], use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(detail, x="horse_name", y="expected_value", color="selected_for_bet", title="EV by horse"), use_container_width=True)
    prob_cols = [c for c in ["probability_market", "probability_raw", "probability_calibrated"] if detail[c].notna().any()]
    if prob_cols:
        long = detail[["horse_name"] + prob_cols].melt(id_vars="horse_name", var_name="probability", value_name="value")
        st.plotly_chart(px.bar(long, x="horse_name", y="value", color="probability", barmode="group", title="Market / Raw / Calibrated Probability"), use_container_width=True)


def analysis(df: pd.DataFrame) -> None:
    st.header("Analysis")
    if df.empty:
        st.warning("No data under current filters.")
        return
    work = df.copy()
    work["year"] = pd.to_datetime(work["race_date"]).dt.year
    work["month"] = pd.to_datetime(work["race_date"]).dt.to_period("M").astype(str)
    work["odds_band"] = pd.cut(work["fuku_odds_low"], bins=[0, 1.5, 2, 3, 5, 10, 999], labels=["<=1.5", "1.5-2", "2-3", "3-5", "5-10", "10+"])
    work["popularity_band"] = pd.cut(work["fuku_ninki"].fillna(work["tan_ninki"]), bins=[0, 1, 3, 5, 10, 999], labels=["1", "2-3", "4-5", "6-10", "11+"])
    work["ev_band"] = pd.cut(work["expected_value"], bins=[0, 1, 1.05, 1.10, 1.15, 999], labels=["<1.00", "1.00-1.05", "1.05-1.10", "1.10-1.15", "1.15+"])
    for label, col in [
        ("年度別ROI", "year"),
        ("月別ROI", "month"),
        ("競馬場別ROI", "racecourse"),
        ("オッズ帯別ROI", "odds_band"),
        ("人気帯別ROI", "popularity_band"),
        ("EV帯別ROI", "ev_band"),
        ("Tier別ROI", "tier"),
    ]:
        table = grouped_roi(work, col)
        st.subheader(label)
        st.dataframe(table, use_container_width=True, hide_index=True)
        if not table.empty:
            st.plotly_chart(px.bar(table, x=col, y="roi", hover_data=["bets", "profit_yen"], title=label), use_container_width=True)
    st.download_button("Download filtered rows CSV", data=work.to_csv(index=False).encode("utf-8-sig"), file_name="current_model_filtered_rows.csv", mime="text/csv")


def model_info(config: dict, inventory_path: str, df: pd.DataFrame) -> None:
    st.header("Model Info")
    info = config["model_info"]
    st.code(
        f"""Champion:
{info['champion']}

CatBoost:
{info['catboost_path']}

CatBoost SHA256:
{info['catboost_sha256']}

Official Platt:
{info['official_platt_path']}

Platt SHA256:
{info['official_platt_sha256']}

live raw prediction:
{info['live_raw_prediction']}

saved prediction visualization:
{info['saved_prediction_visualization']}"""
    )
    st.markdown("このアプリは保存済み予測を読み取り専用で可視化します。モデル再学習、calibrator refit、SQLite更新、予測上書き、実購入は行いません。")
    st.write("Inventory:", inventory_path)
    st.write("Rows:", len(df), "Races:", df["race_id"].nunique() if not df.empty else 0)
    if not df.empty:
        st.dataframe(df.groupby(["source_type", "fixture"], dropna=False).size().reset_index(name="rows"), hide_index=True)


def main() -> None:
    base_config = load_config(str(CONFIG_PATH))
    st.set_page_config(page_title=base_config["app"]["title"], layout="wide")
    config, df, inventory_path = cached_data(str(CONFIG_PATH))
    st.title(config["app"]["title"])
    if df.empty:
        st.error("No saved prediction data found.")
        model_info(config, inventory_path, df)
        return
    filtered = sidebar_filters(df, config)
    tabs = st.tabs(["Dashboard", "Race Calendar", "Analysis", "Model Info"])
    with tabs[0]:
        dashboard(filtered)
    with tabs[1]:
        race_calendar(filtered)
    with tabs[2]:
        analysis(filtered)
    with tabs[3]:
        model_info(config, inventory_path, df)


if __name__ == "__main__":
    main()
