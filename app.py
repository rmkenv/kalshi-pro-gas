import streamlit as st
import requests
import pandas as pd
import traceback
import re as _re
import time
from datetime import datetime, timezone

try:
    from kalshi_pro_gas import ProGasAlgo
except ImportError:
    class ProGasAlgo:
        def __init__(self, **kwargs): pass
        def refresh_data(self, **kwargs):
            return {
                "gas_momentum": {"current": 2.924, "momentum": 0, "trend": 0, "avg_52w": 2.924, "signal": 0},
                "wti":          {"current_wti": 62.53, "lagged_wti": 63.0, "wti_change": -0.007, "optimal_lag": 1, "signal": -0.007},
                "regional":     {"weighted_avg": None, "regional_data": {}, "divergence_signal": 0},
                "seasonal":     {"multiplier": 1.0, "signal": 0.0, "factors": []},
                "refinery":     {"current": None, "signal": 0, "status": "Data Unavailable"},
                "inventory":    {"current": None, "signal": 0, "status": "unknown", "z_score": 0, "wow_change": 0},
            }
        def edge(self, title, yes_price):
            match = _re.search(r'\$(\d+\.\d+)', title)
            strike = float(match.group(1)) if match else 3.00
            base_signal = (2.924 - strike) / 2.924
            combined = base_signal * 0.7 - 0.01
            fair = max(0.05, min(0.95, 0.5 + combined))
            raw_edge = fair - yes_price
            if 0 < raw_edge < 0.15:
                return 0.0
            if raw_edge > 0 and combined < -0.02:
                return -raw_edge
            return raw_edge

    import streamlit as _st
    _st.warning("⚠️ `kalshi_pro_gas` module not found — running in stub mode.")


# ─────────────────────────────────────────
# Constants
# ─────────────────────────────────────────
BASE       = "https://api.elections.kalshi.com/trade-api/v2"
GAS_SERIES = ["KXAAAGASM", "KXAAAGASW", "KXAAGASM", "KXAAGASW"]
AUTO_REFRESH_INTERVAL = 15 * 60  # 15 minutes in seconds


# ─────────────────────────────────────────
# Kalshi helpers
# ─────────────────────────────────────────

def search_kalshi_gas_markets() -> list[dict]:
    results, seen = [], set()
    for series in GAS_SERIES:
        cursor = None
        while True:
            params = {"series_ticker": series, "status": "open", "limit": 100}
            if cursor:
                params["cursor"] = cursor
            try:
                r = requests.get(f"{BASE}/markets", params=params, timeout=10)
                r.raise_for_status()
                data    = r.json()
                markets = data.get("markets", [])
                for m in markets:
                    ticker = m.get("ticker", "")
                    if ticker in seen:
                        continue
                    seen.add(ticker)
                    yes_bid = m.get("yes_bid")
                    yes_ask = m.get("yes_ask")
                    yb = yes_bid / 100.0 if isinstance(yes_bid, (int, float)) else None
                    ya = yes_ask / 100.0 if isinstance(yes_ask, (int, float)) else None
                    mid    = (yb + ya) / 2.0 if yb and ya else None
                    spread = (ya - yb)        if yb and ya else None
                    results.append({
                        "ticker":        ticker,
                        "title":         m.get("title", "").replace("**", ""),
                        "strike":        m.get("floor_strike"),
                        "yes_bid":       yb,
                        "yes_ask":       ya,
                        "mid_price":     mid,
                        "spread_est":    spread,
                        "last_price":    m.get("last_price", 0) / 100.0 if m.get("last_price") else None,
                        "volume":        m.get("volume"),
                        "open_interest": m.get("open_interest"),
                        "close_time":    m.get("close_time"),
                    })
                cursor = data.get("cursor")
                if not cursor or not markets:
                    break
            except Exception as e:
                st.warning(f"Error fetching {series}: {e}")
                break
    results.sort(key=lambda x: x.get("strike") or 0)
    return results


def get_kalshi_orderbook(ticker: str) -> dict | None:
    try:
        r  = requests.get(f"{BASE}/markets/{ticker}/orderbook", timeout=10)
        r.raise_for_status()
        ob = r.json().get("orderbook", {})
        return {"yes_depth": ob.get("yes", [])[:5], "no_depth": ob.get("no", [])[:5]}
    except Exception:
        return None


# ─────────────────────────────────────────
# Kelly sizing
# ─────────────────────────────────────────

def kelly_fraction(edge: float, price: float) -> float:
    """
    Full Kelly fraction, capped 1–25%.
    p = implied win prob from edge
    b = net odds (payout per $1 risked on YES)
    f = (p*b - (1-p)) / b
    """
    if price <= 0 or price >= 1 or edge == 0:
        return 0.0
    p = min(0.95, max(0.05, 0.5 + edge * 0.5))
    b = (1.0 - price) / price          # net odds
    f = (p * b - (1.0 - p)) / b
    return round(min(0.25, max(0.01, f)), 4) if f > 0 else 0.0


# ─────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────

def score_market(market: dict, algo, signals: dict) -> dict:
    mid = market.get("mid_price")
    if mid is None:
        return {**market, "edge": None, "fair_value": None, "kelly": None,
                "signal_score": 0, "recommendation": "NO DATA", "reasons": [], "risks": []}

    edge   = algo.edge(market["title"], mid)
    spread = market.get("spread_est", 0.05) or 0.05
    kelly  = kelly_fraction(edge, mid) if edge > 0 else 0.0

    score, reasons, risks = 0, [], []

    # Edge strength
    if edge >= 0.10:
        score += 3; reasons.append(f"Strong edge ({edge:.1%})")
    elif edge >= 0.05:
        score += 2; reasons.append(f"Moderate edge ({edge:.1%})")
    elif edge > 0:
        score += 1; reasons.append(f"Weak edge ({edge:.1%})")
    else:
        risks.append(f"Negative edge ({edge:.1%})")

    # Edge vs spread
    if edge > 0 and spread > 0:
        if edge >= 2 * spread:
            score += 2; reasons.append("Edge ≥ 2× spread")
        elif edge >= spread:
            score += 1; reasons.append("Edge covers spread")
        else:
            risks.append("Edge < spread")

    # WTI
    wti_chg = (signals.get("wti") or {}).get("wti_change", 0) or 0
    if wti_chg > 0.02:
        score += 1; reasons.append(f"WTI rising ({wti_chg:+.1%})")
    elif wti_chg < -0.02:
        risks.append(f"WTI falling ({wti_chg:+.1%})")

    # Inventory
    inv = signals.get("inventory") or {}
    if inv.get("current") is not None:
        z = inv.get("z_score", 0) or 0
        if z < -0.5:
            score += 1; reasons.append(f"Tight inventory (Z={z:.2f})")
        elif z > 0.5:
            risks.append(f"Ample inventory (Z={z:.2f})")

    # Seasonal
    mult = (signals.get("seasonal") or {}).get("multiplier", 1.0) or 1.0
    if mult >= 1.05:
        score += 1; reasons.append(f"Seasonal tailwind ({mult:.2f}×)")
    elif mult < 1.0:
        risks.append(f"Seasonal headwind ({mult:.2f}×)")

    if score >= 6:   rec = "🟢 STRONG BUY"
    elif score >= 4: rec = "🟡 BUY"
    elif score >= 2 and edge > 0: rec = "🟠 WEAK BUY"
    else:            rec = "🔴 PASS"

    return {
        **market,
        "edge":           edge,
        "fair_value":     max(0.0, min(1.0, mid + edge)),
        "kelly":          kelly,
        "signal_score":   score,
        "recommendation": rec,
        "reasons":        reasons,
        "risks":          risks,
    }


def run_full_scan(api_key: str, force: bool = False) -> tuple[list[dict], dict]:
    """Fetch markets + FRED signals + score everything. Returns (scored, signals)."""
    markets = search_kalshi_gas_markets()
    if not markets:
        return [], {}
    algo    = ProGasAlgo(fred_api_key=api_key)
    signals = algo.refresh_data(force=force)
    scored  = [score_market(m, algo, signals) for m in markets]
    scored  = [m for m in scored if m.get("edge") is not None]
    scored.sort(key=lambda x: x.get("edge", -99), reverse=True)
    return scored, signals


# ─────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────
st.set_page_config(page_title="Pro Gas Algo", page_icon="⛽", layout="wide")


# ─────────────────────────────────────────
# Auto-Refresh Logic
# ─────────────────────────────────────────

# Initialize timestamps
if "last_scan_time" not in st.session_state:
    st.session_state["last_scan_time"] = 0.0
if "auto_refresh_enabled" not in st.session_state:
    st.session_state["auto_refresh_enabled"] = False

now = time.time()
time_since_last = now - st.session_state["last_scan_time"]
next_refresh_in = max(0, AUTO_REFRESH_INTERVAL - time_since_last)

# Trigger auto-rerun if enabled and interval elapsed
if st.session_state["auto_refresh_enabled"] and time_since_last >= AUTO_REFRESH_INTERVAL:
    st.session_state["trigger_auto_scan"] = True


# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────

with st.sidebar:
    st.header("🔑 Configuration")
    api_key = None
    if "FRED_API_KEY" in st.secrets:
        api_key = st.secrets["FRED_API_KEY"]
        st.success("✅ FRED API key loaded.")
    else:
        st.error("❌ Missing `FRED_API_KEY` in `.streamlit/secrets.toml`.")

    st.divider()
    st.header("⏱️ Auto-Refresh")
    auto_enabled = st.toggle(
        "Auto-scan every 15 min",
        value=st.session_state["auto_refresh_enabled"],
        help="Automatically re-runs the full market scan every 15 minutes."
    )
    st.session_state["auto_refresh_enabled"] = auto_enabled

    if auto_enabled:
        if st.session_state["last_scan_time"] > 0:
            mins, secs = divmod(int(next_refresh_in), 60)
            st.info(f"⏳ Next scan in **{mins}m {secs}s**")
            last_dt = datetime.fromtimestamp(st.session_state["last_scan_time"])
            st.caption(f"Last scan: {last_dt.strftime('%H:%M:%S')}")
        else:
            st.info("▶️ Click **Auto-Scan** to start the first scan.")

        if st.button("⏸️ Force Refresh Now"):
            st.session_state["trigger_auto_scan"] = True
            st.rerun()
    else:
        st.caption("Toggle on to enable automatic 15-minute rescanning.")

    st.divider()
    if st.button("🔄 Force Refresh FRED Cache"):
        st.session_state["force_refresh"] = True
        st.success("✅ Will refresh FRED on next scan.")

    force_refresh = st.session_state.pop("force_refresh", False)

    st.divider()
    st.header("📖 How This Works")
    st.markdown("""
**Auto-Scan Mode**

Fetches ALL open gas markets, loads live FRED signals once, scores every market, and ranks by edge.

| Signal | Weight |
|---|---|
| Gas Momentum | 35% |
| WTI Crude (lagged) | 25% |
| Inventory | 20% |
| Refinery Util. | 10% |
| Regional PADD | 5% |
| Seasonal | 5% |

**Kelly Sizing**

`f = (p·b − (1−p)) / b`

where `p = 0.5 + edge×0.5`, capped 1–25%.

**Decision Scores**

- **6+** = Strong Buy
- **4+** = Buy
- **2+** = Weak Buy
- **<2** = Pass

⚠️ Not financial advice.
    """)


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

st.title("⛽ Kalshi Pro Gas Algorithm")
st.markdown("Multi-factor gas price prediction for Kalshi prediction markets — powered by FRED economic data.")

# ── AUTO-SCAN SECTION ──
st.subheader("🚀 Auto-Scan: All Open Markets")
st.caption("Scores every open Kalshi gas market at once and ranks by edge.")

col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
with col_btn1:
    manual_scan = st.button("🚀 Scan Now", type="primary", use_container_width=True)
with col_btn2:
    min_edge = st.slider("Min Edge", 0.00, 0.25, 0.05, 0.01, label_visibility="collapsed")
with col_btn3:
    st.caption(f"Minimum edge threshold: **{min_edge:.0%}**")

# Combine manual + auto triggers
trigger_scan = manual_scan or st.session_state.pop("trigger_auto_scan", False)

if trigger_scan:
    if not api_key:
        st.error("❌ No FRED API key. Add `FRED_API_KEY` to `.streamlit/secrets.toml`.")
    else:
        progress = st.progress(0, text="Fetching markets...")
        with st.spinner("Running full scan..."):
            progress.progress(20, text="Fetching open markets from Kalshi...")
            try:
                scored, signals = run_full_scan(api_key, force=force_refresh)
                progress.progress(100, text="✅ Scan complete!")
                st.session_state["scan_results"]   = scored
                st.session_state["scan_signals"]   = signals
                st.session_state["last_scan_time"] = time.time()
            except Exception as e:
                st.error(f"❌ Scan failed: {e}")
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())

# ── Display Results ──
if "scan_results" in st.session_state:
    scored  = st.session_state["scan_results"]
    signals = st.session_state["scan_signals"]

    buys      = [m for m in scored if "BUY" in m["recommendation"]]
    strong    = [m for m in scored if m["recommendation"] == "🟢 STRONG BUY"]
    above_thr = [m for m in scored if abs(m.get("edge") or 0) >= min_edge and m.get("edge", 0) > 0]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Markets Scanned",    len(scored))
    k2.metric("🟢 Strong Buys",     len(strong))
    k3.metric("Any Buy Signal",     len(buys))
    k4.metric(f"Above {min_edge:.0%} Edge", len(above_thr))

    st.markdown("### 🎯 Ranked Recommendations")

    if above_thr:
        rows = []
        for m in above_thr:
            edge_val = m.get("edge", 0) or 0
            kelly_val = m.get("kelly", 0) or 0
            rows.append({
                "Rec":          m["recommendation"],
                "Score":        m["signal_score"],
                "Ticker":       m["ticker"],
                "Title":        m["title"][:55] + ("…" if len(m["title"]) > 55 else ""),
                "Strike":       f"${m['strike']:.2f}" if m.get("strike") is not None else "N/A",
                "Mid":          f"{m['mid_price']:.0%}"   if m.get("mid_price")  is not None else "N/A",
                "Edge":         f"{edge_val:+.1%}",
                "Fair Value":   f"{m['fair_value']:.0%}"  if m.get("fair_value") is not None else "N/A",
                "Kelly %":      f"{kelly_val:.1%}" if kelly_val else "—",
                "Spread":       f"{m['spread_est']:.0%}"  if m.get("spread_est") is not None else "N/A",
                "Volume":       m.get("volume") or 0,
                "Closes":       m["close_time"][:10] if m.get("close_time") else "N/A",
                "Reasons":      " · ".join(m.get("reasons", [])[:2]),
            })

        rdf = pd.DataFrame(rows)
        st.dataframe(
            rdf,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rec":     st.column_config.TextColumn("Rec",   width="small"),
                "Score":   st.column_config.ProgressColumn("Score", min_value=0, max_value=8, format="%d"),
                "Edge":    st.column_config.TextColumn("Edge",  width="small"),
                "Kelly %": st.column_config.TextColumn("Kelly %", width="small",
                            help="Fraction of bankroll to risk (Full Kelly, capped 1–25%)"),
            }
        )

        csv = rdf.to_csv(index=False)
        dl1, dl2 = st.columns([1, 5])
        with dl1:
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            st.download_button("⬇️ CSV", csv, f"kalshi_gas_scan_{ts}.csv", "text/csv")

    else:
        st.info(f"No buy signals above {min_edge:.0%} edge threshold. Lower the slider to see more.")

    # ── FRED Signal summary ──
    with st.expander("📡 Live FRED Signals"):
        gm  = signals.get("gas_momentum") or {}
        wti = signals.get("wti")          or {}
        inv = signals.get("inventory")    or {}
        sea = signals.get("seasonal")     or {}
        ref = signals.get("refinery")     or {}
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Gas Price",       f"${gm.get('current', 0):.3f}/gal" if gm.get("current") else "N/A")
        s1.metric("4-Wk Momentum",   f"{gm.get('momentum', 0):+.2%}")
        s2.metric("WTI",             f"${wti.get('current_wti', 0):.2f}" if wti.get("current_wti") else "N/A")
        s2.metric("WTI Change",      f"{wti.get('wti_change', 0):+.2%}")
        s3.metric("Inventory",       inv.get("status", "N/A"))
        s3.metric("Z-Score",         f"{inv.get('z_score', 0):.2f}")
        s4.metric("Seasonal Mult.",  f"{sea.get('multiplier', 1.0):.3f}×")
        s4.metric("Seasonal Signal", f"{sea.get('signal', 0):+.3f}")
        s5.metric("Refinery",        ref.get("status", "N/A"))
        s5.metric("Refinery Util.",  f"{ref.get('current', 0):.1f}%" if ref.get("current") else "N/A")

# ── Auto-refresh countdown ticker (re-fires the page every 60s to update countdown) ──
if st.session_state.get("auto_refresh_enabled") and st.session_state.get("last_scan_time", 0) > 0:
    time_left = AUTO_REFRESH_INTERVAL - (time.time() - st.session_state["last_scan_time"])
    if time_left > 0:
        # Sleep 60s then rerun to update sidebar countdown and check if scan needed
        time.sleep(min(60, time_left))
        st.rerun()
    else:
        st.session_state["trigger_auto_scan"] = True
        st.rerun()

st.divider()

# ─────────────────────────────────────────
# MANUAL SINGLE-MARKET (unchanged below)
# ─────────────────────────────────────────
st.subheader("🔍 Manual Single-Market Analysis")

if st.button("🔍 Find Open Gas Markets"):
    with st.spinner("Fetching..."):
        gm_list = search_kalshi_gas_markets()
    st.session_state["gas_markets"] = gm_list
    st.session_state.pop("selected_market", None)

if "gas_markets" in st.session_state:
    gm_list = st.session_state["gas_markets"]
    if not gm_list:
        st.warning("No open gas markets found.")
    else:
        df = pd.DataFrame([{
            "Select": False, "Ticker": m["ticker"], "Title": m["title"],
            "Strike": f"${m['strike']:.2f}" if m.get("strike") is not None else "N/A",
            "Mid":    f"{m['mid_price']:.0%}" if m.get("mid_price") is not None else "N/A",
            "Volume": m.get("volume") or 0,
            "Closes": m["close_time"][:10] if m.get("close_time") else "N/A",
        } for m in gm_list])

        edited = st.data_editor(df,
            column_config={"Select": st.column_config.CheckboxColumn("Select")},
            use_container_width=True, hide_index=True, key="market_table")

        sel = edited[edited["Select"] == True]
        if len(sel) == 1:
            ticker = sel.iloc[0]["Ticker"]
            mkt    = next((m for m in gm_list if m["ticker"] == ticker), None)
            if mkt:
                st.session_state["selected_market"] = mkt
                ob = get_kalshi_orderbook(ticker)
                if ob:
                    with st.expander("📊 Orderbook"):
                        d1, d2 = st.columns(2)
                        with d1:
                            st.markdown("**YES Bids**")
                            for row in ob["yes_depth"]:
                                st.markdown(f"- {row[0]}¢ × {row[1]}")
                        with d2:
                            st.markdown("**NO Bids**")
                            for row in ob["no_depth"]:
                                st.markdown(f"- {row[0]}¢ × {row[1]}")

sel_market = st.session_state.get("selected_market", {})
default_title  = sel_market.get("title", "")
default_mid    = float(sel_market["mid_price"])  if isinstance(sel_market.get("mid_price"),  (int, float)) else 0.50
default_spread = float(sel_market["spread_est"]) if isinstance(sel_market.get("spread_est"), (int, float)) else 0.05

c1, c2 = st.columns(2)
with c1:
    title = st.text_input("Market Title", value=default_title,
                          placeholder="e.g. Will average gas prices be above $3.50?")
with c2:
    price = st.slider("YES Price (Mid)", 0.01, 0.99,
                      value=min(max(round(default_mid, 2), 0.01), 0.99), step=0.01)

spread_input = st.number_input("Bid-Ask Spread", 0.0, 0.50,
                               value=min(max(round(default_spread, 2), 0.0), 0.50),
                               step=0.01, format="%.2f")

if st.button("⚡ Calculate Edge", type="primary"):
    if not api_key:
        st.error("❌ No FRED API key.")
    elif not title.strip():
        st.warning("⚠️ Enter a market title or select one above.")
    else:
        with st.spinner("Calculating..."):
            try:
                algo    = ProGasAlgo(fred_api_key=api_key)
                signals = algo.refresh_data(force=force_refresh)
                result  = score_market(
                    {"ticker": "", "title": title.strip(), "mid_price": price,
                     "spread_est": spread_input, "yes_bid": None, "yes_ask": None,
                     "last_price": None, "volume": None, "open_interest": None,
                     "close_time": None, "strike": None},
                    algo, signals
                )
                edge  = result["edge"]
                kelly = result["kelly"]

                r1, r2, r3, r4 = st.columns(4)
                r1.metric("YES Mid Price",   f"{price:.2%}")
                r2.metric("Calculated Edge", f"{edge:+.2%}")
                r3.metric("Fair Value",      f"{result['fair_value']:.2%}")
                r4.metric("Kelly Bet Size",  f"{kelly:.1%}" if kelly else "—",
                          help="% of bankroll to risk (Full Kelly, capped 25%)")

                rec = result["recommendation"]
                if "STRONG" in rec:   st.success(rec)
                elif "WEAK" in rec:   st.warning(rec)
                elif "BUY" in rec:    st.success(rec)
                else:                 st.error(rec)

                if result["reasons"]:
                    st.markdown("**✅ Supporting:**")
                    for r in result["reasons"]: st.markdown(f"- {r}")
                if result["risks"]:
                    st.markdown("**⚠️ Risks:**")
                    for r in result["risks"]:   st.markdown(f"- {r}")

            except Exception as e:
                st.error(f"❌ {e}")
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())
