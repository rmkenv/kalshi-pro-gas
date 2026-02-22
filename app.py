import streamlit as st
import requests
import pandas as pd
import traceback
import re as _re
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from kalshi_pro_gas import ProGasAlgo
except ImportError:
    class ProGasAlgo:
        """Stub fallback when kalshi_pro_gas module is unavailable."""
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


# =========================
# Kalshi API helpers
# =========================

BASE = "https://api.elections.kalshi.com/trade-api/v2"
GAS_SERIES = ["KXAAAGASM", "KXAAAGASW", "KXAAGASM", "KXAAGASW"]


def search_kalshi_gas_markets() -> list[dict]:
    results = []
    seen_tickers = set()
    for series in GAS_SERIES:
        cursor = None
        while True:
            params = {"series_ticker": series, "status": "open", "limit": 100}
            if cursor:
                params["cursor"] = cursor
            try:
                r = requests.get(f"{BASE}/markets", params=params, timeout=10)
                r.raise_for_status()
                data = r.json()
                markets = data.get("markets", [])
                for m in markets:
                    ticker = m.get("ticker", "")
                    if ticker in seen_tickers:
                        continue
                    seen_tickers.add(ticker)
                    yes_bid = m.get("yes_bid")
                    yes_ask = m.get("yes_ask")
                    yes_bid_f  = yes_bid  / 100.0 if isinstance(yes_bid,  (int, float)) else None
                    yes_ask_f  = yes_ask  / 100.0 if isinstance(yes_ask,  (int, float)) else None
                    mid_price  = (yes_bid_f + yes_ask_f) / 2.0 if yes_bid_f and yes_ask_f else None
                    spread_est = (yes_ask_f - yes_bid_f)        if yes_bid_f and yes_ask_f else None
                    results.append({
                        "ticker":        ticker,
                        "title":         m.get("title", "").replace("**", ""),
                        "strike":        m.get("floor_strike"),
                        "yes_bid":       yes_bid_f,
                        "yes_ask":       yes_ask_f,
                        "mid_price":     mid_price,
                        "spread_est":    spread_est,
                        "last_price":    m.get("last_price", 0) / 100.0 if m.get("last_price") else None,
                        "volume":        m.get("volume"),
                        "open_interest": m.get("open_interest"),
                        "close_time":    m.get("close_time"),
                    })
                cursor = data.get("cursor")
                if not cursor or not markets:
                    break
            except Exception as e:
                st.warning(f"Error fetching series {series}: {e}")
                break
    results.sort(key=lambda x: x.get("strike") or 0)
    return results


def get_kalshi_orderbook(market_ticker: str) -> dict | None:
    try:
        r = requests.get(f"{BASE}/markets/{market_ticker}/orderbook", timeout=10)
        r.raise_for_status()
        ob = r.json().get("orderbook", {})
        return {"yes_depth": ob.get("yes", [])[:5], "no_depth": ob.get("no", [])[:5]}
    except Exception:
        return None


def score_market(market: dict, algo, signals: dict) -> dict:
    """Run the full decision scoring logic for one market. Returns enriched dict."""
    mid = market.get("mid_price")
    if mid is None:
        return {**market, "edge": None, "signal_score": 0, "recommendation": "NO DATA", "reasons": [], "risks": []}

    edge = algo.edge(market["title"], mid)
    spread = market.get("spread_est", 0.05) or 0.05

    signal_score = 0
    reasons, risks = [], []

    if edge >= 0.10:
        signal_score += 3
        reasons.append(f"Strong edge ({edge:.1%})")
    elif edge >= 0.05:
        signal_score += 2
        reasons.append(f"Moderate edge ({edge:.1%})")
    elif edge > 0:
        signal_score += 1
        reasons.append(f"Weak edge ({edge:.1%})")
    else:
        risks.append(f"Negative edge ({edge:.1%})")

    if edge > 0 and spread > 0:
        if edge >= 2 * spread:
            signal_score += 2
            reasons.append(f"Edge ≥ 2× spread")
        elif edge >= spread:
            signal_score += 1
            reasons.append(f"Edge covers spread")
        else:
            risks.append(f"Edge < spread")

    wti = signals.get("wti", {})
    wti_chg = wti.get("wti_change", 0) or 0
    if wti_chg > 0.02:
        signal_score += 1
        reasons.append("WTI bullish")
    elif wti_chg < -0.02:
        risks.append("WTI bearish")

    inv = signals.get("inventory", {})
    if inv.get("current") is not None:
        z = inv.get("z_score", 0) or 0
        if z < -0.5:
            signal_score += 1
            reasons.append(f"Tight inventory (Z={z:.2f})")
        elif z > 0.5:
            risks.append(f"Ample inventory (Z={z:.2f})")

    sea = signals.get("seasonal", {})
    mult = sea.get("multiplier", 1.0) or 1.0
    if mult >= 1.05:
        signal_score += 1
        reasons.append(f"Seasonal tailwind ({mult:.2f}×)")
    elif mult < 1.0:
        risks.append(f"Seasonal headwind ({mult:.2f}×)")

    if signal_score >= 6:
        rec = "🟢 STRONG BUY"
    elif signal_score >= 4:
        rec = "🟡 BUY"
    elif signal_score >= 2 and edge > 0:
        rec = "🟠 WEAK BUY"
    else:
        rec = "🔴 PASS"

    return {
        **market,
        "edge":            edge,
        "fair_value":      max(0.0, min(1.0, mid + edge)),
        "signal_score":    signal_score,
        "recommendation":  rec,
        "reasons":         reasons,
        "risks":           risks,
    }


# =========================
# Page Config
# =========================

st.set_page_config(page_title="Pro Gas Algo", page_icon="⛽", layout="wide")


# =========================
# Sidebar
# =========================

with st.sidebar:
    st.header("🔑 Configuration")
    api_key = None
    if "FRED_API_KEY" in st.secrets:
        api_key = st.secrets["FRED_API_KEY"]
        st.success("✅ FRED API key loaded.")
    else:
        st.error("❌ Missing `FRED_API_KEY` in `.streamlit/secrets.toml`.")

    if st.button("🔄 Force Refresh FRED Data"):
        st.session_state["force_refresh"] = True
        st.success("✅ Will force refresh on next calculation.")

    force_refresh = st.session_state.pop("force_refresh", False)

    st.divider()
    st.header("📖 How This Works")
    st.markdown("""
**Auto-Scan Mode**

Click **🚀 Auto-Scan All Markets** to:
1. Fetch ALL open gas markets
2. Load live FRED signals once
3. Score every market with the algo
4. Rank by edge — top opportunities shown first

---

**Signal Weights**

| Signal | Weight |
|---|---|
| Gas Price Momentum | 35% |
| WTI Crude Oil (lagged) | 25% |
| Inventory Levels | 20% |
| Refinery Utilization | 10% |
| Regional PADD Prices | 5% |
| Seasonal Adjustment | 5% |

---

**Decision Scores**

- **6+** = Strong Buy
- **4+** = Buy
- **2+** = Weak Buy
- **<2** = Pass

---

**⚠️ Disclaimer**

Quantitative model only — not financial advice.
Always size positions appropriately.
    """)


# =========================
# Main App
# =========================

st.title("⛽ Kalshi Pro Gas Algorithm")
st.markdown("Multi-factor gas price prediction for Kalshi prediction markets — powered by FRED economic data.")

# ─────────────────────────────────────────
# AUTO-SCAN SECTION (new primary workflow)
# ─────────────────────────────────────────
st.subheader("🚀 Auto-Scan: All Open Markets")
st.caption("Fetches all open gas markets, runs the algo on every one, and ranks by edge — no manual selection needed.")

col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    run_scan = st.button("🚀 Auto-Scan All Markets", type="primary", use_container_width=True)
with col_btn2:
    min_edge = st.slider("Minimum Edge Threshold", 0.00, 0.25, 0.05, 0.01,
                         help="Only show markets where |edge| ≥ this value")

if run_scan:
    if not api_key:
        st.error("❌ No FRED API key. Add `FRED_API_KEY` to `.streamlit/secrets.toml`.")
    else:
        progress = st.progress(0, text="Step 1/3 — Fetching open markets...")
        with st.spinner(""):
            markets = search_kalshi_gas_markets()

        if not markets:
            st.warning("No open gas markets found on Kalshi right now.")
        else:
            progress.progress(33, text=f"Step 2/3 — Loading FRED signals for {len(markets)} markets...")
            try:
                algo    = ProGasAlgo(fred_api_key=api_key)
                signals = algo.refresh_data(force=force_refresh)
            except Exception as e:
                st.error(f"❌ Failed to load FRED signals: {e}")
                st.stop()

            progress.progress(66, text="Step 3/3 — Scoring all markets with algo...")

            scored = [score_market(m, algo, signals) for m in markets]
            scored = [m for m in scored if m.get("edge") is not None]
            scored.sort(key=lambda x: x.get("edge", -99), reverse=True)

            progress.progress(100, text="✅ Scan complete!")
            st.session_state["scan_results"] = scored
            st.session_state["scan_signals"]  = signals

# ── Display scan results ──
if "scan_results" in st.session_state:
    scored   = st.session_state["scan_results"]
    signals  = st.session_state["scan_signals"]

    buys      = [m for m in scored if "BUY" in m["recommendation"]]
    strong    = [m for m in scored if m["recommendation"] == "🟢 STRONG BUY"]
    passes    = [m for m in scored if m["recommendation"] == "🔴 PASS"]
    above_thr = [m for m in scored if abs(m.get("edge") or 0) >= min_edge]

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Markets Scanned", len(scored))
    kpi2.metric("Strong Buys",  len(strong), delta=f"+{len(strong)}" if strong else "0")
    kpi3.metric("Any Buy Signal", len(buys))
    kpi4.metric(f"Above {min_edge:.0%} Edge", len(above_thr))

    st.markdown("### 🎯 Ranked Recommendations")
    st.caption(f"Showing {len(above_thr)} markets with |edge| ≥ {min_edge:.0%}, sorted by edge (best first)")

    if above_thr:
        rows = []
        for m in above_thr:
            edge_val = m.get("edge", 0) or 0
            rows.append({
                "Rec":           m["recommendation"],
                "Score":         m["signal_score"],
                "Ticker":        m["ticker"],
                "Title":         m["title"][:60] + ("…" if len(m["title"]) > 60 else ""),
                "Strike":        f"${m['strike']:.2f}" if m.get("strike") is not None else "N/A",
                "Mid Price":     f"{m['mid_price']:.0%}"   if m.get("mid_price") is not None else "N/A",
                "Edge":          f"{edge_val:+.1%}",
                "Fair Value":    f"{m['fair_value']:.0%}"  if m.get("fair_value") is not None else "N/A",
                "Spread":        f"{m['spread_est']:.0%}"  if m.get("spread_est") is not None
