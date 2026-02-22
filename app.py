import streamlit as st
import requests
import pandas as pd
import traceback
try:
    from kalshi_pro_gas import ProGasAlgo
except ImportError:
    import streamlit as _st
    _st.error("❌ `kalshi_pro_gas` module not found. Deploy from the correct repo.")
    _st.stop()


# =========================
# Kalshi API helpers
# =========================

BASE = "https://api.elections.kalshi.com/trade-api/v2"
GAS_SERIES = ["KXAAAGASM"]


def search_kalshi_gas_markets() -> list[dict]:
    """Fetch all open markets from known gas series tickers."""
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

                    # yes_price/no_price are always None — use bid/ask instead
                    yes_bid = m.get("yes_bid")   # cents
                    yes_ask = m.get("yes_ask")   # cents
                    no_bid  = m.get("no_bid")
                    no_ask  = m.get("no_ask")

                    yes_bid_f  = yes_bid  / 100.0 if isinstance(yes_bid,  (int, float)) else None
                    yes_ask_f  = yes_ask  / 100.0 if isinstance(yes_ask,  (int, float)) else None
                    no_bid_f   = no_bid   / 100.0 if isinstance(no_bid,   (int, float)) else None
                    mid_price  = (yes_bid_f + yes_ask_f) / 2.0 if yes_bid_f and yes_ask_f else None
                    spread_est = (yes_ask_f - yes_bid_f)        if yes_bid_f and yes_ask_f else None

                    results.append({
                        "ticker":        ticker,
                        "series_ticker": m.get("series_ticker"),
                        "title":         m.get("title", "").replace("**", ""),
                        "strike":        m.get("floor_strike"),
                        "yes_bid":       yes_bid_f,
                        "yes_ask":       yes_ask_f,
                        "no_bid":        no_bid_f,
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
    """Fetch live orderbook depth (top 5 levels) for display only."""
    try:
        r = requests.get(f"{BASE}/markets/{market_ticker}/orderbook", timeout=10)
        r.raise_for_status()
        ob = r.json().get("orderbook", {})
        return {
            "yes_depth": ob.get("yes", [])[:5],
            "no_depth":  ob.get("no",  [])[:5],
        }
    except Exception:
        return None


# =========================
# Page Config (must be first st call)
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
**Step 1 — Browse Gas Markets**

Click **🔍 Find Open Gas Markets** to pull all open Kalshi markets.
Select one from the table to auto-populate the inputs below.

---

**Step 2 — Calculate Edge**

Click **⚡ Calculate Edge**. The algorithm fetches live FRED data
and runs a multi-factor model:

| Signal | Weight |
|---|---|
| Gas Price Momentum | 35% |
| WTI Crude Oil (lagged) | 25% |
| Inventory Levels | 20% |
| Refinery Utilization | 10% |
| Regional PADD Prices | 5% |
| Seasonal Adjustment | 5% |

---

**Step 3 — Read the Edge**

> **Edge = Fair Value − Market Price**

- **Positive edge** → model thinks YES is *underpriced* → potential BUY
- **Negative edge** → model thinks YES is *overpriced* → PASS or fade

---

**Step 4 — Decision Helper**

Scores the trade across 5 checks:

1. Edge strength (≥5% = good, ≥10% = strong)
2. Edge vs bid-ask spread
3. WTI crude momentum
4. Inventory Z-score
5. Seasonal multiplier

**6+** = Strong Buy · **4+** = Buy · **2+** = Weak Buy · **<2** = Pass

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

# --- Market Browser ---
st.subheader("🔍 Browse Open Gas Markets")

if st.button("🔍 Find Open Gas Markets", type="primary"):
    with st.spinner("Fetching open markets from Kalshi..."):
        gas_markets = search_kalshi_gas_markets()
    st.session_state["gas_markets"] = gas_markets
    # Clear previous selection when refreshing
    st.session_state.pop("selected_market", None)
    st.session_state.pop("selected_orderbook", None)

if "gas_markets" in st.session_state:
    gas_markets = st.session_state["gas_markets"]
    if not gas_markets:
        st.warning("No open gas markets found on Kalshi right now.")
    else:
        st.success(f"Found **{len(gas_markets)}** open gas markets.")

        df = pd.DataFrame([
            {
                "Select":        False,
                "Ticker":        m["ticker"],
                "Title":         m["title"],
                "Strike":        f"${m['strike']:.2f}" if m.get("strike") is not None else "N/A",
                "YES Bid":       f"{m['yes_bid']:.0%}"   if m.get("yes_bid")   is not None else "N/A",
                "YES Ask":       f"{m['yes_ask']:.0%}"   if m.get("yes_ask")   is not None else "N/A",
                "Mid":           f"{m['mid_price']:.0%}" if m.get("mid_price") is not None else "N/A",
                "Spread":        f"{m['spread_est']:.0%}" if m.get("spread_est") is not None else "N/A",
                "Last":          f"{m['last_price']:.0%}" if m.get("last_price") is not None else "N/A",
                "Volume":        m.get("volume") or 0,
                "Open Interest": m.get("open_interest") or 0,
                "Closes":        m["close_time"][:10] if m.get("close_time") else "N/A",
            }
            for m in gas_markets
        ])

        edited_df = st.data_editor(
            df,
            column_config={
                "Select": st.column_config.CheckboxColumn("Select", help="Pick one market to analyze"),
            },
            use_container_width=True,
            hide_index=True,
            key="market_table",
        )

        selected_rows = edited_df[edited_df["Select"] == True]
        if len(selected_rows) > 1:
            st.warning("Please select only one market at a time.")
        elif len(selected_rows) == 1:
            selected_ticker = selected_rows.iloc[0]["Ticker"]
            selected_market = next((m for m in gas_markets if m["ticker"] == selected_ticker), None)

            if selected_market:
                st.session_state["selected_market"] = selected_market
                with st.spinner(f"Fetching orderbook depth for {selected_ticker}..."):
                    ob = get_kalshi_orderbook(selected_ticker)
                st.session_state["selected_orderbook"] = ob
                st.success(f"✅ Selected: `{selected_ticker}` — {selected_market.get('title')}")

                mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
                mc1.metric("YES Bid",  f"{selected_market['yes_bid']:.0%}"   if selected_market.get("yes_bid")   is not None else "N/A")
                mc2.metric("YES Ask",  f"{selected_market['yes_ask']:.0%}"   if selected_market.get("yes_ask")   is not None else "N/A")
                mc3.metric("Mid",      f"{selected_market['mid_price']:.0%}" if selected_market.get("mid_price") is not None else "N/A")
                mc4.metric("Spread",   f"{selected_market['spread_est']:.0%}" if selected_market.get("spread_est") is not None else "N/A")
                mc5.metric("Volume",   f"{selected_market.get('volume', 0):,}")
                mc6.metric("Closes",   selected_market["close_time"][:10] if selected_market.get("close_time") else "N/A")

                if ob:
                    with st.expander("📊 Orderbook Depth (Top 5)"):
                        d1, d2 = st.columns(2)
                        with d1:
                            st.markdown("**YES Bids**")
                            for row in ob.get("yes_depth", []):
                                st.markdown(f"- {row[0]}¢ × {row[1]} contracts")
                        with d2:
                            st.markdown("**NO Bids**")
                            for row in ob.get("no_depth", []):
                                st.markdown(f"- {row[0]}¢ × {row[1]} contracts")

# --- Market Input ---
st.subheader("📋 Market Input")

sel_market = st.session_state.get("selected_market", {})
sel_ob     = st.session_state.get("selected_orderbook", {})

default_title  = sel_market.get("title", "")
default_mid    = float(sel_market["mid_price"])  if isinstance(sel_market.get("mid_price"),  (int, float)) else 0.50
default_spread = float(sel_market["spread_est"]) if isinstance(sel_market.get("spread_est"), (int, float)) else 0.05

col1, col2 = st.columns(2)
with col1:
    title = st.text_input(
        "Market Title",
        value=default_title,
        placeholder="e.g. Will average gas prices be above $3.50?"
    )
with col2:
    price = st.slider(
        "Current YES Price (Mid)",
        min_value=0.01, max_value=0.99,
        value=min(max(round(default_mid, 2), 0.01), 0.99),
        step=0.01
    )

spread_input = st.number_input(
    "Bid-Ask Spread",
    min_value=0.0, max_value=0.50,
    value=min(max(round(default_spread, 2), 0.0), 0.50),
    step=0.01, format="%.2f",
    help="Auto-filled from live market data. You can override."
)

# --- Debug Info ---
with st.expander("🔧 Debug Info"):
    st.write(f"**API Key loaded:** {api_key is not None}")
    st.write(f"**Title:** `{title}`")
    st.write(f"**Price:** `{price}`")
    st.write(f"**Spread:** `{spread_input}`")
    st.write(f"**Selected market in state:** {bool(sel_market)}")

# --- Run Algorithm ---
if st.button("⚡ Calculate Edge", type="primary"):
    if not api_key:
        st.error("❌ No FRED API key found. Add `FRED_API_KEY` to `.streamlit/secrets.toml`.")
    elif not title.strip():
        st.warning("⚠️ Please enter a Market Title or select a market from the table above.")
    else:
        with st.spinner("Fetching FRED data and calculating edge..."):
            try:
                algo    = ProGasAlgo(fred_api_key=api_key)
                signals = algo.refresh_data(force=force_refresh)

                if not signals:
                    st.error("❌ FRED data fetch returned empty. Check your API key and internet connection.")
                    st.stop()

                edge = algo.edge(title.strip(), price)

                # --- Edge Result ---
                st.subheader("📊 Edge Result")
                r1, r2, r3 = st.columns(3)
                r1.metric("Market YES Price (Mid)", f"{price:.2%}")
                r2.metric("Calculated Edge",        f"{edge:+.2%}")
                fair_value = max(0.0, min(1.0, price + edge))
                r3.metric("Implied Fair Value",     f"{fair_value:.2%}")

                # --- Decision Helper ---
                st.subheader("🎯 Decision Helper")
                signal_score = 0
                reasons: list[str] = []
                risks:   list[str] = []

                if edge >= 0.10:
                    signal_score += 3
                    reasons.append("Strong positive edge (≥10%)")
                elif edge >= 0.05:
                    signal_score += 2
                    reasons.append("Moderate positive edge (≥5%)")
                elif edge > 0:
                    signal_score += 1
                    reasons.append("Weak positive edge (<5%)")
                else:
                    risks.append("Negative edge — market appears overpriced vs model")

                if edge > 0 and spread_input > 0:
                    if edge >= 2 * spread_input:
                        signal_score += 2
                        reasons.append(f"Edge ({edge:.1%}) is ≥2× the spread ({spread_input:.1%})")
                    elif edge >= spread_input:
                        signal_score += 1
                        reasons.append(f"Edge ({edge:.1%}) covers the spread ({spread_input:.1%})")
                    else:
                        risks.append(f"Edge ({edge:.1%}) < spread ({spread_input:.1%}) — transaction cost likely dominates")

                wti = signals.get("wti", {})
                if wti.get("current_wti") is not None:
                    wti_chg = wti.get("wti_change", 0)
                    if wti_chg > 0.02:
                        signal_score += 1
                        reasons.append(f"WTI rising ({wti_chg:+.1%}) — bullish tailwind")
                    elif wti_chg < -0.02:
                        risks.append(f"WTI falling ({wti_chg:+.1%}) — bearish headwind")

                inv = signals.get("inventory", {})
                if inv.get("current") is not None:
                    z = inv.get("z_score", 0)
                    if z < -0.5:
                        signal_score += 1
                        reasons.append(f"Inventory tight ({inv.get('status')}, Z={z:.2f})")
                    elif z > 0.5:
                        risks.append(f"Inventory ample ({inv.get('status')}, Z={z:.2f})")

                sea = signals.get("seasonal", {})
                if sea.get("multiplier") is not None:
                    mult = sea.get("multiplier", 1.0)
                    if mult >= 1.05:
                        signal_score += 1
                        reasons.append(f"Seasonal tailwind (multiplier {mult:.2f}×)")
                    elif mult < 1.0:
                        risks.append(f"Seasonal headwind (multiplier {mult:.2f}×)")

                if signal_score >= 6:
                    st.success("🟢 STRONG BUY — Multiple signals aligned, edge exceeds spread")
                elif signal_score >= 4:
                    st.success("🟡 BUY — Positive edge with supporting signals")
                elif signal_score >= 2 and edge > 0:
                    st.warning("🟠 WEAK BUY — Small edge, proceed with caution")
                else:
                    st.error("🔴 PASS — No clear edge or risks dominate")

                if reasons:
                    st.markdown("**✅ Supporting Factors:**")
                    for reason in reasons:
                        st.markdown(f"- {reason}")
                if risks:
                    st.markdown("**⚠️ Risk Factors:**")
                    for risk in risks:
                        st.markdown(f"- {risk}")

                # --- Signal Breakdown ---
                st.subheader("🔍 Signal Breakdown")
                c1, c2, c3 = st.columns(3)

                with c1:
                    st.markdown("**💰 Gas Price Momentum**")
                    gm = signals.get("gas_momentum", {})
                    if gm.get("current") is not None:
                        st.metric("Current Price",    f"${gm['current']:.3f}/gal")
                        st.metric("4-Week Momentum",  f"{gm.get('momentum', 0):+.2%}")
                        st.metric("12-Week Trend",    f"{gm.get('trend', 0):+.2%}")
                        st.metric("52-Week Avg",      f"${gm.get('avg_52w', 0):.3f}/gal")
                    else:
                        st.warning("Gas momentum data unavailable")

                with c2:
                    st.markdown("**🛢️ WTI Crude Oil**")
                    if wti.get("current_wti") is not None:
                        st.metric("Current WTI",  f"${wti.get('current_wti', 0):.2f}/bbl")
                        st.metric("Lagged WTI",   f"${wti.get('lagged_wti', 0):.2f}/bbl")
                        st.metric("WTI Change",   f"{wti.get('wti_change', 0):+.2%}")
                        st.metric("Optimal Lag",  f"{wti.get('optimal_lag', 1)} week(s)")
                    else:
                        st.warning("WTI data unavailable")

                with c3:
                    st.markdown("**📦 Inventory & 🏭 Refinery**")
                    ref = signals.get("refinery", {})
                    if inv.get("current") is not None:
                        st.metric("Inventory Status", inv.get("status"))
                        st.metric("Z-Score",          f"{inv.get('z_score', 0):.2f}")
                        st.metric("WoW Change",        f"{inv.get('wow_change', 0):+.2%}")
                    else:
                        st.warning("Inventory data unavailable")
                    if ref.get("current") is not None:
                        st.metric("Refinery Util.",   f"{ref.get('current', 0):.1f}%")
                        st.metric("Refinery Status",  ref.get("status", "unknown"))
                    else:
                        st.info(f"Refinery: {ref.get('status', 'Data unavailable')}")

                # --- Seasonal ---
                st.subheader("📅 Seasonal Adjustment")
                s1, s2 = st.columns(2)
                s1.metric("Multiplier", f"{sea.get('multiplier', 1.0):.3f}x")
                s2.metric("Signal",     f"{sea.get('signal', 0.0):+.3f}")
                if sea.get("factors"):
                    st.markdown("**Active Factors:**")
                    for name, val in sea["factors"]:
                        st.markdown(f"- {name}: `{val:.2f}x`")

                # --- Regional ---
                st.subheader("🗺️ Regional PADD Prices")
                reg = signals.get("regional", {})
                if reg.get("regional_data"):
                    padd_labels = {
                        "padd1": "East Coast",    "padd2": "Midwest",
                        "padd3": "Gulf Coast",    "padd4": "Rocky Mountain",
                        "padd5": "West Coast",
                    }
                    cols = st.columns(5)
                    for i, (padd, data) in enumerate(reg["regional_data"].items()):
                        cols[i].metric(padd_labels.get(padd, padd), f"${data.get('price', 0):.3f}")
                    if reg.get("weighted_avg") is not None:
                        st.metric("Weighted Avg",    f"${reg.get('weighted_avg', 0):.3f}/gal")
                    if reg.get("spread") is not None:
                        st.metric("Regional Spread", f"${reg.get('spread', 0):.3f}")
                else:
                    st.warning("Regional data unavailable")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                with st.expander("🔧 Full Traceback"):
                    st.code(traceback.format_exc())
