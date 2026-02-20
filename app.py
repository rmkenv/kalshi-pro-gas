import streamlit as st
import requests
from kalshi_pro_gas import ProGasAlgo


# =========================
# Kalshi API helpers
# =========================

BASE = "https://api.elections.kalshi.com/trade-api/v2"


def search_kalshi_gas_markets() -> list[dict]:
    """
    Search all open Kalshi markets containing 'gas' in the title.
    Uses cursor-based pagination to get all results.
    """
    results = []
    cursor = None
    seen_tickers = set()

    # Search terms that cover gasoline markets
    for term in ["gas", "gasoline", "fuel"]:
        cursor = None
        while True:
            params = {"status": "open", "limit": 100}
            if cursor:
                params["cursor"] = cursor
            try:
                r = requests.get(f"{BASE}/markets", params=params, timeout=10)
                r.raise_for_status()
                data = r.json()
                markets = data.get("markets", [])

                for m in markets:
                    title = (m.get("title") or "").lower()
                    ticker = m.get("ticker", "")
                    series = (m.get("series_ticker") or "").lower()
                    if (
                        term in title or term in series or term in ticker.lower()
                    ) and ticker not in seen_tickers:
                        seen_tickers.add(ticker)
                        yes_c = m.get("yes_price")
                        no_c = m.get("no_price")
                        yes_price = yes_c / 100.0 if isinstance(yes_c, (int, float)) else None
                        no_price = no_c / 100.0 if isinstance(no_c, (int, float)) else None
                        spread_est = (
                            abs(yes_price - (1.0 - no_price))
                            if yes_price is not None and no_price is not None
                            else None
                        )
                        results.append({
                            "ticker": ticker,
                            "series_ticker": m.get("series_ticker"),
                            "title": m.get("title"),
                            "yes_price": yes_price,
                            "no_price": no_price,
                            "spread_est": spread_est,
                            "volume": m.get("volume"),
                            "open_interest": m.get("open_interest"),
                            "close_time": m.get("close_time"),
                        })

                cursor = data.get("cursor")
                if not cursor or not markets:
                    break
            except Exception:
                break

    # Sort by volume descending
    results.sort(key=lambda x: x.get("volume") or 0, reverse=True)
    return results


def get_kalshi_orderbook(market_ticker: str) -> dict | None:
    """Fetch live orderbook for a specific market ticker."""
    try:
        r = requests.get(f"{BASE}/markets/{market_ticker}/orderbook", timeout=10)
        r.raise_for_status()
        ob = r.json().get("orderbook", {})
        yes_bids = ob.get("yes", [])
        no_bids = ob.get("no", [])
        best_yes = yes_bids[0][0] / 100.0 if yes_bids else None
        best_no = no_bids[0][0] / 100.0 if no_bids else None
        implied_yes_ask = (1.0 - best_no) if best_no is not None else None
        mid = (
            (best_yes + implied_yes_ask) / 2.0
            if best_yes is not None and implied_yes_ask is not None
            else best_yes
        )
        spread = (
            implied_yes_ask - best_yes
            if best_yes is not None and implied_yes_ask is not None
            else None
        )
        return {
            "best_yes_bid": best_yes,
            "best_no_bid": best_no,
            "implied_yes_ask": implied_yes_ask,
            "mid_price": mid,
            "spread": spread,
            "yes_depth": yes_bids[:5],
            "no_depth": no_bids[:5],
        }
    except Exception:
        return None


# =========================
# Sidebar — How it works
# =========================

with st.sidebar:
    st.header("🔑 Configuration")
    api_key = None
    if "FRED_API_KEY" in st.secrets:
        api_key = st.secrets["FRED_API_KEY"]
        st.success("FRED API key loaded.")
    else:
        st.error("Missing `FRED_API_KEY` in `.streamlit/secrets.toml`.")

    force_refresh = st.button("🔄 Force Refresh FRED Data")

    st.divider()

    st.header("📖 How This Works")
    st.markdown("""
**Step 1 — Browse Gas Markets**

Click **🔍 Find Open Gas Markets** to pull all open Kalshi markets
related to gasoline/fuel. Select one from the table to auto-populate
the inputs below.

---

**Step 2 — Calculate Edge**

Click **⚡ Calculate Edge**. The algorithm fetches live FRED data
and runs a multi-factor model:

| Signal | Weight |
|---|---|
| Gas Price Momentum | 20% |
| WTI Crude Oil (lagged) | 35% |
| Refinery Utilization | 15% |
| Inventory Levels | 15% |
| Regional PADD Prices | 5% |
| Seasonal Adjustment | 10% |

---

**Step 3 — Read the Edge**

> **Edge = Fair Value − Market Price**

- **Positive edge** → model thinks market is *underpriced* → potential BUY
- **Negative edge** → model thinks market is *overpriced* → PASS or fade

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

st.set_page_config(page_title="Pro Gas Algo", page_icon="⛽", layout="wide")

st.title("⛽ Kalshi Pro Gas Algorithm")
st.markdown("Multi-factor gas price prediction for Kalshi prediction markets — powered by FRED economic data.")

# --- Market Browser ---
st.subheader("🔍 Browse Open Gas Markets")

if st.button("🔍 Find Open Gas Markets", type="primary"):
    with st.spinner("Searching Kalshi for open gas/gasoline/fuel markets..."):
        gas_markets = search_kalshi_gas_markets()
    st.session_state["gas_markets"] = gas_markets

if "gas_markets" in st.session_state:
    gas_markets = st.session_state["gas_markets"]
    if not gas_markets:
        st.warning("No open gas-related markets found on Kalshi right now.")
    else:
        st.success(f"Found **{len(gas_markets)}** open gas-related markets.")

        # Build display table
        import pandas as pd
        df = pd.DataFrame([
            {
                "Select": False,
                "Ticker": m["ticker"],
                "Title": m["title"],
                "YES Price": f"{m['yes_price']:.2%}" if m.get("yes_price") is not None else "N/A",
                "NO Price": f"{m['no_price']:.2%}" if m.get("no_price") is not None else "N/A",
                "Est. Spread": f"{m['spread_est']:.2%}" if m.get("spread_est") is not None else "N/A",
                "Volume": m.get("volume") or 0,
                "Open Interest": m.get("open_interest") or 0,
                "Closes": m["close_time"][:10] if m.get("close_time") else "N/A",
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
                with st.spinner(f"Fetching orderbook for {selected_ticker}..."):
                    ob = get_kalshi_orderbook(selected_ticker)
                st.session_state["selected_orderbook"] = ob
                st.success(f"✅ Selected: `{selected_ticker}` — {selected_market.get('title')}")

                # Live snapshot
                mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                mc1.metric("YES Price", f"{selected_market['yes_price']:.2%}" if selected_market.get("yes_price") is not None else "N/A")
                mc2.metric("NO Price", f"{selected_market['no_price']:.2%}" if selected_market.get("no_price") is not None else "N/A")
                mc3.metric("Volume", f"{selected_market.get('volume', 0):,}")
                mc4.metric("Open Interest", f"{selected_market.get('open_interest', 0):,}")
                mc5.metric("Closes", selected_market["close_time"][:10] if selected_market.get("close_time") else "N/A")

                if ob:
                    st.markdown("**📒 Live Orderbook (Top of Book)**")
                    ob1, ob2, ob3, ob4 = st.columns(4)
                    ob1.metric("Best YES Bid", f"{ob['best_yes_bid']:.2%}" if ob.get("best_yes_bid") is not None else "N/A")
                    ob2.metric("Implied YES Ask", f"{ob['implied_yes_ask']:.2%}" if ob.get("implied_yes_ask") is not None else "N/A")
                    ob3.metric("Mid Price", f"{ob['mid_price']:.2%}" if ob.get("mid_price") is not None else "N/A")
                    ob4.metric("Spread", f"{ob['spread']:.2%}" if ob.get("spread") is not None else "N/A")

                    with st.expander("📊 Full Orderbook Depth (Top 5)"):
                        d1, d2 = st.columns(2)
                        with d1:
                            st.markdown("**YES Bids**")
                            for row in ob.get("yes_depth", []):
                                st.markdown(f"- {row[0]}¢ × {row[1]} contracts")
                        with d2:
                            st.markdown("**NO Bids**")
                            for row in ob.get("no_depth", []):
                                st.markdown(f"- {row[0]}¢ × {row[1]} contracts")

# --- Market Input (auto-filled from selection) ---
st.subheader("📋 Market Input")

sel_market = st.session_state.get("selected_market", {})
sel_ob = st.session_state.get("selected_orderbook", {})

default_title = sel_market.get("title", "")
default_yes = float(sel_market["yes_price"]) if isinstance(sel_market.get("yes_price"), (int, float)) else 0.45
default_spread = (
    float(sel_ob["spread"]) if sel_ob and isinstance(sel_ob.get("spread"), (int, float))
    else float(sel_market.get("spread_est") or 0.02)
)

col1, col2 = st.columns(2)
with col1:
    title = st.text_input("Market Title", value=default_title, placeholder="Will national gas prices exceed $3.50?")
with col2:
    price = st.slider("Current YES Price", min_value=0.01, max_value=0.99, value=min(max(default_yes, 0.01), 0.99), step=0.01)

spread = st.number_input(
    "Bid-Ask Spread",
    min_value=0.0, max_value=0.20,
    value=min(max(default_spread, 0.0), 0.20),
    step=0.01, format="%.2f",
    help="Auto-filled from live orderbook. You can override."
)

run = st.button("⚡ Calculate Edge", type="primary")

# --- Run Algorithm ---
if run:
    if not api_key:
        st.error("Missing FRED API key. Add it to `.streamlit/secrets.toml` as `FRED_API_KEY`.")
    elif not title:
        st.error("Please enter a market title.")
    else:
        with st.spinner("Fetching FRED data and calculating edge..."):
            try:
                algo = ProGasAlgo(fred_api_key=api_key)
                signals = algo.refresh_data(force=force_refresh)
                edge = algo.edge(title, price)

                st.subheader("📊 Edge Result")
                r1, r2, r3 = st.columns(3)
                r1.metric("Market YES Price", f"{price:.2%}")
                r2.metric("Calculated Edge", f"{edge:+.2%}")
                fair_value = max(0.0, min(1.0, price + edge))
                r3.metric("Implied Fair Value", f"{fair_value:.2%}")

                st.subheader("🎯 Decision Helper")
                signal_score = 0
                reasons: list[str] = []
                risks: list[str] = []

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

                if edge > 0 and spread > 0:
                    if edge >= 2 * spread:
                        signal_score += 2
                        reasons.append(f"Edge ({edge:.1%}) is ≥2× the spread ({spread:.1%})")
                    elif edge >= spread:
                        signal_score += 1
                        reasons.append(f"Edge ({edge:.1%}) covers the spread ({spread:.1%})")
                    else:
                        risks.append(f"Edge ({edge:.1%}) < spread ({spread:.1%}) — transaction cost likely dominates")

                wti = signals.get("wti", {})
                if wti.get("current_wti") is not None:
                    if wti.get("wti_change", 0) > 0.02:
                        signal_score += 1
                        reasons.append(f"WTI rising ({wti.get('wti_change'):+.1%}) — bullish tailwind")
                    elif wti.get("wti_change", 0) < -0.02:
                        risks.append(f"WTI falling ({wti.get('wti_change'):+.1%}) — bearish headwind")

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
                    for r in reasons:
                        st.markdown(f"- {r}")
                if risks:
                    st.markdown("**⚠️ Risk Factors:**")
                    for r in risks:
                        st.markdown(f"- {r}")

                st.subheader("🔍 Signal Breakdown")
                c1, c2, c3 = st.columns(3)

                with c1:
                    st.markdown("**💰 Gas Price Momentum**")
                    gm = signals.get("gas_momentum", {})
                    if gm.get("current") is not None:
                        st.metric("Current Price", f"${gm['current']:.3f}/gal")
                        st.metric("4-Week Momentum", f"{gm.get('momentum', 0):+.2%}")
                        st.metric("12-Week Trend", f"{gm.get('trend', 0):+.2%}")
                        st.metric("52-Week Avg", f"${gm.get('avg_52w', 0):.3f}/gal")
                    else:
                        st.warning("Gas momentum data unavailable")

                with c2:
                    st.markdown("**🛢️ WTI Crude Oil**")
                    if wti.get("current_wti") is not None:
                        st.metric("Current WTI", f"${wti.get('current_wti', 0):.2f}/bbl")
                        st.metric("Lagged WTI", f"${wti.get('lagged_wti', 0):.2f}/bbl")
                        st.metric("WTI Change", f"{wti.get('wti_change', 0):+.2%}")
                        st.metric("Optimal Lag", f"{wti.get('optimal_lag', 1)} week(s)")
                    else:
                        st.warning("WTI data unavailable")

                with c3:
                    st.markdown("**📦 Inventory & 🏭 Refinery**")
                    ref = signals.get("refinery", {})
                    if inv.get("current") is not None:
                        st.metric("Inventory Status", inv.get("status"))
                        st.metric("Z-Score", f"{inv.get('z_score', 0):.2f}")
                        st.metric("WoW Change", f"{inv.get('wow_change', 0):+.2%}")
                    else:
                        st.warning("Inventory data unavailable")
                    if ref.get("current") is not None:
                        st.metric("Refinery Util.", f"{ref.get('current', 0):.1f}%")
                        st.metric("Refinery Status", ref.get("status", "unknown"))
                    else:
                        st.info(f"Refinery: {ref.get('status', 'Data unavailable')}")

                st.subheader("📅 Seasonal Adjustment")
                s1, s2 = st.columns(2)
                s1.metric("Multiplier", f"{sea.get('multiplier', 1.0):.3f}x")
                s2.metric("Signal", f"{sea.get('signal', 0.0):+.3f}")
                if sea.get("factors"):
                    st.markdown("**Active Factors:**")
                    for name, val in sea["factors"]:
                        st.markdown(f"- {name}: `{val:.2f}x`")

                st.subheader("🗺️ Regional PADD Prices")
                reg = signals.get("regional", {})
                if reg.get("regional_data"):
                    padd_labels = {
                        "padd1": "East Coast", "padd2": "Midwest",
                        "padd3": "Gulf Coast", "padd4": "Rocky Mountain", "padd5": "West Coast",
                    }
                    cols = st.columns(5)
                    for i, (padd, data) in enumerate(reg["regional_data"].items()):
                        cols[i].metric(padd_labels.get(padd, padd), f"${data.get('price', 0):.3f}")
                    if reg.get("weighted_avg") is not None:
                        st.metric("Weighted Avg", f"${reg.get('weighted_avg', 0):.3f}/gal")
                    if reg.get("spread") is not None:
                        st.metric("Regional Spread", f"${reg.get('spread', 0):.3f}")
                else:
                    st.warning("Regional data unavailable")

            except ValueError as e:
                st.error(f"Initialization error: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
