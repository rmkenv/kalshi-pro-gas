import streamlit as st
import requests
from kalshi_pro_gas import ProGasAlgo


# =========================
# Kalshi API helpers
# =========================

BASE = "https://api.elections.kalshi.com/trade-api/v2"


def get_kalshi_market(series_ticker: str, status: str = "open") -> dict | None:
    """Fetch first market for a series ticker."""
    try:
        r = requests.get(f"{BASE}/markets", params={"series_ticker": series_ticker, "status": status}, timeout=10)
        r.raise_for_status()
        markets = r.json().get("markets", [])
        if not markets:
            return None
        m = markets[0]
        yes_c = m.get("yes_price")
        no_c = m.get("no_price")
        yes_price = yes_c / 100.0 if isinstance(yes_c, (int, float)) else None
        no_price = no_c / 100.0 if isinstance(no_c, (int, float)) else None
        spread_est = abs(yes_price - (1.0 - no_price)) if yes_price is not None and no_price is not None else None
        return {
            "ticker": m.get("ticker"),
            "title": m.get("title"),
            "yes_price": yes_price,
            "no_price": no_price,
            "volume": m.get("volume"),
            "open_interest": m.get("open_interest"),
            "close_time": m.get("close_time"),
            "spread_est": spread_est,
        }
    except Exception:
        return None


def get_kalshi_orderbook(market_ticker: str) -> dict | None:
    """
    Fetch live orderbook for a specific market ticker.
    Returns best YES bid, best NO bid, mid price, and top-of-book spread.
    Kalshi orderbook only returns BIDS (not asks) for both YES and NO sides.
    """
    try:
        r = requests.get(f"{BASE}/markets/{market_ticker}/orderbook", timeout=10)
        r.raise_for_status()
        ob = r.json().get("orderbook", {})

        yes_bids = ob.get("yes", [])  # [[price_cents, qty], ...]
        no_bids = ob.get("no", [])

        best_yes = yes_bids[0][0] / 100.0 if yes_bids else None
        best_no = no_bids[0][0] / 100.0 if no_bids else None

        # In Kalshi binary markets: implied ask for YES = 1 - best NO bid
        implied_yes_ask = (1.0 - best_no) if best_no is not None else None
        mid = ((best_yes + implied_yes_ask) / 2.0) if best_yes is not None and implied_yes_ask is not None else best_yes
        spread = (implied_yes_ask - best_yes) if best_yes is not None and implied_yes_ask is not None else None

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
**Step 1 — Load a Kalshi Market**

Paste the series ticker from the Kalshi URL (e.g. `KXAAAGASM`) and click **Load from Kalshi**.
The app pulls the live market title, YES price, and full orderbook automatically.

---

**Step 2 — Calculate Edge**

Click **⚡ Calculate Edge**. The algorithm fetches live FRED economic data and runs a
multi-factor model combining:

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

The app scores the trade across 5 checks:

1. Edge strength (≥5% = good, ≥10% = strong)
2. Edge vs bid-ask spread (edge must beat the spread to be worth it)
3. WTI crude momentum (rising crude = bullish for gas)
4. Inventory Z-score (low inventory = tight supply = bullish)
5. Seasonal multiplier (summer driving season = bullish)

A score of **6+** = Strong Buy. **4+** = Buy. **2+** = Weak Buy. Below that = Pass.

---

**⚠️ Disclaimer**

This is a quantitative model, not financial advice. Prediction markets carry risk.
Always size positions appropriately and never trade more than you can afford to lose.
    """)


# =========================
# Main App
# =========================

st.set_page_config(page_title="Pro Gas Algo", page_icon="⛽", layout="wide")

st.title("⛽ Kalshi Pro Gas Algorithm")
st.markdown("Multi-factor gas price prediction for Kalshi prediction markets — powered by FRED economic data.")

# --- Market Auto-Load ---
st.subheader("📌 Load Kalshi Market")

colA, colB, colC = st.columns([2, 1, 1])
with colA:
    series_ticker = st.text_input("Series Ticker", value="KXAAAGASM", help="From the Kalshi URL: kalshi.com/markets/<ticker>/...")
with colB:
    status_filter = st.selectbox("Status", ["open", "all"], index=0)
with colC:
    st.write("")
    st.write("")
    load_market = st.button("🔗 Load from Kalshi", type="primary")

market_data = None
orderbook_data = None

if load_market and series_ticker:
    with st.spinner("Fetching market data from Kalshi..."):
        market_data = get_kalshi_market(series_ticker, status=status_filter)
        if market_data and market_data.get("ticker"):
            orderbook_data = get_kalshi_orderbook(market_data["ticker"])

    if market_data:
        st.success(f"✅ Loaded: `{market_data['ticker']}` — {market_data.get('title', '')}")

        # Show live market snapshot
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("YES Price", f"{market_data['yes_price']:.2%}" if market_data.get("yes_price") is not None else "N/A")
        mc2.metric("NO Price", f"{market_data['no_price']:.2%}" if market_data.get("no_price") is not None else "N/A")
        mc3.metric("Volume", f"{market_data.get('volume', 'N/A'):,}" if isinstance(market_data.get("volume"), int) else "N/A")
        mc4.metric("Open Interest", f"{market_data.get('open_interest', 'N/A'):,}" if isinstance(market_data.get("open_interest"), int) else "N/A")
        mc5.metric("Closes", market_data.get("close_time", "N/A")[:10] if market_data.get("close_time") else "N/A")

        # Orderbook
        if orderbook_data:
            st.markdown("**📒 Live Orderbook (Top of Book)**")
            ob1, ob2, ob3, ob4 = st.columns(4)
            ob1.metric("Best YES Bid", f"{orderbook_data['best_yes_bid']:.2%}" if orderbook_data.get("best_yes_bid") is not None else "N/A")
            ob2.metric("Implied YES Ask", f"{orderbook_data['implied_yes_ask']:.2%}" if orderbook_data.get("implied_yes_ask") is not None else "N/A")
            ob3.metric("Mid Price", f"{orderbook_data['mid_price']:.2%}" if orderbook_data.get("mid_price") is not None else "N/A")
            ob4.metric("Spread", f"{orderbook_data['spread']:.2%}" if orderbook_data.get("spread") is not None else "N/A")

            # Depth table
            with st.expander("📊 Full Orderbook Depth (Top 5)"):
                d1, d2 = st.columns(2)
                with d1:
                    st.markdown("**YES Bids**")
                    for row in orderbook_data.get("yes_depth", []):
                        st.markdown(f"- {row[0]}¢ × {row[1]} contracts")
                with d2:
                    st.markdown("**NO Bids**")
                    for row in orderbook_data.get("no_depth", []):
                        st.markdown(f"- {row[0]}¢ × {row[1]} contracts")
        else:
            st.info("Orderbook data unavailable for this market.")
    else:
        st.warning("No markets found. Try changing status to 'all' or check the ticker.")

# --- Manual / Pre-filled Inputs ---
st.subheader("📋 Market Input")

default_title = market_data["title"] if market_data and market_data.get("title") else ""
default_yes = float(market_data["yes_price"]) if market_data and isinstance(market_data.get("yes_price"), (int, float)) else 0.45
default_spread = float(orderbook_data["spread"]) if orderbook_data and isinstance(orderbook_data.get("spread"), (int, float)) else (
    float(market_data["spread_est"]) if market_data and isinstance(market_data.get("spread_est"), (int, float)) else 0.02
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
    help="Auto-filled from live orderbook if available. You can override."
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

                # --- Edge Result ---
                st.subheader("📊 Edge Result")
                r1, r2, r3 = st.columns(3)
                r1.metric("Market YES Price", f"{price:.2%}")
                r2.metric("Calculated Edge", f"{edge:+.2%}")
                fair_value = max(0.0, min(1.0, price + edge))
                r3.metric("Implied Fair Value", f"{fair_value:.2%}")

                # --- Decision Helper ---
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
                    if edge >= 2  spread:
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
                        reasons.append(f"WTI rising ({wti.get('wti_change'):+.1%}) — bullish tailwind for gas")
                    elif wti.get("wti_change", 0) < -0.02:
                        risks.append(f"WTI falling ({wti.get('wti_change'):+.1%}) — bearish headwind")

                inv = signals.get("inventory", {})
                if inv.get("current") is not None:
                    z = inv.get("z_score", 0)
                    if z < -0.5:
                        signal_score += 1
                        reasons.append(f"Inventory tight ({inv.get('status')}, Z={z:.2f}) — supply pressure")
                    elif z > 0.5:
                        risks.append(f"Inventory ample ({inv.get('status')}, Z={z:.2f}) — supply headwind")

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

                # --- Signal Breakdown ---
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

                # --- Seasonal ---
                st.subheader("📅 Seasonal Adjustment")
                s1, s2 = st.columns(2)
                s1.metric("Multiplier", f"{sea.get('multiplier', 1.0):.3f}x")
                s2.metric("Signal", f"{sea.get('signal', 0.0):+.3f}")
                if sea.get("factors"):
                    st.markdown("**Active Factors:**")
                    for name, val in sea["factors"]:
                        st.markdown(f"- {name}: `{val:.2f}x`")

                # --- Regional ---
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
