import streamlit as st
import requests
from kalshi_pro_gas import ProGasAlgo


# =========================
# Kalshi public market data
# =========================
def get_kalshi_open_market_by_series(series_ticker: str) -> dict | None:
    """
    Pull the first OPEN market for a given Kalshi series ticker using public endpoints.
    Returns a dict with ticker, title, yes/no price (0-1), and an estimated spread (0-1).
    """
    url = "https://api.elections.kalshi.com/trade-api/v2/markets"
    params = {"series_ticker": series_ticker, "status": "open"}

    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        markets = r.json().get("markets", [])
        if not markets:
            return None

        m = markets[0]
        yes_c = m.get("yes_price", None)
        no_c = m.get("no_price", None)

        # Prices are in cents (per docs/examples). Convert to 0-1 probability-like price.
        yes_price = (yes_c / 100.0) if isinstance(yes_c, (int, float)) else None
        no_price = (no_c / 100.0) if isinstance(no_c, (int, float)) else None

        # "Spread" can be approximated from yes/no quotes if present.
        spread = None
        if yes_price is not None and no_price is not None:
            spread = abs(yes_price - (1.0 - no_price))

        return {
            "ticker": m.get("ticker"),
            "title": m.get("title"),
            "yes_price": yes_price,
            "no_price": no_price,
            "volume": m.get("volume"),
            "close_time": m.get("close_time"),
            "spread_est": spread,
            "raw": m,
        }
    except Exception:
        return None


# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="Pro Gas Algo", page_icon="⛽", layout="wide")

st.title("⛽ Kalshi Pro Gas Algorithm")
st.markdown("Multi-factor gas price prediction for Kalshi prediction markets.")

# --- Sidebar: FRED key + refresh ---
with st.sidebar:
    st.header("🔑 Configuration")

    # Prefer secrets; fall back to blank + message
    api_key = None
    if "FRED_API_KEY" in st.secrets:
        api_key = st.secrets["FRED_API_KEY"]
        st.success("FRED API key loaded from `secrets.toml`.")
    else:
        st.error("Missing `FRED_API_KEY` in `.streamlit/secrets.toml`.")

    force_refresh = st.button("🔄 Force Refresh Data")


# --- Market auto-load ---
st.subheader("📌 Kalshi Market (Auto-Load)")

colA, colB, colC = st.columns([2, 1, 1])
with colA:
    series_ticker = st.text_input(
        "Kalshi Series Ticker",
        value="KXAAAGASM",
        help="From your URL: https://kalshi.com/markets/<series>/...  Example: KXAAAGASM",
    )
with colB:
    status_filter = st.selectbox("Status", ["open", "all"], index=0)
with colC:
    load_market = st.button("🔗 Load from Kalshi", type="primary")

market_data = None
if load_market and series_ticker:
    # If user selects "all", just change the param
    if status_filter == "all":
        url = "https://api.elections.kalshi.com/trade-api/v2/markets"
        try:
            r = requests.get(url, params={"series_ticker": series_ticker, "status": "all"}, timeout=10)
            r.raise_for_status()
            mkts = r.json().get("markets", [])
            market_data = None
            if mkts:
                # choose first market returned (often newest first, but not guaranteed)
                m = mkts[0]
                yes_c = m.get("yes_price", None)
                no_c = m.get("no_price", None)
                yes_price = (yes_c / 100.0) if isinstance(yes_c, (int, float)) else None
                no_price = (no_c / 100.0) if isinstance(no_c, (int, float)) else None
                spread_est = None
                if yes_price is not None and no_price is not None:
                    spread_est = abs(yes_price - (1.0 - no_price))
                market_data = {
                    "ticker": m.get("ticker"),
                    "title": m.get("title"),
                    "yes_price": yes_price,
                    "no_price": no_price,
                    "volume": m.get("volume"),
                    "close_time": m.get("close_time"),
                    "spread_est": spread_est,
                    "raw": m,
                }
        except Exception:
            market_data = None
    else:
        market_data = get_kalshi_open_market_by_series(series_ticker)

    if market_data:
        st.success(f"Loaded market: `{market_data.get('ticker')}`")
    else:
        st.warning("Could not load market data (no markets found or request failed).")


# --- Manual / populated inputs ---
st.subheader("📋 Market Input")

default_title = market_data["title"] if market_data and market_data.get("title") else ""
default_yes = market_data["yes_price"] if market_data and isinstance(market_data.get("yes_price"), (int, float)) else 0.45
default_spread = (
    market_data["spread_est"]
    if market_data and isinstance(market_data.get("spread_est"), (int, float))
    else 0.02
)

col1, col2 = st.columns(2)
with col1:
    title = st.text_input("Market Title", value=default_title, placeholder="Will national gas prices exceed $3.50?")
with col2:
    price = st.slider("Current YES Price", min_value=0.01, max_value=0.99, value=float(default_yes), step=0.01)

spread = st.number_input(
    "Bid-Ask Spread (used for decision check)",
    min_value=0.0,
    max_value=0.20,
    value=float(default_spread),
    step=0.01,
    format="%.2f",
    help="If you loaded a market, this is a rough estimate from yes/no quotes. You can override it.",
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

                # --- Buy Signal Interpretation ---
                st.subheader("🎯 Decision Helper")

                signal_score = 0
                reasons: list[str] = []
                risks: list[str] = []

                # Edge strength
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

                # Edge vs spread
                if edge > 0 and spread > 0:
                    if edge >= 2 * spread:
                        signal_score += 2
                        reasons.append(f"Edge ({edge:.1%}) is ≥2× spread ({spread:.1%})")
                    elif edge >= spread:
                        signal_score += 1
                        reasons.append(f"Edge ({edge:.1%}) covers spread ({spread:.1%})")
                    else:
                        risks.append(f"Edge ({edge:.1%}) < spread ({spread:.1%}) — transaction costs likely dominate")

                # WTI
                wti = signals.get("wti", {})
                if wti.get("current_wti") is not None:
                    if wti.get("wti_change", 0) > 0.02:
                        signal_score += 1
                        reasons.append(f"WTI rising ({wti.get('wti_change'):+.1%}) — bullish tailwind")
                    elif wti.get("wti_change", 0) < -0.02:
                        risks.append(f"WTI falling ({wti.get('wti_change'):+.1%}) — bearish headwind")

                # Inventory
                inv = signals.get("inventory", {})
                if inv.get("current") is not None:
                    z = inv.get("z_score", 0)
                    if z < -0.5:
                        signal_score += 1
                        reasons.append(f"Inventory tight ({inv.get('status')}, Z={z:.2f})")
                    elif z > 0.5:
                        risks.append(f"Inventory ample ({inv.get('status')}, Z={z:.2f})")

                # Seasonal
                sea = signals.get("seasonal", {})
                if sea.get("multiplier") is not None:
                    mult = sea.get("multiplier", 1.0)
                    if mult >= 1.05:
                        signal_score += 1
                        reasons.append(f"Seasonal tailwind (multiplier {mult:.2f}×)")
                    elif mult < 1.0:
                        risks.append(f"Seasonal headwind (multiplier {mult:.2f}×)")

                # Final verdict
                if signal_score >= 6:
                    st.success("STRONG BUY (per model + spread check)")
                elif signal_score >= 4:
                    st.success("BUY (per model)")
                elif signal_score >= 2 and edge > 0:
                    st.warning("WEAK BUY (small edge / mixed signals)")
                else:
                    st.error("PASS (no clear edge or risks dominate)")

                if reasons:
                    st.markdown("**Supporting factors:**")
                    for r in reasons:
                        st.markdown(f"- {r}")

                if risks:
                    st.markdown("**Risk factors / cautions:**")
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
                        st.metric("Inventory Status", f"{inv.get('status')}")
                        st.metric("Z-Score", f"{inv.get('z_score', 0):.2f}")
                        st.metric("WoW Change", f"{inv.get('wow_change', 0):+.2%}")
                    else:
                        st.warning("Inventory data unavailable")

                    if ref.get("current") is not None:
                        st.metric("Refinery Util.", f"{ref.get('current', 0):.1f}%")
                        st.metric("Refinery Status", f"{ref.get('status', 'unknown')}")
                    else:
                        st.info(f"Refinery: {ref.get('status', 'Data unavailable')}")

                # Seasonal section
                st.subheader("📅 Seasonal Adjustment")
                s1, s2 = st.columns(2)
                s1.metric("Multiplier", f"{sea.get('multiplier', 1.0):.3f}x")
                s2.metric("Signal", f"{sea.get('signal', 0.0):+.3f}")
                if sea.get("factors"):
                    st.markdown("**Active factors:**")
                    for name, val in sea["factors"]:
                        st.markdown(f"- {name}: `{val:.2f}x`")

                # Regional
                st.subheader("🗺️ Regional PADD Prices")
                reg = signals.get("regional", {})
                if reg.get("regional_data"):
                    padd_labels = {
                        "padd1": "East Coast",
                        "padd2": "Midwest",
                        "padd3": "Gulf Coast",
                        "padd4": "Rocky Mountain",
                        "padd5": "West Coast",
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
