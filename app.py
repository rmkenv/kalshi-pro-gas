import streamlit as st
import os
from kalshi_pro_gas import ProGasAlgo

st.set_page_config(page_title="Pro Gas Algo", page_icon="⛽", layout="wide")

st.title("⛽ Kalshi Pro Gas Algorithm")
st.markdown("Multi-factor gas price prediction for Kalshi prediction markets.")

# --- Sidebar: API Key ---
with st.sidebar:
    st.header("🔑 Configuration")
    api_key = st.secrets["FRED_API_KEY"]
    st.success("API key loaded from secrets.")
    force_refresh = st.button("🔄 Force Refresh Data")

# --- Main Input ---
st.subheader("📋 Market Input")
col1, col2 = st.columns(2)
with col1:
    title = st.text_input("Market Title", placeholder="Will national gas prices exceed $3.50?")
with col2:
    price = st.slider("Current Market Price", min_value=0.01, max_value=0.99, value=0.45, step=0.01)

run = st.button("⚡ Calculate Edge", type="primary")

# --- Run Algorithm ---
if run:
    if not title:
        st.error("Please enter a market title.")
    else:
        with st.spinner("Fetching data and calculating edge..."):
            try:
                algo = ProGasAlgo(fred_api_key=api_key)
                signals = algo.refresh_data(force=force_refresh)
                edge = algo.edge(title, price)

                # --- Edge Result ---
                st.subheader("📊 Edge Result")
                col1, col2, col3 = st.columns(3)
                col1.metric("Market Price", f"{price:.2%}")
                col2.metric("Calculated Edge", f"{edge:+.2%}")
                fv = price + edge
                col3.metric("Implied Fair Value", f"{fv:.2%}")

                # --- Buy Signal Interpretation ---
                st.subheader("🎯 Buy Signal")

                spread = st.number_input(
                    "Kalshi Bid-Ask Spread (optional)",
                    min_value=0.0, max_value=0.20, value=0.02, step=0.01, format="%.2f",
                    help="Enter the spread between Yes/No prices on Kalshi to check if edge beats transaction cost."
                )

                signal_score = 0
                reasons = []
                warnings = []

                # Edge strength
                if edge >= 0.10:
                    signal_score += 3
                    reasons.append("✅ Strong positive edge (≥10%)")
                elif edge >= 0.05:
                    signal_score += 2
                    reasons.append("✅ Moderate positive edge (≥5%)")
                elif edge > 0:
                    signal_score += 1
                    reasons.append("⚠️ Weak positive edge (<5%)")
                else:
                    warnings.append("❌ Negative edge — market appears overpriced")

                # Edge vs spread
                if edge > 0 and spread > 0:
                    if edge >= spread * 2:
                        signal_score += 2
                        reasons.append(f"✅ Edge ({edge:.1%}) is 2x+ the spread ({spread:.1%})")
                    elif edge >= spread:
                        signal_score += 1
                        reasons.append(f"⚠️ Edge ({edge:.1%}) barely covers the spread ({spread:.1%})")
                    else:
                        warnings.append(f"❌ Edge ({edge:.1%}) is smaller than the spread ({spread:.1%}) — not worth it")

                # WTI momentum
                wti = signals['wti']
                if wti['current_wti'] and wti['wti_change'] > 0.02:
                    signal_score += 1
                    reasons.append(f"✅ WTI rising +{wti['wti_change']:.1%} — gas prices likely to follow")
                elif wti['current_wti'] and wti['wti_change'] < -0.02:
                    warnings.append(f"⚠️ WTI falling {wti['wti_change']:.1%} — headwind for gas prices")

                # Inventory
                inv = signals['inventory']
                if inv['current']:
                    if inv['z_score'] < -0.5:
                        signal_score += 1
                        reasons.append(f"✅ Inventory is {inv['status']} (Z={inv['z_score']:.2f}) — supply tight")
                    elif inv['z_score'] > 0.5:
                        warnings.append(f"⚠️ Inventory is {inv['status']} (Z={inv['z_score']:.2f}) — supply ample")

                # Seasonal
                sea = signals['seasonal']
                if sea['multiplier'] >= 1.05:
                    signal_score += 1
                    reasons.append(f"✅ Peak season multiplier ({sea['multiplier']:.2f}x) supports higher prices")
                elif sea['multiplier'] < 1.0:
                    warnings.append(f"⚠️ Off-season multiplier ({sea['multiplier']:.2f}x) — seasonal headwind")

                # Final verdict
                if signal_score >= 6:
                    st.success("🟢 **STRONG BUY** — Multiple signals aligned, edge exceeds spread")
                elif signal_score >= 4:
                    st.success("🟡 **BUY** — Positive edge with supporting signals")
                elif signal_score >= 2 and edge > 0:
                    st.warning("🟠 **WEAK BUY** — Small edge, proceed with caution")
                else:
                    st.error("🔴 **PASS** — No clear edge or signals are unfavorable")

                if reasons:
                    st.markdown("**Supporting Factors:**")
                    for r in reasons:
                        st.markdown(f"- {r}")

                if warnings:
                    st.markdown("**Risk Factors:**")
                    for w in warnings:
                        st.markdown(f"- {w}")

                # --- Signal Breakdown ---
                st.subheader("🔍 Signal Breakdown")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**💰 Gas Price Momentum**")
                    gm = signals['gas_momentum']
                    if gm['current']:
                        st.metric("Current Price", f"${gm['current']:.3f}/gal")
                        st.metric("4-Week Momentum", f"{gm['momentum']:+.2%}")
                        st.metric("12-Week Trend", f"{gm['trend']:+.2%}")
                        st.metric("52-Week Avg", f"${gm['avg_52w']:.3f}/gal")
                    else:
                        st.warning("Data unavailable")

                with col2:
                    st.markdown("**🛢️ WTI Crude Oil**")
                    if wti['current_wti']:
                        st.metric("Current WTI", f"${wti['current_wti']:.2f}/bbl")
                        st.metric("Lagged WTI", f"${wti['lagged_wti']:.2f}/bbl")
                        st.metric("WTI Change", f"{wti['wti_change']:+.2%}")
                        st.metric("Optimal Lag", f"{wti['optimal_lag']} week(s)")
                    else:
                        st.warning("Data unavailable")

                with col3:
                    st.markdown("**📦 Inventory & Refinery**")
                    ref = signals['refinery']
                    if inv['current']:
                        st.metric("Inventory Status", inv['status'])
                        st.metric("Z-Score", f"{inv['z_score']:.2f}")
                        st.metric("WoW Change", f"{inv['wow_change']:+.2%}")
                    if ref['current']:
                        st.metric("Refinery Util.", f"{ref['current']:.1f}%")
                        st.metric("Refinery Status", ref['status'])
                    elif not inv['current']:
                        st.warning("Data unavailable")

                # --- Seasonal ---
                st.subheader("📅 Seasonal Adjustment")
                scol1, scol2 = st.columns(2)
                scol1.metric("Multiplier", f"{sea['multiplier']:.3f}x")
                scol2.metric("Signal", f"{sea['signal']:+.3f}")
                if sea['factors']:
                    st.markdown("**Active Factors:**")
                    for name, val in sea['factors']:
                        st.markdown(f"- {name}: `{val:.2f}x`")

                # --- Regional ---
                st.subheader("🗺️ Regional PADD Prices")
                reg = signals['regional']
                if reg['regional_data']:
                    padd_labels = {
                        'padd1': 'East Coast',
                        'padd2': 'Midwest',
                        'padd3': 'Gulf Coast',
                        'padd4': 'Rocky Mountain',
                        'padd5': 'West Coast'
                    }
                    cols = st.columns(5)
                    for i, (padd, data) in enumerate(reg['regional_data'].items()):
                        cols[i].metric(padd_labels.get(padd, padd), f"${data['price']:.3f}")
                    st.metric("Weighted Avg", f"${reg['weighted_avg']:.3f}/gal")
                    st.metric("Regional Spread", f"${reg['spread']:.3f}")
                else:
                    st.warning("Regional data unavailable")

            except ValueError as e:
                st.error(f"Initialization error: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
