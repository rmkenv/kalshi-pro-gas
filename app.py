import streamlit as st
import os
from kalshi_pro_gas import progas

st.set_page_config(page_title="Pro Gas Algo", page_icon="⛽", layout="wide")

st.title("⛽ Kalshi Pro Gas Algorithm")
st.markdown("Multi-factor gas price prediction for Kalshi prediction markets.")

# --- Sidebar: API Key ---
with st.sidebar:
    st.header("🔑 Configuration")
    api_key = st.text_input("FRED API Key", type="password", placeholder="Enter your FRED API key")
    st.markdown("[Get a free API key](https://fred.stlouisfed.org/docs/api/api_key.html)")
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
    if not api_key:
        st.error("Please enter your FRED API key in the sidebar.")
    elif not title:
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

                if edge > 0.05:
                    st.success("✅ Positive edge — market appears **underpriced**")
                elif edge < -0.05:
                    st.error("❌ Negative edge — market appears **overpriced**")
                else:
                    st.info("⚖️ Near fair value — no strong edge detected")

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
                    wti = signals['wti']
                    if wti['current_wti']:
                        st.metric("Current WTI", f"${wti['current_wti']:.2f}/bbl")
                        st.metric("Lagged WTI", f"${wti['lagged_wti']:.2f}/bbl")
                        st.metric("WTI Change", f"{wti['wti_change']:+.2%}")
                        st.metric("Optimal Lag", f"{wti['optimal_lag']} week(s)")
                    else:
                        st.warning("Data unavailable")

                with col3:
                    st.markdown("**📦 Inventory & Refinery**")
                    inv = signals['inventory']
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
                sea = signals['seasonal']
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
