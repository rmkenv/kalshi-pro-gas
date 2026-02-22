
"""
KALSHI PRO GAS BACKTEST v5.3 - NO-EDGE FOCUS + ANNUALIZED SHARPE
"""

import subprocess
subprocess.run(["pip", "install", "numpy", "pandas", "matplotlib", "-q"], check=False)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# ── STEP 1: ALGO SETUP ──────────────────────────────────────────

class ProGasAlgo:
    """
    Stub algo with v5.3 filters:
    - Focus on NO edge
    - NO dead zone shrunk to capture more edge
    - YES trades disabled in backtest (until we have a real signal)
    """
    def __init__(self, **kwargs):
        pass

    def refresh_data(self, **kwargs):
        return {"gas_momentum": {"current": 2.924}, "wti": {"current_wti": 62.53}}

    def edge(self, title, yes_price: float) -> float:
        title_lower = title.lower()
        match = re.search(r"\$(\d+\.\d+)", title)
        strike = float(match.group(1)) if match else 3.00
        current_gas = 2.924

        # Base signal: how far strike is from current gas
        base_signal = (current_gas - strike) / current_gas

        # Simple seasonal tilt
        seasonal = 0.02 if any(m in title_lower for m in ["jan", "feb", "mar", "dec"]) else -0.01

        combined = base_signal * 0.7 + seasonal

        # Convert combined signal to a "fair" YES probability
        fair = 0.5 + combined
        fair = max(0.05, min(0.95, fair))

        raw_edge = fair - yes_price  # positive: like YES, negative: like NO

        # v5.3: softer thresholds so we get more trades (help Sharpe via LLN)
        # YES filter: keep only strong YES edge, but we won't use them in backtest yet
        if 0 < raw_edge < 0.18:
            return 0.0

        # NO filter: small negative edges are noisy, ignore them
        if -0.05 < raw_edge < 0:
            return 0.0

        # If model says YES but underlying signal is strongly bearish, flip to NO
        if raw_edge > 0 and combined < -0.02:
            return -raw_edge

        return raw_edge


algo = ProGasAlgo()
print("✅ Algo v5.3 Ready | Gas: $2.924")

# ── STEP 2: BUILD 5000 SYNTHETIC MARKETS ────────────────────────

np.random.seed(42)

strike_prices = [2.80, 2.90, 3.00, 3.10, 3.20, 3.30, 3.40, 3.50]
months        = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

markets = []
for i in range(5000):
    strike = np.random.choice(strike_prices)
    month  = np.random.choice(months)
    year   = np.random.choice([2024, 2025, 2026])

    # "True" probability curve: monotone with strike around 2.92 base
    prob_yes = max(0.15, min(0.85, 0.5 - (strike - 2.92) * 1.2))

    # Realized outcome
    settlement = "YES" if np.random.random() < prob_yes else "NO"

    # Market price ~ true prob + noise
    yes_price = float(np.clip(prob_yes + np.random.normal(0, 0.10), 0.10, 0.90))

    markets.append({
        "id":         f"GAS-{i+1:04d}",
        "title":      f"Will US gas exceed ${strike:.2f} in {month} {year}?",
        "yes_price":  round(yes_price, 3),
        "settlement": settlement,
        "year":       year,
    })

print(f"📊 {len(markets)} markets generated")

# ── STEP 3: BACKTEST + KELLY ─────────────────────────────────────

BANKROLL = 10_000.0  # more realistic capital base
trades   = []

for market in markets:
    yes_price = market["yes_price"]
    edge      = algo.edge(market["title"], yes_price)

    if edge == 0:
        continue

    position = "YES" if edge > 0 else "NO"

    # v5.3: disable YES trades for now, focus on NO where we know we have edge
    if position == "YES":
        continue

    entry_price = 1.0 - yes_price  # NO leg
    outcome     = 1 if market["settlement"] == "YES" else 0

    # Payoff on NO
    raw_pnl = (1.0 - entry_price) if outcome == 0 else -entry_price

    # Kelly criterion
    win_prob = float(np.clip(0.5 + edge * 0.8, 0.1, 0.9))
    b        = (1.0 - entry_price) / entry_price if entry_price > 0 else 1.0

    # v5.3: 0.5× Kelly, min 2%, max 15%
    kelly = (win_prob * b - (1 - win_prob)) / b
    kelly = float(np.clip(kelly * 0.5, 0.02, 0.15))

    stake     = BANKROLL * kelly
    kelly_pnl = raw_pnl * stake

    trades.append({
        "id":          market["id"],
        "year":        market["year"],
        "position":    position,
        "entry_price": round(entry_price, 3),
        "edge":        round(edge, 4),
        "win_prob":    round(win_prob, 3),
        "kelly_pct":   round(kelly, 3),
        "stake":       round(stake, 2),
        "raw_pnl":     round(raw_pnl, 3),
        "kelly_pnl":   round(kelly_pnl, 2),
        "win":         raw_pnl > 0,
    })

# ── STEP 4: METRICS (WEEKLY, ANNUALIZED SHARPE) ─────────────────

df = pd.DataFrame(trades)
if df.empty:
    raise SystemExit("No trades generated with current filters.")

total_pnl  = df["kelly_pnl"].sum()
win_rate   = df["win"].mean()
max_dd     = (df["kelly_pnl"].cumsum().cummax() - df["kelly_pnl"].cumsum()).max()
final_bank = BANKROLL + total_pnl
roi        = (total_pnl / BANKROLL) * 100
avg_edge   = df["edge"].mean()
avg_kelly  = df["kelly_pct"].mean()

# Group trades ~10 per "week" and compute annualized Sharpe
df["trade_num"] = range(len(df))
df["week"]      = df["trade_num"] // 10
weekly_pnl      = df.groupby("week")["kelly_pnl"].sum()

if weekly_pnl.std() > 0:
    sharpe_annual = (weekly_pnl.mean() / weekly_pnl.std()) * np.sqrt(52)
else:
    sharpe_annual = 0.0

print("\n" + "="*60)
print("🏆  BACKTEST RESULTS — 5000 SAMPLES (v5.3 NO-EDGE + ANN. SHARPE)")
print("="*60)
print(f"📈  Total Trades:       {len(df)}")
print(f"✅  Win Rate:           {win_rate:.1%}")
print(f"💰  Total Kelly PnL:    ${total_pnl:,.2f}")
print(f"🏦  Final Bankroll:     ${final_bank:,.2f}  (started ${BANKROLL:,.0f})")
print(f"📊  ROI:                {roi:.1f}%")
print(f"⚡  Annualized Sharpe:  {sharpe_annual:.2f}")
print(f"📉  Max Drawdown:       ${max_dd:,.2f}")
print(f"🎯  Avg Edge:           {avg_edge:+.4f}")
print(f"💡  Avg Kelly Bet:      {avg_kelly:.1%}")
print("="*60)

for pos in ["YES", "NO"]:
    s = df[df["position"] == pos]
    if len(s):
        print(f"  {pos}: {len(s)} trades | WR: {s['win'].mean():.0%} | PnL: ${s['kelly_pnl'].sum():,.2f}")

print("\n📅 RESULTS BY YEAR:")
for yr in sorted(df["year"].unique()):
    s = df[df["year"] == yr]
    print(f"  {yr}: {len(s)} trades | WR: {s['win'].mean():.0%} | PnL: ${s['kelly_pnl'].sum():,.2f}")

print("\n💰 TOP 5 TRADES:")
print(df.nlargest(5, "kelly_pnl")[["id","year","position","edge","kelly_pct","kelly_pnl"]].to_string(index=False))

print("\n📉 WORST 5 TRADES:")
print(df.nsmallest(5, "kelly_pnl")[["id","year","position","edge","kelly_pct","kelly_pnl"]].to_string(index=False))

# ── STEP 5: CHARTS ───────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Kalshi Pro Gas — 5000 Samples (v5.3 NO-EDGE FOCUS)", fontsize=16, fontweight="bold")

# 1. Bankroll growth
cum = df["kelly_pnl"].cumsum() + BANKROLL
axes[0, 0].plot(cum.values, "g-", lw=3)
axes[0, 0].set_title("Bankroll Growth")
axes[0, 0].set_ylabel("Bankroll ($)")
axes[0, 0].grid(True, alpha=0.3)

# 2. Edge vs PnL
colors = ["green" if w else "red" for w in df["win"]]
axes[0, 1].scatter(df["edge"], df["kelly_pnl"], c=colors, alpha=0.5, s=20)
axes[0, 1].set_title("Edge vs Kelly PnL")
axes[0, 1].grid(True, alpha=0.3)

# 3. PnL histogram
axes[0, 2].hist(df["kelly_pnl"], bins=30, alpha=0.7, color="steelblue")
axes[0, 2].set_title("PnL Distribution")
axes[0, 2].grid(True, alpha=0.3)

# 4. Kelly bet sizes
axes[1, 0].hist(df["kelly_pct"] * 100, bins=15, alpha=0.7, color="orange")
axes[1, 0].set_title("Kelly Bet Size (%)")
axes[1, 0].grid(True, alpha=0.3)

# 5. PnL by position (should be NO only)
pos_pnl = df.groupby("position")["kelly_pnl"].sum()
axes[1, 1].bar(pos_pnl.index, pos_pnl.values,
               color=["red" if v < 0 else "green" for v in pos_pnl.values])
axes[1, 1].set_title("PnL by Position Type")
axes[1, 1].grid(True, alpha=0.3)

# 6. Rolling win rate (50-trade window)
rolling_wr = df["win"].rolling(50).mean() * 100
axes[1, 2].plot(rolling_wr, "b-")
axes[1, 2].axhline(50, color="red", ls="--")
axes[1, 2].set_title("Rolling Win Rate (50-trade)")
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()# -*- coding: utf-8 -*-
"""
KALSHI PRO GAS BACKTEST v5.3 - NO-EDGE FOCUS + ANNUALIZED SHARPE
"""

import subprocess
subprocess.run(["pip", "install", "numpy", "pandas", "matplotlib", "-q"], check=False)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# ── STEP 1: ALGO SETUP ──────────────────────────────────────────

class ProGasAlgo:
    """
    Stub algo with v5.3 filters:
    - Focus on NO edge
    - NO dead zone shrunk to capture more edge
    - YES trades disabled in backtest (until we have a real signal)
    """
    def __init__(self, **kwargs):
        pass

    def refresh_data(self, **kwargs):
        return {"gas_momentum": {"current": 2.924}, "wti": {"current_wti": 62.53}}

    def edge(self, title, yes_price: float) -> float:
        title_lower = title.lower()
        match = re.search(r"\$(\d+\.\d+)", title)
        strike = float(match.group(1)) if match else 3.00
        current_gas = 2.924

        # Base signal: how far strike is from current gas
        base_signal = (current_gas - strike) / current_gas

        # Simple seasonal tilt
        seasonal = 0.02 if any(m in title_lower for m in ["jan", "feb", "mar", "dec"]) else -0.01

        combined = base_signal * 0.7 + seasonal

        # Convert combined signal to a "fair" YES probability
        fair = 0.5 + combined
        fair = max(0.05, min(0.95, fair))

        raw_edge = fair - yes_price  # positive: like YES, negative: like NO

        # v5.3: softer thresholds so we get more trades (help Sharpe via LLN)
        # YES filter: keep only strong YES edge, but we won't use them in backtest yet
        if 0 < raw_edge < 0.18:
            return 0.0

        # NO filter: small negative edges are noisy, ignore them
        if -0.05 < raw_edge < 0:
            return 0.0

        # If model says YES but underlying signal is strongly bearish, flip to NO
        if raw_edge > 0 and combined < -0.02:
            return -raw_edge

        return raw_edge


algo = ProGasAlgo()
print("✅ Algo v5.3 Ready | Gas: $2.924")

# ── STEP 2: BUILD 5000 SYNTHETIC MARKETS ────────────────────────

np.random.seed(42)

strike_prices = [2.80, 2.90, 3.00, 3.10, 3.20, 3.30, 3.40, 3.50]
months        = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

markets = []
for i in range(5000):
    strike = np.random.choice(strike_prices)
    month  = np.random.choice(months)
    year   = np.random.choice([2024, 2025, 2026])

    # "True" probability curve: monotone with strike around 2.92 base
    prob_yes = max(0.15, min(0.85, 0.5 - (strike - 2.92) * 1.2))

    # Realized outcome
    settlement = "YES" if np.random.random() < prob_yes else "NO"

    # Market price ~ true prob + noise
    yes_price = float(np.clip(prob_yes + np.random.normal(0, 0.10), 0.10, 0.90))

    markets.append({
        "id":         f"GAS-{i+1:04d}",
        "title":      f"Will US gas exceed ${strike:.2f} in {month} {year}?",
        "yes_price":  round(yes_price, 3),
        "settlement": settlement,
        "year":       year,
    })

print(f"📊 {len(markets)} markets generated")

# ── STEP 3: BACKTEST + KELLY ─────────────────────────────────────

BANKROLL = 10_000.0  # more realistic capital base
trades   = []

for market in markets:
    yes_price = market["yes_price"]
    edge      = algo.edge(market["title"], yes_price)

    if edge == 0:
        continue

    position = "YES" if edge > 0 else "NO"

    # v5.3: disable YES trades for now, focus on NO where we know we have edge
    if position == "YES":
        continue

    entry_price = 1.0 - yes_price  # NO leg
    outcome     = 1 if market["settlement"] == "YES" else 0

    # Payoff on NO
    raw_pnl = (1.0 - entry_price) if outcome == 0 else -entry_price

    # Kelly criterion
    win_prob = float(np.clip(0.5 + edge * 0.8, 0.1, 0.9))
    b        = (1.0 - entry_price) / entry_price if entry_price > 0 else 1.0

    # v5.3: 0.5× Kelly, min 2%, max 15%
    kelly = (win_prob * b - (1 - win_prob)) / b
    kelly = float(np.clip(kelly * 0.5, 0.02, 0.15))

    stake     = BANKROLL * kelly
    kelly_pnl = raw_pnl * stake

    trades.append({
        "id":          market["id"],
        "year":        market["year"],
        "position":    position,
        "entry_price": round(entry_price, 3),
        "edge":        round(edge, 4),
        "win_prob":    round(win_prob, 3),
        "kelly_pct":   round(kelly, 3),
        "stake":       round(stake, 2),
        "raw_pnl":     round(raw_pnl, 3),
        "kelly_pnl":   round(kelly_pnl, 2),
        "win":         raw_pnl > 0,
    })

# ── STEP 4: METRICS (WEEKLY, ANNUALIZED SHARPE) ─────────────────

df = pd.DataFrame(trades)
if df.empty:
    raise SystemExit("No trades generated with current filters.")

total_pnl  = df["kelly_pnl"].sum()
win_rate   = df["win"].mean()
max_dd     = (df["kelly_pnl"].cumsum().cummax() - df["kelly_pnl"].cumsum()).max()
final_bank = BANKROLL + total_pnl
roi        = (total_pnl / BANKROLL) * 100
avg_edge   = df["edge"].mean()
avg_kelly  = df["kelly_pct"].mean()

# Group trades ~10 per "week" and compute annualized Sharpe
df["trade_num"] = range(len(df))
df["week"]      = df["trade_num"] // 10
weekly_pnl      = df.groupby("week")["kelly_pnl"].sum()

if weekly_pnl.std() > 0:
    sharpe_annual = (weekly_pnl.mean() / weekly_pnl.std()) * np.sqrt(52)
else:
    sharpe_annual = 0.0

print("\n" + "="*60)
print("🏆  BACKTEST RESULTS — 5000 SAMPLES (v5.3 NO-EDGE + ANN. SHARPE)")
print("="*60)
print(f"📈  Total Trades:       {len(df)}")
print(f"✅  Win Rate:           {win_rate:.1%}")
print(f"💰  Total Kelly PnL:    ${total_pnl:,.2f}")
print(f"🏦  Final Bankroll:     ${final_bank:,.2f}  (started ${BANKROLL:,.0f})")
print(f"📊  ROI:                {roi:.1f}%")
print(f"⚡  Annualized Sharpe:  {sharpe_annual:.2f}")
print(f"📉  Max Drawdown:       ${max_dd:,.2f}")
print(f"🎯  Avg Edge:           {avg_edge:+.4f}")
print(f"💡  Avg Kelly Bet:      {avg_kelly:.1%}")
print("="*60)

for pos in ["YES", "NO"]:
    s = df[df["position"] == pos]
    if len(s):
        print(f"  {pos}: {len(s)} trades | WR: {s['win'].mean():.0%} | PnL: ${s['kelly_pnl'].sum():,.2f}")

print("\n📅 RESULTS BY YEAR:")
for yr in sorted(df["year"].unique()):
    s = df[df["year"] == yr]
    print(f"  {yr}: {len(s)} trades | WR: {s['win'].mean():.0%} | PnL: ${s['kelly_pnl'].sum():,.2f}")

print("\n💰 TOP 5 TRADES:")
print(df.nlargest(5, "kelly_pnl")[["id","year","position","edge","kelly_pct","kelly_pnl"]].to_string(index=False))

print("\n📉 WORST 5 TRADES:")
print(df.nsmallest(5, "kelly_pnl")[["id","year","position","edge","kelly_pct","kelly_pnl"]].to_string(index=False))

# ── STEP 5: CHARTS ───────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Kalshi Pro Gas — 5000 Samples (v5.3 NO-EDGE FOCUS)", fontsize=16, fontweight="bold")

# 1. Bankroll growth
cum = df["kelly_pnl"].cumsum() + BANKROLL
axes[0, 0].plot(cum.values, "g-", lw=3)
axes[0, 0].set_title("Bankroll Growth")
axes[0, 0].set_ylabel("Bankroll ($)")
axes[0, 0].grid(True, alpha=0.3)

# 2. Edge vs PnL
colors = ["green" if w else "red" for w in df["win"]]
axes[0, 1].scatter(df["edge"], df["kelly_pnl"], c=colors, alpha=0.5, s=20)
axes[0, 1].set_title("Edge vs Kelly PnL")
axes[0, 1].grid(True, alpha=0.3)

# 3. PnL histogram
axes[0, 2].hist(df["kelly_pnl"], bins=30, alpha=0.7, color="steelblue")
axes[0, 2].set_title("PnL Distribution")
axes[0, 2].grid(True, alpha=0.3)

# 4. Kelly bet sizes
axes[1, 0].hist(df["kelly_pct"] * 100, bins=15, alpha=0.7, color="orange")
axes[1, 0].set_title("Kelly Bet Size (%)")
axes[1, 0].grid(True, alpha=0.3)

# 5. PnL by position (should be NO only)
pos_pnl = df.groupby("position")["kelly_pnl"].sum()
axes[1, 1].bar(pos_pnl.index, pos_pnl.values,
               color=["red" if v < 0 else "green" for v in pos_pnl.values])
axes[1, 1].set_title("PnL by Position Type")
axes[1, 1].grid(True, alpha=0.3)

# 6. Rolling win rate (50-trade window)
rolling_wr = df["win"].rolling(50).mean() * 100
axes[1, 2].plot(rolling_wr, "b-")
axes[1, 2].axhline(50, color="red", ls="--")
axes[1, 2].set_title("Rolling Win Rate (50-trade)")
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
