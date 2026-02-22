# backtest.py

import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from kalshi_pro_gas import ProGasAlgo  # your existing module


KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"


def fetch_gas_markets(
    start_date: datetime,
    end_date: datetime,
    limit: int = 1000,
) -> List[Dict]:
    """
    Fetch all Kalshi markets matching gas‑price keywords.
    Uses public /markets endpoint (no key).
    """
    markets = []
    cursor = None

    while True:
        params = {
            "limit": limit,
            "cursor": cursor,
        }
        url = f"{KALSHI_BASE}/markets"
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        batch = data["markets"]
        if not batch:
            break

        # Filter gas‑price markets
        for m in batch:
            title = m["title"].upper()
            if "GAS" not in title and "GASOLINE" not in title:
                continue
            if "US" not in title and "NATIONAL" not in title:
                continue

            markets.append(m)

        cursor = data.get("cursor")
        if not cursor:
            break

    # Filter by date range
    filtered = []
    for m in markets:
        created = datetime.fromisoformat(m["created_at"].rstrip("Z"))
        if start_date <= created <= end_date:
            filtered.append(m)

    return filtered


def fetch_candlesticks(ticker: str) -> List[Dict]:
    """
    Fetch candlestick history for a market.
    """
    url = f"{KALSHI_BASE}/markets/{ticker}/candlesticks"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    return data["candlesticks"]


def timestamp_to_datetime(ts: int) -> datetime:
    """
    Kalshi timestamps are in seconds since epoch.
    """
    return datetime.utcfromtimestamp(ts)


def backtest_pro_gas(
    algo: ProGasAlgo,
    markets: List[Dict],
    edge_threshold: float = 0.05,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> Dict:
    """
    Run backtest on historical Kalshi gas markets.
    """
    if start_date is None:
        start_date = datetime(2023, 1, 1)
    if end_date is None:
        end_date = datetime.now()

    trades = []

    for market in markets:
        ticker = market["ticker"]
        title = market["title"]
        settlement_ts = int(market["settlement_ts"])
        settlement_date = timestamp_to_datetime(settlement_ts)

        # Skip if settlement outside our range
        if settlement_date < start_date or settlement_date > end_date:
            continue

        try:
            candles = fetch_candlesticks(ticker)
        except Exception as e:
            print(f"⚠️ Failed to fetch candles for {ticker}: {e}")
            continue

        # For each candle, compute edge and decide to trade
        for candle in candles:
            ts = candle["ts"]
            t = timestamp_to_datetime(ts)

            # Align FRED data to this timestamp
            signals = algo.refresh_data(force=True)  # your method already respects cache

            # Use the current market price (midpoint)
            bid = candle["bid"]
            ask = candle["ask"]
            mid = (bid + ask) / 2.0

            # Compute edge
            edge = algo.edge(title, mid)

            # Skip if no edge or below threshold
            if abs(edge) < edge_threshold:
                continue

            # Decide position
            if edge > 0:
                position = "YES"
                entry_price = mid
            else:
                position = "NO"
                entry_price = 1.0 - mid  # NO side

            # Outcome: 1 = YES, 0 = NO
            outcome = 1 if market["settlement"] == "YES" else 0

            # PnL
            if position == "YES":
                pnl = (1.0 - entry_price) if outcome == 1 else -entry_price
            else:
                pnl = (1.0 - entry_price) if outcome == 0 else -entry_price

            trades.append(
                {
                    "ticker": ticker,
                    "date": t,
                    "settlement_date": settlement_date,
                    "position": position,
                    "entry_price": entry_price,
                    "edge": edge,
                    "bid": bid,
                    "ask": ask,
                    "outcome": outcome,
                    "pnl": pnl,
                    "win": pnl > 0,
                }
            )

    # Compute metrics
    if not trades:
        return {"trades": [], "metrics": {}}

    df = pd.DataFrame(trades)
    metrics = {
        "total_trades": len(df),
        "wins": int(df["win"].sum()),
        "losses": int((~df["win"]).sum()),
        "win_rate": float(df["win"].mean()),
        "total_pnl": float(df["pnl"].sum()),
        "avg_pnl_per_trade": float(df["pnl"].mean()),
        "std_pnl": float(df["pnl"].std()),
        "sharpe_ratio": (
            float(df["pnl"].mean() / df["pnl"].std())
            if df["pnl"].std() > 0
            else 0.0
        ),
        "max_drawdown": float(
            (df["pnl"].cumsum().cummax() - df["pnl"].cumsum()).max()
        ),
        "best_trade": float(df["pnl"].max()),
        "worst_trade": float(df["pnl"].min()),
        "avg_edge": float(df["edge"].mean()),
    }

    return {"trades": trades, "metrics": metrics}


def main():
    # Initialize your algo
    algo = ProGasAlgo(fred_api_key=os.environ.get("FRED_API_KEY"))

    # Fetch markets (2023–2026 gas‑price markets)
    start = datetime(2023, 1, 1)
    end = datetime.now()
    markets = fetch_gas_markets(start, end)

    print(f"Found {len(markets)} gas‑price markets")

    # Run backtest
    result = backtest_pro_gas(
        algo,
        markets,
        edge_threshold=0.05,
        start_date=start,
        end_date=end,
    )

    metrics = result["metrics"]
    print("\n📊 BACKTEST RESULTS (2023–2026 gas markets)")
    print("-" * 60)
    print(f"Total trades:      {metrics['total_trades']}")
    print(f"Wins / Losses:     {metrics['wins']} / {metrics['losses']}")
    print(f"Win rate:          {metrics['win_rate']:.1%}")
    print(f"Total PnL:         ${metrics['total_pnl']:.2f}")
    print(f"Avg PnL/trade:     ${metrics['avg_pnl_per_trade']:.3f}")
    print(f"Sharpe ratio:      {metrics['sharpe_ratio']:.2f}")
    print(f"Max drawdown:      ${metrics['max_drawdown']:.2f}")
    print(f"Avg edge:          {metrics['avg_edge']:+.3f}")


if __name__ == "__main__":
    import os
    main()
