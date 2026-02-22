import os
import sys
import smtplib
import requests
import re as _re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

sys.path.insert(0, ".")

try:
    from kalshi_pro_gas import ProGasAlgo
except ImportError:
    print("❌ kalshi_pro_gas not found")
    sys.exit(1)

# ── Config from env ──
FRED_API_KEY  = os.environ["FRED_API_KEY"]
FROM_EMAIL    = os.environ["FROM_EMAIL"]
TO_EMAIL      = os.environ["TO_EMAIL"]
EMAIL_PASS    = os.environ["EMAIL_PASSWORD"]  # Gmail App Password
MIN_EDGE      = float(os.environ.get("MIN_EDGE", "0.05"))

BASE       = "https://api.elections.kalshi.com/trade-api/v2"
GAS_SERIES = ["KXAAAGASM", "KXAAAGASW", "KXAAGASM", "KXAAGASW"]


def fetch_markets():
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
                    yb = m.get("yes_bid")
                    ya = m.get("yes_ask")
                    yb_f = yb / 100.0 if isinstance(yb, (int, float)) else None
                    ya_f = ya / 100.0 if isinstance(ya, (int, float)) else None
                    mid  = (yb_f + ya_f) / 2.0 if yb_f and ya_f else None
                    sprd = (ya_f - yb_f)        if yb_f and ya_f else None
                    results.append({
                        "ticker":     ticker,
                        "title":      m.get("title", "").replace("**", ""),
                        "strike":     m.get("floor_strike"),
                        "mid_price":  mid,
                        "spread_est": sprd,
                        "volume":     m.get("volume"),
                        "close_time": m.get("close_time"),
                    })
                cursor = data.get("cursor")
                if not cursor or not markets:
                    break
            except Exception as e:
                print(f"Error fetching {series}: {e}")
                break
    return results


def kelly_fraction(edge, price):
    if price <= 0 or price >= 1 or edge == 0:
        return 0.0
    p = min(0.95, max(0.05, 0.5 + edge * 0.5))
    b = (1.0 - price) / price
    f = (p * b - (1.0 - p)) / b
    return round(min(0.25, max(0.01, f)), 4) if f > 0 else 0.0


def score_market(market, algo, signals):
    mid = market.get("mid_price")
    if mid is None:
        return None
    edge   = algo.edge(market["title"], mid)
    spread = market.get("spread_est", 0.05) or 0.05
    kelly  = kelly_fraction(edge, mid) if edge > 0 else 0.0

    score, reasons, risks = 0, [], []

    if edge >= 0.10:   score += 3; reasons.append(f"Strong edge ({edge:.1%})")
    elif edge >= 0.05: score += 2; reasons.append(f"Moderate edge ({edge:.1%})")
    elif edge > 0:     score += 1; reasons.append(f"Weak edge ({edge:.1%})")
    else:              risks.append(f"Negative edge ({edge:.1%})")

    if edge > 0 and spread > 0:
        if edge >= 2 * spread:   score += 2; reasons.append("Edge ≥ 2× spread")
        elif edge >= spread:     score += 1; reasons.append("Edge covers spread")
        else:                    risks.append("Edge < spread")

    wti_chg = (signals.get("wti") or {}).get("wti_change", 0) or 0
    if wti_chg > 0.02:  score += 1; reasons.append(f"WTI rising ({wti_chg:+.1%})")
    elif wti_chg < -0.02: risks.append(f"WTI falling ({wti_chg:+.1%})")

    inv = signals.get("inventory") or {}
    if inv.get("current") is not None:
        z = inv.get("z_score", 0) or 0
        if z < -0.5:   score += 1; reasons.append(f"Tight inventory (Z={z:.2f})")
        elif z > 0.5:  risks.append(f"Ample inventory (Z={z:.2f})")

    mult = (signals.get("seasonal") or {}).get("multiplier", 1.0) or 1.0
    if mult >= 1.05:  score += 1; reasons.append(f"Seasonal tailwind ({mult:.2f}×)")
    elif mult < 1.0:  risks.append(f"Seasonal headwind ({mult:.2f}×)")

    if score >= 6:        rec = "🟢 STRONG BUY"
    elif score >= 4:      rec = "🟡 BUY"
    elif score >= 2 and edge > 0: rec = "🟠 WEAK BUY"
    else:                 rec = "🔴 PASS"

    return {**market, "edge": edge, "fair_value": max(0, min(1, mid + edge)),
            "kelly": kelly, "signal_score": score,
            "recommendation": rec, "reasons": reasons, "risks": risks}


def send_alert(buys):
    """Send an HTML email listing all buy opportunities."""
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    rows = ""
    for m in buys:
        color = "#1a7a1a" if "STRONG" in m["recommendation"] else \
                "#7a6a00" if "WEAK" not in m["recommendation"] else "#7a3a00"
        rows += f"""
        <tr>
          <td style="color:{color};font-weight:bold">{m['recommendation']}</td>
          <td>{m['ticker']}</td>
          <td>{m['title'][:60]}</td>
          <td>${m['strike']:.2f}" if m.get('strike') else "N/A</td>
          <td>{m['mid_price']:.0%}</td>
          <td><b>{m['edge']:+.1%}</b></td>
          <td>{m['fair_value']:.0%}</td>
          <td>{m['kelly']:.1%} of bankroll</td>
          <td>{m['signal_score']}/8</td>
          <td>{', '.join(m['reasons'][:2])}</td>
        </tr>"""

    html = f"""
    <html><body style="font-family:Arial,sans-serif">
    <h2>⛽ Kalshi Pro Gas — Buy Alert</h2>
    <p>Scanned at <b>{now_str}</b> — found <b>{len(buys)}</b> buy signal(s)</p>
    <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;font-size:13px">
      <tr style="background:#f0f0f0">
        <th>Signal</th><th>Ticker</th><th>Title</th><th>Strike</th>
        <th>Mid</th><th>Edge</th><th>Fair Value</th>
        <th>Kelly Size</th><th>Score</th><th>Reasons</th>
      </tr>
      {rows}
    </table>
    <br>
    <p style="color:#888;font-size:11px">
      ⚠️ Quantitative model only — not financial advice.<br>
      <a href="https://kalshi-pro-gas-3a9trpznl8pvqmh3jsuusx.streamlit.app/">
        Open Dashboard →
      </a>
    </p>
    </body></html>"""

    strong_count = sum(1 for m in buys if "STRONG" in m["recommendation"])
    subject = (f"🟢 {strong_count} STRONG BUY{'s' if strong_count != 1 else ''} found — Kalshi Pro Gas"
               if strong_count else
               f"🟡 {len(buys)} Buy Signal{'s' if len(buys) != 1 else ''} found — Kalshi Pro Gas")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = FROM_EMAIL
    msg["To"]      = TO_EMAIL
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(FROM_EMAIL, EMAIL_PASS)
        server.sendmail(FROM_EMAIL, TO_EMAIL, msg.as_string())

    print(f"✅ Alert sent: {subject}")


def main():
    print(f"🔍 Starting scan at {datetime.utcnow().strftime('%H:%M UTC')}...")

    markets = fetch_markets()
    print(f"  Found {len(markets)} open markets")

    algo    = ProGasAlgo(fred_api_key=FRED_API_KEY)
    signals = algo.refresh_data()

    scored  = [score_market(m, algo, signals) for m in markets]
    scored  = [m for m in scored if m is not None and m.get("edge") is not None]

    buys = [m for m in scored
            if "BUY" in m.get("recommendation", "")
            and m.get("edge", 0) >= MIN_EDGE]

    buys.sort(key=lambda x: x.get("edge", 0), reverse=True)

    print(f"  Scored {len(scored)} markets, {len(buys)} buy signals above {MIN_EDGE:.0%}")

    for m in buys:
        print(f"  {m['recommendation']} {m['ticker']} — edge {m['edge']:+.1%} kelly {m['kelly']:.1%}")

    if buys:
        send_alert(buys)
    else:
        print("  No buy signals found — no alert sent.")


if __name__ == "__main__":
    main()
