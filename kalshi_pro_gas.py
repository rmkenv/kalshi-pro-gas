import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os
import json
from scipy import stats

__version__ = "1.2.0"
__author__ = "RMK Solutions"

# ============================================
# PRO GAS DATA HUB
# ============================================

class ProGasDataHub:
    """
    Enhanced data hub specifically for gas market analysis.
    Fetches and processes FRED data for comprehensive gas price modeling.
    """

    SERIES_IDS = {
        'gas_national':  'GASREGW',       # U.S. Regular All Formulations Retail Gas - Weekly
        'wti_crude':     'DCOILWTICO',    # WTI Crude Oil - Daily
        'gas_inventory': 'WGFUPUS2',      # Weekly U.S. Ending Stocks of Finished Motor Gasoline
        'refinery_util': 'WPULEUS3',      # Weekly U.S. Refinery Operable Capacity Utilization
        'padd1':         'GASREGCOVECW',  # East Coast (PADD 1)
        'padd2':         'GASREGMWW',     # Midwest (PADD 2)
        'padd3':         'GASREGGCW',     # Gulf Coast (PADD 3)
        'padd4':         'GASREGRMW',     # Rocky Mountain (PADD 4)
        'padd5':         'GASREGREFWCW',  # West Coast (PADD 5)
    }

    PADD_WEIGHTS = {
        'padd1': 0.18,
        'padd2': 0.21,
        'padd3': 0.29,
        'padd4': 0.05,
        'padd5': 0.27,
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('FRED_API_KEY')
        if not self.api_key:
            raise ValueError(
                "FRED API key required. Set FRED_API_KEY environment variable or pass api_key parameter.\n"
                "Get your free API key at: https://fred.stlouisfed.org/docs/api/api_key.html"
            )
        self.cache = {}

    def fred_series(self, series_id: str, limit: int = 52, start_date: Optional[str] = '2022-01-01') -> List[Dict]:
        cache_key = f"{series_id}_{limit}_{start_date}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json',
            'sort_order': 'desc',
            'limit': limit
        }
        if start_date:
            params['observation_start'] = start_date

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            observations = []
            for obs in data.get('observations', []):
                if obs['value'] != '.':
                    try:
                        observations.append({
                            'date': datetime.strptime(obs['date'], '%Y-%m-%d'),
                            'value': float(obs['value'])
                        })
                    except (ValueError, KeyError):
                        continue

            self.cache[cache_key] = observations
            return observations

        except Exception as e:
            print(f"⚠️ Error fetching {series_id}: {e}")
            return []

    def get_gas_price_momentum(self) -> Dict:
        obs = self.fred_series(self.SERIES_IDS['gas_national'], limit=52)
        if not obs or len(obs) < 4:
            return {'current': None, 'momentum': 0, 'trend': 0, 'signal': 0, 'avg_52w': None}

        current = obs[0]['value']
        prices = [o['value'] for o in obs]

        four_week_ago = prices[3]
        momentum = (current - four_week_ago) / four_week_ago if four_week_ago > 0 else 0

        if len(prices) >= 12:
            twelve_week_ago = prices[11]
            trend = (current - twelve_week_ago) / twelve_week_ago if twelve_week_ago > 0 else 0
        else:
            trend = momentum

        # Weight recent 4w momentum more heavily than 12w trend
        signal = momentum * 0.70 + trend * 0.30

        return {
            'current': current,
            'momentum': momentum,
            'trend': trend,
            'avg_52w': np.mean(prices),
            'signal': signal
        }

    def get_wti_history(self, weeks: int = 8) -> List[Dict]:
        return self.fred_series(self.SERIES_IDS['wti_crude'], limit=weeks)

    def get_regional_prices(self) -> Dict:
        regional_data = {}
        prices = []
        weights = []

        for padd, series_id in self.SERIES_IDS.items():
            if not padd.startswith('padd'):
                continue
            obs = self.fred_series(series_id, limit=4)
            if obs:
                price = obs[0]['value']
                regional_data[padd] = {'price': price, 'weight': self.PADD_WEIGHTS[padd]}
                prices.append(price)
                weights.append(self.PADD_WEIGHTS[padd])

        if not prices:
            return {'weighted_avg': None, 'regional_data': {}, 'divergence_signal': 0}

        weighted_avg = np.average(prices, weights=weights)
        divergence = np.std(prices)
        divergence_signal = -divergence * 0.01

        return {
            'weighted_avg': weighted_avg,
            'regional_data': regional_data,
            'divergence': divergence,
            'divergence_signal': divergence_signal,
            'max_price': max(prices),
            'min_price': min(prices),
            'spread': max(prices) - min(prices)
        }

    def get_refinery_utilization(self) -> Dict:
        try:
            obs = self.fred_series(self.SERIES_IDS['refinery_util'], limit=12)
        except Exception:
            obs = []

        if not obs or len(obs) < 2:
            return {
                'current': None, 'avg_12w': None, 'signal': 0,
                'status': 'Data Unavailable', 'trend': 0,
                'note': 'Refinery utilization data unavailable from FRED'
            }

        current = obs[0]['value']
        values = [o['value'] for o in obs]
        avg_12w = np.mean(values)

        if current >= 93:
            signal, status = 0.05, 'Very Tight'
        elif current >= 90:
            signal, status = 0.03, 'Tight'
        elif current >= 85:
            signal, status = 0.0, 'Normal'
        elif current >= 80:
            signal, status = -0.02, 'Loose'
        else:
            signal, status = -0.04, 'Very Loose'

        recent_trend = 0
        if len(obs) >= 4:
            recent_trend = (obs[0]['value'] - obs[3]['value']) / obs[3]['value']
            signal += recent_trend * 0.5

        return {'current': current, 'avg_12w': avg_12w, 'signal': signal,
                'status': status, 'trend': recent_trend}

    def get_inventory_levels(self) -> Dict:
        obs = self.fred_series(self.SERIES_IDS['gas_inventory'], limit=52)
        if not obs or len(obs) < 4:
            return {'current': None, 'avg_52w': None, 'signal': 0, 'status': 'unknown', 'wow_change': 0}

        current = obs[0]['value']
        values = [o['value'] for o in obs]
        avg_52w = np.mean(values)
        std_52w = np.std(values)

        z_score = (current - avg_52w) / std_52w if std_52w > 0 else 0

        # Low inventory = bullish; high inventory = bearish
        signal = -z_score * 0.02

        # WoW draw/build as secondary signal
        if len(obs) >= 2:
            wow = (obs[0]['value'] - obs[1]['value']) / obs[1]['value']
            signal += -wow * 0.5  # draw (negative wow) = bullish

        if z_score < -1.5:
            status = 'Very Low'
        elif z_score < -0.5:
            status = 'Low'
        elif z_score < 0.5:
            status = 'Normal'
        elif z_score < 1.5:
            status = 'High'
        else:
            status = 'Very High'

        wow_change = (obs[0]['value'] - obs[1]['value']) / obs[1]['value'] if len(obs) >= 2 else 0

        return {
            'current': current, 'avg_52w': avg_52w, 'z_score': z_score,
            'signal': signal, 'status': status, 'wow_change': wow_change
        }


# ============================================
# WTI LAG MODEL
# ============================================

class WTILagModel:
    """Adaptive lag correlation model for WTI crude oil as a leading indicator."""

    def __init__(self, data_hub: ProGasDataHub):
        self.data_hub = data_hub
        self.optimal_lag = 1

    def calculate_optimal_lag(self, gas_prices: List[float],
                              wti_prices: List[float], max_lag: int = 4) -> int:
        if len(gas_prices) < max_lag + 2 or len(wti_prices) < max_lag + 2:
            return 1

        best_lag, best_corr = 1, -1

        for lag in range(1, max_lag + 1):
            if lag >= len(wti_prices):
                continue
            gas_aligned = gas_prices[:-lag]
            wti_lagged = wti_prices[lag:]
            min_len = min(len(gas_aligned), len(wti_lagged))
            if min_len < 3:
                continue
            corr = np.corrcoef(gas_aligned[:min_len], wti_lagged[:min_len])[0, 1]
            if corr > best_corr:
                best_corr = corr
                best_lag = lag

        self.optimal_lag = best_lag
        return best_lag

    def get_wti_signal(self) -> Dict:
        wti_history = self.data_hub.get_wti_history(weeks=8)

        if not wti_history or len(wti_history) < 3:
            return {'current_wti': None, 'lagged_wti': None,
                    'wti_change': 0, 'signal': 0, 'optimal_lag': self.optimal_lag}

        current_wti = wti_history[0]['value']
        lag_idx = min(self.optimal_lag, len(wti_history) - 1)
        lagged_wti = wti_history[lag_idx]['value']
        wti_change = (current_wti - lagged_wti) / lagged_wti if lagged_wti > 0 else 0

        # Amplify bearish signal when WTI is falling
        signal = wti_change * 1.0 if wti_change < -0.005 else wti_change * 0.6

        wti_trend = 0
        if len(wti_history) >= 4:
            recent_prices = [o['value'] for o in wti_history[:4]]
            wti_trend = (recent_prices[0] - recent_prices[-1]) / recent_prices[-1]

        return {
            'current_wti': current_wti, 'lagged_wti': lagged_wti,
            'wti_change': wti_change, 'wti_trend': wti_trend,
            'signal': signal, 'optimal_lag': self.optimal_lag
        }


# ============================================
# SEASONAL ADJUSTMENT
# ============================================

class SeasonalAdjustment:
    """Seasonal adjustment multipliers for gas price predictions."""

    @staticmethod
    def get_seasonal_multiplier(date: Optional[datetime] = None) -> Dict:
        if date is None:
            date = datetime.now()

        month, day = date.month, date.day
        multiplier = 1.0
        factors = []

        if month in [5, 6, 7, 8, 9]:
            if month == 5 and day < 25:
                sf = 1.03
            elif month in [6, 7]:
                sf = 1.12
            elif month == 8:
                sf = 1.08
            elif month == 9:
                sf = 1.02
            else:
                sf = 1.05
            multiplier *= sf
            factors.append(('Summer Driving', sf))
        elif month in [12, 1, 2]:
            multiplier *= 1.05
            factors.append(('Winter Heating', 1.05))
        elif month in [3, 4]:
            multiplier *= 1.02
            factors.append(('Spring Maintenance', 1.02))
        else:
            multiplier *= 0.98
            factors.append(('Fall Transition', 0.98))

        # Holiday adjustments
        if month == 11 and 22 <= day <= 28:
            multiplier *= 1.06; factors.append(('Thanksgiving Travel', 1.06))
        if (month == 12 and day >= 20) or (month == 1 and day <= 2):
            multiplier *= 1.04; factors.append(('Holiday Travel', 1.04))
        if month == 5 and day >= 25:
            multiplier *= 1.08; factors.append(('Memorial Day', 1.08))
        if month == 9 and day <= 7:
            multiplier *= 1.06; factors.append(('Labor Day', 1.06))
        if month == 7 and 1 <= day <= 7:
            multiplier *= 1.07; factors.append(('Independence Day', 1.07))

        return {
            'multiplier': multiplier,
            'signal': multiplier - 1.0,
            'factors': factors,
            'month': month,
            'is_peak_season': month in [6, 7, 8]
        }


# ============================================
# PRO GAS ALGORITHM
# ============================================

class ProGasAlgo:
    """
    Professional Gas Price Prediction Algorithm v1.2.0

    Multi-factor model for predicting gas price movements in Kalshi
    prediction markets using FRED economic data.

    Changelog v1.3.0:
    - Filtered out COVID-era data (2020-2021) to focus on 2022–present regime
    - Raised YES minimum edge threshold to 0.20 (eliminates low-conviction losers)
    - Rebalanced signal weights: more gas_momentum/inventory, less WTI/seasonal
    - Added momentum conflict filter (4w vs 12w disagreement reduces conviction)
    - Added WTI bearish penalty for YES bets
    - Added bearish signal flip: if combined_signal < -0.02, flip YES to NO
    - Reduced Kelly fraction to 0.25 (quarter Kelly) to control drawdown
    - Fixed _simple_edge to use fair - price (not ratio)
    """

    KELLY_FRACTION = 0.25   # Quarter Kelly — controls variance/drawdown
    KELLY_MAX_BET  = 0.10   # Cap individual bet at 10% of bankroll

    def __init__(self, fred_api_key: Optional[str] = None):
        self.data_hub = ProGasDataHub(fred_api_key)
        self.wti_model = WTILagModel(self.data_hub)
        self.seasonal = SeasonalAdjustment()
        self.last_refresh = None
        self.cached_signals = None
        self._last_combined_signal = 0.0

    def refresh_data(self, force: bool = False) -> Dict:
        """Refresh all gas market data. Cached for 1 hour."""
        if not force and self.last_refresh:
            if (datetime.now() - self.last_refresh).total_seconds() < 3600:
                return self.cached_signals

        print("📊 Refreshing Pro Gas data...")

        gas_momentum = self.data_hub.get_gas_price_momentum()
        wti          = self.wti_model.get_wti_signal()
        regional     = self.data_hub.get_regional_prices()
        seasonal     = self.seasonal.get_seasonal_multiplier()
        refinery     = self.data_hub.get_refinery_utilization()
        inventory    = self.data_hub.get_inventory_levels()

        signals = {
            'gas_momentum': gas_momentum,
            'wti':          wti,
            'regional':     regional,
            'seasonal':     seasonal,
            'refinery':     refinery,
            'inventory':    inventory,
            'timestamp':    datetime.now()
        }

        self.last_refresh = datetime.now()
        self.cached_signals = signals

        print("  " + "=" * 70)
        if gas_momentum['current']:
            print(f"  💰 Gas Price: ${gas_momentum['current']:.3f}/gal")
            print(f"     • 4-week momentum: {gas_momentum['momentum']:+.1%}")
            print(f"     • 12-week trend:   {gas_momentum['trend']:+.1%}")
        if wti['current_wti']:
            print(f"  🛢️  WTI Crude: ${wti['current_wti']:.2f}/bbl  (Δ {wti['wti_change']:+.1%})")
        if refinery['current']:
            print(f"  🏭 Refinery Util: {refinery['current']:.1f}% ({refinery['status']})")
        else:
            print(f"  🏭 Refinery Util: {refinery['status']}")
        if inventory['current']:
            print(f"  📦 Inventory: {inventory['current']:.0f} Mbbls ({inventory['status']})")
        print(f"  📅 Seasonal: {seasonal['multiplier']:.3f}x")
        print("  " + "=" * 70)

        return signals

    def edge(self, title: str, price: float,
             base_gas_price: float = 3.25,
             signal_weights: Optional[Dict] = None) -> float:
        """
        Calculate edge for a gas prediction market.

        Args:
            title: Market title/question
            price: Current market price (0–1)
            base_gas_price: Fallback gas price if FRED unavailable
            signal_weights: Optional override for signal weights

        Returns:
            Edge as decimal. Positive = underpriced (bet YES),
            Negative = overpriced (bet NO), 0 = skip.
        """
        if 'GAS' not in title.upper() and 'GASOLINE' not in title.upper():
            return 0.0

        signals = self.refresh_data()
        if not signals:
            return self._simple_edge(title, price, base_gas_price)

        # ── Signal weights (rebalanced v1.2.0) ───────────────────────────────
        if signal_weights is None:
            signal_weights = {
                'gas_momentum': 0.35,  # ↑ most reliable real-time signal
                'wti':          0.25,  # ↓ reduce; bearish WTI was being ignored
                'inventory':    0.20,  # ↑ good contrarian signal
                'refinery':     0.10,  # ↓ often unavailable
                'regional':     0.05,
                'seasonal':     0.05,  # ↓ was over-inflating YES bets
            }

        # ── Extract raw signals ───────────────────────────────────────────────
        gas_sig      = signals['gas_momentum']['signal']
        wti_sig      = signals['wti']['signal']
        regional_sig = signals['regional']['divergence_signal']
        seasonal_sig = signals['seasonal']['signal']
        refinery_sig = signals['refinery']['signal']
        inventory_sig = signals['inventory']['signal']

        combined_signal = (
            gas_sig      * signal_weights['gas_momentum'] +
            wti_sig      * signal_weights['wti'] +
            refinery_sig * signal_weights['refinery'] +
            inventory_sig * signal_weights['inventory'] +
            regional_sig * signal_weights['regional'] +
            seasonal_sig * signal_weights['seasonal']
        )

        self._last_combined_signal = combined_signal

        # ── Fair value & raw edge ─────────────────────────────────────────────
        is_bullish = self._parse_market_direction(title)
        fair_value = 0.50 + combined_signal if is_bullish else 0.50 - combined_signal
        fair_value = max(0.10, min(0.90, fair_value))

        edge = max(-0.50, min(0.50, fair_value - price))

        # ── Backtest-derived filters ──────────────────────────────────────────

        # Filter 1: YES minimum threshold — low-conviction YES bets lose consistently
        if 0 < edge < 0.20:
            return 0.0

        # Filter 2: Bearish signal override — trust combined signal over fair_value nudge
        if edge > 0 and combined_signal < -0.02:
            edge = -abs(edge)

        # Filter 3: Momentum conflict — 4w and 12w disagree → reduce conviction 40%
        momentum_4w = signals['gas_momentum'].get('momentum', 0)
        trend_12w   = signals['gas_momentum'].get('trend', 0)
        if momentum_4w != 0 and trend_12w != 0:
            if (momentum_4w > 0) != (trend_12w > 0):
                edge *= 0.60
                if 0 < edge < 0.20:
                    return 0.0

        # Filter 4: WTI bearish penalty — falling WTI cuts YES conviction by 50%
        wti_change = signals['wti'].get('wti_change', 0)
        if wti_change < -0.005 and edge > 0:
            edge *= 0.50
            if edge < 0.20:
                return 0.0

        return edge

    def _simple_edge(self, title: str, price: float, gas_price: float) -> float:
        """Fallback simple model when FRED data is unavailable."""
        fair_base   = 0.48
        sensitivity = 0.12
        ref_price   = 3.0
        fair = max(0.10, min(0.90, fair_base + (gas_price - ref_price) * sensitivity))
        return fair - price

    def _parse_market_direction(self, title: str) -> bool:
        """
        Determine if market is bullish (YES = price goes up) or bearish.
        Handles negation: 'will NOT exceed', 'won't rise', etc.
        Returns True for bullish, False for bearish.
        """
        import re
        t = title.upper()

        bullish_kws = ['ABOVE', 'OVER', 'EXCEED', 'RISE', 'INCREASE', 'HIGHER', 'UP']
        bearish_kws = ['BELOW', 'UNDER', 'DROP', 'FALL', 'DECREASE', 'LOWER', 'DOWN']

        def is_negated(keyword: str) -> bool:
            pattern = re.compile(
                r'\b(?:NOT|WON\'T|WILL\s+NOT|FAIL\s+TO)\b\s+(?:\w+\s+){0,2}' + keyword
            )
            return bool(pattern.search(t))

        for kw in bullish_kws:
            if kw in t:
                return not is_negated(kw)

        for kw in bearish_kws:
            if kw in t:
                return is_negated(kw)

        return True  # Default: treat as bullish

    def get_diagnostics(self) -> Dict:
        """Return detailed diagnostics for debugging."""
        if not self.cached_signals:
            self.refresh_data()
        return {
            'signals':          self.cached_signals,
            'last_refresh':     self.last_refresh,
            'wti_optimal_lag':  self.wti_model.optimal_lag,
            'last_combined_signal': self._last_combined_signal,
            'cache_age_seconds': (
                (datetime.now() - self.last_refresh).total_seconds()
                if self.last_refresh else None
            )
        }


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("=" * 80)
    print("KALSHI PRO GAS ALGORITHM v1.2.0 — Gas Price Prediction System")
    print("=" * 80)

    try:
        algo = ProGasAlgo()
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\nSet your FRED API key:")
        print("  export FRED_API_KEY='your_api_key_here'")
        exit(1)

    signals = algo.refresh_data()

    example_title = "Will national gas prices exceed $3.50 by March 1?"
    example_price = 0.45
    edge = algo.edge(example_title, example_price)

    print(f"\nMarket: {example_title}")
    print(f"Price:  {example_price:.2f}")
    print(f"Edge:   {edge:+.2%}")

    if edge > 0:
        print("✅ Positive edge — market underpriced (bet YES)")
    elif edge < 0:
        print("❌ Negative edge — market overpriced (bet NO)")
    else:
        print("⚖️  No edge — skip")

    print("\n✅ Algorithm ready.")
    print("  from kalshi_pro_gas import ProGasAlgo")
    print("  algo = ProGasAlgo()")
    print("  edge = algo.edge('Market title', 0.45)")
