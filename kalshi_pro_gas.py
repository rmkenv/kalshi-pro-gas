import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os
import json
from scipy import stats

__version__ = "1.0.0"
__author__ = "RMK Solutions"

# ============================================
# PRO GAS DATA HUB
# ============================================

class ProGasDataHub:
    """
    Enhanced data hub specifically for gas market analysis.
    Fetches and processes FRED data for comprehensive gas price modeling.
    """

    # FRED Series IDs for gas market indicators
    SERIES_IDS = {
        'gas_national': 'GASREGW',           # U.S. Regular All Formulations Retail Gasoline Prices - Weekly
        'wti_crude': 'DCOILWTICO',           # Crude Oil Prices: West Texas Intermediate (WTI) - Daily
        'gas_inventory': 'WGFUPUS2',         # Weekly U.S. Ending Stocks of Finished Motor Gasoline
        'refinery_util': 'WPULEUS3',         # Weekly U.S. Percent Utilization of Refinery Operable Capacity

        # PADD District Gas Prices (Regional) - Weekly prices in $/gallon
        'padd1': 'GASREGCOVECW',             # East Coast (PADD 1)
        'padd2': 'GASREGMWW',                # Midwest (PADD 2)
        'padd3': 'GASREGGCW',                # Gulf Coast (PADD 3) (fixed)
        'padd4': 'GASREGRMW',                # Rocky Mountain (PADD 4)
        'padd5': 'GASREGREFWCW',             # West Coast (PADD 5)
    }

    # PADD District population weights (2020 Census estimates)
    PADD_WEIGHTS = {
        'padd1': 0.18,   # East Coast
        'padd2': 0.21,   # Midwest
        'padd3': 0.29,   # Gulf Coast
        'padd4': 0.05,   # Rocky Mountain
        'padd5': 0.27,   # West Coast
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize with FRED API key.
        
        Args:
            api_key: FRED API key (optional, can use FRED_API_KEY environment variable)
            
        Raises:
            ValueError: If no API key provided
        """
        self.api_key = api_key or os.environ.get('FRED_API_KEY')
        if not self.api_key:
            raise ValueError(
                "FRED API key required. Set FRED_API_KEY environment variable or pass api_key parameter.\n"
                "Get your free API key at: https://fred.stlouisfed.org/docs/api/api_key.html"
            )
        self.cache = {}  # Cache for API responses

    def fred_series(self, series_id: str, limit: int = 52) -> List[Dict]:
        """
        Fetch FRED time series data.

        Args:
            series_id: FRED series identifier
            limit: Number of observations to fetch (default 52 for ~1 year of weekly data)

        Returns:
            List of observations with date and value
        """
        # Check cache first
        cache_key = f"{series_id}_{limit}"
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

            # Cache the results
            self.cache[cache_key] = observations
            return observations

        except Exception as e:
            print(f"⚠️ Error fetching {series_id}: {e}")
            return []

    def get_gas_price_momentum(self) -> Dict:
        """
        Fetch national gas price history and calculate momentum/trend signals.
        
        Returns:
            Dictionary with current price, momentum, trend, and signal
        """
        obs = self.fred_series(self.SERIES_IDS['gas_national'], limit=52)
        if not obs or len(obs) < 4:
            return {'current': None, 'momentum': 0, 'trend': 0, 'signal': 0}

        current = obs[0]['value']
        prices = [o['value'] for o in obs]

        # Calculate momentum (4-week change)
        if len(prices) >= 4:
            four_week_ago = prices[3]
            momentum = (current - four_week_ago) / four_week_ago if four_week_ago > 0 else 0
        else:
            momentum = 0

        # Calculate longer-term trend (12-week)
        if len(prices) >= 12:
            twelve_week_ago = prices[11]
            trend = (current - twelve_week_ago) / twelve_week_ago if twelve_week_ago > 0 else 0
        else:
            trend = momentum

        # Price momentum signal
        signal = momentum * 0.5 + trend * 0.3

        return {
            'current': current,
            'momentum': momentum,
            'trend': trend,
            'avg_52w': np.mean(prices),
            'signal': signal
        }

    def get_wti_history(self, weeks: int = 8) -> List[Dict]:
        """
        Fetch WTI crude oil price history for lag analysis.
        
        Args:
            weeks: Number of weeks of history to fetch
            
        Returns:
            List of WTI price observations
        """
        return self.fred_series(self.SERIES_IDS['wti_crude'], limit=weeks)

    def get_regional_prices(self) -> Dict:
        """
        Fetch PADD district gas prices and calculate regional signals.
        
        Returns:
            Dictionary with regional price data and divergence signals
        """
        regional_data = {}
        prices = []
        weights = []

        for padd, series_id in self.SERIES_IDS.items():
            if not padd.startswith('padd'):
                continue

            obs = self.fred_series(series_id, limit=4)
            if obs:
                price = obs[0]['value']
                regional_data[padd] = {
                    'price': price,
                    'weight': self.PADD_WEIGHTS[padd]
                }
                prices.append(price)
                weights.append(self.PADD_WEIGHTS[padd])

        if not prices:
            return {
                'weighted_avg': None,
                'regional_data': {},
                'divergence_signal': 0
            }

        # Calculate population-weighted average
        weighted_avg = np.average(prices, weights=weights)

        # Calculate regional divergence
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
        """
        Fetch U.S. refinery utilization rate.
        
        Returns:
            Dictionary with utilization data and supply signals
        """
        try:
            obs = self.fred_series(self.SERIES_IDS['refinery_util'], limit=12)
        except:
            obs = []

        if not obs or len(obs) < 2:
            return {
                'current': None,
                'avg_12w': None,
                'signal': 0,
                'status': 'Data Unavailable',
                'trend': 0,
                'note': 'Refinery data requires EIA API (not in FRED)'
            }

        current = obs[0]['value']
        values = [o['value'] for o in obs]
        avg_12w = np.mean(values)

        # Signal calculation based on utilization levels
        if current >= 93:
            signal = 0.05
            status = 'Very Tight'
        elif current >= 90:
            signal = 0.03
            status = 'Tight'
        elif current >= 85:
            signal = 0.0
            status = 'Normal'
        elif current >= 80:
            signal = -0.02
            status = 'Loose'
        else:
            signal = -0.04
            status = 'Very Loose'

        # Add trend component
        recent_trend = 0
        if len(obs) >= 4:
            recent_trend = (obs[0]['value'] - obs[3]['value']) / obs[3]['value']
            signal += recent_trend * 0.5

        return {
            'current': current,
            'avg_12w': avg_12w,
            'signal': signal,
            'status': status,
            'trend': recent_trend
        }

    def get_inventory_levels(self) -> Dict:
        """
        Fetch U.S. gasoline inventory levels.
        
        Returns:
            Dictionary with inventory data and supply/demand signals
        """
        obs = self.fred_series(self.SERIES_IDS['gas_inventory'], limit=52)
        if not obs or len(obs) < 4:
            return {
                'current': None,
                'avg_52w': None,
                'signal': 0,
                'status': 'unknown'
            }

        current = obs[0]['value']
        values = [o['value'] for o in obs]
        avg_52w = np.mean(values)
        std_52w = np.std(values)

        # Z-score: how many standard deviations from average
        if std_52w > 0:
            z_score = (current - avg_52w) / std_52w
        else:
            z_score = 0

        # Signal based on inventory deviation
        signal = -z_score * 0.02

        # Determine status
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

        # Add inventory trend
        if len(obs) >= 2:
            wow_change = (obs[0]['value'] - obs[1]['value']) / obs[1]['value']
        else:
            wow_change = 0

        return {
            'current': current,
            'avg_52w': avg_52w,
            'z_score': z_score,
            'signal': signal,
            'status': status,
            'wow_change': wow_change
        }


# ============================================
# WTI LAG MODEL
# ============================================

class WTILagModel:
    """
    Adaptive lag correlation model for WTI crude oil as a leading indicator.
    """

    def __init__(self, data_hub: ProGasDataHub):
        """
        Initialize WTI lag model.
        
        Args:
            data_hub: ProGasDataHub instance for data access
        """
        self.data_hub = data_hub
        self.optimal_lag = 1  # Default 1 week lag

    def calculate_optimal_lag(self, gas_prices: List[float],
                             wti_prices: List[float],
                             max_lag: int = 4) -> int:
        """
        Determine optimal lag period by analyzing correlation.
        
        Args:
            gas_prices: List of historical gas prices
            wti_prices: List of historical WTI prices
            max_lag: Maximum lag to test (weeks)
            
        Returns:
            Optimal lag in weeks
        """
        if len(gas_prices) < max_lag + 2 or len(wti_prices) < max_lag + 2:
            return 1

        best_lag = 1
        best_corr = -1

        for lag in range(1, max_lag + 1):
            if lag >= len(wti_prices):
                continue

            gas_aligned = gas_prices[:-lag] if lag > 0 else gas_prices
            wti_lagged = wti_prices[lag:]

            min_len = min(len(gas_aligned), len(wti_lagged))
            if min_len < 3:
                continue

            gas_aligned = gas_aligned[:min_len]
            wti_lagged = wti_lagged[:min_len]

            corr = np.corrcoef(gas_aligned, wti_lagged)[0, 1]

            if corr > best_corr:
                best_corr = corr
                best_lag = lag

        self.optimal_lag = best_lag
        return best_lag

    def get_wti_signal(self) -> Dict:
        """
        Calculate WTI-based signal using optimal lag.
        
        Returns:
            Dictionary with WTI prices, changes, and signal
        """
        wti_history = self.data_hub.get_wti_history(weeks=8)

        if not wti_history or len(wti_history) < 3:
            return {
                'current_wti': None,
                'lagged_wti': None,
                'wti_change': 0,
                'signal': 0,
                'optimal_lag': self.optimal_lag
            }

        current_wti = wti_history[0]['value']

        lag_idx = min(self.optimal_lag, len(wti_history) - 1)
        lagged_wti = wti_history[lag_idx]['value']

        wti_change = (current_wti - lagged_wti) / lagged_wti if lagged_wti > 0 else 0

        # WTI signal: positive change = bullish for gas prices
        signal = wti_change * 0.6

        # Get trend
        if len(wti_history) >= 4:
            recent_prices = [o['value'] for o in wti_history[:4]]
            wti_trend = (recent_prices[0] - recent_prices[-1]) / recent_prices[-1]
        else:
            wti_trend = 0

        return {
            'current_wti': current_wti,
            'lagged_wti': lagged_wti,
            'wti_change': wti_change,
            'wti_trend': wti_trend,
            'signal': signal,
            'optimal_lag': self.optimal_lag
        }


# ============================================
# SEASONAL ADJUSTMENT
# ============================================

class SeasonalAdjustment:
    """
    Seasonal adjustment multipliers for gas price predictions.
    """

    @staticmethod
    def get_seasonal_multiplier(date: Optional[datetime] = None) -> Dict:
        """
        Calculate seasonal adjustment multiplier based on date.
        
        Args:
            date: Target date (defaults to current date)
            
        Returns:
            Dictionary with multiplier, signal, and active factors
        """
        if date is None:
            date = datetime.now()

        month = date.month
        day = date.day

        multiplier = 1.0
        factors = []

        # Summer driving season (May-September)
        if month in [5, 6, 7, 8, 9]:
            if month == 5 and day < 25:
                summer_factor = 1.03
            elif month in [6, 7]:
                summer_factor = 1.12
            elif month == 8:
                summer_factor = 1.08
            elif month == 9:
                summer_factor = 1.02  # Late summer / early fall, still elevated
            else:
                summer_factor = 1.05

            multiplier *= summer_factor
            factors.append(('Summer Driving', summer_factor))

        # Winter heating season
        elif month in [12, 1, 2]:
            winter_factor = 1.05
            multiplier *= winter_factor
            factors.append(('Winter Heating', winter_factor))

        # Spring (refinery maintenance)
        elif month in [3, 4]:
            spring_factor = 1.02
            multiplier *= spring_factor
            factors.append(('Spring Maintenance', spring_factor))

        # Fall transition
        else:
            fall_factor = 0.98
            multiplier *= fall_factor
            factors.append(('Fall Transition', fall_factor))

        # Holiday travel adjustments
        if month == 11 and 22 <= day <= 28:
            holiday_factor = 1.06
            multiplier *= holiday_factor
            factors.append(('Thanksgiving Travel', holiday_factor))

        if (month == 12 and day >= 20) or (month == 1 and day <= 2):
            holiday_factor = 1.04
            multiplier *= holiday_factor
            factors.append(('Holiday Travel', holiday_factor))

        if month == 5 and day >= 25:
            holiday_factor = 1.08
            multiplier *= holiday_factor
            factors.append(('Memorial Day', holiday_factor))

        if month == 9 and day <= 7:
            holiday_factor = 1.06
            multiplier *= holiday_factor
            factors.append(('Labor Day', holiday_factor))

        if month == 7 and 1 <= day <= 7:
            holiday_factor = 1.07
            multiplier *= holiday_factor
            factors.append(('Independence Day', holiday_factor))

        signal = multiplier - 1.0

        return {
            'multiplier': multiplier,
            'signal': signal,
            'factors': factors,
            'month': month,
            'is_peak_season': month in [6, 7, 8]
        }


# ============================================
# PRO GAS ALGORITHM
# ============================================

class ProGasAlgo:
    """
    Professional Gas Price Prediction Algorithm.
    
    Comprehensive multi-factor model for predicting gas price movements in
    Kalshi prediction markets using FRED economic data.
    """

    def __init__(self, fred_api_key: Optional[str] = None):
        """
        Initialize ProGasAlgo with data sources.
        
        Args:
            fred_api_key: FRED API key (optional, can use environment variable)
        """
        self.data_hub = ProGasDataHub(fred_api_key)
        self.wti_model = WTILagModel(self.data_hub)
        self.seasonal = SeasonalAdjustment()
        self.last_refresh = None
        self.cached_signals = None

    def refresh_data(self, force: bool = False) -> Dict:
        """
        Refresh all gas market data and calculate signals.
        
        Data is cached for 1 hour by default.
        
        Args:
            force: Force refresh even if cache is valid
            
        Returns:
            Dictionary containing all calculated signals
        """
        # Cache for 1 hour
        if not force and self.last_refresh:
            time_since_refresh = datetime.now() - self.last_refresh
            if time_since_refresh.total_seconds() < 3600:
                return self.cached_signals

        print("📊 Refreshing Pro Gas data...")

        gas_momentum = self.data_hub.get_gas_price_momentum()
        wti = self.wti_model.get_wti_signal()
        regional = self.data_hub.get_regional_prices()
        seasonal = self.seasonal.get_seasonal_multiplier()
        refinery = self.data_hub.get_refinery_utilization()
        inventory = self.data_hub.get_inventory_levels()

        signals = {
            'gas_momentum': gas_momentum,
            'wti': wti,
            'regional': regional,
            'seasonal': seasonal,
            'refinery': refinery,
            'inventory': inventory,
            'timestamp': datetime.now()
        }

        self.last_refresh = datetime.now()
        self.cached_signals = signals

        # Print summary
        print("  " + "=" * 70)
        if gas_momentum['current']:
            print(f"  💰 Gas Price: ${gas_momentum['current']:.3f}/gal")
            print(f"     • 4-week momentum: {gas_momentum['momentum']:+.1%}")
            print(f"     • 12-week trend: {gas_momentum['trend']:+.1%}")

        if wti['current_wti']:
            print(f"  🛢️  WTI Crude: ${wti['current_wti']:.2f}/bbl")
            print(f"     • Change: {wti['wti_change']:+.1%}")

        if refinery['current']:
            print(f"  🏭 Refinery Utilization: {refinery['current']:.1f}% ({refinery['status']})")
        else:
            print(f"  🏭 Refinery Utilization: {refinery['status']}")

        if inventory['current']:
            print(f"  📦 Gasoline Inventory: {inventory['current']:.0f} ({inventory['status']})")

        print(f"  📅 Seasonal: {seasonal['multiplier']:.2f}x multiplier")
        print("  " + "=" * 70)

        return signals

    def edge(self, title: str, price: float,
             base_gas_price: float = 3.25,
             signal_weights: Optional[Dict] = None) -> float:
        """
        Calculate edge for gas markets using comprehensive signal integration.
        
        Args:
            title: Market title/question
            price: Current market price (0-1 range)
            base_gas_price: Base gas price assumption ($/gallon)
            signal_weights: Optional custom signal weights dictionary
            
        Returns:
            Edge as decimal (-1 to 1), where positive = underpriced, negative = overpriced
            
        Example:
            edge = algo.edge("Will national gas prices exceed $3.50?", 0.45)
        """
        # Only analyze gas markets
        if 'GAS' not in title.upper() and 'GASOLINE' not in title.upper():
            return 0

        # Ensure fresh data
        signals = self.refresh_data()
        if not signals:
            return self._simple_edge(title, price, base_gas_price)

        # Signal weights — rebalanced: less WTI dominance, more momentum+inventory
        # Rationale: NO bets win 70% vs YES 43%, WTI bearish (-0.8%) was being
        # overridden by seasonal/momentum. Boost gas_momentum and inventory weight.
        if signal_weights is None:
            signal_weights = {
                'gas_momentum': 0.35,   # was 0.20 — most reliable available signal
                'wti': 0.25,            # was 0.35 — reduce; bearish WTI was ignored
                'refinery': 0.10,       # was 0.15 — often unavailable, reduce weight
                'inventory': 0.20,      # was 0.15 — good contrarian signal
                'regional': 0.05,       # unchanged
                'seasonal': 0.05,       # was 0.10 — seasonal was over-inflating YES
            }

        # Extract signals
        gas_momentum_signal = signals['gas_momentum']['signal']
        wti_signal = signals['wti']['signal']
        regional_signal = signals['regional']['divergence_signal']
        seasonal_signal = signals['seasonal']['signal']
        refinery_signal = signals['refinery']['signal']
        inventory_signal = signals['inventory']['signal']

        # Weighted combination
        combined_signal = (
            gas_momentum_signal * signal_weights['gas_momentum'] +
            wti_signal * signal_weights['wti'] +
            refinery_signal * signal_weights['refinery'] +
            inventory_signal * signal_weights['inventory'] +
            regional_signal * signal_weights['regional'] +
            seasonal_signal * signal_weights['seasonal']
        )

        # Determine market direction
        is_bullish_market = self._parse_market_direction(title)

        # Store combined_signal on self for use in post-edge filters
        self._last_combined_signal = combined_signal

        # Calculate fair value
        fair_value = 0.50 + combined_signal

        if not is_bullish_market:
            fair_value = 0.50 - combined_signal

        # Clamp fair value
        fair_value = max(0.10, min(0.90, fair_value))

        # Calculate edge (standard prediction market convention: fair_value - price)
        edge = fair_value - price

        # Cap edge
        edge = max(-0.50, min(0.50, edge))

        # ── Backtest-derived filters ──────────────────────────────────────────
        # YES bets with edge < 0.15 lose consistently (43% WR on YES overall).
        # NO bets are profitable even at lower edges (70% WR). Skip weak YES.
        if edge > 0 and edge < 0.15:
            return 0.0  # Not enough conviction to bet YES

        # If combined signal is bearish (WTI down, momentum negative) but
        # fair_value still nudges YES, trust the bearish signal and flip to NO.
        if edge > 0 and combined_signal < -0.02:
            edge = -abs(edge)  # Flip to NO signal

        return edge

    def _simple_edge(self, title: str, price: float, gas_price: float) -> float:
        """
        Fallback to simple model if Pro data unavailable.
        
        Args:
            title: Market title
            price: Current market price
            gas_price: Current gas price
            
        Returns:
            Simple edge calculation
        """
        fair_base = 0.48
        sensitivity = 0.12
        ref_price = 3.0

        fair = fair_base + (gas_price - ref_price) * sensitivity
        fair = max(0.10, min(0.90, fair))
        return fair - price  # consistent with main edge() convention

    def _parse_market_direction(self, title: str) -> bool:
        """
        Determine if market is bullish or bearish from title.
        Handles negation (e.g. "NOT exceed", "will NOT rise").
        
        Args:
            title: Market title/question
            
        Returns:
            True if bullish market, False if bearish
        """
        import re
        t = title.upper()

        bullish_keywords = ['ABOVE', 'OVER', 'EXCEED', 'RISE', 'INCREASE', 'HIGHER', 'UP']
        bearish_keywords = ['BELOW', 'UNDER', 'DROP', 'FALL', 'DECREASE', 'LOWER', 'DOWN']

        negation_pattern = re.compile(r'\bNOT\b|\bWON\'T\b|\bWILL NOT\b|\bFAIL TO\b')

        def is_negated(keyword: str) -> bool:
            """Check if a keyword is preceded by a negation within 3 words."""
            pattern = re.compile(
                r'\b(?:NOT|WON\'T|WILL\s+NOT|FAIL\s+TO)\b\s+(?:\w+\s+){0,2}' + keyword
            )
            return bool(pattern.search(t))

        for kw in bullish_keywords:
            if kw in t:
                return False if is_negated(kw) else True

        for kw in bearish_keywords:
            if kw in t:
                return True if is_negated(kw) else False

        return True

    def get_diagnostics(self) -> Dict:
        """
        Get detailed diagnostics including all signals and cache status.
        
        Returns:
            Dictionary with diagnostic information
        """
        if not self.cached_signals:
            self.refresh_data()

        return {
            'signals': self.cached_signals,
            'last_refresh': self.last_refresh,
            'wti_optimal_lag': self.wti_model.optimal_lag,
            'cache_age_seconds': (datetime.now() - self.last_refresh).total_seconds() if self.last_refresh else None
        }


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("=" * 80)
    print("KALSHI PRO GAS ALGORITHM - Gas Price Prediction System")
    print("=" * 80)
    
    # Initialize algorithm
    print("\n🎛️ INITIALIZING...")
    try:
        algo = ProGasAlgo()
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease set your FRED API key:")
        print("  export FRED_API_KEY='your_api_key_here'")
        exit(1)
    
    # Refresh data
    print("\n📊 REFRESHING DATA...")
    signals = algo.refresh_data()
    
    # Example edge calculation
    print("\n💰 EXAMPLE EDGE CALCULATION:")
    example_title = "Will national gas prices exceed $3.50 by March 1?"
    example_price = 0.45
    
    edge = algo.edge(example_title, example_price)
    
    print(f"\nMarket: {example_title}")
    print(f"Current Price: ${example_price:.2f}")
    print(f"Calculated Edge: {edge:+.2%}")
    
    if edge > 0:
        print("✅ Positive edge - market appears underpriced")
    elif edge < 0:
        print("❌ Negative edge - market appears overpriced")
    else:
        print("⚖️ Fair value - no clear edge")
    
    print("\n✅ Algorithm initialized successfully!")
    print("\nUsage:")
    print("  from kalshi_pro_gas import ProGasAlgo")
    print("  algo = ProGasAlgo()")
    print("  edge = algo.edge('Market title', 0.45)")
