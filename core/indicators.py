"""
Technical Indicators Module

Provides calculation of all technical indicators used by the trading strategy:
- Trend: EMA, SMA, ADX
- Momentum: RSI, MACD
- Volatility: ATR, Bollinger Bands
- Volume: Volume MA, OBV

All functions operate on pandas DataFrames and return results as new columns.
"""

import logging
from typing import Optional, Tuple
import pandas as pd
import numpy as np

from config.settings import IndicatorConfig


logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Calculator for all technical indicators.
    
    Design principles:
    - All calculations are vectorized for performance
    - Functions add columns to existing DataFrames
    - Invalid/incomplete data handled gracefully
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        """
        Initialize with configuration.
        
        Args:
            config: Indicator configuration settings
        """
        self.config = config or IndicatorConfig()
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all indicators and add them to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all indicator columns added
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Trend indicators
        df = self.add_ema(df, self.config.ema_fast, 'ema_fast')
        df = self.add_ema(df, self.config.ema_medium, 'ema_medium')
        df = self.add_ema(df, self.config.ema_slow, 'ema_slow')
        df = self.add_sma(df, self.config.ema_slow, 'sma_slow')
        df = self.add_adx(df, self.config.adx_period)
        
        # Momentum indicators
        df = self.add_rsi(df, self.config.rsi_period)
        df = self.add_macd(
            df,
            self.config.macd_fast,
            self.config.macd_slow,
            self.config.macd_signal
        )
        
        # Volatility indicators
        df = self.add_atr(df, self.config.atr_period)
        df = self.add_bollinger_bands(df, self.config.bb_period, self.config.bb_std_dev)
        
        # Add normalized ATR for volatility regime detection
        df = self.add_normalized_atr(df, self.config.atr_volatility_lookback)
        
        # Volume indicators
        df = self.add_volume_ma(df, self.config.volume_ma_period)
        df = self.add_obv(df)
        df = self.add_obv_ma(df, self.config.obv_ma_period)
        
        # Derived signals
        df = self.add_trend_strength(df)
        df = self.add_momentum_score(df)
        
        logger.debug(f"Calculated all indicators for {len(df)} rows")
        return df
    
    # ==================== TREND INDICATORS ====================
    
    def add_ema(self, df: pd.DataFrame, period: int, column_name: str) -> pd.DataFrame:
        """
        Add Exponential Moving Average.
        
        Formula: EMA_t = (Price_t × k) + (EMA_{t-1} × (1 - k))
        where k = 2 / (period + 1)
        """
        df[column_name] = df['close'].ewm(span=period, adjust=False).mean()
        return df
    
    def add_sma(self, df: pd.DataFrame, period: int, column_name: str) -> pd.DataFrame:
        """Add Simple Moving Average."""
        df[column_name] = df['close'].rolling(window=period).mean()
        return df
    
    def add_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Add Average Directional Index (ADX) with +DI and -DI.
        
        ADX measures trend strength (not direction):
        - ADX < 20: No clear trend
        - ADX 20-40: Trending
        - ADX > 40: Strong trend
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smooth with EMA
        atr = true_range.ewm(span=period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / atr
        
        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        df['adx'] = adx
        
        return df
    
    # ==================== MOMENTUM INDICATORS ====================
    
    def add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Add Relative Strength Index.
        
        Formula: RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss
        """
        delta = df['close'].diff()
        
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        # Use exponential moving average for smoothing
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        df['rsi'] = rsi
        
        # Add RSI direction
        df['rsi_prev'] = df['rsi'].shift(1)
        df['rsi_rising'] = df['rsi'] > df['rsi_prev']
        
        return df
    
    def add_macd(
        self,
        df: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> pd.DataFrame:
        """
        Add MACD (Moving Average Convergence Divergence).
        
        Components:
        - MACD Line = EMA(fast) - EMA(slow)
        - Signal Line = EMA(MACD Line, signal_period)
        - Histogram = MACD Line - Signal Line
        """
        ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()
        
        df['macd_line'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd_line'].ewm(span=signal_period, adjust=False).mean()
        df['macd_histogram'] = df['macd_line'] - df['macd_signal']
        
        # Add histogram direction
        df['macd_hist_prev'] = df['macd_histogram'].shift(1)
        df['macd_hist_increasing'] = df['macd_histogram'] > df['macd_hist_prev']
        
        return df
    
    # ==================== VOLATILITY INDICATORS ====================
    
    def add_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Add Average True Range.
        
        True Range = max(High - Low, |High - Prev Close|, |Low - Prev Close|)
        ATR = EMA(True Range, period)
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = true_range.ewm(span=period, adjust=False).mean()
        
        return df
    
    def add_normalized_atr(self, df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
        """
        Add ATR normalized by its historical average.
        
        Used for volatility regime detection:
        - atr_ratio > 2.0 = High volatility
        - atr_ratio < 0.5 = Low volatility
        """
        if 'atr' not in df.columns:
            df = self.add_atr(df)
        
        df['atr_sma'] = df['atr'].rolling(window=lookback).mean()
        df['atr_ratio'] = df['atr'] / df['atr_sma']
        
        return df
    
    def add_bollinger_bands(
        self,
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0
    ) -> pd.DataFrame:
        """
        Add Bollinger Bands.
        
        - Middle Band = SMA(period)
        - Upper Band = Middle + (std_dev × Standard Deviation)
        - Lower Band = Middle - (std_dev × Standard Deviation)
        """
        df['bb_middle'] = df['close'].rolling(window=period).mean()
        rolling_std = df['close'].rolling(window=period).std()
        
        df['bb_upper'] = df['bb_middle'] + (std_dev * rolling_std)
        df['bb_lower'] = df['bb_middle'] - (std_dev * rolling_std)
        
        # Band width (for volatility regime)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Percent B (price position within bands)
        df['bb_percent_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    # ==================== VOLUME INDICATORS ====================
    
    def add_volume_ma(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add Volume Moving Average and volume ratio."""
        df['volume_ma'] = df['volume'].rolling(window=period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        return df
    
    def add_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add On-Balance Volume.
        
        OBV = Previous OBV + (Volume if close > prev_close, -Volume if close < prev_close)
        """
        close = df['close']
        volume = df['volume']
        
        # Direction based on price change
        direction = np.where(close > close.shift(1), 1, 
                           np.where(close < close.shift(1), -1, 0))
        
        df['obv'] = (volume * direction).cumsum()
        
        return df
    
    def add_obv_ma(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add OBV moving average and trend."""
        if 'obv' not in df.columns:
            df = self.add_obv(df)
        
        df['obv_ma'] = df['obv'].rolling(window=period).mean()
        df['obv_trending_up'] = df['obv'] > df['obv_ma']
        
        return df
    
    # ==================== DERIVED SIGNALS ====================
    
    def add_trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add composite trend strength indicator.
        
        Combines EMA alignment and ADX into a single score.
        """
        # EMA alignment score (-1 to 1)
        ema_bullish = (
            (df['ema_fast'] > df['ema_medium']) & 
            (df['ema_medium'] > df['ema_slow'])
        ).astype(int)
        
        ema_bearish = (
            (df['ema_fast'] < df['ema_medium']) & 
            (df['ema_medium'] < df['ema_slow'])
        ).astype(int)
        
        df['ema_alignment'] = ema_bullish - ema_bearish
        
        # Normalized ADX (0 to 1)
        df['adx_normalized'] = df['adx'] / 100
        
        # Composite trend strength
        df['trend_strength'] = df['ema_alignment'] * df['adx_normalized']
        
        return df
    
    def add_momentum_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add composite momentum score.
        
        Combines RSI and MACD into a single momentum indicator.
        """
        # RSI component (-1 to 1)
        rsi_score = (df['rsi'] - 50) / 50
        
        # MACD histogram direction
        macd_direction = df['macd_hist_increasing'].astype(int) * 2 - 1
        
        # Combine
        df['momentum_score'] = (rsi_score + macd_direction) / 2
        
        return df
    
    def detect_divergence(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        indicator_col: str = 'rsi',
        lookback: int = 14
    ) -> pd.DataFrame:
        """
        Detect bullish and bearish divergences between price and indicator.
        
        Bullish divergence: Price makes lower low, indicator makes higher low
        Bearish divergence: Price makes higher high, indicator makes lower high
        """
        df = df.copy()
        
        # Find local extrema
        df['price_min'] = df[price_col].rolling(window=lookback, center=True).min()
        df['price_max'] = df[price_col].rolling(window=lookback, center=True).max()
        df['ind_min'] = df[indicator_col].rolling(window=lookback, center=True).min()
        df['ind_max'] = df[indicator_col].rolling(window=lookback, center=True).max()
        
        # Bullish divergence (price lower low, indicator higher low)
        price_lower_low = df[price_col] < df['price_min'].shift(lookback)
        ind_higher_low = df[indicator_col] > df['ind_min'].shift(lookback)
        df['bullish_divergence'] = price_lower_low & ind_higher_low
        
        # Bearish divergence (price higher high, indicator lower high)
        price_higher_high = df[price_col] > df['price_max'].shift(lookback)
        ind_lower_high = df[indicator_col] < df['ind_max'].shift(lookback)
        df['bearish_divergence'] = price_higher_high & ind_lower_high
        
        # Clean up temporary columns
        df.drop(columns=['price_min', 'price_max', 'ind_min', 'ind_max'], inplace=True)
        
        return df


def calculate_indicators(df: pd.DataFrame, config: Optional[IndicatorConfig] = None) -> pd.DataFrame:
    """
    Convenience function to calculate all indicators.
    
    Args:
        df: DataFrame with OHLCV data
        config: Optional indicator configuration
        
    Returns:
        DataFrame with all indicators calculated
    """
    calculator = TechnicalIndicators(config)
    return calculator.calculate_all(df)
