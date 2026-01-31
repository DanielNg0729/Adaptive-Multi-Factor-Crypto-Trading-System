"""
Strategy Logic Module

Implements the multi-factor confirmation trading strategy:
- Entry signal generation with regime awareness
- Exit signal generation with volatility-based stops
- Signal validation and freshness checks

This module contains the BRAIN of the trading system.
"""

import logging
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import pandas as pd
import numpy as np

from config.settings import (
    TradingConfig,
    EntryConfig,
    ExitConfig,
    IndicatorConfig,
    MarketRegime,
    OrderSide,
    PositionSide
)
from core.regime import RegimeDetector, RegimeAnalysis


logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Type of trading signal."""
    LONG_ENTRY = "long_entry"
    SHORT_ENTRY = "short_entry"
    LONG_EXIT = "long_exit"
    SHORT_EXIT = "short_exit"
    NO_SIGNAL = "no_signal"


@dataclass
class TradingSignal:
    """Complete trading signal with all necessary information."""
    signal_type: SignalType
    timestamp: datetime
    price: float
    confidence: float  # 0 to 1
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size_modifier: float = 1.0  # Reduce for uncertain conditions
    reasons: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []
        if self.warnings is None:
            self.warnings = []


class SignalGenerator:
    """
    Generates trading signals based on multi-factor confirmation.
    
    Key principles:
    1. Regime awareness - adapt to market conditions
    2. Multi-factor confirmation - require agreement across indicator types
    3. Conservative defaults - prefer missing trades to bad entries
    4. Clear exit rules - every entry has defined stop and target
    """
    
    def __init__(self, config: Optional[TradingConfig] = None):
        """
        Initialize signal generator.
        
        Args:
            config: Complete trading configuration
        """
        self.config = config or TradingConfig()
        self.entry_config = self.config.entry
        self.exit_config = self.config.exit
        self.indicator_config = self.config.indicators
        
        self.regime_detector = RegimeDetector(self.indicator_config)
        
        # Track signal history for freshness checks
        self._last_signal_bar: Dict[str, int] = {}
        self._signal_prices: Dict[str, float] = {}
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        current_position: Optional[PositionSide] = None
    ) -> TradingSignal:
        """
        Generate trading signal based on current market data.
        
        Args:
            df: DataFrame with OHLCV data and calculated indicators
            current_position: Current position (None, LONG, or SHORT)
            
        Returns:
            TradingSignal with entry/exit information
        """
        if len(df) < self.config.data.min_historical_bars:
            logger.warning("Insufficient data for signal generation")
            return TradingSignal(
                signal_type=SignalType.NO_SIGNAL,
                timestamp=datetime.now(),
                price=0,
                confidence=0,
                reasons=["Insufficient historical data"]
            )
        
        latest = df.iloc[-1]
        timestamp = latest['timestamp'] if 'timestamp' in latest else datetime.now()
        price = latest['close']
        
        # First, detect market regime
        regime_analysis = self.regime_detector.detect(df)
        
        # Check if it's safe to trade
        is_safe, safety_reason = self.regime_detector.is_safe_to_trade(regime_analysis)
        
        # If we have a position, always check exit signals first
        if current_position is not None:
            exit_signal = self._check_exit_signals(
                df, current_position, regime_analysis
            )
            if exit_signal.signal_type != SignalType.NO_SIGNAL:
                return exit_signal
        
        # If not safe to trade, don't generate entry signals
        if not is_safe:
            return TradingSignal(
                signal_type=SignalType.NO_SIGNAL,
                timestamp=timestamp,
                price=price,
                confidence=0,
                reasons=[safety_reason]
            )
        
        # Generate entry signals based on regime
        if regime_analysis.current_regime == MarketRegime.TRENDING_UP:
            return self._check_long_entry(df, regime_analysis)
        
        elif regime_analysis.current_regime == MarketRegime.TRENDING_DOWN:
            return self._check_short_entry(df, regime_analysis)
        
        elif regime_analysis.current_regime == MarketRegime.HIGH_VOLATILITY:
            # In high volatility, we might still trade but with reduced size
            signal = self._check_trend_following_entry(df, regime_analysis)
            if signal.signal_type != SignalType.NO_SIGNAL:
                signal.position_size_modifier = 0.5
                signal.warnings.append("High volatility - position size reduced")
            return signal
        
        # Ranging market - mostly stay out
        return TradingSignal(
            signal_type=SignalType.NO_SIGNAL,
            timestamp=timestamp,
            price=price,
            confidence=0,
            reasons=["Ranging market - no trend-following setup"]
        )
    
    def _check_long_entry(
        self,
        df: pd.DataFrame,
        regime: RegimeAnalysis
    ) -> TradingSignal:
        """
        Check for long entry conditions.
        
        Required confirmations:
        1. Trend: ADX > threshold, EMAs aligned bullish
        2. Momentum: RSI or MACD confirmation
        3. Volume: Above average
        4. Not overextended
        """
        latest = df.iloc[-1]
        recent = df.tail(5)
        
        timestamp = latest['timestamp'] if 'timestamp' in latest else datetime.now()
        price = latest['close']
        atr = latest['atr']
        
        reasons = []
        warnings = []
        
        # ==================== TREND CONFIRMATION ====================
        
        trend_confirmed = False
        
        # ADX check
        adx = latest['adx']
        if adx < self.entry_config.min_adx_for_entry:
            return TradingSignal(
                signal_type=SignalType.NO_SIGNAL,
                timestamp=timestamp,
                price=price,
                confidence=0,
                reasons=[f"ADX {adx:.1f} below threshold {self.entry_config.min_adx_for_entry}"]
            )
        reasons.append(f"ADX: {adx:.1f}")
        
        # EMA alignment
        if self.entry_config.require_ema_alignment:
            ema_fast = latest['ema_fast']
            ema_medium = latest['ema_medium']
            ema_slow = latest['ema_slow']
            
            if not (ema_fast > ema_medium > ema_slow):
                return TradingSignal(
                    signal_type=SignalType.NO_SIGNAL,
                    timestamp=timestamp,
                    price=price,
                    confidence=0,
                    reasons=["EMAs not aligned for uptrend"]
                )
            reasons.append("EMAs aligned bullish")
        
        # Price above fast EMA
        if self.entry_config.require_price_above_fast_ema:
            if price <= latest['ema_fast']:
                return TradingSignal(
                    signal_type=SignalType.NO_SIGNAL,
                    timestamp=timestamp,
                    price=price,
                    confidence=0,
                    reasons=["Price below fast EMA"]
                )
            reasons.append("Price above fast EMA")
        
        # ADX consistency check
        if self.entry_config.min_trending_bars > 0:
            trending_bars = (recent['adx'] > self.entry_config.min_adx_for_entry).sum()
            if trending_bars < self.entry_config.min_trending_bars:
                return TradingSignal(
                    signal_type=SignalType.NO_SIGNAL,
                    timestamp=timestamp,
                    price=price,
                    confidence=0,
                    reasons=[f"Trend not established (only {trending_bars} bars trending)"]
                )
        
        trend_confirmed = True
        
        # ==================== MOMENTUM CONFIRMATION ====================
        
        momentum_confirmed = False
        
        if self.entry_config.require_momentum_confirmation:
            rsi = latest['rsi']
            rsi_rising = latest.get('rsi_rising', False)
            macd_hist = latest['macd_histogram']
            macd_increasing = latest.get('macd_hist_increasing', False)
            
            # Option 1: RSI recovering from oversold
            if rsi < self.indicator_config.rsi_neutral_low and rsi_rising:
                momentum_confirmed = True
                reasons.append(f"RSI {rsi:.1f} recovering from oversold")
            
            # Option 2: RSI neutral with positive MACD
            elif (self.indicator_config.rsi_neutral_low <= rsi <= self.indicator_config.rsi_neutral_high 
                  and macd_hist > 0 and macd_increasing):
                momentum_confirmed = True
                reasons.append(f"RSI {rsi:.1f} neutral, MACD histogram positive and increasing")
            
            # Option 3: RSI above 50 with MACD cross
            elif (rsi > 50 and latest['macd_line'] > latest['macd_signal'] 
                  and df.iloc[-2]['macd_line'] <= df.iloc[-2]['macd_signal']):
                momentum_confirmed = True
                reasons.append(f"RSI {rsi:.1f} bullish, MACD just crossed up")
            
            if not momentum_confirmed:
                return TradingSignal(
                    signal_type=SignalType.NO_SIGNAL,
                    timestamp=timestamp,
                    price=price,
                    confidence=0,
                    reasons=["Momentum not confirmed"]
                )
        else:
            momentum_confirmed = True
        
        # ==================== VOLATILITY FILTER ====================
        
        if self.entry_config.filter_high_volatility:
            atr_ratio = latest.get('atr_ratio', 1.0)
            if atr_ratio > self.entry_config.max_atr_multiple:
                return TradingSignal(
                    signal_type=SignalType.NO_SIGNAL,
                    timestamp=timestamp,
                    price=price,
                    confidence=0,
                    reasons=[f"Volatility too high (ATR ratio: {atr_ratio:.2f})"]
                )
        
        # Check overextension
        ema_distance = abs(price - latest['ema_medium']) / atr
        if ema_distance > self.entry_config.max_distance_from_ema_atr:
            return TradingSignal(
                signal_type=SignalType.NO_SIGNAL,
                timestamp=timestamp,
                price=price,
                confidence=0,
                reasons=[f"Price overextended ({ema_distance:.1f} ATR from EMA)"]
            )
        
        # ==================== VOLUME CONFIRMATION ====================
        
        if self.entry_config.require_volume_confirmation:
            volume_ratio = latest.get('volume_ratio', 1.0)
            if volume_ratio < self.entry_config.min_volume_ratio:
                return TradingSignal(
                    signal_type=SignalType.NO_SIGNAL,
                    timestamp=timestamp,
                    price=price,
                    confidence=0,
                    reasons=[f"Volume too low (ratio: {volume_ratio:.2f})"]
                )
            reasons.append(f"Volume confirmed (ratio: {volume_ratio:.2f})")
            
            # Check OBV trend
            if 'obv_trending_up' in latest and not latest['obv_trending_up']:
                warnings.append("OBV not confirming - potential divergence")
        
        # ==================== CALCULATE STOPS AND TARGETS ====================
        
        stop_loss = price - (atr * self.exit_config.stop_loss_atr_mult)
        risk = price - stop_loss
        take_profit = price + (risk * self.exit_config.target_risk_reward_ratio)
        
        reasons.append(f"Stop: {stop_loss:.2f}, Target: {take_profit:.2f}")
        
        # Calculate confidence based on multiple factors
        confidence = self._calculate_entry_confidence(
            adx=adx,
            regime_confidence=regime.confidence,
            volume_ratio=latest.get('volume_ratio', 1.0),
            momentum_confirmed=momentum_confirmed
        )
        
        # Store signal for freshness tracking
        self._last_signal_bar['long'] = len(df)
        self._signal_prices['long'] = price
        
        return TradingSignal(
            signal_type=SignalType.LONG_ENTRY,
            timestamp=timestamp,
            price=price,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasons=reasons,
            warnings=warnings
        )
    
    def _check_short_entry(
        self,
        df: pd.DataFrame,
        regime: RegimeAnalysis
    ) -> TradingSignal:
        """
        Check for short entry conditions.
        
        Mirror of long entry logic with inverted conditions.
        """
        latest = df.iloc[-1]
        recent = df.tail(5)
        
        timestamp = latest['timestamp'] if 'timestamp' in latest else datetime.now()
        price = latest['close']
        atr = latest['atr']
        
        reasons = []
        warnings = []
        
        # ==================== TREND CONFIRMATION ====================
        
        # ADX check
        adx = latest['adx']
        if adx < self.entry_config.min_adx_for_entry:
            return TradingSignal(
                signal_type=SignalType.NO_SIGNAL,
                timestamp=timestamp,
                price=price,
                confidence=0,
                reasons=[f"ADX {adx:.1f} below threshold"]
            )
        reasons.append(f"ADX: {adx:.1f}")
        
        # EMA alignment (bearish)
        if self.entry_config.require_ema_alignment:
            ema_fast = latest['ema_fast']
            ema_medium = latest['ema_medium']
            ema_slow = latest['ema_slow']
            
            if not (ema_fast < ema_medium < ema_slow):
                return TradingSignal(
                    signal_type=SignalType.NO_SIGNAL,
                    timestamp=timestamp,
                    price=price,
                    confidence=0,
                    reasons=["EMAs not aligned for downtrend"]
                )
            reasons.append("EMAs aligned bearish")
        
        # Price below fast EMA
        if self.entry_config.require_price_above_fast_ema:
            if price >= latest['ema_fast']:
                return TradingSignal(
                    signal_type=SignalType.NO_SIGNAL,
                    timestamp=timestamp,
                    price=price,
                    confidence=0,
                    reasons=["Price above fast EMA"]
                )
            reasons.append("Price below fast EMA")
        
        # ADX consistency
        if self.entry_config.min_trending_bars > 0:
            trending_bars = (recent['adx'] > self.entry_config.min_adx_for_entry).sum()
            if trending_bars < self.entry_config.min_trending_bars:
                return TradingSignal(
                    signal_type=SignalType.NO_SIGNAL,
                    timestamp=timestamp,
                    price=price,
                    confidence=0,
                    reasons=[f"Trend not established"]
                )
        
        # ==================== MOMENTUM CONFIRMATION ====================
        
        momentum_confirmed = False
        
        if self.entry_config.require_momentum_confirmation:
            rsi = latest['rsi']
            rsi_rising = latest.get('rsi_rising', False)
            macd_hist = latest['macd_histogram']
            macd_increasing = latest.get('macd_hist_increasing', False)
            
            # RSI recovering from overbought (falling)
            if rsi > self.indicator_config.rsi_neutral_high and not rsi_rising:
                momentum_confirmed = True
                reasons.append(f"RSI {rsi:.1f} falling from overbought")
            
            # RSI neutral with negative MACD
            elif (self.indicator_config.rsi_neutral_low <= rsi <= self.indicator_config.rsi_neutral_high
                  and macd_hist < 0 and not macd_increasing):
                momentum_confirmed = True
                reasons.append(f"RSI {rsi:.1f} neutral, MACD histogram negative")
            
            # RSI below 50 with MACD cross down
            elif (rsi < 50 and latest['macd_line'] < latest['macd_signal']
                  and df.iloc[-2]['macd_line'] >= df.iloc[-2]['macd_signal']):
                momentum_confirmed = True
                reasons.append(f"RSI {rsi:.1f} bearish, MACD just crossed down")
            
            if not momentum_confirmed:
                return TradingSignal(
                    signal_type=SignalType.NO_SIGNAL,
                    timestamp=timestamp,
                    price=price,
                    confidence=0,
                    reasons=["Momentum not confirmed for short"]
                )
        else:
            momentum_confirmed = True
        
        # ==================== VOLATILITY FILTER ====================
        
        if self.entry_config.filter_high_volatility:
            atr_ratio = latest.get('atr_ratio', 1.0)
            if atr_ratio > self.entry_config.max_atr_multiple:
                return TradingSignal(
                    signal_type=SignalType.NO_SIGNAL,
                    timestamp=timestamp,
                    price=price,
                    confidence=0,
                    reasons=[f"Volatility too high"]
                )
        
        # Check overextension
        ema_distance = abs(price - latest['ema_medium']) / atr
        if ema_distance > self.entry_config.max_distance_from_ema_atr:
            return TradingSignal(
                signal_type=SignalType.NO_SIGNAL,
                timestamp=timestamp,
                price=price,
                confidence=0,
                reasons=[f"Price overextended"]
            )
        
        # ==================== VOLUME CONFIRMATION ====================
        
        if self.entry_config.require_volume_confirmation:
            volume_ratio = latest.get('volume_ratio', 1.0)
            if volume_ratio < self.entry_config.min_volume_ratio:
                return TradingSignal(
                    signal_type=SignalType.NO_SIGNAL,
                    timestamp=timestamp,
                    price=price,
                    confidence=0,
                    reasons=[f"Volume too low"]
                )
            reasons.append(f"Volume confirmed")
            
            if 'obv_trending_up' in latest and latest['obv_trending_up']:
                warnings.append("OBV showing strength - potential divergence")
        
        # ==================== CALCULATE STOPS AND TARGETS ====================
        
        stop_loss = price + (atr * self.exit_config.stop_loss_atr_mult)
        risk = stop_loss - price
        take_profit = price - (risk * self.exit_config.target_risk_reward_ratio)
        
        reasons.append(f"Stop: {stop_loss:.2f}, Target: {take_profit:.2f}")
        
        confidence = self._calculate_entry_confidence(
            adx=adx,
            regime_confidence=regime.confidence,
            volume_ratio=latest.get('volume_ratio', 1.0),
            momentum_confirmed=momentum_confirmed
        )
        
        self._last_signal_bar['short'] = len(df)
        self._signal_prices['short'] = price
        
        return TradingSignal(
            signal_type=SignalType.SHORT_ENTRY,
            timestamp=timestamp,
            price=price,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasons=reasons,
            warnings=warnings
        )
    
    def _check_trend_following_entry(
        self,
        df: pd.DataFrame,
        regime: RegimeAnalysis
    ) -> TradingSignal:
        """
        Check for trend-following entry in ambiguous conditions.
        
        Used when regime is unclear but we might still want to trade.
        """
        if regime.trending_score > 0.3:
            return self._check_long_entry(df, regime)
        elif regime.trending_score < -0.3:
            return self._check_short_entry(df, regime)
        
        latest = df.iloc[-1]
        return TradingSignal(
            signal_type=SignalType.NO_SIGNAL,
            timestamp=latest['timestamp'] if 'timestamp' in latest else datetime.now(),
            price=latest['close'],
            confidence=0,
            reasons=["No clear trend direction"]
        )
    
    def _check_exit_signals(
        self,
        df: pd.DataFrame,
        position: PositionSide,
        regime: RegimeAnalysis
    ) -> TradingSignal:
        """
        Check for exit signals on existing position.
        
        Exit triggers:
        1. Stop-loss hit
        2. Take-profit hit
        3. Trailing stop hit (if enabled)
        4. Regime change
        5. Time-based exit
        """
        latest = df.iloc[-1]
        timestamp = latest['timestamp'] if 'timestamp' in latest else datetime.now()
        price = latest['close']
        
        # This is a simplified check - actual stops are managed by the order executor
        # Here we check for regime changes and other strategic exits
        
        reasons = []
        
        # Regime change exit
        if position == PositionSide.LONG:
            if regime.current_regime == MarketRegime.TRENDING_DOWN:
                return TradingSignal(
                    signal_type=SignalType.LONG_EXIT,
                    timestamp=timestamp,
                    price=price,
                    confidence=0.8,
                    reasons=["Regime changed to downtrend"]
                )
            
            # Momentum loss
            if latest['macd_line'] < latest['macd_signal'] and latest['rsi'] < 40:
                reasons.append("Momentum weakening significantly")
                if regime.current_regime != MarketRegime.TRENDING_UP:
                    return TradingSignal(
                        signal_type=SignalType.LONG_EXIT,
                        timestamp=timestamp,
                        price=price,
                        confidence=0.6,
                        reasons=reasons
                    )
        
        elif position == PositionSide.SHORT:
            if regime.current_regime == MarketRegime.TRENDING_UP:
                return TradingSignal(
                    signal_type=SignalType.SHORT_EXIT,
                    timestamp=timestamp,
                    price=price,
                    confidence=0.8,
                    reasons=["Regime changed to uptrend"]
                )
            
            # Momentum loss (for shorts, this means strength returning)
            if latest['macd_line'] > latest['macd_signal'] and latest['rsi'] > 60:
                reasons.append("Momentum shifting bullish")
                if regime.current_regime != MarketRegime.TRENDING_DOWN:
                    return TradingSignal(
                        signal_type=SignalType.SHORT_EXIT,
                        timestamp=timestamp,
                        price=price,
                        confidence=0.6,
                        reasons=reasons
                    )
        
        return TradingSignal(
            signal_type=SignalType.NO_SIGNAL,
            timestamp=timestamp,
            price=price,
            confidence=0,
            reasons=["No exit signal"]
        )
    
    def _calculate_entry_confidence(
        self,
        adx: float,
        regime_confidence: float,
        volume_ratio: float,
        momentum_confirmed: bool
    ) -> float:
        """
        Calculate overall entry confidence score.
        
        Factors:
        - ADX strength
        - Regime detection confidence
        - Volume confirmation strength
        - Momentum confirmation
        """
        # ADX component (0 to 0.3)
        adx_score = min(adx / 100, 0.3)
        
        # Regime confidence component (0 to 0.3)
        regime_score = regime_confidence * 0.3
        
        # Volume component (0 to 0.2)
        volume_score = min(volume_ratio / 3, 0.2)
        
        # Momentum component (0 or 0.2)
        momentum_score = 0.2 if momentum_confirmed else 0
        
        total = adx_score + regime_score + volume_score + momentum_score
        return min(total, 1.0)
    
    def check_signal_freshness(
        self,
        df: pd.DataFrame,
        signal_type: str
    ) -> Tuple[bool, str]:
        """
        Check if a signal is still fresh (not stale).
        
        Prevents entering trades too late after the signal formed.
        """
        current_bar = len(df)
        
        if signal_type not in self._last_signal_bar:
            return True, "No previous signal"
        
        bars_since_signal = current_bar - self._last_signal_bar[signal_type]
        
        if bars_since_signal > self.entry_config.max_bars_since_signal:
            return False, f"Signal is {bars_since_signal} bars old"
        
        # Check price movement since signal
        if signal_type in self._signal_prices:
            signal_price = self._signal_prices[signal_type]
            current_price = df.iloc[-1]['close']
            atr = df.iloc[-1]['atr']
            
            price_move = abs(current_price - signal_price) / atr
            
            if price_move > self.entry_config.max_price_move_since_signal_atr:
                return False, f"Price moved {price_move:.1f} ATR since signal"
        
        return True, "Signal is fresh"


def generate_signal(
    df: pd.DataFrame,
    config: Optional[TradingConfig] = None,
    current_position: Optional[PositionSide] = None
) -> TradingSignal:
    """
    Convenience function to generate trading signals.
    
    Args:
        df: DataFrame with OHLCV and indicators
        config: Trading configuration
        current_position: Current position if any
        
    Returns:
        TradingSignal with entry/exit decision
    """
    generator = SignalGenerator(config)
    return generator.generate_signal(df, current_position)
