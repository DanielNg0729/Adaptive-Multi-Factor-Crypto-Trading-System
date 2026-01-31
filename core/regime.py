"""
Market Regime Detection Module

Identifies current market conditions to adapt trading strategy:
- TRENDING_UP: Strong uptrend with momentum
- TRENDING_DOWN: Strong downtrend with momentum
- RANGING: No clear trend, choppy price action
- HIGH_VOLATILITY: Extreme volatility, reduce exposure

The regime determines which indicators and entry rules to prioritize.
"""

import logging
from typing import Optional, Dict, List
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np

from config.settings import MarketRegime, IndicatorConfig


logger = logging.getLogger(__name__)


@dataclass
class RegimeAnalysis:
    """Complete regime analysis result."""
    current_regime: MarketRegime
    confidence: float  # 0 to 1
    trending_score: float  # -1 (strong down) to 1 (strong up)
    volatility_level: str  # 'low', 'normal', 'high', 'extreme'
    trend_direction: Optional[str]  # 'up', 'down', None
    regime_duration: int  # Bars in current regime
    details: Dict[str, float]


class RegimeDetector:
    """
    Detects market regime based on multiple factors.
    
    Regime detection is crucial because:
    - Trend-following works in trending markets, fails in ranges
    - Mean-reversion works in ranges, fails in trends
    - High volatility requires position size reduction
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        """
        Initialize regime detector.
        
        Args:
            config: Indicator configuration
        """
        self.config = config or IndicatorConfig()
        self._regime_history: List[MarketRegime] = []
    
    def detect(self, df: pd.DataFrame) -> RegimeAnalysis:
        """
        Analyze current market regime.
        
        Args:
            df: DataFrame with calculated indicators
            
        Returns:
            RegimeAnalysis with regime classification and confidence
        """
        if len(df) < 50:
            logger.warning("Insufficient data for regime detection")
            return RegimeAnalysis(
                current_regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                trending_score=0.0,
                volatility_level='unknown',
                trend_direction=None,
                regime_duration=0,
                details={}
            )
        
        latest = df.iloc[-1]
        
        # Analyze trend component
        trend_analysis = self._analyze_trend(df)
        
        # Analyze volatility component
        volatility_analysis = self._analyze_volatility(df)
        
        # Determine regime
        regime, confidence = self._classify_regime(trend_analysis, volatility_analysis)
        
        # Calculate regime duration
        duration = self._calculate_regime_duration(regime)
        
        # Update history
        self._regime_history.append(regime)
        if len(self._regime_history) > 100:
            self._regime_history = self._regime_history[-100:]
        
        analysis = RegimeAnalysis(
            current_regime=regime,
            confidence=confidence,
            trending_score=trend_analysis['score'],
            volatility_level=volatility_analysis['level'],
            trend_direction=trend_analysis['direction'],
            regime_duration=duration,
            details={
                'adx': latest.get('adx', 0),
                'atr_ratio': latest.get('atr_ratio', 1),
                'ema_alignment': latest.get('ema_alignment', 0),
                'bb_width': latest.get('bb_width', 0),
            }
        )
        
        logger.debug(f"Regime: {regime.value}, Confidence: {confidence:.2f}")
        return analysis
    
    def _analyze_trend(self, df: pd.DataFrame) -> Dict:
        """
        Analyze trend strength and direction.
        
        Uses:
        - ADX for trend strength
        - EMA alignment for trend direction
        - Price position relative to EMAs
        """
        latest = df.iloc[-1]
        recent = df.tail(10)
        
        # ADX-based trend strength
        adx = latest.get('adx', 0)
        
        if adx < self.config.adx_no_trend:
            trend_strength = 'none'
        elif adx < self.config.adx_trend_threshold:
            trend_strength = 'weak'
        elif adx < self.config.adx_strong_trend:
            trend_strength = 'moderate'
        else:
            trend_strength = 'strong'
        
        # EMA alignment for direction
        ema_fast = latest.get('ema_fast', 0)
        ema_medium = latest.get('ema_medium', 0)
        ema_slow = latest.get('ema_slow', 0)
        price = latest['close']
        
        if ema_fast > ema_medium > ema_slow and price > ema_fast:
            direction = 'up'
            alignment_score = 1.0
        elif ema_fast < ema_medium < ema_slow and price < ema_fast:
            direction = 'down'
            alignment_score = -1.0
        else:
            direction = None
            alignment_score = 0.0
        
        # Calculate composite trend score
        # Scale ADX to 0-1, then multiply by direction
        adx_normalized = min(adx / 50, 1.0)
        trend_score = alignment_score * adx_normalized
        
        # Check consistency over recent bars
        adx_consistent = (recent['adx'] > self.config.adx_trend_threshold).sum() / len(recent)
        
        return {
            'strength': trend_strength,
            'direction': direction,
            'score': trend_score,
            'adx': adx,
            'consistency': adx_consistent,
            'alignment': alignment_score
        }
    
    def _analyze_volatility(self, df: pd.DataFrame) -> Dict:
        """
        Analyze volatility level.
        
        Uses:
        - ATR ratio (current ATR vs historical average)
        - Bollinger Band width
        - Recent price range
        """
        latest = df.iloc[-1]
        recent = df.tail(20)
        
        # ATR ratio
        atr_ratio = latest.get('atr_ratio', 1.0)
        
        # Bollinger Band width
        bb_width = latest.get('bb_width', 0.02)
        bb_width_avg = recent['bb_width'].mean() if 'bb_width' in recent else 0.02
        
        # Price range analysis
        high_low_range = (recent['high'].max() - recent['low'].min()) / latest['close']
        
        # Classify volatility level
        if atr_ratio > self.config.atr_high_volatility_mult * 1.5:
            level = 'extreme'
            vol_score = 1.0
        elif atr_ratio > self.config.atr_high_volatility_mult:
            level = 'high'
            vol_score = 0.7
        elif atr_ratio < 0.5:
            level = 'low'
            vol_score = 0.2
        else:
            level = 'normal'
            vol_score = 0.4
        
        return {
            'level': level,
            'score': vol_score,
            'atr_ratio': atr_ratio,
            'bb_width': bb_width,
            'range': high_low_range
        }
    
    def _classify_regime(
        self,
        trend: Dict,
        volatility: Dict
    ) -> tuple[MarketRegime, float]:
        """
        Classify market regime based on trend and volatility analysis.
        
        Priority:
        1. HIGH_VOLATILITY overrides everything (safety first)
        2. Strong trend with proper alignment → TRENDING
        3. Weak trend or no alignment → RANGING
        """
        confidence = 0.0
        
        # Check for high volatility first
        if volatility['level'] in ('high', 'extreme'):
            confidence = min(volatility['atr_ratio'] / 3, 1.0)
            return MarketRegime.HIGH_VOLATILITY, confidence
        
        # Check for trending
        trend_strength = trend['strength']
        trend_direction = trend['direction']
        
        if trend_strength in ('moderate', 'strong') and trend_direction is not None:
            # Calculate confidence based on ADX and consistency
            adx_conf = min(trend['adx'] / 50, 1.0)
            consistency_conf = trend['consistency']
            confidence = (adx_conf + consistency_conf) / 2
            
            if trend_direction == 'up':
                return MarketRegime.TRENDING_UP, confidence
            else:
                return MarketRegime.TRENDING_DOWN, confidence
        
        # Default to ranging
        # Higher confidence if ADX is very low
        if trend['adx'] < self.config.adx_no_trend:
            confidence = 0.8
        else:
            confidence = 0.5
        
        return MarketRegime.RANGING, confidence
    
    def _calculate_regime_duration(self, current_regime: MarketRegime) -> int:
        """Calculate how many bars we've been in the current regime."""
        if not self._regime_history:
            return 1
        
        duration = 1
        for regime in reversed(self._regime_history):
            if regime == current_regime:
                duration += 1
            else:
                break
        
        return duration
    
    def get_regime_statistics(self, df: pd.DataFrame, lookback: int = 100) -> Dict:
        """
        Calculate regime statistics over recent history.
        
        Useful for understanding overall market conditions.
        """
        if len(df) < lookback:
            lookback = len(df)
        
        recent = df.tail(lookback)
        
        # Trend statistics
        trending_bars = (recent['adx'] > self.config.adx_trend_threshold).sum()
        bullish_bars = (recent['ema_alignment'] > 0).sum() if 'ema_alignment' in recent else 0
        bearish_bars = (recent['ema_alignment'] < 0).sum() if 'ema_alignment' in recent else 0
        
        # Volatility statistics
        high_vol_bars = (recent['atr_ratio'] > self.config.atr_high_volatility_mult).sum() if 'atr_ratio' in recent else 0
        avg_volatility = recent['atr_ratio'].mean() if 'atr_ratio' in recent else 1.0
        
        return {
            'trending_percent': trending_bars / lookback,
            'bullish_percent': bullish_bars / lookback,
            'bearish_percent': bearish_bars / lookback,
            'high_volatility_percent': high_vol_bars / lookback,
            'average_volatility': avg_volatility,
            'lookback_bars': lookback
        }
    
    def is_safe_to_trade(self, analysis: RegimeAnalysis) -> tuple[bool, str]:
        """
        Determine if current regime is suitable for trading.
        
        Returns:
            Tuple of (is_safe, reason)
        """
        # Never trade in extreme volatility
        if analysis.volatility_level == 'extreme':
            return False, "Extreme volatility detected"
        
        # Avoid unknown regimes
        if analysis.current_regime == MarketRegime.UNKNOWN:
            return False, "Unable to determine market regime"
        
        # Low confidence = uncertain
        if analysis.confidence < 0.4:
            return False, f"Low regime confidence ({analysis.confidence:.2f})"
        
        # Ranging markets are difficult
        if analysis.current_regime == MarketRegime.RANGING:
            if analysis.confidence > 0.7:
                return False, "Strong ranging conditions - avoid trend trading"
            return True, "Mild ranging - trade with caution"
        
        return True, "Regime suitable for trading"


def detect_regime(df: pd.DataFrame, config: Optional[IndicatorConfig] = None) -> RegimeAnalysis:
    """
    Convenience function to detect market regime.
    
    Args:
        df: DataFrame with calculated indicators
        config: Optional indicator configuration
        
    Returns:
        RegimeAnalysis result
    """
    detector = RegimeDetector(config)
    return detector.detect(df)
