"""
Trading Bot Configuration Settings

This module contains all configurable parameters for the trading system.
Modify these values to adjust strategy behavior, risk management, and execution.

WARNING: Changing these values can significantly impact performance and risk.
Test thoroughly on testnet before deploying with real funds.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class MarketRegime(Enum):
    """Market regime classification."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    UNKNOWN = "unknown"


class OrderSide(Enum):
    """Order direction."""
    BUY = "buy"
    SELL = "sell"


class PositionSide(Enum):
    """Position direction for futures."""
    LONG = "long"
    SHORT = "short"


@dataclass
class IndicatorConfig:
    """Configuration for technical indicators."""
    
    # EMA Settings
    ema_fast: int = 9
    ema_medium: int = 21
    ema_slow: int = 50
    
    # ADX Settings
    adx_period: int = 14
    adx_trend_threshold: float = 25.0  # ADX above this = trending
    adx_strong_trend: float = 35.0     # ADX above this = strong trend
    adx_no_trend: float = 20.0         # ADX below this = ranging
    
    # RSI Settings
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    rsi_neutral_low: float = 40.0
    rsi_neutral_high: float = 60.0
    
    # MACD Settings
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # ATR Settings
    atr_period: int = 14
    atr_volatility_lookback: int = 50  # For calculating "normal" ATR
    atr_high_volatility_mult: float = 2.0  # ATR > this × average = high vol
    
    # Bollinger Bands Settings
    bb_period: int = 20
    bb_std_dev: float = 2.0
    
    # Volume Settings
    volume_ma_period: int = 20
    volume_confirmation_mult: float = 1.0  # Volume > this × MA to confirm
    
    # OBV Settings
    obv_ma_period: int = 20


@dataclass
class EntryConfig:
    """Configuration for entry logic."""
    
    # Trend confirmation
    require_ema_alignment: bool = True
    require_price_above_fast_ema: bool = True
    min_adx_for_entry: float = 25.0
    
    # Momentum confirmation
    require_momentum_confirmation: bool = True
    
    # Volume confirmation
    require_volume_confirmation: bool = True
    min_volume_ratio: float = 1.0  # Current vol / Vol MA
    
    # Volatility filter
    filter_high_volatility: bool = True
    max_atr_multiple: float = 2.0
    
    # Overextension filter
    max_distance_from_ema_atr: float = 2.0  # Max ATR distance from EMA(21)
    
    # Freshness check
    max_bars_since_signal: int = 3
    max_price_move_since_signal_atr: float = 1.5
    
    # Chop zone avoidance
    min_trending_bars: int = 3  # ADX must be > threshold for this many bars


@dataclass
class ExitConfig:
    """Configuration for exit logic."""
    
    # Stop-loss settings
    stop_loss_atr_mult: float = 2.0
    use_volatility_adjusted_stop: bool = True
    
    # Take-profit settings
    min_risk_reward_ratio: float = 2.0
    target_risk_reward_ratio: float = 2.5
    
    # Trailing stop settings
    use_trailing_stop: bool = True
    trailing_activation_profit_mult: float = 1.5  # Activate after 1.5x risk in profit
    trailing_stop_atr_mult: float = 1.5
    min_adx_for_trailing: float = 35.0  # Only trail in strong trends
    
    # Time-based exit
    max_hold_periods: int = 50  # Exit if position doesn't move in this many bars


@dataclass
class RiskConfig:
    """Configuration for risk management."""
    
    # Position sizing
    risk_per_trade_percent: float = 0.01  # 1% of account per trade
    max_risk_per_trade_percent: float = 0.02  # Hard cap at 2%
    
    # Position limits
    max_concurrent_positions: int = 3
    max_position_size_percent: float = 0.20  # Max 20% of account in single position
    
    # Daily limits
    daily_loss_limit_percent: float = 0.03  # Stop trading after 3% daily loss
    daily_profit_target_percent: float = 0.05  # Optional: reduce size after 5% daily gain
    
    # Correlation management
    max_correlated_positions: int = 2
    correlation_threshold: float = 0.7
    correlated_position_size_reduction: float = 0.5  # Reduce size by 50% if correlated
    
    # Drawdown management
    reduce_size_drawdown_threshold: float = 0.10  # Reduce size if 10% drawdown
    size_reduction_factor: float = 0.5
    
    # Volatility adjustment
    reduce_size_in_high_volatility: bool = True
    high_volatility_size_reduction: float = 0.5


@dataclass
class ExecutionConfig:
    """Configuration for order execution."""
    
    # Order type
    use_limit_orders: bool = True
    limit_order_offset_percent: float = 0.0005  # 0.05% from current price
    limit_order_timeout_seconds: int = 30
    
    # Slippage protection
    max_slippage_percent: float = 0.005  # 0.5% max slippage
    
    # Retry logic
    max_order_retries: int = 3
    retry_delay_seconds: float = 1.0
    
    # Position management
    use_isolated_margin: bool = True
    default_leverage: int = 2  # Conservative leverage
    max_leverage: int = 3
    
    # Order cooldown
    min_time_between_orders_seconds: int = 60


@dataclass
class DataConfig:
    """Configuration for data fetching and processing."""
    
    # Timeframe
    timeframe: str = "1h"  # Default to 1-hour candles
    
    # Historical data
    min_historical_bars: int = 200  # Need at least this many bars for indicators
    max_historical_bars: int = 1000
    
    # Data quality
    max_missing_candles_percent: float = 0.05  # Alert if > 5% missing
    
    # Symbols
    default_symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    
    # Costs
    maker_fee_percent: float = 0.0002  # 0.02% maker fee
    taker_fee_percent: float = 0.0004  # 0.04% taker fee
    slippage_percent: float = 0.001    # 0.1% assumed slippage
    
    # Funding rate (for perpetuals)
    include_funding: bool = True
    avg_funding_rate_8h: float = 0.0001  # 0.01% average funding
    
    # Initial capital
    initial_capital: float = 10000.0
    
    # Reporting
    report_frequency: int = 100  # Print progress every N trades
    
    # Walk-forward
    walk_forward_train_months: int = 6
    walk_forward_test_months: int = 3


@dataclass
class TradingConfig:
    """Master configuration combining all settings."""
    
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    entry: EntryConfig = field(default_factory=EntryConfig)
    exit: ExitConfig = field(default_factory=ExitConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    
    # Environment
    testnet: bool = True  # ALWAYS default to testnet
    log_level: str = "INFO"
    enable_notifications: bool = False
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of warnings/errors."""
        warnings = []
        
        # Risk checks
        if self.risk.risk_per_trade_percent > 0.02:
            warnings.append("WARNING: Risk per trade > 2% is very aggressive")
        
        if self.risk.daily_loss_limit_percent > 0.05:
            warnings.append("WARNING: Daily loss limit > 5% may cause significant drawdowns")
        
        if self.execution.default_leverage > 5:
            warnings.append("WARNING: Leverage > 5x significantly increases risk")
        
        # Indicator checks
        if self.indicators.adx_trend_threshold < 20:
            warnings.append("WARNING: ADX threshold < 20 will generate many false signals")
        
        if self.exit.min_risk_reward_ratio < 1.5:
            warnings.append("WARNING: R:R ratio < 1.5 requires high win rate to be profitable")
        
        # Entry checks
        if not self.entry.require_volume_confirmation:
            warnings.append("WARNING: Disabling volume confirmation may increase false signals")
        
        if not self.testnet:
            warnings.append("CRITICAL: Testnet is DISABLED - using REAL funds!")
        
        return warnings


# Default configuration instance
DEFAULT_CONFIG = TradingConfig()


def load_config_from_env() -> TradingConfig:
    """Load configuration with environment variable overrides."""
    import os
    
    config = TradingConfig()
    
    # Override testnet setting from environment
    testnet_env = os.getenv("BINANCE_TESTNET", "true").lower()
    config.testnet = testnet_env in ("true", "1", "yes")
    
    # Override risk settings from environment
    if os.getenv("RISK_PER_TRADE"):
        config.risk.risk_per_trade_percent = float(os.getenv("RISK_PER_TRADE"))
    
    if os.getenv("MAX_POSITIONS"):
        config.risk.max_concurrent_positions = int(os.getenv("MAX_POSITIONS"))
    
    if os.getenv("DAILY_LOSS_LIMIT"):
        config.risk.daily_loss_limit_percent = float(os.getenv("DAILY_LOSS_LIMIT"))
    
    return config
