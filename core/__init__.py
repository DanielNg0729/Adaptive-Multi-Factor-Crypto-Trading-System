"""Core trading modules."""

from .indicators import TechnicalIndicators, calculate_indicators
from .regime import RegimeDetector, RegimeAnalysis, detect_regime
from .strategy import SignalGenerator, TradingSignal, SignalType, generate_signal
from .executor import OrderExecutor, Order, OrderStatus, ExecutionResult

__all__ = [
    "TechnicalIndicators",
    "calculate_indicators",
    "RegimeDetector",
    "RegimeAnalysis",
    "detect_regime",
    "SignalGenerator",
    "TradingSignal",
    "SignalType",
    "generate_signal",
    "OrderExecutor",
    "Order",
    "OrderStatus",
    "ExecutionResult",
]
