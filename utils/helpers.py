"""
Utility Functions

Helper functions used across the trading system.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
import hashlib
import json


logger = logging.getLogger(__name__)


def format_price(price: float, precision: int = 2) -> str:
    """Format price with proper precision."""
    return f"{price:,.{precision}f}"


def format_pnl(pnl: float) -> str:
    """Format P&L with color indicators."""
    sign = "+" if pnl >= 0 else ""
    return f"{sign}{pnl:,.2f}"


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values."""
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100


def timestamp_to_string(ts: datetime) -> str:
    """Convert timestamp to ISO format string."""
    return ts.strftime('%Y-%m-%d %H:%M:%S')


def string_to_timestamp(s: str) -> datetime:
    """Convert ISO format string to timestamp."""
    return datetime.strptime(s, '%Y-%m-%d %H:%M:%S')


def calculate_risk_reward_ratio(
    entry: float,
    stop_loss: float,
    take_profit: float
) -> float:
    """Calculate risk-reward ratio."""
    risk = abs(entry - stop_loss)
    reward = abs(take_profit - entry)
    
    if risk == 0:
        return 0.0
    
    return reward / risk


def round_to_precision(value: float, precision: int) -> float:
    """Round value to specified decimal precision."""
    multiplier = 10 ** precision
    return round(value * multiplier) / multiplier


def generate_trade_id(
    symbol: str,
    timestamp: datetime,
    side: str
) -> str:
    """Generate unique trade ID."""
    data = f"{symbol}_{timestamp.isoformat()}_{side}"
    return hashlib.md5(data.encode()).hexdigest()[:12]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    if denominator == 0:
        return default
    return numerator / denominator


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp value between min and max."""
    return max(min_value, min(max_value, value))


def serialize_config(config: Any) -> Dict:
    """Serialize configuration object to dictionary."""
    if hasattr(config, '__dict__'):
        result = {}
        for key, value in config.__dict__.items():
            if hasattr(value, '__dict__'):
                result[key] = serialize_config(value)
            else:
                result[key] = value
        return result
    return {}


def load_json_file(filepath: str) -> Optional[Dict]:
    """Load JSON file safely."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return None


def save_json_file(data: Dict, filepath: str) -> bool:
    """Save dictionary to JSON file."""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception as e:
        logger.error(f"Error saving {filepath}: {e}")
        return False


class MovingAverage:
    """Simple online moving average calculator."""
    
    def __init__(self, period: int):
        self.period = period
        self.values = []
    
    def update(self, value: float) -> float:
        """Add value and return current average."""
        self.values.append(value)
        if len(self.values) > self.period:
            self.values.pop(0)
        return sum(self.values) / len(self.values)
    
    def current(self) -> float:
        """Get current average."""
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)


class ExponentialMovingAverage:
    """Online EMA calculator."""
    
    def __init__(self, period: int):
        self.period = period
        self.multiplier = 2 / (period + 1)
        self.ema = None
    
    def update(self, value: float) -> float:
        """Add value and return current EMA."""
        if self.ema is None:
            self.ema = value
        else:
            self.ema = (value * self.multiplier) + (self.ema * (1 - self.multiplier))
        return self.ema
    
    def current(self) -> float:
        """Get current EMA."""
        return self.ema if self.ema is not None else 0.0
