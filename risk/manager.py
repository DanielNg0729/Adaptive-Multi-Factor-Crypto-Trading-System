"""
Risk Management Module

Implements comprehensive risk management:
- Position sizing based on account risk
- Maximum position limits
- Daily loss limits (circuit breaker)
- Drawdown management
- Correlation checks

Capital preservation is the #1 priority.
"""

import logging
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum

import pandas as pd
import numpy as np

from config.settings import RiskConfig, PositionSide


logger = logging.getLogger(__name__)


class RiskAction(Enum):
    """Risk manager decisions."""
    ALLOW = "allow"
    REDUCE_SIZE = "reduce_size"
    BLOCK = "block"


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    side: PositionSide
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    unrealized_pnl: float = 0.0
    highest_price: float = 0.0  # For trailing stop
    lowest_price: float = float('inf')  # For trailing stop


@dataclass
class RiskState:
    """Current risk state of the account."""
    account_balance: float
    available_balance: float
    total_position_value: float
    open_positions: List[Position]
    daily_realized_pnl: float
    daily_unrealized_pnl: float
    max_drawdown: float
    current_drawdown: float
    daily_trades: int


@dataclass
class RiskDecision:
    """Result of risk evaluation."""
    action: RiskAction
    allowed_size: float
    reason: str
    warnings: List[str] = field(default_factory=list)


class RiskManager:
    """
    Manages all risk-related decisions and tracking.
    
    Key responsibilities:
    1. Calculate position sizes based on risk parameters
    2. Enforce position and exposure limits
    3. Track daily P&L and enforce circuit breakers
    4. Monitor drawdown and adjust risk accordingly
    """
    
    def __init__(self, config: Optional[RiskConfig] = None):
        """
        Initialize risk manager.
        
        Args:
            config: Risk configuration settings
        """
        self.config = config or RiskConfig()
        
        # Track positions
        self._positions: Dict[str, Position] = {}
        
        # Daily tracking
        self._daily_pnl: float = 0.0
        self._daily_trades: int = 0
        self._last_reset_date: date = date.today()
        
        # Drawdown tracking
        self._peak_balance: float = 0.0
        self._current_drawdown: float = 0.0
        
        # Trade history for correlation analysis
        self._recent_trades: List[Dict] = []
        
        logger.info("RiskManager initialized")
    
    def calculate_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss: float,
        symbol: str = "default"
    ) -> RiskDecision:
        """
        Calculate appropriate position size based on risk parameters.
        
        Formula:
        position_size = (account_balance × risk_per_trade) / |entry_price - stop_loss|
        
        Args:
            account_balance: Current account value
            entry_price: Planned entry price
            stop_loss: Stop-loss price
            symbol: Trading symbol for correlation check
            
        Returns:
            RiskDecision with allowed size and any warnings
        """
        self._check_daily_reset()
        
        warnings = []
        
        # Calculate base position size
        risk_amount = account_balance * self.config.risk_per_trade_percent
        stop_distance = abs(entry_price - stop_loss)
        
        if stop_distance == 0:
            return RiskDecision(
                action=RiskAction.BLOCK,
                allowed_size=0,
                reason="Invalid stop-loss (same as entry)"
            )
        
        base_size = risk_amount / stop_distance
        
        # Apply position size cap
        max_position_value = account_balance * self.config.max_position_size_percent
        max_size_by_value = max_position_value / entry_price
        
        if base_size > max_size_by_value:
            base_size = max_size_by_value
            warnings.append(f"Size capped by max position value ({self.config.max_position_size_percent:.0%})")
        
        # Check circuit breaker
        is_blocked, block_reason = self._check_circuit_breaker(account_balance)
        if is_blocked:
            return RiskDecision(
                action=RiskAction.BLOCK,
                allowed_size=0,
                reason=block_reason
            )
        
        # Check position count
        if len(self._positions) >= self.config.max_concurrent_positions:
            return RiskDecision(
                action=RiskAction.BLOCK,
                allowed_size=0,
                reason=f"Max positions ({self.config.max_concurrent_positions}) reached"
            )
        
        # Check drawdown and adjust size
        adjusted_size = base_size
        if self._current_drawdown > self.config.reduce_size_drawdown_threshold:
            adjusted_size *= self.config.size_reduction_factor
            warnings.append(f"Size reduced due to {self._current_drawdown:.1%} drawdown")
        
        # Check for correlated positions
        correlation_factor = self._check_correlation(symbol)
        if correlation_factor < 1.0:
            adjusted_size *= correlation_factor
            warnings.append(f"Size reduced due to correlation with existing positions")
        
        return RiskDecision(
            action=RiskAction.ALLOW if adjusted_size > 0 else RiskAction.BLOCK,
            allowed_size=adjusted_size,
            reason="Position size calculated successfully",
            warnings=warnings
        )
    
    def evaluate_trade(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        symbol: str
    ) -> RiskDecision:
        """
        Comprehensive trade evaluation including all risk checks.
        
        Args:
            account_balance: Current account value
            entry_price: Planned entry price
            stop_loss: Stop-loss price
            take_profit: Take-profit price
            symbol: Trading symbol
            
        Returns:
            RiskDecision with final verdict
        """
        self._check_daily_reset()
        
        # Calculate risk-reward ratio
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if risk == 0:
            return RiskDecision(
                action=RiskAction.BLOCK,
                allowed_size=0,
                reason="Invalid risk (zero)"
            )
        
        risk_reward = reward / risk
        
        if risk_reward < 1.5:
            return RiskDecision(
                action=RiskAction.BLOCK,
                allowed_size=0,
                reason=f"Risk-reward ratio {risk_reward:.2f} below minimum 1.5"
            )
        
        # Get position size decision
        size_decision = self.calculate_position_size(
            account_balance, entry_price, stop_loss, symbol
        )
        
        if size_decision.action == RiskAction.BLOCK:
            return size_decision
        
        # Additional sanity checks
        warnings = size_decision.warnings.copy()
        
        # Check if stop is too close (less than 0.5% from entry)
        stop_percent = risk / entry_price
        if stop_percent < 0.005:
            warnings.append("Stop very close to entry - may get hit by noise")
        
        # Check if stop is too far (more than 5% from entry)
        if stop_percent > 0.05:
            warnings.append("Stop far from entry - consider tighter risk management")
        
        return RiskDecision(
            action=size_decision.action,
            allowed_size=size_decision.allowed_size,
            reason=size_decision.reason,
            warnings=warnings
        )
    
    def register_position(self, position: Position) -> None:
        """
        Register a new position.
        
        Args:
            position: Position to register
        """
        self._positions[position.symbol] = position
        self._daily_trades += 1
        
        logger.info(
            f"Position registered: {position.symbol} {position.side.value} "
            f"size={position.size:.6f} entry={position.entry_price:.2f}"
        )
    
    def close_position(
        self,
        symbol: str,
        exit_price: float,
        realized_pnl: float
    ) -> None:
        """
        Close a position and update tracking.
        
        Args:
            symbol: Symbol to close
            exit_price: Exit price
            realized_pnl: Realized P&L
        """
        if symbol not in self._positions:
            logger.warning(f"Position {symbol} not found")
            return
        
        position = self._positions.pop(symbol)
        
        # Update daily P&L
        self._daily_pnl += realized_pnl
        
        # Update drawdown tracking
        # Note: This is simplified - real implementation would track equity curve
        
        # Record trade
        self._recent_trades.append({
            'symbol': symbol,
            'side': position.side.value,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'pnl': realized_pnl,
            'timestamp': datetime.now()
        })
        
        # Keep only recent trades
        if len(self._recent_trades) > 100:
            self._recent_trades = self._recent_trades[-100:]
        
        logger.info(
            f"Position closed: {symbol} pnl={realized_pnl:.2f} "
            f"daily_pnl={self._daily_pnl:.2f}"
        )
    
    def update_position_pnl(
        self,
        symbol: str,
        current_price: float
    ) -> Optional[float]:
        """
        Update unrealized P&L for a position.
        
        Args:
            symbol: Symbol to update
            current_price: Current market price
            
        Returns:
            Unrealized P&L or None if position not found
        """
        if symbol not in self._positions:
            return None
        
        position = self._positions[symbol]
        
        if position.side == PositionSide.LONG:
            unrealized = (current_price - position.entry_price) * position.size
        else:
            unrealized = (position.entry_price - current_price) * position.size
        
        position.unrealized_pnl = unrealized
        
        # Update high/low for trailing stop
        position.highest_price = max(position.highest_price, current_price)
        position.lowest_price = min(position.lowest_price, current_price)
        
        return unrealized
    
    def get_risk_state(self, account_balance: float) -> RiskState:
        """
        Get current risk state summary.
        
        Args:
            account_balance: Current account balance
            
        Returns:
            RiskState with all risk metrics
        """
        self._check_daily_reset()
        
        total_position_value = sum(
            p.entry_price * p.size for p in self._positions.values()
        )
        
        total_unrealized = sum(
            p.unrealized_pnl for p in self._positions.values()
        )
        
        return RiskState(
            account_balance=account_balance,
            available_balance=account_balance - total_position_value,
            total_position_value=total_position_value,
            open_positions=list(self._positions.values()),
            daily_realized_pnl=self._daily_pnl,
            daily_unrealized_pnl=total_unrealized,
            max_drawdown=self._current_drawdown,
            current_drawdown=self._current_drawdown,
            daily_trades=self._daily_trades
        )
    
    def _check_circuit_breaker(self, account_balance: float) -> Tuple[bool, str]:
        """
        Check if circuit breaker should be triggered.
        
        Returns:
            Tuple of (is_triggered, reason)
        """
        # Check daily loss limit
        daily_loss_limit = account_balance * self.config.daily_loss_limit_percent
        
        total_unrealized = sum(
            p.unrealized_pnl for p in self._positions.values()
        )
        
        total_daily_pnl = self._daily_pnl + total_unrealized
        
        if total_daily_pnl <= -daily_loss_limit:
            return True, f"Daily loss limit reached ({total_daily_pnl:.2f})"
        
        return False, ""
    
    def _check_correlation(self, symbol: str) -> float:
        """
        Check correlation with existing positions.
        
        Returns factor to multiply position size by (1.0 = no reduction).
        """
        if not self._positions:
            return 1.0
        
        # Simple correlation check based on symbol
        # In production, use actual price correlation
        
        # Check if we already have positions in similar assets
        existing_bases = set()
        for pos_symbol in self._positions.keys():
            # Extract base currency (e.g., BTC from BTCUSDT)
            base = pos_symbol.replace("USDT", "").replace("/", "")
            existing_bases.add(base)
        
        new_base = symbol.replace("USDT", "").replace("/", "")
        
        # Major crypto correlations (simplified)
        high_correlation_groups = [
            {'BTC', 'ETH'},  # Major caps often correlated
            {'SOL', 'AVAX', 'DOT'},  # Layer 1s
            {'DOGE', 'SHIB'},  # Meme coins
        ]
        
        for group in high_correlation_groups:
            if new_base in group:
                for existing in existing_bases:
                    if existing in group:
                        return self.config.correlated_position_size_reduction
        
        return 1.0
    
    def _check_daily_reset(self) -> None:
        """Reset daily tracking if day has changed."""
        today = date.today()
        if today != self._last_reset_date:
            logger.info(f"Daily reset: previous P&L = {self._daily_pnl:.2f}")
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._last_reset_date = today
    
    def update_drawdown(self, current_equity: float) -> float:
        """
        Update drawdown tracking.
        
        Args:
            current_equity: Current total equity
            
        Returns:
            Current drawdown percentage
        """
        if current_equity > self._peak_balance:
            self._peak_balance = current_equity
            self._current_drawdown = 0.0
        else:
            self._current_drawdown = (self._peak_balance - current_equity) / self._peak_balance
        
        return self._current_drawdown
    
    def should_reduce_exposure(self) -> Tuple[bool, str]:
        """
        Determine if overall exposure should be reduced.
        
        Returns:
            Tuple of (should_reduce, reason)
        """
        # Check drawdown
        if self._current_drawdown > self.config.reduce_size_drawdown_threshold:
            return True, f"Drawdown at {self._current_drawdown:.1%}"
        
        # Check if approaching daily limit
        # (implemented in real version with proper equity tracking)
        
        return False, ""
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self._positions.get(symbol)
    
    def get_all_positions(self) -> List[Position]:
        """Get all open positions."""
        return list(self._positions.values())
    
    def calculate_total_exposure(self) -> float:
        """Calculate total position value as percentage of peak balance."""
        if self._peak_balance == 0:
            return 0.0
        
        total_value = sum(
            p.entry_price * p.size for p in self._positions.values()
        )
        
        return total_value / self._peak_balance


def calculate_position_size(
    account_balance: float,
    entry_price: float,
    stop_loss: float,
    risk_percent: float = 0.01
) -> float:
    """
    Simple position size calculation.
    
    Args:
        account_balance: Account value
        entry_price: Entry price
        stop_loss: Stop-loss price
        risk_percent: Risk per trade (default 1%)
        
    Returns:
        Position size in base currency
    """
    risk_amount = account_balance * risk_percent
    stop_distance = abs(entry_price - stop_loss)
    
    if stop_distance == 0:
        return 0.0
    
    return risk_amount / stop_distance
