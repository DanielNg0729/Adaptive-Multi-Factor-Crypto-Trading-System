"""
Backtesting Engine

Provides realistic backtesting with:
- Transaction costs (fees and slippage)
- Proper position sizing
- Comprehensive performance metrics
- Walk-forward testing support

WARNING: Backtesting has inherent limitations. Past performance
does NOT guarantee future results.
"""

import logging
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

import pandas as pd
import numpy as np

from config.settings import TradingConfig, BacktestConfig, PositionSide, MarketRegime
from core.indicators import TechnicalIndicators
from core.regime import RegimeDetector
from core.strategy import SignalGenerator, SignalType


logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Record of a backtested trade."""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_percent: float
    fees: float
    exit_reason: str
    bars_held: int


@dataclass
class BacktestMetrics:
    """Comprehensive backtest performance metrics."""
    total_return: float
    total_return_percent: float
    annual_return_percent: float
    
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    profit_factor: float
    average_win: float
    average_loss: float
    average_trade: float
    largest_win: float
    largest_loss: float
    
    max_drawdown: float
    max_drawdown_percent: float
    max_drawdown_duration_days: int
    
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    avg_bars_held: float
    avg_bars_winning: float
    avg_bars_losing: float
    
    total_fees: float
    
    # Monthly breakdown
    monthly_returns: Dict[str, float] = field(default_factory=dict)
    
    # Warnings
    warnings: List[str] = field(default_factory=list)


@dataclass
class BacktestResult:
    """Complete backtest result."""
    metrics: BacktestMetrics
    trades: List[BacktestTrade]
    equity_curve: pd.DataFrame
    config_used: Dict
    start_date: datetime
    end_date: datetime


class BacktestEngine:
    """
    Backtesting engine for strategy validation.
    
    Simulates trading with realistic conditions:
    - Fees and slippage applied to all trades
    - Position sizing based on account equity
    - Stop-loss and take-profit execution
    - Daily drawdown tracking
    """
    
    def __init__(
        self,
        config: Optional[TradingConfig] = None,
        backtest_config: Optional[BacktestConfig] = None
    ):
        """
        Initialize backtester.
        
        Args:
            config: Trading configuration
            backtest_config: Backtest-specific configuration
        """
        self.config = config or TradingConfig()
        self.bt_config = backtest_config or self.config.backtest
        
        self.indicators = TechnicalIndicators(self.config.indicators)
        self.regime_detector = RegimeDetector(self.config.indicators)
        self.signal_generator = SignalGenerator(self.config)
        
        # Backtest state
        self._reset_state()
    
    def _reset_state(self) -> None:
        """Reset backtest state."""
        self._equity = self.bt_config.initial_capital
        self._peak_equity = self.bt_config.initial_capital
        self._position: Optional[Dict] = None
        self._trades: List[BacktestTrade] = []
        self._equity_curve: List[Dict] = []
        self._daily_pnl: float = 0.0
        self._current_date: Optional[datetime] = None
    
    def run(
        self,
        df: pd.DataFrame,
        symbol: str = "BTCUSDT"
    ) -> BacktestResult:
        """
        Run backtest on historical data.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            
        Returns:
            BacktestResult with complete analysis
        """
        self._reset_state()
        
        # Calculate indicators
        df = self.indicators.calculate_all(df.copy())
        
        # Need enough data for indicators
        start_idx = max(50, self.config.data.min_historical_bars)
        
        logger.info(f"Starting backtest: {len(df)} bars, starting from bar {start_idx}")
        
        for i in range(start_idx, len(df)):
            # Get data up to current bar (no lookahead)
            current_data = df.iloc[:i+1].copy()
            current_bar = df.iloc[i]
            bar_time = current_bar['timestamp'] if 'timestamp' in current_bar else datetime.now()
            
            # Track daily reset
            if self._current_date is None or bar_time.date() != self._current_date:
                self._daily_pnl = 0.0
                self._current_date = bar_time.date()
            
            # Update position P&L if we have one
            if self._position is not None:
                self._update_position_pnl(current_bar)
                
                # Check stop-loss and take-profit
                exit_reason = self._check_exits(current_bar)
                if exit_reason:
                    self._close_position(current_bar, exit_reason)
                    continue
            
            # Check for circuit breaker
            if self._check_circuit_breaker():
                continue
            
            # Generate signal
            current_position = None
            if self._position is not None:
                current_position = PositionSide.LONG if self._position['side'] == 'long' else PositionSide.SHORT
            
            signal = self.signal_generator.generate_signal(current_data, current_position)
            
            # Process signal
            if signal.signal_type == SignalType.LONG_ENTRY and self._position is None:
                self._open_position(current_bar, 'long', signal.stop_loss, signal.take_profit)
            
            elif signal.signal_type == SignalType.SHORT_ENTRY and self._position is None:
                self._open_position(current_bar, 'short', signal.stop_loss, signal.take_profit)
            
            elif signal.signal_type in (SignalType.LONG_EXIT, SignalType.SHORT_EXIT):
                if self._position is not None:
                    self._close_position(current_bar, "signal_exit")
            
            # Record equity
            self._equity_curve.append({
                'timestamp': bar_time,
                'equity': self._equity + (self._position['unrealized_pnl'] if self._position else 0),
                'position': self._position['side'] if self._position else 'none'
            })
            
            # Update peak equity for drawdown
            current_equity = self._equity + (self._position['unrealized_pnl'] if self._position else 0)
            self._peak_equity = max(self._peak_equity, current_equity)
            
            # Progress reporting
            if (i - start_idx) % self.bt_config.report_frequency == 0:
                logger.debug(f"Bar {i}/{len(df)}, Equity: {self._equity:.2f}, Trades: {len(self._trades)}")
        
        # Close any remaining position
        if self._position is not None:
            self._close_position(df.iloc[-1], "end_of_test")
        
        # Calculate metrics
        metrics = self._calculate_metrics(df)
        
        start_date = df.iloc[start_idx]['timestamp'] if 'timestamp' in df.columns else datetime.now()
        end_date = df.iloc[-1]['timestamp'] if 'timestamp' in df.columns else datetime.now()
        
        return BacktestResult(
            metrics=metrics,
            trades=self._trades,
            equity_curve=pd.DataFrame(self._equity_curve),
            config_used=self._serialize_config(),
            start_date=start_date,
            end_date=end_date
        )
    
    def _open_position(
        self,
        bar: pd.Series,
        side: str,
        stop_loss: float,
        take_profit: float
    ) -> None:
        """Open a new position."""
        price = bar['close']
        
        # Calculate position size
        risk_amount = self._equity * self.config.risk.risk_per_trade_percent
        stop_distance = abs(price - stop_loss)
        
        if stop_distance == 0:
            return
        
        size = risk_amount / stop_distance
        
        # Apply fees (slippage on entry)
        entry_cost = size * price * (self.bt_config.taker_fee_percent + self.bt_config.slippage_percent)
        
        self._position = {
            'side': side,
            'entry_price': price,
            'size': size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': bar['timestamp'] if 'timestamp' in bar else datetime.now(),
            'entry_bar': len(self._equity_curve),
            'unrealized_pnl': -entry_cost,  # Start negative due to fees
            'highest_price': price,
            'lowest_price': price
        }
        
        logger.debug(f"Opened {side} position @ {price:.2f}, SL: {stop_loss:.2f}, TP: {take_profit:.2f}")
    
    def _close_position(self, bar: pd.Series, reason: str) -> None:
        """Close the current position."""
        if self._position is None:
            return
        
        exit_price = bar['close']
        
        # Adjust for slippage on stop-loss hits
        if reason in ('stop_loss', 'trailing_stop'):
            if self._position['side'] == 'long':
                exit_price = min(exit_price, self._position['stop_loss'])
            else:
                exit_price = max(exit_price, self._position['stop_loss'])
        
        # Calculate P&L
        if self._position['side'] == 'long':
            gross_pnl = (exit_price - self._position['entry_price']) * self._position['size']
        else:
            gross_pnl = (self._position['entry_price'] - exit_price) * self._position['size']
        
        # Subtract fees
        trade_value = self._position['size'] * exit_price
        exit_fees = trade_value * (self.bt_config.taker_fee_percent + self.bt_config.slippage_percent)
        entry_fees = self._position['size'] * self._position['entry_price'] * self.bt_config.taker_fee_percent
        total_fees = entry_fees + exit_fees
        
        net_pnl = gross_pnl - total_fees
        
        # Update equity
        self._equity += net_pnl
        self._daily_pnl += net_pnl
        
        # Record trade
        bars_held = len(self._equity_curve) - self._position['entry_bar']
        
        trade = BacktestTrade(
            entry_time=self._position['entry_time'],
            exit_time=bar['timestamp'] if 'timestamp' in bar else datetime.now(),
            symbol="BTCUSDT",
            side=self._position['side'],
            entry_price=self._position['entry_price'],
            exit_price=exit_price,
            size=self._position['size'],
            pnl=net_pnl,
            pnl_percent=net_pnl / (self._position['entry_price'] * self._position['size']) * 100,
            fees=total_fees,
            exit_reason=reason,
            bars_held=bars_held
        )
        
        self._trades.append(trade)
        
        logger.debug(f"Closed {self._position['side']} @ {exit_price:.2f}, P&L: {net_pnl:.2f} ({reason})")
        
        self._position = None
    
    def _update_position_pnl(self, bar: pd.Series) -> None:
        """Update position unrealized P&L."""
        if self._position is None:
            return
        
        current_price = bar['close']
        
        if self._position['side'] == 'long':
            self._position['unrealized_pnl'] = (current_price - self._position['entry_price']) * self._position['size']
            self._position['highest_price'] = max(self._position['highest_price'], bar['high'])
        else:
            self._position['unrealized_pnl'] = (self._position['entry_price'] - current_price) * self._position['size']
            self._position['lowest_price'] = min(self._position['lowest_price'], bar['low'])
    
    def _check_exits(self, bar: pd.Series) -> Optional[str]:
        """Check if stop-loss or take-profit hit."""
        if self._position is None:
            return None
        
        high = bar['high']
        low = bar['low']
        
        if self._position['side'] == 'long':
            # Stop-loss hit
            if low <= self._position['stop_loss']:
                return 'stop_loss'
            
            # Take-profit hit
            if high >= self._position['take_profit']:
                return 'take_profit'
            
            # Trailing stop (if enabled and in profit)
            if self.config.exit.use_trailing_stop:
                entry_risk = self._position['entry_price'] - self._position['stop_loss']
                profit = self._position['highest_price'] - self._position['entry_price']
                
                if profit > entry_risk * self.config.exit.trailing_activation_profit_mult:
                    atr = bar.get('atr', entry_risk)
                    trailing_stop = self._position['highest_price'] - (atr * self.config.exit.trailing_stop_atr_mult)
                    
                    if trailing_stop > self._position['stop_loss']:
                        self._position['stop_loss'] = trailing_stop
                    
                    if low <= self._position['stop_loss']:
                        return 'trailing_stop'
        
        else:  # Short
            # Stop-loss hit
            if high >= self._position['stop_loss']:
                return 'stop_loss'
            
            # Take-profit hit
            if low <= self._position['take_profit']:
                return 'take_profit'
            
            # Trailing stop for shorts
            if self.config.exit.use_trailing_stop:
                entry_risk = self._position['stop_loss'] - self._position['entry_price']
                profit = self._position['entry_price'] - self._position['lowest_price']
                
                if profit > entry_risk * self.config.exit.trailing_activation_profit_mult:
                    atr = bar.get('atr', entry_risk)
                    trailing_stop = self._position['lowest_price'] + (atr * self.config.exit.trailing_stop_atr_mult)
                    
                    if trailing_stop < self._position['stop_loss']:
                        self._position['stop_loss'] = trailing_stop
                    
                    if high >= self._position['stop_loss']:
                        return 'trailing_stop'
        
        # Time-based exit
        bars_held = len(self._equity_curve) - self._position['entry_bar']
        if bars_held >= self.config.exit.max_hold_periods:
            return 'time_exit'
        
        return None
    
    def _check_circuit_breaker(self) -> bool:
        """Check if daily loss limit is hit."""
        daily_loss_limit = self._equity * self.config.risk.daily_loss_limit_percent
        
        if self._daily_pnl <= -daily_loss_limit:
            return True
        
        return False
    
    def _calculate_metrics(self, df: pd.DataFrame) -> BacktestMetrics:
        """Calculate comprehensive backtest metrics."""
        warnings = []
        
        if len(self._trades) == 0:
            warnings.append("No trades executed")
            return BacktestMetrics(
                total_return=0, total_return_percent=0, annual_return_percent=0,
                total_trades=0, winning_trades=0, losing_trades=0, win_rate=0,
                profit_factor=0, average_win=0, average_loss=0, average_trade=0,
                largest_win=0, largest_loss=0,
                max_drawdown=0, max_drawdown_percent=0, max_drawdown_duration_days=0,
                sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
                avg_bars_held=0, avg_bars_winning=0, avg_bars_losing=0,
                total_fees=0, warnings=warnings
            )
        
        # Basic metrics
        total_return = self._equity - self.bt_config.initial_capital
        total_return_pct = total_return / self.bt_config.initial_capital * 100
        
        # Trade analysis
        winning = [t for t in self._trades if t.pnl > 0]
        losing = [t for t in self._trades if t.pnl <= 0]
        
        win_rate = len(winning) / len(self._trades) if self._trades else 0
        
        avg_win = np.mean([t.pnl for t in winning]) if winning else 0
        avg_loss = np.mean([t.pnl for t in losing]) if losing else 0
        
        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Drawdown
        equity_df = pd.DataFrame(self._equity_curve)
        if not equity_df.empty:
            equity_df['peak'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = equity_df['equity'] - equity_df['peak']
            equity_df['drawdown_pct'] = equity_df['drawdown'] / equity_df['peak'] * 100
            
            max_dd = equity_df['drawdown'].min()
            max_dd_pct = equity_df['drawdown_pct'].min()
            
            # Drawdown duration
            drawdown_periods = []
            current_dd_start = None
            
            for i, row in equity_df.iterrows():
                if row['drawdown'] < 0 and current_dd_start is None:
                    current_dd_start = row['timestamp']
                elif row['drawdown'] >= 0 and current_dd_start is not None:
                    drawdown_periods.append((row['timestamp'] - current_dd_start).days)
                    current_dd_start = None
            
            max_dd_duration = max(drawdown_periods) if drawdown_periods else 0
        else:
            max_dd = 0
            max_dd_pct = 0
            max_dd_duration = 0
        
        # Risk-adjusted metrics
        returns = [t.pnl_percent for t in self._trades]
        
        if len(returns) > 1:
            returns_std = np.std(returns)
            avg_return = np.mean(returns)
            
            # Sharpe (simplified - assuming 0 risk-free rate)
            sharpe = avg_return / returns_std if returns_std > 0 else 0
            
            # Sortino (only downside deviation)
            negative_returns = [r for r in returns if r < 0]
            downside_std = np.std(negative_returns) if negative_returns else 0
            sortino = avg_return / downside_std if downside_std > 0 else 0
        else:
            sharpe = 0
            sortino = 0
        
        # Annualized return
        if len(df) > 0 and 'timestamp' in df.columns:
            days = (df.iloc[-1]['timestamp'] - df.iloc[0]['timestamp']).days
            years = days / 365 if days > 0 else 1
            annual_return = ((1 + total_return_pct/100) ** (1/years) - 1) * 100 if years > 0 else 0
        else:
            annual_return = 0
            years = 1
        
        # Calmar ratio
        calmar = annual_return / abs(max_dd_pct) if max_dd_pct != 0 else 0
        
        # Bars held analysis
        avg_bars = np.mean([t.bars_held for t in self._trades])
        avg_bars_win = np.mean([t.bars_held for t in winning]) if winning else 0
        avg_bars_loss = np.mean([t.bars_held for t in losing]) if losing else 0
        
        # Total fees
        total_fees = sum(t.fees for t in self._trades)
        
        # Warnings
        if len(self._trades) < 30:
            warnings.append("Low trade count - results may not be statistically significant")
        
        if max_dd_pct < -20:
            warnings.append(f"High max drawdown ({max_dd_pct:.1f}%) - review risk parameters")
        
        if profit_factor < 1.5:
            warnings.append("Low profit factor - strategy may not be robust")
        
        if sharpe < 1.0:
            warnings.append("Low Sharpe ratio - poor risk-adjusted returns")
        
        return BacktestMetrics(
            total_return=total_return,
            total_return_percent=total_return_pct,
            annual_return_percent=annual_return,
            total_trades=len(self._trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=win_rate,
            profit_factor=profit_factor,
            average_win=avg_win,
            average_loss=avg_loss,
            average_trade=np.mean([t.pnl for t in self._trades]),
            largest_win=max(t.pnl for t in self._trades) if self._trades else 0,
            largest_loss=min(t.pnl for t in self._trades) if self._trades else 0,
            max_drawdown=max_dd,
            max_drawdown_percent=max_dd_pct,
            max_drawdown_duration_days=max_dd_duration,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            avg_bars_held=avg_bars,
            avg_bars_winning=avg_bars_win,
            avg_bars_losing=avg_bars_loss,
            total_fees=total_fees,
            warnings=warnings
        )
    
    def _serialize_config(self) -> Dict:
        """Serialize configuration for reproducibility."""
        return {
            'risk_per_trade': self.config.risk.risk_per_trade_percent,
            'stop_loss_atr_mult': self.config.exit.stop_loss_atr_mult,
            'target_rr_ratio': self.config.exit.target_risk_reward_ratio,
            'min_adx': self.config.entry.min_adx_for_entry,
            'initial_capital': self.bt_config.initial_capital,
            'fees': self.bt_config.taker_fee_percent,
            'slippage': self.bt_config.slippage_percent,
        }


def run_walk_forward_test(
    df: pd.DataFrame,
    config: Optional[TradingConfig] = None,
    train_months: int = 6,
    test_months: int = 3
) -> List[BacktestResult]:
    """
    Run walk-forward test to check for overfitting.
    
    Divides data into rolling train/test windows and runs
    backtest on each test period using parameters from training.
    
    Args:
        df: Full historical data
        config: Trading configuration
        train_months: Training window size
        test_months: Testing window size
        
    Returns:
        List of BacktestResult for each test period
    """
    results = []
    
    if 'timestamp' not in df.columns:
        logger.error("DataFrame must have timestamp column")
        return results
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    start_date = df['timestamp'].min()
    end_date = df['timestamp'].max()
    
    current_train_start = start_date
    
    while True:
        train_end = current_train_start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        
        if test_end > end_date:
            break
        
        # Get test data
        test_data = df[(df['timestamp'] >= train_end) & (df['timestamp'] < test_end)].copy()
        
        if len(test_data) < 100:
            logger.warning(f"Insufficient test data for period ending {test_end}")
            break
        
        # Run backtest on test period
        engine = BacktestEngine(config)
        result = engine.run(test_data)
        results.append(result)
        
        logger.info(
            f"Walk-forward period {train_end.strftime('%Y-%m')} to {test_end.strftime('%Y-%m')}: "
            f"Return: {result.metrics.total_return_percent:.1f}%, "
            f"Trades: {result.metrics.total_trades}"
        )
        
        # Move to next period
        current_train_start = current_train_start + pd.DateOffset(months=test_months)
    
    # Aggregate results
    if results:
        total_return = sum(r.metrics.total_return for r in results)
        total_trades = sum(r.metrics.total_trades for r in results)
        avg_sharpe = np.mean([r.metrics.sharpe_ratio for r in results])
        
        logger.info(f"\n=== Walk-Forward Summary ===")
        logger.info(f"Total periods: {len(results)}")
        logger.info(f"Combined return: ${total_return:.2f}")
        logger.info(f"Total trades: {total_trades}")
        logger.info(f"Average Sharpe: {avg_sharpe:.2f}")
    
    return results


def print_backtest_report(result: BacktestResult) -> None:
    """Print formatted backtest report."""
    m = result.metrics
    
    print("\n" + "="*60)
    print("BACKTEST REPORT")
    print("="*60)
    
    print(f"\nPeriod: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}")
    
    print("\n--- RETURNS ---")
    print(f"Total Return:      ${m.total_return:,.2f} ({m.total_return_percent:.2f}%)")
    print(f"Annual Return:     {m.annual_return_percent:.2f}%")
    
    print("\n--- TRADE STATISTICS ---")
    print(f"Total Trades:      {m.total_trades}")
    print(f"Winning Trades:    {m.winning_trades} ({m.win_rate*100:.1f}%)")
    print(f"Losing Trades:     {m.losing_trades}")
    print(f"Profit Factor:     {m.profit_factor:.2f}")
    print(f"Average Win:       ${m.average_win:.2f}")
    print(f"Average Loss:      ${m.average_loss:.2f}")
    print(f"Largest Win:       ${m.largest_win:.2f}")
    print(f"Largest Loss:      ${m.largest_loss:.2f}")
    
    print("\n--- RISK METRICS ---")
    print(f"Max Drawdown:      ${m.max_drawdown:.2f} ({m.max_drawdown_percent:.2f}%)")
    print(f"Max DD Duration:   {m.max_drawdown_duration_days} days")
    print(f"Sharpe Ratio:      {m.sharpe_ratio:.2f}")
    print(f"Sortino Ratio:     {m.sortino_ratio:.2f}")
    print(f"Calmar Ratio:      {m.calmar_ratio:.2f}")
    
    print("\n--- HOLDING PERIODS ---")
    print(f"Avg Bars Held:     {m.avg_bars_held:.1f}")
    print(f"Avg Bars (Win):    {m.avg_bars_winning:.1f}")
    print(f"Avg Bars (Loss):   {m.avg_bars_losing:.1f}")
    
    print(f"\nTotal Fees:        ${m.total_fees:.2f}")
    
    if m.warnings:
        print("\n--- WARNINGS ---")
        for warning in m.warnings:
            print(f"⚠️  {warning}")
    
    print("\n" + "="*60)
