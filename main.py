#!/usr/bin/env python3
"""
Adaptive Multi-Factor Crypto Trading Bot

Main entry point for the trading system.

DISCLAIMER: This software is for educational purposes only.
Trading cryptocurrencies is risky and you can lose money.
Never trade with funds you cannot afford to lose.

Usage:
    # Testnet (default - ALWAYS start here)
    python main.py --symbol BTCUSDT --timeframe 1h
    
    # Paper trading (log trades without executing)
    python main.py --paper --symbol BTCUSDT --timeframe 1h
    
    # Live trading (DANGER - uses real funds)
    python main.py --live --symbol BTCUSDT --timeframe 1h
    
    # Backtest
    python main.py --backtest --symbol BTCUSDT --start 2023-01-01 --end 2024-01-01
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from config import TradingConfig, load_config_from_env, PositionSide
from data import DataFetcher
from core import (
    TechnicalIndicators,
    SignalGenerator,
    SignalType,
    OrderExecutor,
    detect_regime
)
from risk import RiskManager, Position
from backtest import BacktestEngine, print_backtest_report


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_bot.log')
    ]
)

logger = logging.getLogger(__name__)


class TradingBot:
    """
    Main trading bot orchestrator.
    
    Coordinates all components:
    - Data fetching
    - Signal generation
    - Risk management
    - Order execution
    """
    
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        config: Optional[TradingConfig] = None,
        paper_trading: bool = False
    ):
        """
        Initialize the trading bot.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            timeframe: Candle timeframe (e.g., "1h")
            config: Trading configuration
            paper_trading: If True, log trades without executing
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.config = config or load_config_from_env()
        self.paper_trading = paper_trading
        
        # Validate configuration
        warnings = self.config.validate()
        for warning in warnings:
            logger.warning(warning)
        
        # Get API credentials
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        if not api_key or not api_secret:
            raise ValueError("BINANCE_API_KEY and BINANCE_API_SECRET must be set")
        
        # Initialize components
        logger.info(f"Initializing bot for {symbol} on {timeframe}")
        logger.info(f"Testnet: {self.config.testnet}, Paper trading: {paper_trading}")
        
        self.data_fetcher = DataFetcher(
            api_key=api_key,
            api_secret=api_secret,
            testnet=self.config.testnet,
            data_config=self.config.data,
            execution_config=self.config.execution
        )
        
        self.indicators = TechnicalIndicators(self.config.indicators)
        self.signal_generator = SignalGenerator(self.config)
        self.risk_manager = RiskManager(self.config.risk)
        
        if not paper_trading:
            self.executor = OrderExecutor(
                self.data_fetcher.exchange,
                self.config.execution
            )
        else:
            self.executor = None
        
        # State tracking
        self._running = False
        self._current_position: Optional[PositionSide] = None
        self._last_signal_time: Optional[datetime] = None
    
    def start(self) -> None:
        """Start the trading bot main loop."""
        logger.info("="*50)
        logger.info("STARTING TRADING BOT")
        logger.info("="*50)
        
        # Test connection
        if not self.data_fetcher.test_connection():
            logger.error("Failed to connect to exchange")
            return
        
        logger.info("Exchange connection successful")
        
        # Get initial balance
        balance = self.data_fetcher.fetch_balance()
        usdt_balance = float(balance.get('USDT', {}).get('free', 0))
        logger.info(f"Available USDT balance: ${usdt_balance:,.2f}")
        
        self._running = True
        
        # Calculate sleep time based on timeframe
        sleep_seconds = self._get_sleep_seconds()
        
        logger.info(f"Entering main loop (checking every {sleep_seconds}s)")
        
        try:
            while self._running:
                try:
                    self._run_iteration()
                except Exception as e:
                    logger.error(f"Error in trading iteration: {e}", exc_info=True)
                
                # Wait until next candle
                time.sleep(sleep_seconds)
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            self.stop()
    
    def _run_iteration(self) -> None:
        """Run a single trading iteration."""
        logger.debug(f"Running iteration at {datetime.now()}")
        
        # Fetch latest data
        df = self.data_fetcher.fetch_ohlcv(
            self.symbol,
            self.timeframe,
            limit=self.config.data.min_historical_bars
        )
        
        if df.empty:
            logger.warning("No data received")
            return
        
        # Calculate indicators
        df = self.indicators.calculate_all(df)
        
        # Detect regime
        regime = detect_regime(df, self.config.indicators)
        logger.info(f"Market regime: {regime.current_regime.value} (confidence: {regime.confidence:.2f})")
        
        # Get current position
        self._update_current_position()
        
        # Generate signal
        signal = self.signal_generator.generate_signal(df, self._current_position)
        
        logger.info(f"Signal: {signal.signal_type.value}")
        if signal.reasons:
            for reason in signal.reasons:
                logger.debug(f"  - {reason}")
        
        # Process signal
        if signal.signal_type in (SignalType.LONG_ENTRY, SignalType.SHORT_ENTRY):
            self._process_entry_signal(signal, df)
        
        elif signal.signal_type in (SignalType.LONG_EXIT, SignalType.SHORT_EXIT):
            self._process_exit_signal(signal)
    
    def _process_entry_signal(self, signal, df) -> None:
        """Process an entry signal."""
        if self._current_position is not None:
            logger.debug("Already in position, skipping entry")
            return
        
        latest = df.iloc[-1]
        price = latest['close']
        
        # Get account balance
        balance = self.data_fetcher.fetch_balance()
        usdt_balance = float(balance.get('USDT', {}).get('free', 0))
        
        # Evaluate trade with risk manager
        risk_decision = self.risk_manager.evaluate_trade(
            account_balance=usdt_balance,
            entry_price=price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            symbol=self.symbol
        )
        
        if risk_decision.action.value == 'block':
            logger.warning(f"Trade blocked: {risk_decision.reason}")
            return
        
        for warning in risk_decision.warnings:
            logger.warning(f"Risk warning: {warning}")
        
        # Apply position size modifier from signal
        position_size = risk_decision.allowed_size * signal.position_size_modifier
        
        side = PositionSide.LONG if signal.signal_type == SignalType.LONG_ENTRY else PositionSide.SHORT
        
        logger.info(f"Entry signal confirmed: {side.value} {position_size:.6f} @ {price:.2f}")
        logger.info(f"  Stop-loss: {signal.stop_loss:.2f}")
        logger.info(f"  Take-profit: {signal.take_profit:.2f}")
        logger.info(f"  Confidence: {signal.confidence:.2f}")
        
        if self.paper_trading:
            logger.info("[PAPER] Would execute entry order")
            self._current_position = side
        elif self.executor:
            result = self.executor.execute_entry(
                symbol=self.symbol,
                side=side,
                size=position_size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            )
            
            if result.success:
                logger.info(f"Entry order executed: {result.order.exchange_id}")
                self._current_position = side
                
                # Register with risk manager
                position = Position(
                    symbol=self.symbol,
                    side=side,
                    entry_price=result.order.filled_price,
                    size=position_size,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    entry_time=datetime.now()
                )
                self.risk_manager.register_position(position)
            else:
                logger.error(f"Entry order failed: {result.message}")
    
    def _process_exit_signal(self, signal) -> None:
        """Process an exit signal."""
        if self._current_position is None:
            logger.debug("No position to exit")
            return
        
        logger.info(f"Exit signal for {self._current_position.value} position")
        logger.info(f"  Reason: {', '.join(signal.reasons)}")
        
        if self.paper_trading:
            logger.info("[PAPER] Would execute exit order")
            self._current_position = None
        elif self.executor:
            # Get position size from exchange
            positions = self.data_fetcher.fetch_positions(self.symbol)
            if not positions:
                logger.warning("No position found on exchange")
                self._current_position = None
                return
            
            size = abs(float(positions[0].get('contracts', 0)))
            
            result = self.executor.execute_exit(
                symbol=self.symbol,
                side=self._current_position,
                size=size
            )
            
            if result.success:
                logger.info(f"Exit order executed: {result.order.exchange_id}")
                
                # Calculate P&L and close in risk manager
                pnl = 0  # Would calculate from position
                self.risk_manager.close_position(
                    self.symbol,
                    result.order.filled_price,
                    pnl
                )
                
                self._current_position = None
            else:
                logger.error(f"Exit order failed: {result.message}")
    
    def _update_current_position(self) -> None:
        """Update current position from exchange."""
        try:
            positions = self.data_fetcher.fetch_positions(self.symbol)
            
            for pos in positions:
                contracts = float(pos.get('contracts', 0))
                if contracts > 0:
                    self._current_position = PositionSide.LONG
                    return
                elif contracts < 0:
                    self._current_position = PositionSide.SHORT
                    return
            
            self._current_position = None
            
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
    
    def _get_sleep_seconds(self) -> int:
        """Calculate sleep time based on timeframe."""
        tf = self.timeframe
        
        if tf.endswith('m'):
            minutes = int(tf[:-1])
            return minutes * 60
        elif tf.endswith('h'):
            hours = int(tf[:-1])
            return hours * 3600
        elif tf.endswith('d'):
            return 86400
        else:
            return 3600  # Default to 1 hour
    
    def stop(self) -> None:
        """Stop the trading bot."""
        logger.info("Stopping trading bot...")
        self._running = False
        
        if self.data_fetcher:
            self.data_fetcher.close()
        
        logger.info("Trading bot stopped")


def run_backtest(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    config: Optional[TradingConfig] = None
) -> None:
    """Run backtest on historical data."""
    config = config or TradingConfig()
    
    # Get API credentials
    api_key = os.getenv('BINANCE_API_KEY', 'dummy')
    api_secret = os.getenv('BINANCE_API_SECRET', 'dummy')
    
    logger.info(f"Running backtest: {symbol} {timeframe}")
    logger.info(f"Period: {start_date} to {end_date}")
    
    # Fetch historical data
    fetcher = DataFetcher(
        api_key=api_key,
        api_secret=api_secret,
        testnet=True  # Always use testnet for backtest data
    )
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    df = fetcher.fetch_historical_ohlcv(symbol, timeframe, start, end)
    fetcher.close()
    
    if df.empty:
        logger.error("No data fetched for backtest")
        return
    
    logger.info(f"Fetched {len(df)} candles")
    
    # Run backtest
    engine = BacktestEngine(config)
    result = engine.run(df, symbol)
    
    # Print report
    print_backtest_report(result)
    
    # Save trades to CSV
    if result.trades:
        import pandas as pd
        trades_df = pd.DataFrame([
            {
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'side': t.side,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'pnl': t.pnl,
                'exit_reason': t.exit_reason
            }
            for t in result.trades
        ])
        trades_df.to_csv('backtest_trades.csv', index=False)
        logger.info("Trades saved to backtest_trades.csv")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Adaptive Multi-Factor Crypto Trading Bot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Testnet trading (safe - uses fake money)
  python main.py --symbol BTCUSDT --timeframe 1h
  
  # Paper trading (logs trades without executing)
  python main.py --paper --symbol BTCUSDT --timeframe 1h
  
  # Live trading (DANGER - uses real money!)
  python main.py --live --symbol BTCUSDT --timeframe 1h
  
  # Backtest
  python main.py --backtest --symbol BTCUSDT --start 2023-01-01 --end 2024-01-01
        """
    )
    
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                        help='Trading symbol (default: BTCUSDT)')
    parser.add_argument('--timeframe', type=str, default='1h',
                        help='Candle timeframe (default: 1h)')
    
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--testnet', action='store_true', default=True,
                           help='Use testnet (default)')
    mode_group.add_argument('--paper', action='store_true',
                           help='Paper trading mode')
    mode_group.add_argument('--live', action='store_true',
                           help='Live trading (DANGER!)')
    mode_group.add_argument('--backtest', action='store_true',
                           help='Run backtest')
    
    parser.add_argument('--start', type=str,
                        help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str,
                        help='Backtest end date (YYYY-MM-DD)')
    
    parser.add_argument('--risk', type=float, default=0.01,
                        help='Risk per trade (default: 0.01 = 1%%)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config_from_env()
    config.risk.risk_per_trade_percent = args.risk
    
    if args.backtest:
        if not args.start or not args.end:
            parser.error("--backtest requires --start and --end dates")
        
        run_backtest(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=args.start,
            end_date=args.end,
            config=config
        )
    else:
        # Set testnet based on mode
        if args.live:
            config.testnet = False
            print("\n" + "!"*60)
            print("WARNING: LIVE TRADING MODE - REAL MONEY AT RISK!")
            print("!"*60)
            response = input("Are you sure? Type 'YES' to continue: ")
            if response != 'YES':
                print("Aborted.")
                return
        else:
            config.testnet = True
        
        # Create and start bot
        bot = TradingBot(
            symbol=args.symbol,
            timeframe=args.timeframe,
            config=config,
            paper_trading=args.paper
        )
        
        bot.start()


if __name__ == '__main__':
    main()
