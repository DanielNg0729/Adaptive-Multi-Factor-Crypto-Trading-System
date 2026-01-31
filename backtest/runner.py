#!/usr/bin/env python3
"""
Backtest Runner Script

Standalone script for running backtests with various options.

Usage:
    python -m backtest.runner --symbol BTCUSDT --start 2023-01-01 --end 2024-01-01
    python -m backtest.runner --symbol ETHUSDT --start 2023-01-01 --end 2024-01-01 --walk-forward
"""

import argparse
import logging
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from config import TradingConfig
from data import DataFetcher
from backtest import BacktestEngine, print_backtest_report, run_walk_forward_test


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Backtest Runner')
    
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                        help='Trading symbol')
    parser.add_argument('--timeframe', type=str, default='1h',
                        help='Candle timeframe')
    parser.add_argument('--start', type=str, required=True,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=10000,
                        help='Initial capital')
    parser.add_argument('--risk', type=float, default=0.01,
                        help='Risk per trade')
    parser.add_argument('--walk-forward', action='store_true',
                        help='Run walk-forward analysis')
    parser.add_argument('--output', type=str,
                        help='Output file for trades CSV')
    
    args = parser.parse_args()
    
    # Configure
    config = TradingConfig()
    config.risk.risk_per_trade_percent = args.risk
    config.backtest.initial_capital = args.capital
    
    # Fetch data
    api_key = os.getenv('BINANCE_API_KEY', 'dummy')
    api_secret = os.getenv('BINANCE_API_SECRET', 'dummy')
    
    logger.info(f"Fetching data for {args.symbol} {args.timeframe}")
    logger.info(f"Period: {args.start} to {args.end}")
    
    fetcher = DataFetcher(
        api_key=api_key,
        api_secret=api_secret,
        testnet=True
    )
    
    start = datetime.strptime(args.start, '%Y-%m-%d')
    end = datetime.strptime(args.end, '%Y-%m-%d')
    
    df = fetcher.fetch_historical_ohlcv(args.symbol, args.timeframe, start, end)
    fetcher.close()
    
    if df.empty:
        logger.error("No data fetched")
        return
    
    logger.info(f"Fetched {len(df)} candles")
    
    if args.walk_forward:
        logger.info("Running walk-forward analysis...")
        results = run_walk_forward_test(df, config)
        
        if results:
            print("\n=== WALK-FORWARD RESULTS ===")
            for i, result in enumerate(results):
                print(f"\nPeriod {i+1}:")
                print(f"  Return: {result.metrics.total_return_percent:.2f}%")
                print(f"  Trades: {result.metrics.total_trades}")
                print(f"  Win Rate: {result.metrics.win_rate*100:.1f}%")
                print(f"  Sharpe: {result.metrics.sharpe_ratio:.2f}")
    else:
        logger.info("Running backtest...")
        engine = BacktestEngine(config)
        result = engine.run(df, args.symbol)
        
        print_backtest_report(result)
        
        # Save trades
        output_file = args.output or f"backtest_{args.symbol}_{args.start}_{args.end}.csv"
        if result.trades:
            import pandas as pd
            trades_df = pd.DataFrame([
                {
                    'entry_time': t.entry_time,
                    'exit_time': t.exit_time,
                    'side': t.side,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'size': t.size,
                    'pnl': t.pnl,
                    'pnl_percent': t.pnl_percent,
                    'fees': t.fees,
                    'exit_reason': t.exit_reason,
                    'bars_held': t.bars_held
                }
                for t in result.trades
            ])
            trades_df.to_csv(output_file, index=False)
            logger.info(f"Trades saved to {output_file}")
        
        # Save equity curve
        if not result.equity_curve.empty:
            result.equity_curve.to_csv(f"equity_curve_{args.symbol}.csv", index=False)
            logger.info(f"Equity curve saved")


if __name__ == '__main__':
    main()
