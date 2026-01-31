"""
Data Fetching Module

Handles all data retrieval from Binance API including:
- Historical OHLCV data
- Real-time price updates
- Account information
- Position data

Uses CCXT library for standardized exchange interaction.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import time

import pandas as pd
import numpy as np
import ccxt

from config.settings import DataConfig, ExecutionConfig


logger = logging.getLogger(__name__)


@dataclass
class OHLCV:
    """Single OHLCV candle."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class DataFetcher:
    """
    Handles all data fetching operations from Binance Futures.
    
    Features:
    - Automatic reconnection on failure
    - Rate limiting compliance
    - Data validation and cleaning
    - Caching to reduce API calls
    """
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        data_config: Optional[DataConfig] = None,
        execution_config: Optional[ExecutionConfig] = None
    ):
        """
        Initialize the data fetcher.
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: If True, use testnet endpoint
            data_config: Data configuration settings
            execution_config: Execution configuration settings
        """
        self.testnet = testnet
        self.data_config = data_config or DataConfig()
        self.execution_config = execution_config or ExecutionConfig()
        
        # Initialize CCXT exchange
        exchange_config = {
            'apiKey': api_key,
            'secret': api_secret,
            'sandbox': testnet,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',  # Use USDT-M futures
                'adjustForTimeDifference': True,
            }
        }
        
        self.exchange = ccxt.binance(exchange_config)
        
        # Cache for market data
        self._markets_cache: Optional[Dict] = None
        self._markets_cache_time: Optional[datetime] = None
        
        # Rate limiting
        self._last_request_time: Dict[str, float] = {}
        self._min_request_interval = 0.1  # 100ms between requests
        
        logger.info(f"DataFetcher initialized (testnet={testnet})")
    
    def _rate_limit(self, endpoint: str = "default") -> None:
        """Implement rate limiting to avoid API bans."""
        current_time = time.time()
        last_time = self._last_request_time.get(endpoint, 0)
        
        elapsed = current_time - last_time
        if elapsed < self._min_request_interval:
            sleep_time = self._min_request_interval - elapsed
            time.sleep(sleep_time)
        
        self._last_request_time[endpoint] = time.time()
    
    def load_markets(self, force_refresh: bool = False) -> Dict:
        """
        Load market information with caching.
        
        Args:
            force_refresh: If True, bypass cache
            
        Returns:
            Dictionary of market information
        """
        cache_valid = (
            self._markets_cache is not None and
            self._markets_cache_time is not None and
            datetime.now() - self._markets_cache_time < timedelta(hours=1) and
            not force_refresh
        )
        
        if cache_valid:
            return self._markets_cache
        
        try:
            self._rate_limit("markets")
            self._markets_cache = self.exchange.load_markets()
            self._markets_cache_time = datetime.now()
            logger.info(f"Loaded {len(self._markets_cache)} markets")
            return self._markets_cache
        except Exception as e:
            logger.error(f"Failed to load markets: {e}")
            raise
    
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 200,
        since: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV (candlestick) data.
        
        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            timeframe: Candle timeframe (e.g., "1h", "4h", "1d")
            limit: Number of candles to fetch (max 1000)
            since: Start timestamp in milliseconds
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        # Normalize symbol format
        if "/" not in symbol:
            symbol = symbol.replace("USDT", "/USDT")
        
        try:
            self._rate_limit("ohlcv")
            
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=min(limit, 1000),
                since=since
            )
            
            if not ohlcv:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Ensure numeric types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Validate data quality
            self._validate_ohlcv(df, symbol)
            
            logger.debug(f"Fetched {len(df)} candles for {symbol}")
            return df
            
        except ccxt.NetworkError as e:
            logger.error(f"Network error fetching OHLCV: {e}")
            raise
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error fetching OHLCV: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching OHLCV: {e}")
            raise
    
    def fetch_historical_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch extended historical OHLCV data by paginating through API.
        
        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            start_date: Start of historical period
            end_date: End of historical period (default: now)
            
        Returns:
            DataFrame with complete historical data
        """
        if end_date is None:
            end_date = datetime.now()
        
        # Normalize symbol
        if "/" not in symbol:
            symbol = symbol.replace("USDT", "/USDT")
        
        all_data = []
        current_time = int(start_date.timestamp() * 1000)
        end_time = int(end_date.timestamp() * 1000)
        
        # Calculate timeframe in milliseconds for pagination
        tf_ms = self._timeframe_to_ms(timeframe)
        
        while current_time < end_time:
            try:
                self._rate_limit("ohlcv")
                
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=1000,
                    since=current_time
                )
                
                if not ohlcv:
                    break
                
                all_data.extend(ohlcv)
                
                # Move to next batch
                last_timestamp = ohlcv[-1][0]
                current_time = last_timestamp + tf_ms
                
                logger.debug(f"Fetched {len(ohlcv)} candles, total: {len(all_data)}")
                
                # Break if we've caught up to current time
                if len(ohlcv) < 1000:
                    break
                    
            except Exception as e:
                logger.error(f"Error fetching historical data: {e}")
                time.sleep(1)  # Wait before retry
                continue
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(
            all_data,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove duplicates and sort
        df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
        
        # Filter to requested date range
        df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
        
        logger.info(f"Fetched {len(df)} historical candles for {symbol}")
        return df
    
    def _timeframe_to_ms(self, timeframe: str) -> int:
        """Convert timeframe string to milliseconds."""
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        
        multipliers = {
            'm': 60 * 1000,
            'h': 60 * 60 * 1000,
            'd': 24 * 60 * 60 * 1000,
            'w': 7 * 24 * 60 * 60 * 1000,
        }
        
        return value * multipliers.get(unit, 60 * 60 * 1000)
    
    def _validate_ohlcv(self, df: pd.DataFrame, symbol: str) -> None:
        """Validate OHLCV data quality."""
        if df.empty:
            return
        
        # Check for missing values
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_pct > self.data_config.max_missing_candles_percent:
            logger.warning(f"{symbol}: {missing_pct:.2%} missing data")
        
        # Check for invalid prices
        if (df['low'] > df['high']).any():
            logger.warning(f"{symbol}: Found candles with low > high")
        
        if (df['close'] < 0).any() or (df['open'] < 0).any():
            logger.warning(f"{symbol}: Found negative prices")
        
        # Check for suspicious volume
        if (df['volume'] == 0).sum() / len(df) > 0.1:
            logger.warning(f"{symbol}: >10% of candles have zero volume")
    
    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current ticker information.
        
        Args:
            symbol: Trading pair
            
        Returns:
            Dictionary with bid, ask, last, etc.
        """
        if "/" not in symbol:
            symbol = symbol.replace("USDT", "/USDT")
        
        try:
            self._rate_limit("ticker")
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            raise
    
    def fetch_orderbook(self, symbol: str, limit: int = 10) -> Dict[str, Any]:
        """
        Fetch current order book.
        
        Args:
            symbol: Trading pair
            limit: Depth of order book
            
        Returns:
            Dictionary with bids and asks
        """
        if "/" not in symbol:
            symbol = symbol.replace("USDT", "/USDT")
        
        try:
            self._rate_limit("orderbook")
            orderbook = self.exchange.fetch_order_book(symbol, limit=limit)
            return orderbook
        except Exception as e:
            logger.error(f"Error fetching orderbook for {symbol}: {e}")
            raise
    
    def fetch_balance(self) -> Dict[str, Any]:
        """
        Fetch account balance.
        
        Returns:
            Dictionary with balance information
        """
        try:
            self._rate_limit("balance")
            balance = self.exchange.fetch_balance()
            return balance
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            raise
    
    def fetch_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetch open positions.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of position dictionaries
        """
        try:
            self._rate_limit("positions")
            positions = self.exchange.fetch_positions(symbols=[symbol] if symbol else None)
            
            # Filter to only non-zero positions
            active_positions = [
                p for p in positions 
                if float(p.get('contracts', 0)) != 0 or float(p.get('notional', 0)) != 0
            ]
            
            return active_positions
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            raise
    
    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetch open orders.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of open order dictionaries
        """
        if symbol and "/" not in symbol:
            symbol = symbol.replace("USDT", "/USDT")
        
        try:
            self._rate_limit("orders")
            orders = self.exchange.fetch_open_orders(symbol=symbol)
            return orders
        except Exception as e:
            logger.error(f"Error fetching open orders: {e}")
            raise
    
    def fetch_funding_rate(self, symbol: str) -> float:
        """
        Fetch current funding rate for perpetual futures.
        
        Args:
            symbol: Trading pair
            
        Returns:
            Current funding rate as decimal
        """
        if "/" not in symbol:
            symbol = symbol.replace("USDT", "/USDT")
        
        try:
            self._rate_limit("funding")
            funding = self.exchange.fetch_funding_rate(symbol)
            return funding.get('fundingRate', 0.0)
        except Exception as e:
            logger.error(f"Error fetching funding rate for {symbol}: {e}")
            return 0.0
    
    def get_min_order_size(self, symbol: str) -> Tuple[float, float]:
        """
        Get minimum order size and precision for a symbol.
        
        Args:
            symbol: Trading pair
            
        Returns:
            Tuple of (min_amount, amount_precision)
        """
        if "/" not in symbol:
            symbol = symbol.replace("USDT", "/USDT")
        
        markets = self.load_markets()
        
        if symbol not in markets:
            logger.warning(f"Symbol {symbol} not found in markets")
            return 0.001, 3  # Default values
        
        market = markets[symbol]
        min_amount = market.get('limits', {}).get('amount', {}).get('min', 0.001)
        precision = market.get('precision', {}).get('amount', 3)
        
        return float(min_amount), int(precision)
    
    def test_connection(self) -> bool:
        """
        Test API connection and authentication.
        
        Returns:
            True if connection successful
        """
        try:
            self._rate_limit("test")
            
            # Test public endpoint
            self.exchange.fetch_time()
            logger.info("Public API connection successful")
            
            # Test authenticated endpoint
            self.fetch_balance()
            logger.info("Authenticated API connection successful")
            
            return True
            
        except ccxt.AuthenticationError as e:
            logger.error(f"Authentication failed: {e}")
            return False
        except ccxt.NetworkError as e:
            logger.error(f"Network error: {e}")
            return False
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def close(self) -> None:
        """Clean up resources."""
        try:
            self.exchange.close()
            logger.info("Exchange connection closed")
        except Exception as e:
            logger.error(f"Error closing connection: {e}")
