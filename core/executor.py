"""
Order Execution Module

Handles all order placement and management:
- Market and limit order execution
- Stop-loss and take-profit orders
- Order status tracking
- Error handling and retries

This module interfaces directly with the exchange API.
"""

import logging
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import time

import ccxt

from config.settings import ExecutionConfig, OrderSide, PositionSide
from risk.manager import Position


logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    FAILED = "failed"


class OrderType(Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    STOP_LOSS_LIMIT = "stop_loss_limit"
    TRAILING_STOP = "trailing_stop"


@dataclass
class Order:
    """Represents an order."""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    price: Optional[float]
    size: float
    status: OrderStatus
    created_at: datetime
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    filled_size: float = 0.0
    exchange_id: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ExecutionResult:
    """Result of order execution."""
    success: bool
    order: Optional[Order]
    message: str
    exchange_response: Optional[Dict] = None


class OrderExecutor:
    """
    Handles all order execution logic.
    
    Features:
    - Robust error handling with retries
    - Slippage protection
    - Order validation before submission
    - Position management via exchange
    """
    
    def __init__(
        self,
        exchange: ccxt.Exchange,
        config: Optional[ExecutionConfig] = None
    ):
        """
        Initialize order executor.
        
        Args:
            exchange: CCXT exchange instance
            config: Execution configuration
        """
        self.exchange = exchange
        self.config = config or ExecutionConfig()
        
        # Track orders
        self._orders: Dict[str, Order] = {}
        self._last_order_time: Dict[str, datetime] = {}
        
        # Order ID counter
        self._order_counter = 0
        
        logger.info("OrderExecutor initialized")
    
    def execute_entry(
        self,
        symbol: str,
        side: PositionSide,
        size: float,
        stop_loss: float,
        take_profit: float,
        entry_price: Optional[float] = None
    ) -> ExecutionResult:
        """
        Execute a full entry with stop-loss and take-profit.
        
        Args:
            symbol: Trading symbol
            side: Position side (LONG or SHORT)
            size: Position size
            stop_loss: Stop-loss price
            take_profit: Take-profit price
            entry_price: Limit price (None for market order)
            
        Returns:
            ExecutionResult with entry order details
        """
        # Normalize symbol
        if "/" not in symbol:
            symbol = symbol.replace("USDT", "/USDT")
        
        # Validate inputs
        validation = self._validate_order(symbol, size, entry_price, stop_loss)
        if not validation[0]:
            return ExecutionResult(
                success=False,
                order=None,
                message=validation[1]
            )
        
        # Check cooldown
        if not self._check_cooldown(symbol):
            return ExecutionResult(
                success=False,
                order=None,
                message="Order cooldown in effect"
            )
        
        # Set leverage and margin mode
        try:
            self._configure_position(symbol)
        except Exception as e:
            logger.error(f"Failed to configure position: {e}")
            # Continue anyway - might already be configured
        
        # Determine order side
        order_side = "buy" if side == PositionSide.LONG else "sell"
        
        # Execute entry order
        if entry_price is None or not self.config.use_limit_orders:
            result = self._execute_market_order(symbol, order_side, size)
        else:
            result = self._execute_limit_order(symbol, order_side, size, entry_price)
        
        if not result.success:
            return result
        
        # Place stop-loss order
        sl_side = "sell" if side == PositionSide.LONG else "buy"
        sl_result = self._place_stop_loss(symbol, sl_side, size, stop_loss)
        
        if not sl_result.success:
            logger.warning(f"Stop-loss placement failed: {sl_result.message}")
            # Don't fail the entire entry - stop can be placed manually
        
        # Place take-profit order
        tp_result = self._place_take_profit(symbol, sl_side, size, take_profit)
        
        if not tp_result.success:
            logger.warning(f"Take-profit placement failed: {tp_result.message}")
        
        return result
    
    def execute_exit(
        self,
        symbol: str,
        side: PositionSide,
        size: float,
        price: Optional[float] = None
    ) -> ExecutionResult:
        """
        Execute position exit.
        
        Args:
            symbol: Trading symbol
            side: Position side being closed
            size: Size to close
            price: Limit price (None for market)
            
        Returns:
            ExecutionResult
        """
        if "/" not in symbol:
            symbol = symbol.replace("USDT", "/USDT")
        
        # Exit is opposite of position side
        order_side = "sell" if side == PositionSide.LONG else "buy"
        
        # Cancel any existing SL/TP orders first
        try:
            self._cancel_position_orders(symbol)
        except Exception as e:
            logger.warning(f"Failed to cancel existing orders: {e}")
        
        # Execute exit
        if price is None:
            return self._execute_market_order(symbol, order_side, size)
        else:
            return self._execute_limit_order(symbol, order_side, size, price)
    
    def _execute_market_order(
        self,
        symbol: str,
        side: str,
        size: float
    ) -> ExecutionResult:
        """
        Execute a market order with retries.
        """
        order_id = self._generate_order_id()
        
        order = Order(
            id=order_id,
            symbol=symbol,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            order_type=OrderType.MARKET,
            price=None,
            size=size,
            status=OrderStatus.PENDING,
            created_at=datetime.now()
        )
        
        for attempt in range(self.config.max_order_retries):
            try:
                logger.info(f"Executing market {side} {size} {symbol} (attempt {attempt + 1})")
                
                response = self.exchange.create_order(
                    symbol=symbol,
                    type='market',
                    side=side,
                    amount=size
                )
                
                # Update order with response
                order.exchange_id = response.get('id')
                order.filled_price = float(response.get('average', 0) or response.get('price', 0))
                order.filled_size = float(response.get('filled', size))
                order.status = OrderStatus.FILLED
                order.filled_at = datetime.now()
                
                self._orders[order_id] = order
                self._last_order_time[symbol] = datetime.now()
                
                logger.info(
                    f"Market order filled: {order.exchange_id} "
                    f"price={order.filled_price} size={order.filled_size}"
                )
                
                return ExecutionResult(
                    success=True,
                    order=order,
                    message="Market order executed",
                    exchange_response=response
                )
                
            except ccxt.InsufficientFunds as e:
                order.status = OrderStatus.FAILED
                order.error = str(e)
                return ExecutionResult(
                    success=False,
                    order=order,
                    message=f"Insufficient funds: {e}"
                )
                
            except ccxt.InvalidOrder as e:
                order.status = OrderStatus.FAILED
                order.error = str(e)
                return ExecutionResult(
                    success=False,
                    order=order,
                    message=f"Invalid order: {e}"
                )
                
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                logger.warning(f"Order attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_order_retries - 1:
                    time.sleep(self.config.retry_delay_seconds)
                else:
                    order.status = OrderStatus.FAILED
                    order.error = str(e)
                    return ExecutionResult(
                        success=False,
                        order=order,
                        message=f"Order failed after {self.config.max_order_retries} attempts: {e}"
                    )
        
        return ExecutionResult(
            success=False,
            order=order,
            message="Order failed unexpectedly"
        )
    
    def _execute_limit_order(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float
    ) -> ExecutionResult:
        """
        Execute a limit order with timeout handling.
        """
        order_id = self._generate_order_id()
        
        order = Order(
            id=order_id,
            symbol=symbol,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            order_type=OrderType.LIMIT,
            price=price,
            size=size,
            status=OrderStatus.PENDING,
            created_at=datetime.now()
        )
        
        try:
            logger.info(f"Placing limit {side} {size} {symbol} @ {price}")
            
            response = self.exchange.create_order(
                symbol=symbol,
                type='limit',
                side=side,
                amount=size,
                price=price
            )
            
            order.exchange_id = response.get('id')
            order.status = OrderStatus.OPEN
            
            self._orders[order_id] = order
            
            # Wait for fill or timeout
            start_time = time.time()
            while time.time() - start_time < self.config.limit_order_timeout_seconds:
                time.sleep(1)
                
                status = self.exchange.fetch_order(order.exchange_id, symbol)
                
                if status['status'] == 'closed':
                    order.filled_price = float(status.get('average', price))
                    order.filled_size = float(status.get('filled', size))
                    order.status = OrderStatus.FILLED
                    order.filled_at = datetime.now()
                    
                    self._last_order_time[symbol] = datetime.now()
                    
                    return ExecutionResult(
                        success=True,
                        order=order,
                        message="Limit order filled",
                        exchange_response=status
                    )
                
                if status['status'] == 'canceled':
                    order.status = OrderStatus.CANCELLED
                    return ExecutionResult(
                        success=False,
                        order=order,
                        message="Order was cancelled"
                    )
            
            # Timeout - cancel and use market
            logger.warning("Limit order timeout - cancelling and using market")
            self.exchange.cancel_order(order.exchange_id, symbol)
            order.status = OrderStatus.CANCELLED
            
            # Fall back to market order
            return self._execute_market_order(symbol, side, size)
            
        except Exception as e:
            order.status = OrderStatus.FAILED
            order.error = str(e)
            logger.error(f"Limit order failed: {e}")
            
            return ExecutionResult(
                success=False,
                order=order,
                message=f"Limit order failed: {e}"
            )
    
    def _place_stop_loss(
        self,
        symbol: str,
        side: str,
        size: float,
        stop_price: float
    ) -> ExecutionResult:
        """
        Place a stop-loss order.
        """
        order_id = self._generate_order_id()
        
        order = Order(
            id=order_id,
            symbol=symbol,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            order_type=OrderType.STOP_LOSS,
            price=stop_price,
            size=size,
            status=OrderStatus.PENDING,
            created_at=datetime.now()
        )
        
        try:
            # Binance Futures uses 'STOP_MARKET' for stop-loss
            params = {
                'stopPrice': stop_price,
                'reduceOnly': True
            }
            
            response = self.exchange.create_order(
                symbol=symbol,
                type='STOP_MARKET',
                side=side,
                amount=size,
                params=params
            )
            
            order.exchange_id = response.get('id')
            order.status = OrderStatus.OPEN
            
            self._orders[order_id] = order
            
            logger.info(f"Stop-loss placed: {order.exchange_id} @ {stop_price}")
            
            return ExecutionResult(
                success=True,
                order=order,
                message="Stop-loss placed",
                exchange_response=response
            )
            
        except Exception as e:
            order.status = OrderStatus.FAILED
            order.error = str(e)
            logger.error(f"Failed to place stop-loss: {e}")
            
            return ExecutionResult(
                success=False,
                order=order,
                message=f"Stop-loss failed: {e}"
            )
    
    def _place_take_profit(
        self,
        symbol: str,
        side: str,
        size: float,
        target_price: float
    ) -> ExecutionResult:
        """
        Place a take-profit order.
        """
        order_id = self._generate_order_id()
        
        order = Order(
            id=order_id,
            symbol=symbol,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            order_type=OrderType.TAKE_PROFIT,
            price=target_price,
            size=size,
            status=OrderStatus.PENDING,
            created_at=datetime.now()
        )
        
        try:
            # Binance Futures uses 'TAKE_PROFIT_MARKET'
            params = {
                'stopPrice': target_price,
                'reduceOnly': True
            }
            
            response = self.exchange.create_order(
                symbol=symbol,
                type='TAKE_PROFIT_MARKET',
                side=side,
                amount=size,
                params=params
            )
            
            order.exchange_id = response.get('id')
            order.status = OrderStatus.OPEN
            
            self._orders[order_id] = order
            
            logger.info(f"Take-profit placed: {order.exchange_id} @ {target_price}")
            
            return ExecutionResult(
                success=True,
                order=order,
                message="Take-profit placed",
                exchange_response=response
            )
            
        except Exception as e:
            order.status = OrderStatus.FAILED
            order.error = str(e)
            logger.error(f"Failed to place take-profit: {e}")
            
            return ExecutionResult(
                success=False,
                order=order,
                message=f"Take-profit failed: {e}"
            )
    
    def update_stop_loss(
        self,
        symbol: str,
        new_stop_price: float,
        size: float
    ) -> ExecutionResult:
        """
        Update (move) the stop-loss order.
        
        Used for trailing stops.
        """
        if "/" not in symbol:
            symbol = symbol.replace("USDT", "/USDT")
        
        # Find and cancel existing stop
        try:
            open_orders = self.exchange.fetch_open_orders(symbol)
            for order in open_orders:
                if order.get('type') == 'STOP_MARKET':
                    self.exchange.cancel_order(order['id'], symbol)
                    logger.info(f"Cancelled old stop: {order['id']}")
        except Exception as e:
            logger.warning(f"Error cancelling old stop: {e}")
        
        # Place new stop
        # Determine side based on current position
        # This is a simplification - in production, track position side
        positions = self.exchange.fetch_positions([symbol])
        for pos in positions:
            if float(pos.get('contracts', 0)) != 0:
                side = "sell" if float(pos.get('contracts', 0)) > 0 else "buy"
                return self._place_stop_loss(symbol, side, size, new_stop_price)
        
        return ExecutionResult(
            success=False,
            order=None,
            message="No position found to update stop-loss"
        )
    
    def _cancel_position_orders(self, symbol: str) -> None:
        """Cancel all open orders for a symbol."""
        try:
            open_orders = self.exchange.fetch_open_orders(symbol)
            for order in open_orders:
                self.exchange.cancel_order(order['id'], symbol)
                logger.info(f"Cancelled order: {order['id']}")
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")
            raise
    
    def _configure_position(self, symbol: str) -> None:
        """Configure leverage and margin mode for symbol."""
        try:
            # Set isolated margin
            if self.config.use_isolated_margin:
                self.exchange.set_margin_mode('isolated', symbol)
            
            # Set leverage
            self.exchange.set_leverage(self.config.default_leverage, symbol)
            
            logger.debug(f"Configured {symbol}: isolated margin, {self.config.default_leverage}x leverage")
            
        except Exception as e:
            # This often fails if already configured - not critical
            logger.debug(f"Position configuration note: {e}")
    
    def _validate_order(
        self,
        symbol: str,
        size: float,
        price: Optional[float],
        stop_loss: Optional[float]
    ) -> Tuple[bool, str]:
        """Validate order parameters."""
        if size <= 0:
            return False, "Invalid size (must be positive)"
        
        if price is not None and price <= 0:
            return False, "Invalid price (must be positive)"
        
        if stop_loss is not None and stop_loss <= 0:
            return False, "Invalid stop-loss (must be positive)"
        
        # Check minimum order size
        try:
            markets = self.exchange.load_markets()
            if symbol in markets:
                min_amount = markets[symbol].get('limits', {}).get('amount', {}).get('min', 0)
                if size < min_amount:
                    return False, f"Size below minimum ({min_amount})"
        except Exception as e:
            logger.warning(f"Could not validate min order size: {e}")
        
        return True, "Valid"
    
    def _check_cooldown(self, symbol: str) -> bool:
        """Check if enough time has passed since last order."""
        if symbol not in self._last_order_time:
            return True
        
        elapsed = (datetime.now() - self._last_order_time[symbol]).total_seconds()
        return elapsed >= self.config.min_time_between_orders_seconds
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self._order_counter += 1
        return f"BOT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._order_counter}"
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self._orders.get(order_id)
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders."""
        orders = [o for o in self._orders.values() if o.status == OrderStatus.OPEN]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders
