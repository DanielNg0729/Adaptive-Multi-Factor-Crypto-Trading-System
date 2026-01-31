# Adaptive Multi-Factor Crypto Trading System

## ⚠️ CRITICAL DISCLAIMER

**THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSES ONLY.**

- **No guarantee of profits.** Trading cryptocurrencies is inherently risky, and most retail traders lose money.
- **Past performance does not predict future results.** Backtested strategies often fail in live markets.
- **You can lose your entire investment.** Never trade with money you cannot afford to lose.
- **This is not financial advice.** Consult a licensed financial advisor before trading.
- **Test extensively on paper trading before using real funds.**

---

## 1. Strategy Overview (Plain English)

This trading system is designed around the principle of **regime-adaptive trading** combined with **multi-factor confirmation**. Rather than blindly following indicator signals, we first identify the current market regime (trending, ranging, or high-volatility), then apply indicator logic appropriate to that regime.

### Core Philosophy

1. **Capital Preservation First**: We'd rather miss profitable trades than enter losing ones. The system is designed to be conservative.

2. **Regime Awareness**: Markets behave differently in trends vs. ranges. A moving average crossover works in trends but fails in choppy markets. Our system detects regime first.

3. **Multi-Factor Confirmation**: We require agreement across trend, momentum, volatility, and volume indicators before entering. This reduces false signals but may miss some opportunities.

4. **Dynamic Risk Management**: Position sizing and stop-losses adapt to current volatility using ATR (Average True Range).

5. **No Revenge Trading**: Daily loss limits trigger circuit breakers. If we hit our daily max loss, trading stops for the day.

### When We Trade (and When We Don't)

**We ENTER long positions when:**
- Market is in a confirmed UPTREND (ADX > 25, price above EMAs)
- Momentum confirms strength (RSI recovering from oversold OR positive MACD histogram)
- Volatility is not extreme (ATR within normal range)
- Volume confirms participation (above average volume)
- Price is not overextended (within 2 ATR of recent support)

**We ENTER short positions when:**
- Market is in a confirmed DOWNTREND (ADX > 25, price below EMAs)
- Momentum confirms weakness (RSI falling from overbought OR negative MACD histogram)
- Volatility is not extreme
- Volume confirms participation
- Price is not overextended

**We STAY OUT when:**
- ADX < 20 (no clear trend - choppy market)
- Conflicting signals between indicators
- Extreme volatility (ATR > 2x normal)
- Low volume periods (typically 0:00-08:00 UTC for crypto)
- Daily loss limit reached
- Maximum concurrent positions reached

---

## 2. Mathematical Logic Behind Indicators

### 2.1 Trend Indicators

#### Exponential Moving Averages (EMA)

EMAs give more weight to recent prices, making them more responsive than simple moving averages.

**Formula:**
```
EMA_today = (Price_today × k) + (EMA_yesterday × (1 - k))
where k = 2 / (N + 1)
N = number of periods
```

**Our Implementation:**
- EMA(9): Fast EMA - captures short-term momentum
- EMA(21): Medium EMA - intermediate trend
- EMA(50): Slow EMA - overall trend direction

**Signal Logic:**
- Bullish: EMA(9) > EMA(21) > EMA(50) AND price > EMA(9)
- Bearish: EMA(9) < EMA(21) < EMA(50) AND price < EMA(9)
- Neutral: EMAs not properly aligned

#### Average Directional Index (ADX)

ADX measures trend STRENGTH (not direction). It ranges from 0 to 100.

**Components:**
- +DI (Positive Directional Indicator): Measures upward movement
- -DI (Negative Directional Indicator): Measures downward movement
- ADX: Smoothed average of |+DI - -DI| / (+DI + -DI)

**Our Interpretation:**
- ADX < 20: No clear trend (ranging market) → AVOID trend-following strategies
- ADX 20-25: Trend possibly emerging → Prepare but don't act
- ADX 25-40: Strong trend → Trade with trend
- ADX 40-50: Very strong trend → Tight trailing stops
- ADX > 50: Extremely strong trend → Watch for exhaustion

### 2.2 Momentum Indicators

#### Relative Strength Index (RSI)

RSI measures the speed and magnitude of recent price changes.

**Formula:**
```
RSI = 100 - (100 / (1 + RS))
where RS = Average Gain / Average Loss over N periods
```

**Our Implementation (14-period RSI):**

| RSI Range | Interpretation | Action |
|-----------|---------------|--------|
| < 30 | Oversold | Look for long entries in uptrends |
| 30-40 | Recovering | Confirmation for long in uptrend |
| 40-60 | Neutral | No signal |
| 60-70 | Strong momentum | Confirmation for existing positions |
| > 70 | Overbought | Look for short entries in downtrends |

**Critical Note:** In strong uptrends, RSI can stay overbought for extended periods. We use RSI as confirmation, NOT as a reversal signal.

#### MACD (Moving Average Convergence Divergence)

MACD shows the relationship between two EMAs.

**Formula:**
```
MACD Line = EMA(12) - EMA(26)
Signal Line = EMA(9) of MACD Line
Histogram = MACD Line - Signal Line
```

**Our Signal Logic:**
- Bullish: MACD line above signal line AND histogram increasing
- Bearish: MACD line below signal line AND histogram decreasing
- Divergence: Price makes new high/low but MACD doesn't → potential reversal warning

### 2.3 Volatility Indicators

#### Average True Range (ATR)

ATR measures market volatility using the true range of price movement.

**Formula:**
```
True Range = max(High - Low, |High - Previous Close|, |Low - Previous Close|)
ATR = EMA(True Range, N periods)
```

**Our Uses:**
1. **Stop-Loss Placement**: Stop = Entry ± (ATR × multiplier)
2. **Position Sizing**: Larger ATR = smaller position
3. **Regime Detection**: ATR > 2× 50-period average = high volatility regime
4. **Take-Profit Targets**: TP = Entry + (ATR × risk_reward_ratio × multiplier)

**Default ATR multiplier:** 2.0 (gives room for normal price noise)

#### Bollinger Bands

Bollinger Bands measure volatility using standard deviation.

**Formula:**
```
Middle Band = SMA(20)
Upper Band = Middle Band + (2 × Standard Deviation)
Lower Band = Middle Band - (2 × Standard Deviation)
```

**Our Uses:**
- Band Width: Narrow bands = low volatility, potential breakout coming
- Mean Reversion: In ranging markets, fade moves to outer bands
- Trend Confirmation: In trending markets, price "walking the bands" confirms trend

### 2.4 Volume Indicators

#### Volume Moving Average

Simple moving average of volume over N periods.

**Our Logic:**
- Current Volume > 1.5 × Volume MA(20) = High volume confirmation
- Current Volume < 0.5 × Volume MA(20) = Low participation, avoid entries

#### On-Balance Volume (OBV)

Cumulative volume indicator that relates volume to price change.

**Formula:**
```
If Close > Previous Close: OBV = Previous OBV + Volume
If Close < Previous Close: OBV = Previous OBV - Volume
If Close = Previous Close: OBV = Previous OBV
```

**Our Use:**
- OBV trend should confirm price trend
- Divergence between price and OBV = warning signal

---

## 3. Market Regime Detection

Our system identifies three primary regimes:

### TRENDING Regime
**Characteristics:**
- ADX > 25
- EMAs properly aligned (9 > 21 > 50 for uptrend)
- Price consistently making higher highs/lows (uptrend) or lower highs/lows (downtrend)
- Bollinger Band width expanding or stable

**Trading Approach:**
- Follow the trend
- Use trend indicators for entries
- Use momentum for confirmation
- Wider stops (2-3 ATR)
- Trail stops on winning trades

### RANGING Regime
**Characteristics:**
- ADX < 20
- EMAs flat and intertwined
- Price bouncing between support and resistance
- Bollinger Band width contracted

**Trading Approach:**
- **Mostly stay out** - ranging markets are difficult
- If trading: fade extremes (buy near support, sell near resistance)
- Tighter stops (1-1.5 ATR)
- Quick profits, no trailing

### HIGH VOLATILITY Regime
**Characteristics:**
- ATR > 2× 50-period average
- Large candles (body > 2× average)
- Wide Bollinger Bands expanding
- Often accompanies news events

**Trading Approach:**
- **Reduce position size by 50%**
- Wider stops required
- Avoid entries during news events
- Wait for volatility to normalize

---

## 4. Entry Logic (Detailed Conditions)

### Long Entry Conditions (ALL must be true)

```
1. TREND CONFIRMATION
   - ADX(14) > 25 (strong trend present)
   - EMA(9) > EMA(21) > EMA(50) (aligned for uptrend)
   - Current price > EMA(9) (above fast MA)

2. MOMENTUM CONFIRMATION (at least ONE)
   - RSI(14) < 40 AND RSI rising (recovering from oversold)
   - RSI(14) between 40-60 AND MACD histogram > 0 and increasing
   - RSI(14) > 50 AND MACD line just crossed above signal line

3. VOLATILITY FILTER
   - ATR(14) < 2 × ATR_SMA(50) (not extremely volatile)
   - Price within 2 ATR of EMA(21) (not overextended)

4. VOLUME CONFIRMATION
   - Current volume > Volume_SMA(20) (above average participation)
   - OBV trend direction matches price trend (no divergence)

5. RISK FILTERS
   - Not in "chop zone" (ADX < 20 in past 3 bars)
   - Current drawdown < 50% of daily loss limit
   - Position count < max concurrent positions
   - Not within 1 hour of major news events (if news filter enabled)
```

### Short Entry Conditions (ALL must be true)

Mirror of long conditions with inverted signals:
- EMA(9) < EMA(21) < EMA(50)
- Price < EMA(9)
- RSI recovering from overbought OR below 50 with negative MACD histogram
- Same volatility, volume, and risk filters

### Avoiding Late Entries

We implement "freshness" checks:
- Signal must be within 3 bars of trigger
- Don't enter if price already moved > 1.5 ATR from signal candle
- Don't chase after large moves

### Avoiding Chop Zones

- ADX must be > 25 for at least 3 consecutive bars
- Skip first 2 signals after regime change
- Volume must confirm (prevents false breakouts in low-liquidity)

---

## 5. Exit Logic

### Stop-Loss (Volatility-Based)

```
Long Stop = Entry Price - (ATR(14) × Stop_Multiplier)
Short Stop = Entry Price + (ATR(14) × Stop_Multiplier)

Default Stop_Multiplier = 2.0
```

**Rationale:** Fixed percentage stops don't account for varying volatility. A 2% stop might be too tight in volatile conditions and too loose in calm markets.

### Take-Profit (Risk-Reward Based)

```
Risk = |Entry Price - Stop Loss|
Minimum Risk-Reward = 2:1

Long TP = Entry Price + (Risk × Risk_Reward_Ratio)
Short TP = Entry Price - (Risk × Risk_Reward_Ratio)

Default Risk_Reward_Ratio = 2.5
```

### Trailing Stop (Optional)

Activated after position is in profit by 1.5× Risk:

```
Trailing Stop = Highest High since entry - (ATR × Trail_Multiplier)

Default Trail_Multiplier = 1.5 (tighter than initial stop)
```

**When Trailing Helps:**
- Strong trends where price continues beyond TP
- Captures extended moves

**When Trailing Hurts:**
- Choppy conditions (gets stopped out repeatedly)
- Markets that spike then reverse

**Our Implementation:** Trailing stops only activate in STRONG TRENDING regimes (ADX > 35).

### Time-Based Exit

If position doesn't hit TP or SL within `max_hold_periods` (default: 50 bars), exit at market. This prevents capital from being tied up in stagnant positions.

---

## 6. Risk Management

### Position Sizing Formula

```python
def calculate_position_size(account_balance, risk_per_trade, entry_price, stop_loss):
    """
    Kelly-fraction-inspired position sizing based on dollar risk.
    
    Args:
        account_balance: Total account value in quote currency (e.g., USDT)
        risk_per_trade: Maximum % of account to risk (e.g., 0.01 for 1%)
        entry_price: Planned entry price
        stop_loss: Stop-loss price
    
    Returns:
        Position size in base currency
    """
    # Dollar amount we're willing to lose
    risk_amount = account_balance * risk_per_trade
    
    # Price distance to stop
    stop_distance = abs(entry_price - stop_loss)
    
    # Position size that risks exactly risk_amount if stop is hit
    position_size = risk_amount / stop_distance
    
    return position_size
```

**Example:**
- Account: $10,000 USDT
- Risk per trade: 1% ($100)
- Entry: $50,000 (BTC)
- Stop: $48,000 (2 ATR below)
- Stop distance: $2,000

Position size = $100 / $2,000 = 0.05 BTC

If stop is hit, we lose exactly $100 (1% of account).

### Maximum Concurrent Positions

Default: 3 positions

**Rationale:** Diversification across multiple setups while maintaining concentrated risk management.

### Daily Loss Limit (Circuit Breaker)

```
max_daily_loss = account_balance × daily_loss_limit_percent

If realized_loss_today >= max_daily_loss:
    STOP TRADING FOR THE DAY
```

Default: 3% daily loss limit

**Implementation:**
- Tracks all closed trades for the day
- Includes any open position unrealized P&L
- Resets at 00:00 UTC

### Correlation Check

Before opening a new position, check correlation with existing positions:
- If new symbol correlation > 0.7 with existing position, reduce size by 50%
- Prevents over-concentration in correlated assets

---

## 7. API Selection: Binance Futures (USDT-M)

We use **Binance USDT-M Futures** for several reasons:

1. **Leverage Control**: Can use low leverage (1-3x) to efficiently use capital
2. **Short Selling**: Native short positions without borrowing
3. **Liquidity**: Deep order books on major pairs
4. **Lower Fees**: 0.02% maker / 0.04% taker (with BNB discount)
5. **Isolated Margin**: Limit risk to the position, not entire account

**Important Settings:**
- Use ISOLATED margin mode (not Cross)
- Set leverage conservatively (2-3x maximum for this system)
- Use USDT-M perpetual contracts

---

## 8. Deployment Instructions

### Step 1: Environment Setup

```bash
# Create virtual environment
python -m venv trading_env
source trading_env/bin/activate  # Linux/Mac
# or: trading_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure API Keys

1. Create a Binance account (or use testnet for paper trading)
2. Enable Futures trading
3. Create API keys with Futures permissions
4. **IMPORTANT:** Restrict API to your IP address
5. **NEVER** enable withdrawal permissions for trading bots

Create `.env` file:
```
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_secret_here
BINANCE_TESTNET=true  # Set to false for live trading
```

### Step 3: Test on Testnet First

```bash
# Run in paper trading mode
python main.py --testnet --symbol BTCUSDT --timeframe 1h
```

Testnet endpoint: https://testnet.binancefuture.com

**Test for at least 30 days before considering real money.**

### Step 4: Backtest Historical Data

```bash
# Run backtest
python -m backtest.runner --symbol BTCUSDT --start 2023-01-01 --end 2024-01-01
```

Review all metrics carefully. If Sharpe < 1.0 or max drawdown > 15%, DO NOT proceed.

### Step 5: Start Paper Trading

Run the bot but with `execute_trades=False` to log what trades it WOULD make:

```bash
python main.py --paper --symbol BTCUSDT --timeframe 1h
```

### Step 6: Gradual Live Deployment

If paper trading results are satisfactory after 30+ days:

1. Start with 10% of intended capital
2. Monitor closely for 1-2 weeks
3. Gradually increase if performance matches expectations
4. Have kill switch ready (can manually close all positions via Binance app)

---

## 9. Backtest Metrics Explanation

### Win Rate
Percentage of trades that are profitable. Aim for > 40% with 2:1+ risk-reward.

### Profit Factor
Total gross profit / Total gross loss. Should be > 1.5.

### Maximum Drawdown
Largest peak-to-trough decline. Should be < 15% for conservative strategies.

### Sharpe Ratio
Risk-adjusted return. Sharpe = (Return - Risk Free Rate) / Standard Deviation
- Sharpe < 1.0: Poor risk-adjusted returns
- Sharpe 1.0-2.0: Acceptable
- Sharpe > 2.0: Excellent (but verify not overfitting)

### Sortino Ratio
Like Sharpe but only penalizes downside volatility. Should be higher than Sharpe.

### Calmar Ratio
Annual Return / Max Drawdown. Higher is better.

---

## 10. Overfitting Warnings

**Signs of Overfitting:**
1. Spectacular backtest returns (>100% annual with low drawdown)
2. Many optimized parameters (>10)
3. Performance drops significantly on out-of-sample data
4. Strategy only works on one specific symbol/timeframe

**Our Mitigations:**
1. Use robust, well-known indicators (not exotic)
2. Limit parameters to essential ones
3. Use walk-forward testing
4. Test on multiple symbols and timeframes
5. Add transaction costs and slippage to backtests
6. Paper trade for extended period before live trading

### Walk-Forward Testing Protocol

1. Divide data into multiple segments (e.g., 6 months each)
2. Train on segment 1, test on segment 2
3. Train on segments 1-2, test on segment 3
4. Continue expanding training set
5. Aggregate out-of-sample results
6. If out-of-sample performance << in-sample, you're overfitting

---

## 11. Future Improvements

### Safe Optimizations
- Add sentiment analysis from funding rates
- Incorporate open interest data
- Multi-timeframe confirmation (e.g., 4h trend + 1h entry)
- Order flow analysis using bid/ask imbalance

### Machine Learning Integration (Careful!)
- Use ML for regime detection only
- Don't let ML generate entry signals directly
- Ensemble approaches with traditional indicators
- Always maintain interpretability

### Infrastructure Improvements
- Move to colocated servers for lower latency
- Implement proper database for trade logging
- Set up monitoring and alerting
- Add automatic hedging capabilities

---

## 12. Known Limitations

1. **Latency**: This system is not suitable for high-frequency trading
2. **Flash Crashes**: Extreme volatility can cause significant slippage
3. **API Limits**: Binance has rate limits that constrain polling frequency
4. **Exchange Risk**: Centralized exchange could have outages or failures
5. **Regulatory Risk**: Crypto regulations are evolving rapidly
6. **Liquidity**: Less liquid pairs may have execution issues
7. **Correlation**: Crypto markets are highly correlated in downturns

---

## Contact and Support

This is open-source educational software. Use at your own risk. No support is provided. Consider this a starting point for your own research, not a production-ready system.
