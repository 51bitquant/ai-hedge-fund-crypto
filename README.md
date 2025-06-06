# AI Hedge Fund Crypto

[![English](https://img.shields.io/badge/Language-English-blue)](README.md)
[![中文](https://img.shields.io/badge/语言-中文-red)](README_CN.md)

A next-generation algorithmic trading framework that leverages graph-based workflow architecture, 
ensemble technical analysis, and AI language models to make data-driven cryptocurrency trading decisions. 
This system employs a directed acyclic graph (DAG) of specialized nodes for multi-timeframe analysis, 
enabling sophisticated signal generation through weighted combinations of diverse trading strategies.

At its core, the system builds upon LangGraph's computational graph architecture to process market data through 
a pipeline of technical analysis nodes. Each strategy implements a BaseNode interface that processes multi-interval data 
for multiple assets simultaneously. The framework then aggregates these signals using adaptive weighting mechanisms, 
evaluates risk parameters, and formulates position management decisions through large language model (LLM) analysis.

The system stands out through its:
- **AI-Enhanced Decision Making**: Integration of large language models (LLMs) for portfolio management decisions, combining technical signals with sophisticated reasoning
- **Compositional Architecture**: Distinct nodes for data fetching, strategy execution, risk management, and portfolio management
- **Signal Ensemble Approach**: Weighted aggregation of multiple technical strategies (trend following, mean reversion, momentum, volatility, and statistical arbitrage)
- **Multi-Timeframe Analysis**: Simultaneous processing across various time intervals for more robust signal generation
- **Dynamic Strategy Visualization**: Automatic generation of computational graph visualizations to better understand the decision flow
- **Comprehensive Backtesting**: Robust historical performance evaluation with detailed metrics and visualizations

## Backtest Results

Sample backtesting results showcasing how well-designed trading strategies can generate high-quality signals 
when implemented in this framework. The system's performance is driven by the quality of your strategies, 
with the framework enabling efficient implementation, LLM-based decision refinement, and multi-timeframe analysis:

![Backtest Results](imgs/backtest1.png)
![Backtest Results](imgs/backtest2.png)
![Portfolio Performance](imgs/backtest3.png)

These results demonstrate the system's ability to:
- Generate profitable trading signals across different market conditions
- Manage risk effectively with position sizing rules
- Adapt to changing market dynamics with multi-timeframe analysis
- Provide detailed performance metrics for strategy evaluation

## Architecture

The system is built around a highly configurable directed acyclic graph (DAG) architecture, where data flows through various processing nodes. 
This architecture provides exceptional flexibility, allowing users to customize both the available strategies and the timeframes for analysis without modifying the core code.

### Node-Based Workflow System

At its heart, the system uses LangGraph to create a dynamic computational graph where:

1. **Start Node**: Initializes the workflow and prepares the state for processing
2. **Data Nodes**: Process market data for specific intervals (e.g., 30m, 1h, 4h) 
3. **Strategy Nodes**: Apply technical analysis algorithms to the processed data
4. **Risk Management Node**: Evaluates position limits and exposure
5. **Portfolio Management Node**: Makes final trading decisions using LLM reasoning

The key innovation is that both the data nodes and strategy nodes are fully configurable via the `config.yaml` file, 
allowing you to add, remove, or modify these nodes without changing the core architecture.

![Graph Demo1](imgs/graph1.png)

### Configurable Timeframes

You can specify multiple timeframes for analysis in your configuration:

```yaml
signals:
  intervals: ["5m", "15m", "30m", "1h", "4h", "1d"]
```

The system will dynamically create separate data processing nodes for each timeframe, allowing strategies to analyze market behavior across multiple time horizons simultaneously. 
This multi-timeframe approach provides more robust signals by capturing both short-term and long-term market trends.

![Graph Demo2](imgs/graph2.png)

### Configurable Strategies

Similarly, you can specify which trading strategies to include:

```yaml
signals:
  strategies: ['MacdStrategy', 'RSIStrategy', 'BollingerStrategy']
```

The system will dynamically load and integrate only the specified strategies into the workflow graph. Each strategy is implemented as an independent node that:

1. Receives the aggregated multi-timeframe data
2. Applies its specific technical analysis algorithms
3. Generates trading signals with confidence levels
4. Passes these signals to the risk management node

This modular approach allows you to easily experiment with different combinations of strategies without rewriting any code. 
You can also create your own custom strategy modules and add them to the configuration.

![Graph Demo3](imgs/graph3.png)

### Data Flow Architecture

The complete data flow works as follows:

1. The start node initializes the workflow state
2. Multiple timeframe nodes fetch and process data in parallel for different intervals
3. The merge node combines the multi-timeframe data into a unified state
4. Multiple strategy nodes analyze this unified data and generate signals
5. The risk management node applies position sizing and exposure limits
6. The portfolio management node makes final trading decisions using all available signals and LLM reasoning

This architecture provides several advantages:
- **Flexibility**: Change strategies or timeframes without code modifications
- **Parallelization**: Process multiple timeframes concurrently for improved performance
- **Isolation**: Maintain separation of concerns between different components
- **Extensibility**: Add new strategy nodes without affecting existing functionality
- **Visualization**: Generate visual representations of the workflow for better understanding

## Features

- **Strategy-Based Architecture**: Implement and backtest multiple trading strategies
- **Multi-Timeframe Analysis**: Analyze multiple intervals (1h, 4h, 1d, etc.) simultaneously
- **Multiple Technical Indicators**: MACD, RSI, and custom indicator support
- **Comprehensive Backtesting**: Test strategies against historical data
- **Portfolio Management**: Manage positions with support for both long and short trades
- **Performance Metrics**: Detailed performance statistics and visualization
- **Strategy Visualization**: Automatically generate strategy workflow graphs

## Table of Contents
- [Setup](#setup)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Running Backtest Mode](#running-backtest-mode)
  - [Running Live Mode](#running-live-mode)
- [Creating Custom Strategies](#creating-custom-strategies)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Star History](#star-history)
- [License](#license)
- [Disclaimer](#disclaimer)
- [Referral Links](#referral-links)

## Setup

### Prerequisites
- Python 3.9 or higher (Python 3.12 recommended — used during development)
- Binance account (required for accessing market data)

⚠️ While Python 3.9+ should work, we recommend using Python 3.12 for full compatibility with the development environment.

### Installation

1. Clone the repository:
```bash
git clone https://github.com/51bitquant/ai-hedge-fund-crypto.git
cd ai-hedge-fund-crypto
```

2. Set up using uv (recommended):
```bash
# Install uv if you don't have it
curl -fsSL https://install.lunarvim.org/uv.sh | sh

# create an venv with python 3.12
uv venv --python 3.12

# Activate the virtual environment
# For macOS/Linux:
source .venv/bin/activate
# For Windows (PowerShell):
.venv\Scripts\Activate.ps1

# For Windows (CMD):
.venv\Scripts\activate.bat
# Install dependencies using pyproject.toml and lockfile
uv pip sync
```
If you have more questions, feel free to check out the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/)

3. Copy the example configuration:
```bash
cp config.example.yaml config.yaml
```

4. Set up environment variables:
```bash
cp .env.example .env
```

5. Add your API keys to the `.env` file:
```
# Binance API keys (required for data access)
BINANCE_API_KEY=your-binance-api-key
BINANCE_API_SECRET=your-binance-api-secret

# LLM API keys (if using AI assistants)
OPENAI_API_KEY=your-openai-api-key
GROQ_API_KEY=your-groq-api-key
OPENROUTER_API_KEY=your-openrouter-api-key
```

## Configuration

The system is configured through the `config.yaml` file. Here's what each setting means:

```yaml
mode: backtest  # Options: backtest, live
start_date: 2025-04-20  # Start date for backtesting
end_date: 2025-05-01  # End date for backtesting
primary_interval: 1h  # Main timeframe for analysis
initial_cash: 100000  # Starting capital
margin_requirement: 0.0  # Margin requirements for short positions
show_reasoning: false  # Whether to show strategy reasoning
show_agent_graph: true  # Whether to show the agent workflow graph
signals:
  intervals: ["30m", "1h", "4h"]  # Timeframes to analyze
  tickers: ["BTCUSDT", "ETHUSDT"]  # Trading pairs
  strategies: ['MacdStrategy']  # Strategies to use
```

## Usage

### Running Backtest Mode

The backtester allows you to test your trading strategies against historical market data.

1. Configure your settings in `config.yaml`:
```yaml
mode: backtest
start_date: 2025-01-01
end_date: 2025-02-01
```

2. Run the backtest:
```bash
uv run backtest.py
```
or 

```bash
uv run main.py
```

This will execute the backtest using the strategies and settings defined in your config file. Results will be displayed in the console and performance charts will be shown.

#### Backtest Results

The backtest generates detailed performance metrics and visualizations:

![Backtest Results](imgs/backtest1.png)
![Backtest Results](imgs/backtest2.png)
![Portfolio Performance](imgs/backtest3.png)

### Running Live Mode

The system supports two live trading modes: signal generation and real trading execution.

#### Signal Generation Only

For real-time analysis and trading signals without executing actual trades:

1. Configure your settings in `config.yaml`:
```yaml
mode: live
```

2. Run the main script:
```bash
uv run main.py
```

This will fetch current market data, run it through your configured strategies, 
and generate trading signals without sending any orders to exchanges.

#### Real Trading Execution

For live trading with actual order execution on Binance:

1. Configure your settings in `config.yaml`:
```yaml
mode: live
```

2. Ensure your API keys have trading permissions in your `.env` file:
```
BINANCE_API_KEY=your-binance-api-key-with-trading-permissions
BINANCE_API_SECRET=your-binance-api-secret
```

3. Run the main script:
```bash
uv run main.py
```
## Real Trading Integration (Planned)
The system is designed to support real trading through a Binance gateway client, which can send live orders based on strategy-generated signals. The gateway module extends the functionality of the python-binance library with features optimized for algorithmic trading.

While full trading execution is not yet implemented, the current setup focuses on generating structured trading signals. These signals can later be converted into actual orders as part of a future enhancement.

Here's an example of how to run the agent and retrieve signals:
```python

result = Agent.run(
            primary_interval=settings.primary_interval,
            intervals=settings.signals.intervals,
            tickers=settings.signals.tickers,
            end_date=datetime.now(),
            portfolio=portfolio,
            strategies=settings.signals.strategies,
            show_reasoning=settings.show_reasoning,
            show_agent_graph=settings.show_agent_graph
        )
print(result)
```
Sample output:
```json
{
  "decisions": {
    "BTCUSDT": {
      "action": "hold",
      "quantity": 0,
      "confidence": 20,
      "reasoning": "Signals are mixed with bearish trends dominating in the shorter timeframes. No long position held, and the bearish outlook indicates avoiding purchases."
    },
    "ETHUSDT": {
      "action": "buy",
      "quantity": 10.87547580206634,
      "confidence": 23,
      "reasoning": "Despite signals being neutral with a slight bullish trend on longer timeframes, the maximum shares can be purchased. The overall market sentiment is uncertain, but ETH shows potential for upward movement."
    }
  }
}

```

> **IMPORTANT**: Live trading involves real financial risk. Start with small amounts and thoroughly test your strategies in backtesting mode before deploying with real capital.

## Creating Custom Strategies

Creating your own trading strategy is straightforward with the system's modular architecture:

1. **Create a Strategy Class**:
   Create a new Python file in the `src/strategies/` directory (e.g., `my_strategy.py`):

```python
from typing import Dict, Any
import json
import pandas as pd
from langchain_core.messages import HumanMessage
from src.graph import AgentState, BaseNode

class MyStrategy(BaseNode):
    def __call__(self, state: AgentState) -> Dict[str, Any]:
        """
        Custom strategy implementation that processes market data across multiple timeframes.
        """
        # Access the state data
        data = state.get("data", {})
        data['name'] = "MyStrategy"  # Set strategy name for visualization
        
        # Get tickers and intervals from the state
        tickers = data.get("tickers", [])
        intervals = data.get("intervals", [])
        
        # Initialize analysis dictionary to store results
        technical_analysis = {}
        for ticker in tickers:
            technical_analysis[ticker] = {}
        
        # Process each ticker and interval combination
        for ticker in tickers:
            for interval in intervals:
                # Access the price data for this ticker and interval
                df = data.get(f"{ticker}_{interval.value}", pd.DataFrame())
                
                if df.empty:
                    continue
                
                # Implement your custom technical analysis here
                # Example: Simple moving average crossover strategy
                df['sma_fast'] = df['close'].rolling(window=10).mean()
                df['sma_slow'] = df['close'].rolling(window=30).mean()
                
                # Generate signal based on your strategy logic
                signal = "neutral"
                confidence = 50
                
                if df['sma_fast'].iloc[-1] > df['sma_slow'].iloc[-1]:
                    signal = "bullish"
                    confidence = 70
                elif df['sma_fast'].iloc[-1] < df['sma_slow'].iloc[-1]:
                    signal = "bearish"
                    confidence = 70
                
                # Store analysis results
                technical_analysis[ticker][interval.value] = {
                    "signal": signal,
                    "confidence": confidence,
                    "strategy_signals": {
                        "simple_ma_crossover": {
                            "signal": signal,
                            "confidence": confidence,
                            "metrics": {
                                "sma_fast": float(df['sma_fast'].iloc[-1]),
                                "sma_slow": float(df['sma_slow'].iloc[-1]),
                                "price": float(df['close'].iloc[-1])
                            }
                        }
                    }
                }
        
        # Create message with analysis results
        message = HumanMessage(
            content=json.dumps(technical_analysis),
            name="my_strategy_agent",
        )
        
        # Update the state with the analysis
        state["data"]["analyst_signals"]["my_strategy_agent"] = technical_analysis
        
        # Return the updated state
        return {
            "messages": [message],
            "data": data,
        }
```

2. **Register Your Strategy**:
   Add your strategy to `src/strategies/__init__.py`:

```python
from .macd_strategy import MacdStrategy
from .rsi_strategy import RSIStrategy
from .my_strategy import MyStrategy

__all__ = [
    "MacdStrategy",
    "RSIStrategy",
    "MyStrategy",
]
```

3. **Configure Your Strategy**:
   Edit `config.yaml` to include your strategy:

```yaml
signals:
  intervals: ["1h", "4h"]
  tickers: ["BTCUSDT", "ETHUSDT"]
  strategies: ['MyStrategy']
```

4. **Run the System**:
   Execute the system to see your strategy in action:

```bash
uv run backtest.py
```

Your strategy will be automatically integrated into the workflow graph and will contribute to the final trading decisions. The system will generate a visualization of the workflow including your strategy node.

## Project Structure

The project follows a modular, organized structure:

```
ai-hedge-fund-crypto/
├── src/                       # Source code directory
│   ├── agent/                 # Agent system components
│   │   ├── workflow.py        # Workflow creation and management
│   │   └── agent.py           # Main agent implementation
│   ├── backtest/              # Backtesting framework
│   │   └── backtester.py      # Backtester implementation
│   ├── graph/                 # Workflow graph components
│   │   ├── base_node.py       # Base node interface
│   │   ├── start_node.py      # Workflow initialization node
│   │   ├── data_node.py       # Data processing nodes
│   │   ├── empty_node.py      # do nothing.
│   │   ├── risk_management_node.py  # Risk management
│   │   └── portfolio_management_node.py  # Decision making
│   │   └── state.py           # defined Agent state.
│   ├── indicators/            # Technical indicators
│   │   ├── general_indicators.py # some indicators.
│   │   ├── momentum.py        # Momentum indicators (RSI, etc.)
│   │   └── volatility.py      # Volatility indicators (BB, etc.)
│   ├── llm/                   # Language model integration
│   │   └── __init__.py        # LLM setup and configuration
│   ├── strategies/            # Trading strategies
│   │   ├── macd_strategy.py   # MACD-based strategy
│   │   ├── rsi_strategy.py    # RSI-based strategy
│   │   └── __init__.py        # Strategy registry
│   ├── utils/                 # Utility functions
│   │   ├── binance_data_provider.py  # Data fetching
│   │   └── config.py          # Configuration management
│   └── gateway/               # Exchange connectivity
│       └── binance            # Enhanced Binance client
├── imgs/                      # Strategy visualizations and results
├── cache/                     # Data cache for efficiency
├── backtest.py                # Backtest entry point
├── main.py                    # Main application entry point
├── config.yaml                # Main configuration file
├── config.example.yaml        # Example configuration
├── uv.lock                    # UV dependency lock file
└── pyproject.toml             # Project metadata and dependencies
```

Key Files and Directories:

- **backtest.py**: Main entry point for running backtests
- **main.py**: Main entry point for running live analysis
- **config.yaml**: Central configuration file where you set tickers, intervals, and strategies
- **src/strategies/**: Directory for all trading strategy implementations
- **src/graph/**: Components for building the computational graph workflow
- **src/indicators/**: Technical indicators used by strategies
- **src/llm/**: Language model integration for decision making

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Star History
<a href="https://www.star-history.com/#51bitquant/ai-hedge-fund-crypto&Date" target="_blank" rel="noopener noreferrer">
  <picture>
    <source 
      media="(prefers-color-scheme: dark)" 
      srcset="https://api.star-history.com/svg?repos=51bitquant/ai-hedge-fund-crypto&type=Date&theme=dark" />
    <source 
      media="(prefers-color-scheme: light)" 
      srcset="https://api.star-history.com/svg?repos=51bitquant/ai-hedge-fund-crypto&type=Date" />
    <img 
      alt="Star History Chart" 
      src="https://api.star-history.com/svg?repos=51bitquant/ai-hedge-fund-crypto&type=Date" 
      loading="lazy" />
  </picture>
</a>

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Disclaimer

This project is provided "as is" without any warranty. Use at your own risk.


## Referral Links

If you're interested in trading cryptocurrency on Binance, you can use the referral link below to receive a discount on trading fees:

[Binance Futures Referral Link](https://www.binance.com/futures/ref/51bitquant) - Sign up with this link to get a fee discount when trading on Binance Futures.