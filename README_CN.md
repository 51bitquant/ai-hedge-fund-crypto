# AI Hedge Fund Crypto (加密货币AI对冲基金)

[![English](https://img.shields.io/badge/Language-English-blue)](README.md)
[![中文](https://img.shields.io/badge/语言-中文-red)](README_CN.md)

一个新一代算法交易框架，利用基于图的工作流架构、集成技术分析和AI语言模型做出数据驱动的加密货币交易决策。
该系统采用有向无环图(DAG)架构，通过专门的节点进行多时间框架分析，
通过各种交易策略的加权组合实现复杂的信号生成。

系统核心基于LangGraph的计算图架构，通过技术分析节点管道处理市场数据。
每个策略都实现了BaseNode接口，可同时处理多个资产的多区间数据。
框架通过自适应加权机制聚合这些信号，评估风险参数，并通过大型语言模型(LLM)分析制定仓位管理决策。

系统的突出特点包括：
- **AI增强决策**：集成大型语言模型(LLMs)用于投资组合管理决策，将技术信号与复杂推理相结合
- **组合式架构**：区分数据获取、策略执行、风险管理和投资组合管理的独立节点
- **信号集成方法**：多种技术策略的加权聚合（趋势跟踪、均值回归、动量、波动性和统计套利）
- **多时间框架分析**：同时处理多个时间区间，生成更稳健的交易信号
- **动态策略可视化**：自动生成计算图可视化，更好地理解决策流程
- **全面回测功能**：通过详细指标和可视化进行稳健的历史表现评估

## 回测结果

以下是回测样例结果，展示了精心设计的交易策略在此框架中能够生成高质量信号的能力。系统性能由策略质量决定，
框架提供高效实现、基于LLM的决策优化和多时间框架分析：

![回测结果](imgs/backtest1.png)
![回测结果](imgs/backtest2.png)
![投资组合表现](imgs/backtest3.png)

## 架构

系统基于高度可配置的有向无环图(DAG)架构，数据通过各种处理节点流动。
这种架构提供了卓越的灵活性，允许用户在不修改核心代码的情况下自定义可用策略和分析时间框架。

### 基于节点的工作流系统

系统核心使用LangGraph创建动态计算图，包括：

1. **起始节点**：初始化工作流并准备处理状态
2. **数据节点**：处理特定时间间隔的市场数据（如30分钟、1小时、4小时）
3. **策略节点**：对处理后的数据应用技术分析算法
4. **风险管理节点**：评估仓位限制和风险敞口
5. **投资组合管理节点**：使用LLM推理做出最终交易决策

关键创新在于数据节点和策略节点可通过`config.yaml`文件完全配置，
允许在不更改核心架构的情况下添加、删除或修改这些节点。

![图表示例1](imgs/graph1.png)

### 可配置时间框架

您可以在配置中指定多个分析时间框架：

```yaml
signals:
  intervals: ["5m", "15m", "30m", "1h", "4h", "1d"]
```

系统将为每个时间框架动态创建独立的数据处理节点，使策略能够同时分析多个时间周期的市场行为。
这种多时间框架方法通过捕捉短期和长期市场趋势提供更稳健的信号。

![图表示例2](imgs/graph2.png)

### 可配置策略

同样，您可以指定要包含的交易策略：

```yaml
signals:
  strategies: ['MacdStrategy', 'RSIStrategy', 'BollingerStrategy']
```

系统将动态加载并仅将指定的策略整合到工作流图中。每个策略作为独立节点实现，用于：

1. 接收汇总的多时间框架数据
2. 应用特定的技术分析算法
3. 生成具有置信度的交易信号
4. 将这些信号传递给风险管理节点

这种模块化方法使您可以轻松尝试不同的策略组合，无需重写任何代码。
您还可以创建自己的自定义策略模块并将其添加到配置中。

![图表示例3](imgs/graph3.png)

### 数据流架构

完整的数据流程如下：

1. 起始节点初始化工作流状态
2. 多个时间框架节点并行获取和处理不同间隔的数据
3. 合并节点将多时间框架数据合并为统一状态
4. 多个策略节点分析这些统一数据并生成信号
5. 风险管理节点应用仓位大小和风险敞口限制
6. 投资组合管理节点使用所有可用信号和LLM推理做出最终交易决策

这种架构提供了几个优势：
- **灵活性**：无需代码修改即可更改策略或时间框架
- **并行处理**：同时处理多个时间框架以提高性能
- **隔离性**：维护不同组件之间的关注点分离
- **可扩展性**：添加新策略节点而不影响现有功能
- **可视化**：生成工作流的可视化表示以便更好地理解

## 特性

- **基于策略的架构**：实现和回测多种交易策略
- **多时间框架分析**：同时分析多个区间（1小时、4小时、1天等）
- **多种技术指标**：支持MACD、RSI和自定义指标
- **全面回测**：对历史数据测试策略
- **投资组合管理**：管理多头和空头仓位
- **性能指标**：详细的性能统计和可视化
- **策略可视化**：自动生成策略工作流图表

## 目录
- [安装设置](#安装设置)
- [配置](#配置)
- [使用方法](#使用方法)
  - [运行回测模式](#运行回测模式)
  - [运行实时模式](#运行实时模式)
- [创建自定义策略](#创建自定义策略)
- [项目结构](#项目结构)
- [贡献](#贡献)
- [星标历史](#星标历史)
- [许可证](#许可证)
- [免责声明](#免责声明)
- [推荐链接](#推荐链接)

## 安装设置

### 前提条件
- Python 3.9或更高版本（推荐Python 3.12 — 开发时使用）
- 币安账户（获取市场数据必需）

⚠️ 虽然Python 3.9+应该可以工作，但我们建议使用Python 3.12以获得与开发环境的完全兼容性。

### 安装

1. 克隆仓库：
```bash
git clone https://github.com/51bitquant/ai-hedge-fund-crypto.git
cd ai-hedge-fund-crypto
```

2. 使用uv设置（推荐）：
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
如果你在使用uv过程中，遇到问题，可以参考它们的[安装和使用文档](https://docs.astral.sh/uv/getting-started/installation/) for detailed guidance.


3. 复制示例配置：
```bash
cp config.example.yaml config.yaml
```

4. 设置环境变量：
```bash
cp .env.example .env
```

5. 在`.env`文件中添加API密钥：
```
# 币安API密钥（获取数据必需）
BINANCE_API_KEY=your-binance-api-key
BINANCE_API_SECRET=your-binance-api-secret

# LLM API密钥（如果使用AI助手）
OPENAI_API_KEY=your-openai-api-key
GROQ_API_KEY=your-groq-api-key
OPENROUTER_API_KEY=your-openrouter-api-key
```

## 配置

系统通过`config.yaml`文件配置。以下是每个设置的含义：

```yaml
mode: backtest  # 选项：backtest（回测）, live（实时）
start_date: 2025-04-20  # 回测开始日期
end_date: 2025-05-01  # 回测结束日期
primary_interval: 1h  # 主要时间框架
initial_cash: 100000  # 初始资金
margin_requirement: 0.0  # 空头保证金要求
show_reasoning: false  # 是否显示策略推理
show_agent_graph: true  # 是否显示agent工作流图
signals:
  intervals: ["30m", "1h", "4h"]  # 要分析的时间框架
  tickers: ["BTCUSDT", "ETHUSDT"]  # 交易对
  strategies: ['MacdStrategy']  # 使用的策略
```

## 使用方法

### 运行回测模式

回测器允许您对历史市场数据测试交易策略。

1. 在`config.yaml`中配置设置：
```yaml
mode: backtest
start_date: 2025-01-01
end_date: 2025-02-01
```

2. 运行回测：
```bash
uv run backtest.py
```
或

```bash
uv run main.py
```

这将使用配置文件中定义的策略和设置执行回测。结果将显示在控制台中，并生成性能图表。

#### 回测结果

回测生成详细的性能指标和可视化：

![回测结果](imgs/backtest1.png)
![回测结果](imgs/backtest2.png)
![投资组合表现](imgs/backtest3.png)

### 运行实时模式

系统支持两种实时交易模式：信号生成和实际交易执行。

#### 仅信号生成

对于不执行实际交易的实时分析和交易信号：

1. 在`config.yaml`中配置设置：
```yaml
mode: live
```

2. 运行主脚本：
```bash
uv run main.py
```

这将获取当前市场数据，通过您配置的策略运行，并生成交易信号，但不会向交易所发送订单。

## 实盘交易集成（计划中）

该系统设计为通过币安网关客户端支持实盘交易，可以根据策略生成的信号发送实时订单。网关模块扩展了python-binance库的功能，针对算法交易进行了优化。

虽然目前尚未实现完整的交易执行，但当前设置专注于生成结构化交易信号。这些信号可以在未来的增强中转换为实际订单。

以下是运行代理并检索信号的示例：
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

示例输出：
```json
{
  "decisions": {
    "BTCUSDT": {
      "action": "hold",
      "quantity": 0,
      "confidence": 20,
      "reasoning": "信号混合，短时间框架呈现熊市趋势。没有持有多头仓位，熊市展望表明应避免购买。"
    },
    "ETHUSDT": {
      "action": "buy",
      "quantity": 10.87547580206634,
      "confidence": 23,
      "reasoning": "尽管信号在较长时间框架上呈中性略带看涨趋势，可以购买最大股份。整体市场情绪不确定，但ETH显示出上行潜力。"
    }
  }
}
```

> **重要提示**：实盘交易涉及真实财务风险。在部署实盘资金前，请从小额开始并在回测模式中彻底测试您的策略。

## 创建自定义策略

使用系统的模块化架构创建自己的交易策略非常简单：

1. **创建策略类**：
   在`src/strategies/`目录中创建新的Python文件（例如`my_strategy.py`）：

```python
from typing import Dict, Any
import json
import pandas as pd
from langchain_core.messages import HumanMessage
from src.graph import AgentState, BaseNode

class MyStrategy(BaseNode):
    def __call__(self, state: AgentState) -> Dict[str, Any]:
        """
        处理多时间框架市场数据的自定义策略实现。
        """
        # 访问状态数据
        data = state.get("data", {})
        data['name'] = "MyStrategy"  # 设置策略名称用于可视化
        
        # 从状态中获取交易对和时间间隔
        tickers = data.get("tickers", [])
        intervals = data.get("intervals", [])
        
        # 初始化分析字典以存储结果
        technical_analysis = {}
        for ticker in tickers:
            technical_analysis[ticker] = {}
        
        # 处理每个交易对和时间间隔组合
        for ticker in tickers:
            for interval in intervals:
                # 访问此交易对和时间间隔的价格数据
                df = data.get(f"{ticker}_{interval.value}", pd.DataFrame())
                
                if df.empty:
                    continue
                
                # 在此实现您的自定义技术分析
                # 示例：简单移动平均线交叉策略
                df['sma_fast'] = df['close'].rolling(window=10).mean()
                df['sma_slow'] = df['close'].rolling(window=30).mean()
                
                # 基于策略逻辑生成信号
                signal = "neutral"
                confidence = 50
                
                if df['sma_fast'].iloc[-1] > df['sma_slow'].iloc[-1]:
                    signal = "bullish"
                    confidence = 70
                elif df['sma_fast'].iloc[-1] < df['sma_slow'].iloc[-1]:
                    signal = "bearish"
                    confidence = 70
                
                # 存储分析结果
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
        
        # 创建带有分析结果的消息
        message = HumanMessage(
            content=json.dumps(technical_analysis),
            name="my_strategy_agent",
        )
        
        # 用分析更新状态
        state["data"]["analyst_signals"]["my_strategy_agent"] = technical_analysis
        
        # 返回更新后的状态
        return {
            "messages": [message],
            "data": data,
        }
```

2. **注册您的策略**：
   将您的策略添加到`src/strategies/__init__.py`：

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

3. **配置您的策略**：
   编辑`config.yaml`以包含您的策略：

```yaml
signals:
  intervals: ["1h", "4h"]
  tickers: ["BTCUSDT", "ETHUSDT"]
  strategies: ['MyStrategy']
```

4. **运行系统**：
   执行系统以查看策略效果：

```bash
uv run backtest.py
```

您的策略将自动集成到工作流图中，并将为最终交易决策做出贡献。系统将生成包含您策略节点的工作流可视化。

## 项目结构

项目遵循模块化、有组织的结构：

```
ai-hedge-fund-crypto/
├── src/                       # 源代码目录
│   ├── agent/                 # 代理系统组件
│   │   ├── workflow.py        # 工作流创建和管理
│   │   └── agent.py           # 主代理实现
│   ├── backtest/              # 回测框架
│   │   └── backtester.py      # 回测器实现
│   ├── graph/                 # 工作流图组件
│   │   ├── base_node.py       # 基本节点接口
│   │   ├── start_node.py      # 工作流初始化节点
│   │   ├── data_node.py       # 数据处理节点
│   │   ├── empty_node.py      # 空操作节点
│   │   ├── risk_management_node.py  # 风险管理
│   │   └── portfolio_management_node.py  # 决策制定
│   │   └── state.py           # 定义Agent状态
│   ├── indicators/            # 技术指标
│   │   ├── general_indicators.py # 一般指标
│   │   ├── momentum.py        # 动量指标（RSI等）
│   │   └── volatility.py      # 波动率指标（布林带等）
│   ├── llm/                   # 语言模型集成
│   │   └── __init__.py        # LLM设置和配置
│   ├── strategies/            # 交易策略
│   │   ├── macd_strategy.py   # 基于MACD的策略
│   │   ├── rsi_strategy.py    # 基于RSI的策略
│   │   └── __init__.py        # 策略注册
│   ├── utils/                 # 实用函数
│   │   ├── binance_data_provider.py  # 数据获取
│   │   └── config.py          # 配置管理
│   └── gateway/               # 交易所连接
│       └── binance            # 增强的币安客户端
├── imgs/                      # 策略可视化和结果
├── cache/                     # 数据缓存以提高效率
├── backtest.py                # 回测入口点
├── main.py                    # 主应用程序入口点
├── config.yaml                # 主配置文件
├── config.example.yaml        # 示例配置
├── uv.lock                    # UV依赖锁定文件
└── pyproject.toml             # 项目元数据和依赖
```

关键文件和目录：

- **backtest.py**：运行回测的主入口点
- **main.py**：运行实时分析的主入口点
- **config.yaml**：设置交易对、时间间隔和策略的中央配置文件
- **src/strategies/**：所有交易策略实现的目录
- **src/graph/**：构建计算图工作流的组件
- **src/indicators/**：策略使用的技术指标
- **src/llm/**：用于决策制定的语言模型集成

## 贡献
1. Fork仓库
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 星标历史
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

## 许可证

本项目基于MIT许可证。有关更多信息，请参阅[LICENSE](LICENSE)文件。

## 免责声明

本项目按"原样"提供，不提供任何保证。使用风险自负。

## 推荐链接

如果您对在币安交易加密货币感兴趣，可以使用下面的推荐链接获得交易费折扣：

[币安期货推荐链接](https://www.binance.com/futures/ref/51bitquant) - 使用此链接注册可获得币安期货交易费折扣。 