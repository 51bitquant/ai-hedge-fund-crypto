import itertools
from datetime import datetime
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from colorama import Fore, Style
from utils import Interval, QUANTITY_DECIMALS, format_backtest_row, print_backtest_results
from agent import Agent
from utils.binance_data_provider import BinanceDataProvider
from utils.okx_data_provider import OkxDataProvider
from utils import settings
import matplotlib.pyplot as plt


class Backtester:
    def __init__(
            self,
            primary_interval: Interval,
            intervals: List[Interval],
            tickers: List[str],
            start_date: datetime,
            end_date: datetime,
            initial_capital: float,
            strategies: List[str],
            model_name: str = "gpt-4o",
            model_provider: str = "openai",
            model_base_url: Optional[str] = None,
            initial_margin_requirement: float = 0.0,
            show_agent_graph: bool = False,
            show_reasoning: bool = False
    ):
        """
        Backtester
        :param primary_interval:
        :param intervals:
        :param tickers:
        :param start_date:
        :param end_date:
        :param initial_capital:
        :param strategies:
        :param model_name:
        :param model_provider:
        :param model_base_url: model base url
        :param initial_margin_requirement:
        :param show_agent_graph:
        :param show_reasoning:
        """
        self.primary_interval = primary_interval
        self.tickers = tickers
        self.intervals = intervals
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.strategies = strategies
        self.model_name = model_name
        self.model_provider = model_provider
        self.model_base_url = model_base_url
        self.show_agent_graph = show_agent_graph
        self.show_reasoning = show_reasoning
        if settings.exchange.lower() == "okx":
            self.data_provider = OkxDataProvider()
        else:
            self.data_provider = BinanceDataProvider()
        self.klines: Dict[str, pd.DataFrame]() = {}

        # Initialize portfolio with support for long/short positions
        self.portfolio_values = []
        self.portfolio = {
            "cash": initial_capital,
            "margin_requirement": initial_margin_requirement,  # The margin ratio required for shorts
            "margin_used": 0.0,  # total margin usage across all short positions
            "positions": {
                ticker: {
                    "long": 0.0,
                    "short": 0.0,
                    "long_cost_basis": 0.0,
                    "short_cost_basis": 0.0,
                    "short_margin_used": 0.0
                }
                for ticker in tickers
            },
            "realized_gains": {
                ticker: {
                    "long": 0.0,  # Realized gains from long positions
                    "short": 0.0,  # Realized gains from short positions
                }
                for ticker in tickers
            },
        }

    def execute_trade(self, ticker: str, action: str, quantity: float, current_price: float):
        """
        Execute trades with support for both long and short positions.
        `quantity` is the number of shares the agent wants to buy/sell/short/cover.
        We will only trade integer shares to keep it simple.
        """
        if quantity <= 0.0:
            return 0.0

        quantity = round(float(quantity), QUANTITY_DECIMALS)  # force to keep just 0.001
        position = self.portfolio["positions"][ticker]

        if action == "buy":
            cost = quantity * current_price
            if cost <= self.portfolio["cash"]:
                # Weighted average cost basis for the new total
                old_shares = position["long"]
                old_cost_basis = position["long_cost_basis"]
                new_shares = quantity
                total_shares = old_shares + new_shares

                if total_shares > 0.0:
                    total_old_cost = old_cost_basis * old_shares
                    total_new_cost = cost
                    position["long_cost_basis"] = (total_old_cost + total_new_cost) / total_shares

                position["long"] += quantity
                self.portfolio["cash"] -= cost
                return quantity
            else:
                # Calculate maximum affordable quantity
                max_quantity = round(float(self.portfolio["cash"] / current_price), QUANTITY_DECIMALS)
                if max_quantity > 0.0:
                    cost = max_quantity * current_price
                    old_shares = position["long"]
                    old_cost_basis = position["long_cost_basis"]
                    total_shares = old_shares + max_quantity

                    if total_shares > 0.0:
                        total_old_cost = old_cost_basis * old_shares
                        total_new_cost = cost
                        position["long_cost_basis"] = (total_old_cost + total_new_cost) / total_shares

                    position["long"] += max_quantity
                    self.portfolio["cash"] -= cost
                    return max_quantity
                return 0.0

        elif action == "sell":
            # You can only sell as many as you own
            quantity = min(quantity, position["long"])
            if quantity > 0.0:
                # Realized gain/loss using average cost basis
                avg_cost_per_share = position["long_cost_basis"] if position["long"] > 0.0 else 0.0
                realized_gain = (current_price - avg_cost_per_share) * quantity
                self.portfolio["realized_gains"][ticker]["long"] += realized_gain

                position["long"] -= quantity
                self.portfolio["cash"] += quantity * current_price

                if position["long"] == 0.0:
                    position["long_cost_basis"] = 0.0

                return quantity

        elif action == "short":
            """
            Typical short sale flow:
              1) Receive proceeds = current_price * quantity
              2) Post margin_required = proceeds * margin_ratio
              3) Net effect on cash = +proceeds - margin_required
            """
            proceeds = current_price * quantity
            margin_required = proceeds * self.portfolio["margin_requirement"]
            if margin_required <= self.portfolio["cash"]:
                # Weighted average short cost basis
                old_short_shares = position["short"]
                old_cost_basis = position["short_cost_basis"]
                new_shares = quantity
                total_shares = old_short_shares + new_shares

                if total_shares > 0.0:
                    total_old_cost = old_cost_basis * old_short_shares
                    total_new_cost = current_price * new_shares
                    position["short_cost_basis"] = (total_old_cost + total_new_cost) / total_shares

                position["short"] += quantity

                # Update margin usage
                position["short_margin_used"] += margin_required
                self.portfolio["margin_used"] += margin_required

                # Increase cash by proceeds, then subtract the required margin
                self.portfolio["cash"] += proceeds
                self.portfolio["cash"] -= margin_required
                return quantity
            else:
                # Calculate maximum shortable quantity
                margin_ratio = self.portfolio["margin_requirement"]
                if margin_ratio > 0.0:
                    max_quantity = int(self.portfolio["cash"] / (current_price * margin_ratio))
                else:
                    max_quantity = 0.0

                if max_quantity > 0.0:
                    proceeds = current_price * max_quantity
                    margin_required = proceeds * margin_ratio

                    old_short_shares = position["short"]
                    old_cost_basis = position["short_cost_basis"]
                    total_shares = old_short_shares + max_quantity

                    if total_shares > 0.0:
                        total_old_cost = old_cost_basis * old_short_shares
                        total_new_cost = current_price * max_quantity
                        position["short_cost_basis"] = (total_old_cost + total_new_cost) / total_shares

                    position["short"] += max_quantity
                    position["short_margin_used"] += margin_required
                    self.portfolio["margin_used"] += margin_required

                    self.portfolio["cash"] += proceeds
                    self.portfolio["cash"] -= margin_required
                    return max_quantity
                return 0.0

        elif action == "cover":
            """
            When covering shares:
              1) Pay cover cost = current_price * quantity
              2) Release a proportional share of the margin
              3) Net effect on cash = -cover_cost + released_margin
            """
            quantity = min(quantity, position["short"])
            if quantity > 0.0:
                cover_cost = quantity * current_price
                avg_short_price = position["short_cost_basis"] if position["short"] > 0.0 else 0.0
                realized_gain = (avg_short_price - current_price) * quantity

                if position["short"] > 0.0:
                    portion = quantity / position["short"]
                else:
                    portion = 1.0

                margin_to_release = portion * position["short_margin_used"]

                position["short"] -= quantity
                position["short_margin_used"] -= margin_to_release
                self.portfolio["margin_used"] -= margin_to_release

                # Pay the cost to cover, but get back the released margin
                self.portfolio["cash"] += margin_to_release
                self.portfolio["cash"] -= cover_cost

                self.portfolio["realized_gains"][ticker]["short"] += realized_gain

                if position["short"] == 0.0:
                    position["short_cost_basis"] = 0.0
                    position["short_margin_used"] = 0.0

                return quantity

        return 0.0

    def calculate_portfolio_value(self, current_prices):
        """
        Calculate total portfolio value, including:
          - cash
          - market value of long positions
          - unrealized gains/losses for short positions
        """
        total_value = self.portfolio["cash"]

        for ticker in self.tickers:
            position = self.portfolio["positions"][ticker]
            price = current_prices[ticker]

            # Long position value
            long_value = position["long"] * price
            total_value += long_value

            # Short position unrealized PnL = short_shares * (short_cost_basis - current_price)
            if position["short"] > 0.0:
                total_value -= position["short"] * price

        return total_value

    def prefetch_data(self):
        """Pre-fetch all data needed for the backtest period."""
        print("\nPre-fetching data for the entire backtest period...")
        for ticker in self.tickers:
            # Fetch price data for the entire period
            data = self.data_provider.get_historical_klines(
                symbol=ticker,
                timeframe=self.primary_interval.value,
                start_date=self.start_date,
                end_date=self.end_date,
            )
            self.klines[ticker] = data

        print("Data pre-fetch complete.")

    def run_backtest(self):
        # Pre-fetch all data at the start
        self.prefetch_data()

        # Check all are DataFrames and collect lengths
        lengths = []
        for ticker in self.tickers:
            df = self.klines.get(ticker)
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"Data for {ticker} is not a DataFrame: {type(df)}")
            lengths.append(len(df))

        # Check if all lengths are equal
        if len(set(lengths)) != 1:
            raise ValueError(f"DataFrames have mismatched lengths: {dict(zip(self.tickers, lengths))}")

        ticker = self.tickers[0]
        data_df: pd.DataFrame = self.klines[ticker]
        # dates = pd.date_range(self.start_date, self.end_date, freq="B")
        table_rows = []
        performance_metrics = {"sharpe_ratio": None, "sortino_ratio": None, "max_drawdown": None,
                               "long_short_ratio": None, "gross_exposure": None, "net_exposure": None}

        print("\nStarting backtest...")

        # Initialize portfolio values list with initial capital
        if len(data_df) > 0:
            self.portfolio_values = [{"Date": data_df.loc[0, 'open_time'], "Portfolio Value": self.initial_capital}]
        else:
            self.portfolio_values = []

        # print(self.portfolio_values)
        for row in data_df.itertuples(index=True):

            index = row.Index
            current_time = row.close_time
            current_prices = {}
            for ticker in self.tickers:
                price_data = self.klines[ticker]
                current_prices[ticker] = price_data.iloc[index]["close"]

            # ---------------------------------------------------------------
            # 1) Execute the agent's trades
            # ---------------------------------------------------------------
            output = Agent.run(
                primary_interval=self.primary_interval,
                intervals=self.intervals,
                tickers=self.tickers,
                end_date=current_time,
                portfolio=self.portfolio,
                strategies=self.strategies,
                model_name=self.model_name,
                model_provider=self.model_provider,
                model_base_url=self.model_base_url,
                show_agent_graph=self.show_agent_graph,
                show_reasoning=self.show_reasoning,
            )

            decisions = output.get("decisions")
            analyst_signals = output["analyst_signals"]

            # Execute trades for each ticker
            executed_trades = {}
            for ticker in self.tickers:
                decision = decisions.get(ticker, {"action": "hold", "quantity": 0.0})
                action, quantity = decision.get("action", "hold"), decision.get("quantity", 0.0)

                executed_quantity = self.execute_trade(ticker, action, quantity, current_prices[ticker])
                executed_trades[ticker] = executed_quantity

            # ---------------------------------------------------------------
            # 2) Now that trades have executed trades, recalculate the final
            #    portfolio value for this day.
            # ---------------------------------------------------------------
            total_value = self.calculate_portfolio_value(current_prices)

            # Also compute long/short exposures for final post‐trade state
            long_exposure = sum(self.portfolio["positions"][t]["long"] * current_prices[t] for t in self.tickers)
            short_exposure = sum(self.portfolio["positions"][t]["short"] * current_prices[t] for t in self.tickers)

            # Calculate gross and net exposures
            gross_exposure = long_exposure + short_exposure
            net_exposure = long_exposure - short_exposure
            long_short_ratio = long_exposure / short_exposure if short_exposure > 1e-9 else float("inf")

            # Track each day's portfolio value in self.portfolio_values
            self.portfolio_values.append(
                {"Date": current_time, "Portfolio Value": total_value, "Long Exposure": long_exposure,
                 "Short Exposure": short_exposure, "Gross Exposure": gross_exposure, "Net Exposure": net_exposure,
                 "Long/Short Ratio": long_short_ratio})

            # ---------------------------------------------------------------
            # 3) Build the table rows to display
            # ---------------------------------------------------------------
            date_rows = []

            # For each ticker, record signals/trades
            for ticker in self.tickers:
                ticker_signals = {}
                for agent_name, signals in analyst_signals.items():
                    if ticker in signals:
                        ticker_signals[agent_name] = signals[ticker]

                bullish_count = len([s for s in ticker_signals.values() if s.get("signal", "").lower() == "bullish"])
                bearish_count = len([s for s in ticker_signals.values() if s.get("signal", "").lower() == "bearish"])
                neutral_count = len([s for s in ticker_signals.values() if s.get("signal", "").lower() == "neutral"])

                # Calculate net position value
                pos = self.portfolio["positions"][ticker]
                long_val = pos["long"] * current_prices[ticker]
                short_val = pos["short"] * current_prices[ticker]
                net_position_value = long_val - short_val

                # Get the action and quantity from the decisions
                action = decisions.get(ticker, {}).get("action", "hold")
                quantity = executed_trades.get(ticker, 0.0)

                # Append the agent action to the table rows
                date_rows.append(
                    format_backtest_row(
                        date=current_time,
                        ticker=ticker,
                        action=action,
                        quantity=quantity,
                        price=current_prices[ticker],
                        shares_owned=pos["long"] - pos["short"],  # net shares
                        position_value=net_position_value,
                        bullish_count=bullish_count,
                        bearish_count=bearish_count,
                        neutral_count=neutral_count,
                    )
                )
            # ---------------------------------------------------------------
            # 4) Calculate performance summary metrics
            # ---------------------------------------------------------------
            # Calculate portfolio return vs. initial capital
            # The realized gains are already reflected in cash balance, so we don't add them separately
            portfolio_return = (total_value / self.initial_capital - 1) * 100

            # Add summary row for this day
            date_rows.append(
                format_backtest_row(
                    date=current_time,
                    ticker="",
                    action="",
                    quantity=0,
                    price=0,
                    shares_owned=0,
                    position_value=0,
                    bullish_count=0,
                    bearish_count=0,
                    neutral_count=0,
                    is_summary=True,
                    total_value=total_value,
                    return_pct=portfolio_return,
                    cash_balance=self.portfolio["cash"],
                    total_position_value=total_value - self.portfolio["cash"],
                    sharpe_ratio=performance_metrics["sharpe_ratio"],
                    sortino_ratio=performance_metrics["sortino_ratio"],
                    max_drawdown=performance_metrics["max_drawdown"],
                ),
            )

            table_rows.extend(date_rows)
            print_backtest_results(table_rows)

            # Update performance metrics if we have enough data
            if len(self.portfolio_values) > 3:
                self._update_performance_metrics(performance_metrics)

        # Store the final performance metrics for reference in analyze_performance
        self.performance_metrics = performance_metrics
        return performance_metrics

    def _update_performance_metrics(self, performance_metrics):
        """Helper method to update performance metrics using daily returns."""
        values_df = pd.DataFrame(self.portfolio_values).set_index("Date")
        values_df["Daily Return"] = values_df["Portfolio Value"].pct_change()
        clean_returns = values_df["Daily Return"].dropna()

        if len(clean_returns) < 2:
            return  # not enough data points

        # Assumes 365 trading days/year
        daily_risk_free_rate = 0.0434 / 365
        excess_returns = clean_returns - daily_risk_free_rate
        mean_excess_return = excess_returns.mean()
        std_excess_return = excess_returns.std()

        # Sharpe ratio
        if std_excess_return > 1e-12:
            performance_metrics["sharpe_ratio"] = np.sqrt(365) * (mean_excess_return / std_excess_return)
        else:
            performance_metrics["sharpe_ratio"] = 0.0

        # Sortino ratio
        negative_returns = excess_returns[excess_returns < 0]
        if len(negative_returns) > 0:
            downside_std = negative_returns.std()
            if downside_std > 1e-12:
                performance_metrics["sortino_ratio"] = np.sqrt(365) * (mean_excess_return / downside_std)
            else:
                performance_metrics["sortino_ratio"] = float("inf") if mean_excess_return > 0 else 0
        else:
            performance_metrics["sortino_ratio"] = float("inf") if mean_excess_return > 0 else 0

        # Maximum drawdown (ensure it's stored as a negative percentage)
        rolling_max = values_df["Portfolio Value"].cummax()
        drawdown = (values_df["Portfolio Value"] - rolling_max) / rolling_max

        if len(drawdown) > 0:
            min_drawdown = drawdown.min()
            # Store as a negative percentage
            performance_metrics["max_drawdown"] = min_drawdown * 100

            # Store the date of max drawdown for reference
            if min_drawdown < 0:
                performance_metrics["max_drawdown_date"] = drawdown.idxmin().strftime("%Y-%m-%d")
            else:
                performance_metrics["max_drawdown_date"] = None
        else:
            performance_metrics["max_drawdown"] = 0.0
            performance_metrics["max_drawdown_date"] = None

    def analyze_performance(self):
        """Creates a performance DataFrame, prints summary stats, and plots equity curve."""
        if not self.portfolio_values:
            print("No portfolio data found. Please run the backtest first.")
            return pd.DataFrame()

        performance_df = pd.DataFrame(self.portfolio_values).set_index("Date")
        if performance_df.empty:
            print("No valid performance data to analyze.")
            return performance_df

        final_portfolio_value = performance_df["Portfolio Value"].iloc[-1]
        total_return = ((final_portfolio_value - self.initial_capital) / self.initial_capital) * 100

        print(f"\n{Fore.WHITE}{Style.BRIGHT}PORTFOLIO PERFORMANCE SUMMARY:{Style.RESET_ALL}")
        print(f"Total Return: {Fore.GREEN if total_return >= 0 else Fore.RED}{total_return:.2f}%{Style.RESET_ALL}")

        # Print realized P&L for informational purposes only
        total_realized_gains = sum(
            self.portfolio["realized_gains"][ticker]["long"] + self.portfolio["realized_gains"][ticker]["short"] for
            ticker in self.tickers)
        print(
            f"Total Realized Gains/Losses: {Fore.GREEN if total_realized_gains >= 0 else Fore.RED}${total_realized_gains:,.2f}{Style.RESET_ALL}")

        # Plot the portfolio value over time
        plt.figure(figsize=(12, 6))
        plt.plot(performance_df.index, performance_df["Portfolio Value"], color="blue")
        plt.title("Portfolio Value Over Time")
        plt.ylabel("Portfolio Value ($)")
        plt.xlabel("Date")
        plt.grid(True)
        plt.show()

        # Compute daily returns
        performance_df["Daily Return"] = performance_df["Portfolio Value"].pct_change().fillna(0)
        daily_rf = 0.0434 / 365  # daily risk-free rate
        mean_daily_return = performance_df["Daily Return"].mean()
        std_daily_return = performance_df["Daily Return"].std()

        # Annualized Sharpe Ratio
        if std_daily_return != 0:
            annualized_sharpe = np.sqrt(365) * ((mean_daily_return - daily_rf) / std_daily_return)
        else:
            annualized_sharpe = 0
        print(f"\nSharpe Ratio: {Fore.YELLOW}{annualized_sharpe:.2f}{Style.RESET_ALL}")

        # Use the max drawdown value calculated during the backtest if available
        max_drawdown = getattr(self, "performance_metrics", {}).get("max_drawdown")
        max_drawdown_date = getattr(self, "performance_metrics", {}).get("max_drawdown_date")

        # If no value exists yet, calculate it
        if max_drawdown is None:
            rolling_max = performance_df["Portfolio Value"].cummax()
            drawdown = (performance_df["Portfolio Value"] - rolling_max) / rolling_max
            max_drawdown = drawdown.min() * 100
            max_drawdown_date = drawdown.idxmin().strftime("%Y-%m-%d") if pd.notnull(drawdown.idxmin()) else None

        if max_drawdown_date:
            print(f"Maximum Drawdown: {Fore.RED}{abs(max_drawdown):.2f}%{Style.RESET_ALL} (on {max_drawdown_date})")
        else:
            print(f"Maximum Drawdown: {Fore.RED}{abs(max_drawdown):.2f}%{Style.RESET_ALL}")

        # Win Rate
        winning_days = len(performance_df[performance_df["Daily Return"] > 0])
        total_days = max(len(performance_df) - 1, 1)
        win_rate = (winning_days / total_days) * 100
        print(f"Win Rate: {Fore.GREEN}{win_rate:.2f}%{Style.RESET_ALL}")

        # Average Win/Loss Ratio
        positive_returns = performance_df[performance_df["Daily Return"] > 0]["Daily Return"]
        negative_returns = performance_df[performance_df["Daily Return"] < 0]["Daily Return"]
        avg_win = positive_returns.mean() if not positive_returns.empty else 0
        avg_loss = abs(negative_returns.mean()) if not negative_returns.empty else 0
        if avg_loss != 0:
            win_loss_ratio = avg_win / avg_loss
        else:
            win_loss_ratio = float("inf") if avg_win > 0 else 0
        print(f"Win/Loss Ratio: {Fore.GREEN}{win_loss_ratio:.2f}{Style.RESET_ALL}")

        # Maximum Consecutive Wins / Losses
        returns_binary = (performance_df["Daily Return"] > 0).astype(int)
        if len(returns_binary) > 0:
            max_consecutive_wins = max((len(list(g)) for k, g in itertools.groupby(returns_binary) if k == 1),
                                       default=0)
            max_consecutive_losses = max((len(list(g)) for k, g in itertools.groupby(returns_binary) if k == 0),
                                         default=0)
        else:
            max_consecutive_wins = 0
            max_consecutive_losses = 0

        print(f"Max Consecutive Wins: {Fore.GREEN}{max_consecutive_wins}{Style.RESET_ALL}")
        print(f"Max Consecutive Losses: {Fore.RED}{max_consecutive_losses}{Style.RESET_ALL}")

        return performance_df
