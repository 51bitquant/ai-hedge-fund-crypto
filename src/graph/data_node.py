"""
Data Fetching Module

This module handles the first step in the workflow: fetching data from the data provider.
"""

from datetime import datetime, timedelta
from typing import Dict, Any

from src.utils import BinanceDataProvider, OkxDataProvider, Interval, settings
from .base_node import BaseNode, AgentState

# Initialize data provider
if settings.exchange.lower() == "okx":
    data_provider = OkxDataProvider()
else:
    data_provider = BinanceDataProvider()


class DataNode(BaseNode):
    def __init__(self, interval: Interval = Interval.DAY_1):
        self.interval = interval

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        """
        Fetch data for all required timeframes using the configured data provider.

        Args:
            state: The current state with symbol information

        Returns:
            Updated state with timeframes data
        """
        data = state.get('data', {})
        data['name'] = "DataNode"
        timeframe: str = self.interval.value
        tickers = data.get('tickers', [])
        end_time = data.get('end_date', datetime.now()) + timedelta(milliseconds=500)

        for ticker in tickers:
            df = data_provider.get_history_klines_with_end_time(symbol=ticker, timeframe=timeframe, end_time=end_time)
            if df is not None and not df.empty:
                data[f"{ticker}_{timeframe}"] = df
            else:
                print(f"[Warning] No data returned for {ticker} at interval {timeframe}")

        return state
