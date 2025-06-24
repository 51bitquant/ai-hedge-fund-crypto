from typing import Optional
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from okx.api import Market

from .constants import COLUMNS, NUMERIC_COLUMNS


class OkxDataProvider:
    """Data provider for fetching OHLCV data from OKX."""

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, passphrase: Optional[str] = None):
        self.client = Market()
        self.cache_dir = Path("./cache")
        self.cache_dir.mkdir(exist_ok=True)

    @staticmethod
    def _format_timeframe(timeframe: str) -> str:
        mapping = {
            "1m": "1m",
            "3m": "3m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1H",
            "2h": "2H",
            "4h": "4H",
            "6h": "6H",
            "12h": "12H",
            "1d": "1D",
            "1w": "1W",
            "1M": "1M",
        }
        return mapping.get(timeframe, timeframe)

    @staticmethod
    def _timeframe_ms(timeframe: str) -> int:
        mapping = {
            "1m": 60_000,
            "3m": 3 * 60_000,
            "5m": 5 * 60_000,
            "15m": 15 * 60_000,
            "30m": 30 * 60_000,
            "1h": 60 * 60_000,
            "2h": 2 * 60 * 60_000,
            "4h": 4 * 60 * 60_000,
            "6h": 6 * 60 * 60_000,
            "12h": 12 * 60 * 60_000,
            "1d": 24 * 60 * 60_000,
            "1w": 7 * 24 * 60 * 60_000,
            "1M": 30 * 24 * 60 * 60_000,
        }
        return mapping.get(timeframe, 60_000)

    def _to_df(self, data: list, timeframe: str) -> pd.DataFrame:
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(
            data,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_volume",
                "_vol_quote",
                "confirm",
            ],
        )
        df["open_time"] = pd.to_datetime(pd.to_numeric(df["open_time"]), unit="ms")
        df["close_time"] = df["open_time"] + pd.to_timedelta(self._timeframe_ms(timeframe), unit="ms")
        df["count"] = 0
        df["taker_buy_volume"] = 0
        df["taker_buy_quote_volume"] = 0
        df["ignore"] = 0
        df = df[COLUMNS]
        for col in NUMERIC_COLUMNS:
            df[col] = pd.to_numeric(df[col])
        return df

    def get_history_klines_with_end_time(
        self,
        symbol: str,
        timeframe: str,
        end_time: datetime,
        limit: int = 500,
    ) -> pd.DataFrame:
        bar = self._format_timeframe(timeframe)
        formatted_symbol = symbol.replace("/", "-")
        res = self.client.get_history_candles(
            instId=formatted_symbol,
            bar=bar,
            before=str(int(end_time.timestamp() * 1000)),
            limit=str(limit),
        )
        if res.get("code") != "0":
            return pd.DataFrame()
        return self._to_df(res.get("data", []), timeframe)

    def get_latest_data(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        bar = self._format_timeframe(timeframe)
        formatted_symbol = symbol.replace("/", "-")
        res = self.client.get_candles(instId=formatted_symbol, bar=bar, limit=str(limit))
        if res.get("code") != "0":
            return pd.DataFrame()
        return self._to_df(res.get("data", []), timeframe)

    def get_historical_klines(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        formatted_symbol = symbol.replace("/", "-")
        print(f"formatted_symbol: {formatted_symbol}")
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        cache_file = self.cache_dir / f"{formatted_symbol}_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        if use_cache and cache_file.exists():
            return pd.read_csv(cache_file, parse_dates=["open_time", "close_time"])
        bar = self._format_timeframe(timeframe)
        start_ms = int(start_date.timestamp() * 1000)
        before = int(end_date.timestamp() * 1000)
        all_rows = []
        while True:
            print(f"get_history_candles instId: {formatted_symbol}, bar: {bar}, after: {start_ms}, before: {before}, limit: 100")
            res = self.client.get_history_candles(
                instId=formatted_symbol,
                bar=bar,
                # after=str(start_ms),
                before=str(before),
                limit="100",
            )

            print(f"res: {res}")

            if res.get("code") != "0":
                break
            data = res.get("data", [])
            if not data:
                break
            all_rows.extend(data)
            last_ts = int(data[-1][0])
            if last_ts <= start_ms or len(data) < 100:
                break
            before = last_ts
        df = self._to_df(all_rows, timeframe)
        df = df.sort_values("open_time")
        if use_cache and not df.empty:
            df.to_csv(cache_file, index=False)
        return df
