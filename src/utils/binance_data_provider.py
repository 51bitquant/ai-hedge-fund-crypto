"""
Binance Data Provider Module

This module handles retrieving data from Binance and preparing it for the trading system.
"""

from typing import Dict, List, Optional
import pandas as pd
import logging
import time
import asyncio
import threading
from datetime import datetime, timedelta
from pathlib import Path

from src.gateway.binance.client import Client
from src.gateway.binance.async_client import AsyncClient
from src.utils.constants import COLUMNS, NUMERIC_COLUMNS


class BinanceDataProvider:
    """
    Class to handle data retrieval from Binance and prepare it for the trading system.
    """

    def __init__(
            self, 
            api_key: Optional[str] = None, 
            api_secret: Optional[str] = None,
            use_websocket_by_default: bool = False,
            websocket_timeout: int = 5,
            websocket_collection_time: int = 20,
            max_retries: int = 1,
            testnet: bool = False
        ):
        """
        Initialize the BinanceDataProvider with API credentials.

        Args:
            api_key: Binance API key (optional for public data)
            api_secret: Binance API secret (optional for public data)
            use_websocket_by_default: Whether to use WebSocket for data retrieval by default
            websocket_timeout: Timeout in seconds for WebSocket initial connection
            websocket_collection_time: Time in seconds to collect data from WebSocket
            max_retries: Maximum number of retries for WebSocket connection
            testnet: Whether to use testnet (default: False)
        """
        self.client = Client(api_key=api_key, api_secret=api_secret, testnet=testnet)
        self.async_client = None  # Lazy initialization
        self.use_websocket_by_default = use_websocket_by_default
        self.websocket_timeout = websocket_timeout
        self.websocket_collection_time = websocket_collection_time
        self.max_retries = max_retries
        self.testnet = testnet
        self.api_key = api_key
        self.api_secret = api_secret
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create a dedicated event loop for WebSocket
        self._ws_loop = None
        self._ws_thread = None
        self._ws_loop_lock = threading.Lock()

        # Create cache directory if it doesn't exist
        self.cache_dir = Path("./cache")
        self.cache_dir.mkdir(exist_ok=True)

    def _ensure_ws_event_loop(self):
        """Ensure WebSocket event loop is running"""
        with self._ws_loop_lock:
            if self._ws_loop is None or self._ws_thread is None or not self._ws_thread.is_alive():
                def run_event_loop(loop):
                    asyncio.set_event_loop(loop)
                    loop.run_forever()
                
                self._ws_loop = asyncio.new_event_loop()
                self._ws_thread = threading.Thread(target=run_event_loop, args=(self._ws_loop,), daemon=True)
                self._ws_thread.start()
                self.logger.info("Started new WebSocket event loop")
                
                # Initialize async client in the new event loop
                if self.async_client is None:
                    future = asyncio.run_coroutine_threadsafe(
                        AsyncClient.create(
                            api_key=self.api_key, 
                            api_secret=self.api_secret,
                            testnet=self.testnet
                        ), 
                        self._ws_loop
                    )
                    try:
                        self.async_client = future.result(timeout=self.websocket_timeout)
                        self.logger.info("AsyncClient initialized successfully")
                    except Exception as e:
                        self.logger.error(f"Failed to initialize AsyncClient: {e}")
                        raise e
    
    def _run_coroutine_threadsafe(self, coro):
        """Run coroutine in WebSocket event loop"""
        self._ensure_ws_event_loop()
        try:
            future = asyncio.run_coroutine_threadsafe(coro, self._ws_loop)
            try:
                return future.result(timeout=self.websocket_timeout)
            except asyncio.TimeoutError:
                self.logger.error(f"WebSocket request timed out after {self.websocket_timeout} seconds")
                future.cancel()  # Cancel the timed out task
                raise TimeoutError(f"WebSocket request timed out after {self.websocket_timeout} seconds")
            except Exception as e:
                self.logger.error(f"Error during WebSocket request: {e}")
                raise
        except Exception as e:
            self.logger.error(f"Failed to schedule coroutine on WebSocket event loop: {e}")
            raise

    def _format_timeframe(self, timeframe: str) -> str:
        """
        Convert our timeframe format to Binance's format.

        Args:
            timeframe: Timeframe in format like '1h', '5m', '1d'

        Returns:
            Binance format of the timeframe
        """
        # Binance uses the same format as our system
        return timeframe

    def get_historical_data_via_websocket(
            self,
            symbol: str,
            timeframe: str,
            end_time: Optional[datetime] = None,
            limit: int = 500
    ) -> pd.DataFrame:
        """
        Get historical klines data using WebSocket API.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Time interval (e.g., '1h', '5m', '1d')
            end_time: End time for historical data
            limit: Number of candles to retrieve

        Returns:
            DataFrame with historical price data
        """
        formatted_symbol = symbol.replace("/", "")
        
        # Log the WebSocket request
        if end_time:
            self.logger.info(f"Using WebSocket to get historical data for {formatted_symbol} {timeframe} ending at {end_time}")
        else:
            self.logger.info(f"Using WebSocket to get historical data for {formatted_symbol} {timeframe}")
        
        self.logger.info(f"Collecting WebSocket data for {self.websocket_collection_time} seconds")
        
        retry_count = 0
        while retry_count <= self.max_retries:
            try:
                # Convert parameters for WebSocket request
                params = {
                    "symbol": formatted_symbol,
                    "interval": self._format_timeframe(timeframe),
                    "limit": limit
                }
                
                if end_time:
                    params["endTime"] = int(end_time.timestamp() * 1000)
                
                # Execute WebSocket request in dedicated event loop
                async def ws_request():
                    # Get data using async client's WebSocket API
                    if self.async_client is None:
                        raise ValueError("AsyncClient not initialized")
                    return await self.async_client.ws_get_klines(**params)
                
                try:
                    # Run request in WebSocket event loop
                    ws_data = self._run_coroutine_threadsafe(ws_request())
                    
                    # Log WebSocket returned data in detail for debugging
                    ws_data = self._log_websocket_data(ws_data, formatted_symbol, timeframe)
                    
                    # Extract kline data, supporting different return formats
                    klines = []
                    
                    if ws_data is None:
                        self.logger.warning(f"No data received for {formatted_symbol} {timeframe}")
                    elif isinstance(ws_data, list):
                        # If directly returned a list of klines
                        self.logger.debug(f"WebSocket returned a list with {len(ws_data)} items")
                        klines = ws_data
                    elif isinstance(ws_data, dict):
                        # If returned a dictionary containing result
                        if "result" in ws_data:
                            klines = ws_data["result"]
                            self.logger.debug(f"WebSocket returned a dict with result containing {len(klines)} items")
                        else:
                            # Log received keys for debugging
                            self.logger.warning(f"WebSocket returned a dict without 'result' key. Keys: {list(ws_data.keys())}")
                            if "data" in ws_data:  # Some APIs might use data instead of result
                                klines = ws_data["data"]
                            elif len(ws_data) == 1 and isinstance(list(ws_data.values())[0], list):
                                # Maybe only one key contains kline data
                                klines = list(ws_data.values())[0]
                    else:
                        # Unknown return type
                        self.logger.warning(f"Unexpected WebSocket response type: {type(ws_data)}")
                    
                    # If no data received, retry or return empty DataFrame
                    if not klines:
                        if retry_count < self.max_retries:
                            retry_count += 1
                            self.logger.warning(f"No klines data received for {formatted_symbol} {timeframe}, retrying {retry_count}/{self.max_retries}")
                            time.sleep(1)  # Wait before retrying
                            continue
                        else:
                            self.logger.warning(f"No klines data received for {formatted_symbol} {timeframe} after {self.max_retries} retries")
                            return pd.DataFrame()
                    
                    # Process data to DataFrame
                    df = pd.DataFrame(klines, columns=COLUMNS)
                    
                    # Convert types
                    for col in NUMERIC_COLUMNS:
                        df[col] = pd.to_numeric(df[col])
                    
                    # Convert timestamps to datetime
                    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
                    
                    return df
                    
                except Exception as e:
                    self.logger.error(f"Error processing WebSocket data: {e}")
                    raise e
                    
            except Exception as e:
                if retry_count < self.max_retries:
                    retry_count += 1
                    self.logger.warning(f"Error fetching WebSocket data for {formatted_symbol} {timeframe}: {e}, retrying {retry_count}/{self.max_retries}")
                    time.sleep(1)  # Wait before retrying
                else:
                    self.logger.error(f"Failed to get WebSocket data for {formatted_symbol} {timeframe} after {self.max_retries} retries: {e}")
                    return pd.DataFrame()
        
        return pd.DataFrame()

    def get_historical_klines(
            self,
            symbol: str,
            timeframe: str,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None,
            use_cache: bool = True,
            use_websocket: Optional[bool] = None
    ) -> pd.DataFrame:
        """
        Get historical klines (candlestick data) for a symbol and timeframe.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Time interval (e.g., '1h', '5m', '1d')
            start_date: Start date for historical data
            end_date: End date for historical data
            use_cache: Whether to use cached data if available
            use_websocket: Whether to use WebSocket (defaults to class setting)

        Returns:
            DataFrame with historical price data
        """
        # Determine whether to use WebSocket
        if use_websocket is None:
            use_websocket = self.use_websocket_by_default

        # Format the symbol if needed (remove / if present)
        formatted_symbol = symbol.replace("/", "")

        # Default to 30 days of data if no start date is provided
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)

        if end_date is None:
            end_date = datetime.now()

        # Create cache file path
        cache_file = self.cache_dir / f"{formatted_symbol}_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"

        # Check if cache file exists and is fresh
        if use_cache and cache_file.exists():
            self.logger.info(f"Loading cached data for {formatted_symbol} {timeframe}")
            return pd.read_csv(cache_file, parse_dates=['open_time', 'close_time'])

        self.logger.info(f"Fetching historical data for {formatted_symbol} {timeframe}")

        # Try using WebSocket if enabled
        if use_websocket:
            try:
                # First try WebSocket
                df = self.get_historical_data_via_websocket(
                    symbol=formatted_symbol, 
                    timeframe=timeframe,
                    end_time=end_date
                )
                
                if not df.empty:
                    # Filter by start_date if provided
                    if start_date:
                        df = df[df['open_time'] >= pd.Timestamp(start_date)]
                    
                    # Cache the data
                    if use_cache:
                        df.to_csv(cache_file, index=False)
                    
                    return df
                else:
                    self.logger.warning(f"WebSocket data fetch failed, falling back to REST API for {formatted_symbol} {timeframe}")
            except Exception as e:
                self.logger.warning(f"WebSocket error, falling back to REST API: {e}")
        
        # Fall back to REST API
        try:
            # Convert datetime to milliseconds timestamp
            start_ts = int(start_date.timestamp() * 1000)
            end_ts = int(end_date.timestamp() * 1000)

            # Use the client to get historical klines
            klines = self.client.get_historical_klines(
                symbol=formatted_symbol,
                interval=self._format_timeframe(timeframe),
                start_str=start_ts,
                end_str=end_ts
            )

            df = pd.DataFrame(klines, columns=COLUMNS)
            # Convert types
            for col in NUMERIC_COLUMNS:
                df[col] = pd.to_numeric(df[col])

            # Convert timestamps to datetime
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

            # Cache the data
            if use_cache:
                df.to_csv(cache_file, index=False)

            return df

        except Exception as e:
            self.logger.error(f"Error fetching historical data for {formatted_symbol} {timeframe}: {e}")
            return pd.DataFrame()

    def get_multiple_timeframes_with_end_time(
            self,
            symbol: str,
            timeframes: List[str],
            end_time: str,
            limit: int = 500,
            use_websocket: Optional[bool] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple timeframes for a single symbol.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframes: List of timeframes (e.g., ['5m', '15m', '1h'])
            end_time: end time for multiple timeframes
            limit: Maximum number of timeframes to fetch
            use_websocket: Whether to use WebSocket (defaults to class setting)
        Returns:
            Dictionary of DataFrames for each timeframe
        """
        result = {}
        for timeframe in timeframes:
            df = self.get_history_klines_with_end_time(
                symbol=symbol, 
                timeframe=timeframe, 
                end_time=end_time,
                limit=limit,
                use_websocket=use_websocket
            )
            result[timeframe] = df

        return result

    def get_history_klines_with_end_time(
            self,
            symbol: str,
            timeframe: str,
            end_time: datetime,
            limit: int = 500,
            use_websocket: Optional[bool] = None
    ) -> pd.DataFrame:
        """
        Get historical klines with specific end time.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: time interval (e.g., '1h', '5m', '1d')
            end_time: end time for timeframes data
            limit: Maximum number of timeframes to fetch
            use_websocket: Whether to use WebSocket (defaults to class setting)
        Returns:
            DataFrame with kline data
        """
        # Determine whether to use WebSocket
        if use_websocket is None:
            use_websocket = self.use_websocket_by_default
            
        formatted_symbol = symbol.replace("/", "")
        
        # Try using WebSocket if enabled
        if use_websocket:
            try:
                # First try WebSocket
                df = self.get_historical_data_via_websocket(
                    symbol=formatted_symbol, 
                    timeframe=timeframe,
                    end_time=end_time,
                    limit=limit
                )
                
                if not df.empty:
                    return df
                else:
                    self.logger.warning(f"WebSocket data fetch failed, falling back to REST API for {formatted_symbol} {timeframe}")
            except Exception as e:
                self.logger.warning(f"WebSocket error, falling back to REST API: {e}")
        
        # Fall back to REST API
        try:
            # Use the client to get klines
            klines = self.client.futures_historical_klines_with_end_time(
                symbol=formatted_symbol,
                interval=self._format_timeframe(timeframe),
                end_str=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                limit=limit
            )

            df = pd.DataFrame(klines, columns=COLUMNS)

            for col in NUMERIC_COLUMNS:
                df[col] = pd.to_numeric(df[col])

            # Convert timestamps to datetime
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

            return df

        except Exception as e:
            self.logger.error(f"Error fetching latest data for {formatted_symbol} {timeframe}: {e}")
            return pd.DataFrame()

    def get_latest_multi_timeframe_data(
            self,
            symbol: str,
            timeframes: List[str],
            use_websocket: Optional[bool] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple timeframes for a single symbol.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframes: List of timeframes (e.g., ['5m', '15m', '1h'])
            use_websocket: Whether to use WebSocket (defaults to class setting)
        Returns:
            Dictionary of DataFrames for each timeframe
        """
        result = {}

        for timeframe in timeframes:
            df = self.get_latest_data(
                symbol=symbol,
                timeframe=timeframe,
                use_websocket=use_websocket
            )

            if not df.empty:
                result[timeframe] = df
            else:
                self.logger.warning(f"Warning: No data retrieved for {symbol} {timeframe}")
        return result

    def get_multi_timeframe_data(
            self,
            symbol: str,
            timeframes: List[str],
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None,
            use_websocket: Optional[bool] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple timeframes for a single symbol.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframes: List of timeframes (e.g., ['5m', '15m', '1h'])
            start_date: Start date for historical data
            end_date: End date for historical data
            use_websocket: Whether to use WebSocket (defaults to class setting)

        Returns:
            Dictionary of DataFrames for each timeframe
        """
        result = {}

        for timeframe in timeframes:
            df = self.get_historical_klines(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                use_websocket=use_websocket
            )

            if not df.empty:
                result[timeframe] = df
            else:
                self.logger.warning(f"Warning: No data retrieved for {symbol} {timeframe}")

        return result

    def get_latest_data(
            self, 
            symbol: str, 
            timeframe: str, 
            limit: int = 1000,
            use_websocket: Optional[bool] = None
    ) -> pd.DataFrame:
        """
        Get the latest candlestick data for a symbol and timeframe.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Time interval (e.g., '1h', '5m', '1d')
            limit: Number of candles to retrieve
            use_websocket: Whether to use WebSocket (defaults to class setting)

        Returns:
            DataFrame with the latest price data
        """
        # Determine whether to use WebSocket
        if use_websocket is None:
            use_websocket = self.use_websocket_by_default
            
        # Format the symbol if needed
        formatted_symbol = symbol.replace("/", "")
        
        # Try using WebSocket if enabled
        if use_websocket:
            try:
                # First try WebSocket
                df = self.get_historical_data_via_websocket(
                    symbol=formatted_symbol, 
                    timeframe=timeframe,
                    limit=limit
                )
                
                if not df.empty:
                    return df
                else:
                    self.logger.warning(f"WebSocket data fetch failed, falling back to REST API for {formatted_symbol} {timeframe}")
            except Exception as e:
                self.logger.warning(f"WebSocket error, falling back to REST API: {e}")

        # Fall back to REST API
        try:
            # Use the client to get klines
            klines = self.client.get_klines(
                symbol=formatted_symbol,
                interval=self._format_timeframe(timeframe),
                limit=limit
            )

            # Create a DataFrame from the klines data
            df = pd.DataFrame(klines, columns=COLUMNS)

            # Convert types
            for col in NUMERIC_COLUMNS:
                df[col] = pd.to_numeric(df[col])

            # Convert timestamps to datetime
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

            return df

        except Exception as e:
            self.logger.error(f"Error fetching latest data for {formatted_symbol} {timeframe}: {e}")
            return pd.DataFrame()

    def _log_websocket_data(self, ws_data, symbol, timeframe):
        """Log WebSocket returned data for debugging"""
        # Log the type and basic information of the returned data
        self.logger.debug(f"WebSocket data for {symbol} {timeframe}: Type={type(ws_data)}")
        
        # More detailed logging for different types of data
        if ws_data is None:
            self.logger.debug("WebSocket returned None")
        elif isinstance(ws_data, list):
            self.logger.debug(f"WebSocket returned list with {len(ws_data)} items")
            if ws_data and len(ws_data) > 0:
                self.logger.debug(f"First item type: {type(ws_data[0])}")
                # Log the first few elements of the first item
                if isinstance(ws_data[0], list) and len(ws_data[0]) > 0:
                    sample = ws_data[0][:min(5, len(ws_data[0]))]
                    self.logger.debug(f"Sample of first item: {sample}")
        elif isinstance(ws_data, dict):
            self.logger.debug(f"WebSocket returned dict with keys: {list(ws_data.keys())}")
            # Check common keys
            for key in ['result', 'data', 'id', 'status']:
                if key in ws_data:
                    value = ws_data[key]
                    if isinstance(value, list):
                        self.logger.debug(f"Key '{key}' contains a list with {len(value)} items")
                    else:
                        self.logger.debug(f"Key '{key}' contains: {value}")
        else:
            # For other types, log string representation
            try:
                self.logger.debug(f"WebSocket returned: {str(ws_data)[:500]}")
            except Exception as e:
                self.logger.debug(f"Could not stringify WebSocket data: {e}")
                
        return ws_data

    def close_connections(self):
        """Close all WebSocket and AsyncClient connections"""
        if self._ws_loop is not None and self.async_client is not None:
            try:
                # Create a close task
                future = asyncio.run_coroutine_threadsafe(
                    self.async_client.close_connection(),
                    self._ws_loop
                )
                # Wait for completion
                future.result(timeout=5)
                self.logger.info("AsyncClient connections closed successfully")
            except Exception as e:
                self.logger.error(f"Error closing AsyncClient connections: {e}")
            
            # Stop the event loop
            try:
                self._ws_loop.call_soon_threadsafe(self._ws_loop.stop)
                self.logger.info("WebSocket event loop stopped")
            except Exception as e:
                self.logger.error(f"Error stopping WebSocket event loop: {e}")
        
        self.async_client = None
    
    def __del__(self):
        """Destructor, ensure resources are properly released"""
        try:
            self.close_connections()
        except Exception as e:
            # Cannot raise exceptions in destructor
            print(f"Error during BinanceDataProvider cleanup: {e}")


# Simple test function
def test_data_provider():
    # Use default HTTP API configuration
    provider = BinanceDataProvider()
    symbol = "BTCUSDT"
    timeframe = "1h"

    # Get latest data
    print("=== Testing with HTTP API ===")
    df = provider.get_latest_data(symbol, timeframe, limit=10, use_websocket=False)
    print(f"Latest data for {symbol} {timeframe}:")
    print(df.head())

    # Get historical data
    start_date = datetime.now() - timedelta(days=7)
    df_hist = provider.get_historical_klines(symbol, timeframe, start_date, use_websocket=False)
    print(f"\nHistorical data for {symbol} {timeframe}:")
    print(f"Retrieved {len(df_hist)} records")
    print(df_hist.head())

    # Test WebSocket API
    print("\n=== Testing with WebSocket API ===")
    # Create data provider with WebSocket support
    ws_provider = BinanceDataProvider(
        use_websocket_by_default=True,
        websocket_timeout=10,
        websocket_collection_time=30,
        max_retries=2
    )
    
    # Get latest data
    df_ws = ws_provider.get_latest_data(symbol, timeframe, limit=10)
    print(f"WebSocket latest data for {symbol} {timeframe}:")
    print(f"Retrieved {len(df_ws)} records")
    if not df_ws.empty:
        print(df_ws.head())
    else:
        print("No data received from WebSocket")


if __name__ == "__main__":
    test_data_provider()
