import asyncio
import time
import unittest
from unittest.mock import patch, MagicMock, AsyncMock

from src.gateway.binance.ws.threaded_stream import ThreadedApiManager


class TestThreadedApiManager(unittest.TestCase):
    @patch("src.gateway.binance.ws.threaded_stream.AsyncClient.create", new_callable=AsyncMock)
    def test_thread_lifecycle_and_loop_management(self, mock_async_client):
        """Tests the thread's startup, shutdown, and event loop cleanup."""
        manager = ThreadedApiManager(api_key="test", api_secret="test")

        self.assertIsNone(manager._loop)
        self.assertTrue(manager._running)

        manager.start()
        time.sleep(0.1)  # Give the thread time to start and create the loop

        self.assertTrue(manager.is_alive())
        self.assertIsNotNone(manager._loop)
        self.assertTrue(manager._loop.is_running())

        manager.stop()
        manager.join(timeout=5)  # Wait for the thread to terminate

        self.assertFalse(manager.is_alive())
        self.assertFalse(manager._running)
        # After the thread stops, the loop should be closed.
        self.assertTrue(manager._loop.is_closed())
        mock_async_client.assert_called_once()

    @patch("src.gateway.binance.ws.threaded_stream.AsyncClient.create", new_callable=AsyncMock)
    def test_callback_is_called_on_message(self, mock_async_client):
        """Tests that the callback is invoked when a message is received."""
        manager = ThreadedApiManager(api_key="test", api_secret="test")

        # Mock the async context manager for the socket and its recv method
        mock_socket = AsyncMock()
        mock_recv = mock_socket.__aenter__.return_value.recv

        # Setup the mock to simulate receiving one message
        test_msg = {"data": "test_message"}
        mock_recv.return_value = test_msg

        mock_callback = MagicMock()
        path = "test_path"

        # This wrapper callback will call the mock and then stop the listener loop
        def callback_with_side_effect(msg):
            mock_callback(msg)
            manager.stop_socket(path)

        manager._socket_running[path] = True

        async def test_runner():
            # This coroutine will be run in the manager's event loop
            await manager.start_listener(mock_socket, path, callback_with_side_effect)

        manager.start()
        # Wait until the loop is actually running to avoid a race condition
        while not (manager._loop and manager._loop.is_running()):
            time.sleep(0.01)

        # Run the test coroutine in the thread's event loop
        future = asyncio.run_coroutine_threadsafe(test_runner(), manager._loop)
        future.result(timeout=5)  # Wait for completion

        manager.stop()
        manager.join()

        # Assertions
        mock_callback.assert_called_once_with(test_msg)
        self.assertFalse(path in manager._socket_running)


if __name__ == "__main__":
    unittest.main() 