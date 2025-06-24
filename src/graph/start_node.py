"""
start node
"""

from typing import Dict, Any

from .base_node import BaseNode, AgentState
from src.utils import BinanceDataProvider, OkxDataProvider, settings

# Initialize data provider
if settings.exchange.lower() == "okx":
    data_provider = OkxDataProvider()
else:
    data_provider = BinanceDataProvider()


class StartNode(BaseNode):
    """
    start node
    """
    def __call__(self, state: AgentState) -> Dict[str, Any]:
        data = state['data']
        data['name'] = "StartNode"
        return state
