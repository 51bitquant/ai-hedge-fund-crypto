import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from dotenv import load_dotenv
from src.utils import settings
from src.backtest.backtester import Backtester


load_dotenv()


if __name__ == "__main__":

    backtester = Backtester(
        primary_interval=settings.primary_interval,
        intervals=settings.signals.intervals,
        tickers=settings.signals.tickers,
        start_date=settings.start_date,
        end_date=settings.end_date,
        initial_capital=settings.initial_cash,
        strategies=settings.signals.strategies,
        show_agent_graph=settings.show_agent_graph,
        show_reasoning=settings.show_reasoning,
        model_name=settings.model.name,
        model_provider=settings.model.provider,
    )

    print("Starting backtest...")
    performance_metrics = backtester.run_backtest()
    performance_df = backtester.analyze_performance()