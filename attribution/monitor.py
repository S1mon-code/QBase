"""Real-time strategy monitoring interface (future implementation).

This module defines the interface for real-time monitoring that will be
implemented when strategies move to paper trading / live trading.
"""
import numpy as np


class StrategyMonitor:
    """Monitors strategy health in real-time."""

    def on_daily_close(self, portfolio_state: dict) -> dict:
        """Called after each trading day closes.

        Checks: rolling Sharpe, drawdown, trade frequency, regime shift.

        Args:
            portfolio_state: dict with keys like 'equity', 'positions',
                'date', 'strategy_equities', etc.

        Returns:
            dict with alerts and recommendations:
                {
                    "alerts": [{"level": "warning"|"critical", "msg": str}],
                    "metrics": {"rolling_sharpe": float, "current_dd": float, ...},
                    "recommendations": [str],
                }
        """
        raise NotImplementedError("Implement when entering paper/live trading phase")

    def check_degradation(self, strategy_version: str, recent_equity: np.ndarray) -> dict:
        """Check if a strategy is degrading based on recent performance.

        Compares recent rolling Sharpe / win-rate against historical baseline.

        Args:
            strategy_version: e.g. "v12"
            recent_equity: recent equity curve (e.g. last 60 bars)

        Returns:
            dict with:
                "degraded": bool,
                "confidence": float (0-1),
                "metrics": {"recent_sharpe": float, "baseline_sharpe": float, ...},
                "recommendation": str,
        """
        raise NotImplementedError("Implement when entering paper/live trading phase")

    def check_regime_shift(self, recent_prices: np.ndarray, recent_volumes: np.ndarray) -> dict:
        """Detect if market regime has shifted significantly.

        Compares current regime indicators (ADX, ATR percentile, volume ratio)
        against the regime during which the portfolio was optimized.

        Args:
            recent_prices: recent close prices (e.g. last 60 bars)
            recent_volumes: recent volume data (e.g. last 60 bars)

        Returns:
            dict with:
                "shifted": bool,
                "current_regime": {"trend": str, "volatility": str, "volume": str},
                "baseline_regime": {"trend": str, "volatility": str, "volume": str},
                "recommendation": str,
        """
        raise NotImplementedError("Implement when entering paper/live trading phase")
