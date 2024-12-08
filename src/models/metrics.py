import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class TradeMetrics:
    entry_price: float
    exit_price: float
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    position_size: float
    pnl: float
    return_pct: float
    trade_duration: pd.Timedelta

class MetricsCalculator:
    def __init__(self, risk_free_rate: float = 0.03):
        self.risk_free_rate = risk_free_rate
        
    def calculate_trade_metrics(self, trades: pd.DataFrame) -> List[TradeMetrics]:
        """Calculate metrics for individual trades"""
        trade_metrics = []
        
        for _, trade in trades.iterrows():
            metrics = TradeMetrics(
                entry_price=trade['entry_price'],
                exit_price=trade['exit_price'],
                entry_time=trade['entry_time'],
                exit_time=trade['exit_time'],
                position_size=trade['position_size'],
                pnl=trade['net_pnl'],
                return_pct=trade['return'],
                trade_duration=trade['exit_time'] - trade['entry_time']
            )
            trade_metrics.append(metrics)
            
        return trade_metrics
        
    def calculate_portfolio_metrics(self, trades: List[TradeMetrics]) -> Dict:
        """Calculate portfolio-level metrics"""
        if not trades:
            return {}
            
        returns = [t.return_pct for t in trades]
        pnls = [t.pnl for t in trades]
        
        metrics = {
            'total_pnl': sum(pnls),
            'total_return': np.prod(1 + np.array(returns)) - 1,
            'win_rate': len([t for t in trades if t.pnl > 0]) / len(trades),
            'avg_trade_duration': pd.Timedelta(sum([t.trade_duration for t in trades])) / len(trades),
            'sharpe_ratio': self._calculate_sharpe(returns),
            'sortino_ratio': self._calculate_sortino(returns),
            'max_drawdown': self._calculate_max_drawdown(returns)
        }
        
        return metrics
        
    def _calculate_sharpe(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if not returns:
            return 0
        returns_arr = np.array(returns)
        excess_returns = returns_arr - self.risk_free_rate/252
        return np.sqrt(252) * excess_returns.mean() / returns_arr.std()
        
    def _calculate_sortino(self, returns: List[float]) -> float:
        """Calculate Sortino ratio"""
        if not returns:
            return 0
        returns_arr = np.array(returns)
        downside_returns = returns_arr[returns_arr < 0]
        if len(downside_returns) == 0:
            return np.inf
        excess_returns = returns_arr - self.risk_free_rate/252
        downside_std = np.sqrt(np.mean(downside_returns**2))
        return np.sqrt(252) * excess_returns.mean() / downside_std
        
    @staticmethod
    def _calculate_max_drawdown(returns: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not returns:
            return 0
        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        return drawdowns.min()