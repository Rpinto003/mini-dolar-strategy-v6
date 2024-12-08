from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Optional
from loguru import logger

class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self):
        self.metrics = {}
        self.trades = []
        
    @abstractmethod
    def initialize_components(self):
        """Initialize strategy components"""
        pass
        
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data requirements"""
        required_columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_columns if col not in data.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
            
        if data.empty:
            logger.error("Empty dataset provided")
            return False
            
        return True
        
    def calculate_metrics(self, results: pd.DataFrame) -> Dict:
        """Calculate strategy performance metrics"""
        trades = results[results['signal'] != 0].copy()
        
        if len(trades) == 0:
            logger.warning("No trades found for metric calculation")
            return {}
            
        metrics = {
            'total_trades': len(trades),
            'win_rate': (trades['net_pnl'] > 0).mean(),
            'avg_profit': trades['net_pnl'].mean(),
            'max_drawdown': self.calculate_max_drawdown(trades),
            'sharpe_ratio': self.calculate_sharpe_ratio(trades),
            'profit_factor': self.calculate_profit_factor(trades)
        }
        
        return metrics
        
    @staticmethod
    def calculate_max_drawdown(trades: pd.DataFrame) -> float:
        cumulative = (1 + trades['net_pnl']).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        return drawdowns.min() if len(drawdowns) > 0 else 0
        
    @staticmethod
    def calculate_sharpe_ratio(trades: pd.DataFrame, risk_free_rate: float = 0.03) -> float:
        returns = trades['net_pnl']
        if len(returns) < 2:
            return 0
        excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
        return np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
        
    @staticmethod
    def calculate_profit_factor(trades: pd.DataFrame) -> float:
        profits = trades[trades['net_pnl'] > 0]['net_pnl'].sum()
        losses = abs(trades[trades['net_pnl'] < 0]['net_pnl'].sum())
        return profits / losses if losses != 0 else 0