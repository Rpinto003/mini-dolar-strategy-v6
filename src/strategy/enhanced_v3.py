from typing import Dict, Optional
import pandas as pd
import numpy as np
from loguru import logger

from src.analysis.market_structure import MarketStructure
from src.analysis.volatility import VolatilityAnalyzer
from src.analysis.signals import SignalGenerator
from src.analysis.risk_management import RiskManager
from src.strategy.base import BaseStrategy
from src.models.metrics import MetricsCalculator

class EnhancedStrategyV3(BaseStrategy):
    def __init__(self, config: Dict = None):
        super().__init__()
        self.config = config or {}
        self.initialize_components()
        logger.info("Initialized EnhancedStrategyV3")

    def initialize_components(self):
        self.market_structure = MarketStructure(
            lookback_period=self.config.get('lookback_period', 20)
        )
        self.volatility_analyzer = VolatilityAnalyzer(
            atr_period=self.config.get('atr_period', 14)
        )
        self.signal_generator = SignalGenerator(
            risk_free_rate=self.config.get('risk_free_rate', 0.05)
        )
        self.risk_manager = RiskManager(
            initial_capital=self.config.get('initial_capital', 100000),
            max_risk_per_trade=self.config.get('max_risk_per_trade', 0.01)
        )

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run the complete strategy pipeline"""
        logger.info("Starting strategy execution")
        
        try:
            # Validate input data
            if not self.validate_data(data):
                logger.error("Data validation failed")
                return pd.DataFrame()
                
            # 1. Market Structure Analysis
            df = self.market_structure.identify_structure(data)
            if df.empty:
                logger.error("Market structure analysis failed")
                return pd.DataFrame()
                
            # Log data state
            logger.debug(f"Columns after market structure: {df.columns.tolist()}")
            
            # 2. Generate signals
            df = self.signal_generator.generate_signals(df)
            n_signals = len(df[df['signal'] != 0]) if 'signal' in df.columns else 0
            logger.info(f"Generated {n_signals} signals")
            
            # 3. Apply risk management
            if n_signals > 0:
                df = self.risk_manager.apply_risk_management(df)
                logger.info("Risk management applied")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in strategy execution: {str(e)}")
            logger.debug(f"Available columns: {data.columns.tolist()}")
            return pd.DataFrame()
 
    def calculate_strategy_metrics(self, data: pd.DataFrame):
        """Calculate and store strategy performance metrics"""
        try:
            trades = data[data['signal'] != 0].copy()
            
            # Calculate trade metrics
            trade_metrics = self.metrics_calculator.calculate_trade_metrics(trades)
            
            # Calculate portfolio metrics
            portfolio_metrics = self.metrics_calculator.calculate_portfolio_metrics(trade_metrics)
            
            # Store metrics
            self.metrics = portfolio_metrics
            
            # Log key metrics
            logger.info("Strategy Performance:")
            logger.info(f"Total Trades: {portfolio_metrics['total_trades']}")
            logger.info(f"Win Rate: {portfolio_metrics['win_rate']:.2%}")
            logger.info(f"Total Return: {portfolio_metrics['total_return']:.2%}")
            logger.info(f"Sharpe Ratio: {portfolio_metrics['sharpe_ratio']:.2f}")
            
        except Exception as e:
            logger.error(f"Error calculating strategy metrics: {str(e)}")

    def optimize_parameters(self, data: pd.DataFrame, param_grid: Dict) -> Dict:
        """
        Optimize strategy parameters using grid search
        """
        best_metrics = None
        best_params = None
        best_sharpe = float('-inf')
        
        logger.info("Starting parameter optimization")
        
        for params in self._generate_param_combinations(param_grid):
            # Update configuration
            self.config.update(params)
            self.initialize_components()
            
            # Run strategy
            results = self.run(data)
            
            # Check performance
            if self.metrics.get('sharpe_ratio', float('-inf')) > best_sharpe:
                best_sharpe = self.metrics['sharpe_ratio']
                best_metrics = self.metrics.copy()
                best_params = params.copy()
                
        logger.info(f"Optimization completed. Best Sharpe: {best_sharpe:.2f}")
        return {'params': best_params, 'metrics': best_metrics}

    def analyze_features(self, data: pd.DataFrame):
        """Analyze feature importance and characteristics"""
        try:
            # Get signals
            signals = data[data['signal'] != 0].copy()
            
            if len(signals) == 0:
                return
                
            # Calculate returns
            signals['return'] = signals.apply(
                lambda x: (x['close'].shift(-1) - x['close']) / x['close'] if x['signal'] == 1
                else (x['close'] - x['close'].shift(-1)) / x['close'],
                axis=1
            )
            
            # Separate winning and losing trades
            winning_trades = signals[signals['return'] > 0]
            losing_trades = signals[signals['return'] <= 0]
            
            logger.info("\nFeature Analysis for Trades:")
            logger.info(f"Total Trades: {len(signals)}")
            logger.info(f"Winning Trades: {len(winning_trades)}")
            logger.info(f"Losing Trades: {len(losing_trades)}")
            
            # Analyze features for winning vs losing trades
            features = ['rsi', 'macd', 'volume_ratio', 'price_position', 'trend_direction']
            
            logger.info("\nFeature Characteristics for Winning Trades:")
            for feature in features:
                if len(winning_trades) > 0:
                    avg_win = winning_trades[feature].mean()
                    std_win = winning_trades[feature].std()
                    logger.info(f"{feature}: mean={avg_win:.2f}, std={std_win:.2f}")
            
            logger.info("\nFeature Characteristics for Losing Trades:")
            for feature in features:
                if len(losing_trades) > 0:
                    avg_lose = losing_trades[feature].mean()
                    std_lose = losing_trades[feature].std()
                    logger.info(f"{feature}: mean={avg_lose:.2f}, std={std_lose:.2f}")
                    
        except Exception as e:
            logger.error(f"Error in feature analysis: {str(e)}")

    @staticmethod
    def _generate_param_combinations(param_grid: Dict):
        """Generate combinations of parameters for optimization"""
        import itertools
        
        keys = param_grid.keys()
        values = param_grid.values()
        for instance in itertools.product(*values):
            yield dict(zip(keys, instance))