import yaml
from pathlib import Path
import pandas as pd
from loguru import logger

from src.strategy.enhanced_v3 import EnhancedStrategyV3
from src.data.loaders.market_data import MarketDataLoader

def load_config(config_path: str = "config/strategy_config.yaml") -> dict:
    """Load strategy configuration"""
    with open(config_path) as f:
        return yaml.safe_load(f)

def main():
    # Load configuration
    config = load_config()
    
    # Initialize data loader
    data_loader = MarketDataLoader(
        db_path=config['data']['db_path'],
        table_name=config['data']['table_name']
    )
    
    # Load training data
    data = data_loader.get_minute_data(
        interval=config['data']['interval'],
        start_date=config['backtest']['start_date'],
        end_date=config['backtest']['end_date']
    )
    
    if data.empty:
        logger.error("No data loaded for training")
        return
        
    logger.info(f"Loaded {len(data)} records for training")
    
    # Initialize strategy
    strategy = EnhancedStrategyV3(config=config['strategy'])
    
    try:
        # Run parameter optimization if specified
        if config.get('optimize_parameters', False):
            param_grid = {
                'lookback_period': [10, 20, 30],
                'atr_period': [10, 14, 21],
                'risk_per_trade': [0.005, 0.01, 0.02]
            }
            
            optimization_results = strategy.optimize_parameters(data, param_grid)
            logger.info(f"Optimal parameters found: {optimization_results['params']}")
            
            # Update strategy with optimal parameters
            strategy.config.update(optimization_results['params'])
            strategy.initialize_components()
        
        # Run strategy with final parameters
        results = strategy.run(data)
        
        if results.empty:
            logger.error("Strategy returned no results")
            return
            
        # Save results
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Save trade results
        trades_df = results[results['signal'] != 0] if 'signal' in results.columns else pd.DataFrame()
        if not trades_df.empty:
            trades_df.to_csv(output_dir / "trades.csv", index=False)
            logger.info(f"Saved {len(trades_df)} trades to trades.csv")
        
        # Save metrics
        if hasattr(strategy, 'metrics') and strategy.metrics:
            pd.DataFrame([strategy.metrics]).to_csv(
                output_dir / "metrics.csv", index=False
            )
            logger.info("Saved strategy metrics to metrics.csv")
        
        logger.info("Strategy training and evaluation completed")
        
    except Exception as e:
        logger.error(f"Error during strategy training: {str(e)}")
        raise

if __name__ == "__main__":
    main()