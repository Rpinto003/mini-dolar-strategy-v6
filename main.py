import yaml
from pathlib import Path
from loguru import logger

from src.strategy.enhanced_v3 import EnhancedStrategyV3
from src.data.loaders.market_data import MarketDataLoader

def load_config():
    config_path = Path("config/strategy_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

def main():
    # Load configuration
    config = load_config()
    
    # Initialize components
    data_loader = MarketDataLoader(
        db_path=config['data']['db_path'],
        table_name=config['data']['table_name']
    )
    
    strategy = EnhancedStrategyV3(config=config['strategy'])
    
    # Load and process data
    data = data_loader.get_minute_data(
        interval=config['data']['interval'],
        start_date=config['backtest']['start_date'],
        end_date=config['backtest']['end_date']
    )
    
    # Run strategy
    results = strategy.run(data)
    
    # Generate and save reports
    strategy.generate_reports(results)

if __name__ == "__main__":
    main()