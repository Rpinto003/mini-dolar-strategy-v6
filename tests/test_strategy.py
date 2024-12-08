import pytest
import pandas as pd
import numpy as np
from src.strategy.enhanced_v3 import EnhancedStrategyV3

@pytest.fixture
def sample_data():
    """Create sample market data for testing"""
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1min')
    data = pd.DataFrame({
        'time': dates,
        'open': np.random.randn(len(dates)).cumsum() + 100,
        'high': np.random.randn(len(dates)).cumsum() + 101,
        'low': np.random.randn(len(dates)).cumsum() + 99,
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, len(dates))
    })
    return data

def test_strategy_initialization():
    """Test strategy initialization"""
    strategy = EnhancedStrategyV3()
    assert strategy.market_structure is not None
    assert strategy.volatility_analyzer is not None
    assert strategy.signal_generator is not None
    assert strategy.risk_manager is not None

def test_data_validation(sample_data):
    """Test input data validation"""
    strategy = EnhancedStrategyV3()
    assert strategy.validate_data(sample_data) == True
    
    # Test missing columns
    invalid_data = sample_data.drop('volume', axis=1)
    assert strategy.validate_data(invalid_data) == False

def test_signal_generation(sample_data):
    """Test signal generation"""
    strategy = EnhancedStrategyV3()
    results = strategy.run(sample_data)
    
    assert 'signal' in results.columns
    assert set(results['signal'].unique()).issubset({-1, 0, 1})

def test_risk_management(sample_data):
    """Test risk management calculations"""
    strategy = EnhancedStrategyV3()
    results = strategy.run(sample_data)
    
    signals = results[results['signal'] != 0]
    if len(signals) > 0:
        assert 'position_size' in results.columns
        assert 'stop_loss' in results.columns
        assert 'take_profit' in results.columns