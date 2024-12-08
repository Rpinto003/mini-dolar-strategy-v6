import pytest
import pandas as pd
import numpy as np
from src.analysis.market_structure import MarketStructure

@pytest.fixture
def sample_market_data():
    # Create sample data with known patterns
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1min')
    data = pd.DataFrame({
        'time': dates,
        'open': np.random.randn(len(dates)).cumsum() + 100,
        'high': np.random.randn(len(dates)).cumsum() + 101,
        'low': np.random.randn(len(dates)).cumsum() + 99,
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, len(dates))
    })
    
    # Add known trend
    data['ma_fast'] = data['close'].rolling(9).mean()
    data['ma_slow'] = data['close'].rolling(21).mean()
    return data

def test_structure_initialization():
    ms = MarketStructure(lookback_period=20)
    assert ms.lookback_period == 20

def test_support_resistance_levels(sample_market_data):
    ms = MarketStructure(lookback_period=20)
    result = ms.identify_structure(sample_market_data)
    
    assert 'support' in result.columns
    assert 'resistance' in result.columns
    assert result['support'].max() <= result['close'].max()
    assert result['resistance'].min() >= result['close'].min()

def test_trend_identification(sample_market_data):
    ms = MarketStructure(lookback_period=20)
    result = ms.identify_structure(sample_market_data)
    
    assert 'trend_direction' in result.columns
    assert set(result['trend_direction'].unique()).issubset({-1, 0, 1})

def test_price_position(sample_market_data):
    ms = MarketStructure(lookback_period=20)
    result = ms.identify_structure(sample_market_data)
    
    assert 'price_position' in result.columns
    assert result['price_position'].min() >= 0
    assert result['price_position'].max() <= 1

def test_missing_data_handling():
    # Create data with missing values
    data = pd.DataFrame({
        'time': pd.date_range(start='2024-01-01', end='2024-01-10', freq='1min'),
        'open': [100] * 10,
        'high': [101] * 10,
        'low': [99] * 10,
        'close': [100] * 10,
        'volume': [1000] * 10
    })
    data.loc[5, ['high', 'low', 'close']] = np.nan
    
    ms = MarketStructure(lookback_period=5)
    result = ms.identify_structure(data)
    
    assert not result['support'].isna().any()
    assert not result['resistance'].isna().any()