import pytest
import pandas as pd
import numpy as np
from src.analysis.volatility import VolatilityAnalyzer

@pytest.fixture
def sample_volatility_data():
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1min')
    n = len(dates)
    
    # Create data with known volatility patterns
    base_price = 100
    volatility = np.concatenate([
        np.ones(n//3) * 0.1,  # Low volatility
        np.ones(n//3) * 0.5,  # High volatility
        np.ones(n//3) * 0.2   # Normal volatility
    ])
    
    prices = base_price + np.random.randn(n) * volatility
    data = pd.DataFrame({
        'time': dates,
        'open': prices,
        'high': prices + volatility,
        'low': prices - volatility,
        'close': prices,
        'volume': np.random.randint(1000, 10000, n)
    })
    return data

def test_volatility_initialization():
    va = VolatilityAnalyzer(atr_period=14, vol_threshold=1.5)
    assert va.atr_period == 14
    assert va.vol_threshold == 1.5

def test_atr_calculation(sample_volatility_data):
    va = VolatilityAnalyzer()
    result = va.analyze_volatility(sample_volatility_data)
    
    assert 'atr' in result.columns
    assert 'atr_pct' in result.columns
    assert not result['atr'].isna().all()

def test_volatility_regime_identification(sample_volatility_data):
    va = VolatilityAnalyzer()
    result = va.analyze_volatility(sample_volatility_data)
    
    assert 'vol_regime' in result.columns
    assert set(result['vol_regime'].unique()).issubset({'low', 'normal', 'high'})

def test_extreme_volatility_handling():
    # Create data with extreme volatility
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
    data = pd.DataFrame({
        'time': dates,
        'open': [100] * 100,
        'high': [100] * 100,
        'low': [100] * 100,
        'close': [100] * 100,
        'volume': [1000] * 100
    })
    
    # Add extreme volatility spike
    data.loc[50, ['high', 'low']] = [200, 50]
    
    va = VolatilityAnalyzer()
    result = va.analyze_volatility(data)
    
    assert result['vol_regime'].value_counts()['high'] < 10  # Spike should be identified but not dominate

def test_zero_volatility_handling():
    # Create data with zero volatility
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
    data = pd.DataFrame({
        'time': dates,
        'open': [100] * 100,
        'high': [100] * 100,
        'low': [100] * 100,
        'close': [100] * 100,
        'volume': [1000] * 100
    })
    
    va = VolatilityAnalyzer()
    result = va.analyze_volatility(data)
    
    assert not result['atr'].isna().any()
    assert (result['atr'] >= 0).all()