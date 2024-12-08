import pytest
import pandas as pd
import numpy as np
from src.analysis.signals import SignalGenerator

@pytest.fixture
def sample_signal_data():
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1min')
    data = pd.DataFrame({
        'time': dates,
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'volume_ratio': np.random.uniform(0.5, 1.5, len(dates)),
        'rsi': np.random.uniform(30, 70, len(dates)),
        'trend_direction': np.random.choice([-1, 0, 1], len(dates)),
        'vol_regime': np.random.choice(['low', 'normal', 'high'], len(dates)),
        'price_position': np.random.uniform(0, 1, len(dates)),
        'session_active': np.random.choice([True, False], len(dates))
    })
    return data

def test_signal_generation(sample_signal_data):
    sg = SignalGenerator()
    result = sg.generate_signals(sample_signal_data)
    
    assert 'signal' in result.columns
    assert set(result['signal'].unique()).issubset({-1, 0, 1})

def test_signal_conditions(sample_signal_data):
    sg = SignalGenerator()
    result = sg.generate_signals(sample_signal_data)
    
    # Test long signals
    long_signals = result[result['signal'] == 1]
    if len(long_signals) > 0:
        assert (long_signals['price_position'] < 0.3).all()
        assert (long_signals['rsi'] < 40).all()
        assert (long_signals['volume_ratio'] > 1.2).all()
        assert long_signals['session_active'].all()

    # Test short signals
    short_signals = result[result['signal'] == -1]
    if len(short_signals) > 0:
        assert (short_signals['price_position'] > 0.7).all()
        assert (short_signals['rsi'] > 60).all()
        assert (short_signals['volume_ratio'] > 1.2).all()
        assert short_signals['session_active'].all()

def test_session_filter(sample_signal_data):
    sg = SignalGenerator()
    
    # Set all session_active to False
    test_data = sample_signal_data.copy()
    test_data['session_active'] = False
    
    result = sg.generate_signals(test_data)
    assert (result['signal'] == 0).all()

def test_extreme_conditions():
    # Create data with extreme conditions
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
    extreme_data = pd.DataFrame({
        'time': dates,
        'close': [100] * 100,
        'volume_ratio': [2.0] * 100,  # Very high volume
        'rsi': [10] * 100,            # Very oversold
        'trend_direction': [1] * 100,  # Strong uptrend
        'vol_regime': ['normal'] * 100,
        'price_position': [0.1] * 100, # Near support
        'session_active': [True] * 100
    })
    
    sg = SignalGenerator()
    result = sg.generate_signals(extreme_data)
    
    # Should still maintain reasonable signal frequency
    signal_ratio = len(result[result['signal'] != 0]) / len(result)
    assert signal_ratio < 0.3  # No more than 30% of periods should have signals