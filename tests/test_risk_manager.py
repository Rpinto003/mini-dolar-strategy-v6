import pytest
import pandas as pd
import numpy as np
from src.analysis.risk_management import RiskManager

@pytest.fixture
def sample_trade_data():
    return pd.DataFrame({
        'close': [100] * 10,
        'atr': [1.0] * 10,
        'atr_pct': [0.01] * 10,
        'vol_regime': ['normal'] * 10,
        'signal': [1, -1, 0, 1, -1, 0, 1, -1, 0, 0]
    })

def test_risk_manager_initialization():
    rm = RiskManager(initial_capital=100000, max_risk_per_trade=0.01)
    assert rm.initial_capital == 100000
    assert rm.max_risk_per_trade == 0.01

def test_position_size_calculation():
    rm = RiskManager()
    position_size = rm.calculate_position_size(
        entry_price=100,
        stop_loss=98,
        volatility=0.01
    )
    
    assert position_size > 0
    assert position_size < rm.initial_capital / 100  # Reasonable size check

def test_risk_management_application(sample_trade_data):
    rm = RiskManager()
    result = rm.apply_risk_management(sample_trade_data)
    
    assert 'stop_loss' in result.columns
    assert 'take_profit' in result.columns
    assert 'position_size' in result.columns
    
    # Check risk-reward ratio
    signals = result[result['signal'] != 0]
    if len(signals) > 0:
        for _, row in signals.iterrows():
            risk = abs(row['close'] - row['stop_loss'])
            reward = abs(row['take_profit'] - row['close'])
            assert reward > risk  # Reward should be greater than risk

def test_volatility_adjustment():
    rm = RiskManager()
    
    # Create test data with different volatility regimes
    test_data = pd.DataFrame({
        'close': [100] * 3,
        'atr': [1.0] * 3,
        'atr_pct': [0.01] * 3,
        'vol_regime': ['low', 'normal', 'high'],
        'signal': [1, 1, 1]
    })
    
    result = rm.apply_risk_management(test_data)
    
    # Position sizes should be inversely related to volatility
    pos_sizes = result['position_size'].tolist()
    assert pos_sizes[0] > pos_sizes[1] > pos_sizes[2]

def test_maximum_position_size():
    rm = RiskManager(initial_capital=100000)
    
    # Test with very tight stop loss
    position_size = rm.calculate_position_size(
        entry_price=100,
        stop_loss=99.99,
        volatility=0.01
    )
    
    # Position size should be capped
    assert position_size <= rm.initial_capital * 0.1  # Maximum 10% of capital