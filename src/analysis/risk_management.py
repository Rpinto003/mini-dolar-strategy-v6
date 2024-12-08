from typing import Tuple, Dict
import pandas as pd
import numpy as np
from loguru import logger

class RiskManager:
    def __init__(self, initial_capital=100000, max_risk_per_trade=0.01):
        self.initial_capital = initial_capital
        self.max_risk_per_trade = max_risk_per_trade

    def calculate_position_size(self, entry_price, stop_loss, volatility):
        # Risk amount in currency
        risk_amount = self.initial_capital * self.max_risk_per_trade
        
        # Point risk (distance to stop)
        point_risk = abs(entry_price - stop_loss)
        
        # Adjust position size based on volatility
        vol_factor = 1 / (1 + volatility)
        
        # Calculate base position size
        position_size = (risk_amount / point_risk) * vol_factor
        
        return position_size

    def apply_risk_management(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            data = df.copy()
            
            # Dynamic ATR multiplier based on volatility
            data['atr_multiplier'] = data['atr_pct'].apply(
                lambda x: 2.0 if x < 0.2 else (1.5 if x < 0.3 else 1.0)
            )
            
            # Stop loss calculation
            data['stop_distance'] = data['atr'] * data['atr_multiplier']
            
            for idx in data[data['signal'] != 0].index:
                signal = data.loc[idx, 'signal']
                strength = data.loc[idx, 'signal_strength']
                
                # Base position size on signal strength
                base_size = min(0.02 * strength, 0.1)  # Max 10% of capital
                
                # Adjust for volatility
                vol_factor = 1 / (data.loc[idx, 'atr_pct'] * 10)
                final_size = base_size * vol_factor
                
                data.loc[idx, 'position_size'] = final_size
                
            return data
            
        except Exception as e:
            logger.error(f"Error in risk management: {str(e)}")
            return df