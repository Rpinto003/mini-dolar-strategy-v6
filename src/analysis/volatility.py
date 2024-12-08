import pandas as pd
import numpy as np
import talib
from loguru import logger

class VolatilityAnalyzer:
    def __init__(self, atr_period=14, vol_threshold=1.5):
        self.atr_period = atr_period
        self.vol_threshold = vol_threshold

    def analyze_volatility(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # ATR and Normalized ATR
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=self.atr_period)
        df['atr_pct'] = df['atr'] / df['close'] * 100
        
        # Volatility Regime
        df['vol_ma'] = df['atr_pct'].rolling(20).mean()
        df['vol_std'] = df['atr_pct'].rolling(20).std()
        df['vol_regime'] = 'normal'
        df.loc[df['atr_pct'] > df['vol_ma'] + self.vol_threshold * df['vol_std'], 'vol_regime'] = 'high'
        df.loc[df['atr_pct'] < df['vol_ma'] - self.vol_threshold * df['vol_std'], 'vol_regime'] = 'low'
        
        return df