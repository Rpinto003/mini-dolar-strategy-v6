import pandas as pd
import numpy as np
import talib
from loguru import logger

class TechnicalIndicators:
    def __init__(self):
        pass

    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Calculating technical indicators...")
            data = df.copy()
            
            # EMAs for trend detection
            data['ema_fast'] = talib.EMA(data['close'], timeperiod=8)
            data['ema_medium'] = talib.EMA(data['close'], timeperiod=21)
            data['ema_slow'] = talib.EMA(data['close'], timeperiod=50)
            
            # RSI
            data['rsi'] = talib.RSI(data['close'], timeperiod=14)
            logger.info(f"RSI range: {data['rsi'].min():.2f} to {data['rsi'].max():.2f}")
            
            # Trend strength
            data['short_trend'] = (data['ema_fast'] > data['ema_medium']).astype(int)
            data['long_trend'] = (data['ema_medium'] > data['ema_slow']).astype(int)
            data['trend_strength'] = data['short_trend'] + data['long_trend']
            
            # Volatility
            data['atr'] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14)
            data['atr_pct'] = data['atr'] / data['close'] * 100
            
            # Volume analysis
            data['volume_sma'] = data['volume'].rolling(window=20).mean()
            data['volume_ratio'] = data['volume'] / data['volume_sma']
            data['volume_trend'] = (
                data['volume'].rolling(window=5).mean() > 
                data['volume'].rolling(window=20).mean()
            ).astype(int)
            
            # Price action
            data['body_size'] = abs(data['close'] - data['open'])
            data['upper_wick'] = data['high'] - data['close'].where(data['close'] > data['open'], data['open'])
            data['lower_wick'] = data['close'].where(data['close'] < data['open'], data['open']) - data['low']
            
            # Fill NaN values
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            logger.info("Technical indicators calculated successfully")
            logger.debug(f"Calculated indicators: {[col for col in data.columns if col not in df.columns]}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error in indicator calculation: {str(e)}")
            return df