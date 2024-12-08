from loguru import logger
import pandas as pd
from src.analysis.technical_indicators import TechnicalIndicators

class MarketStructure:
    def __init__(self, lookback_period: int = 20):
        self.lookback_period = lookback_period
        self.technical_indicators = TechnicalIndicators()  # Initialize here

    def identify_structure(self, data: pd.DataFrame) -> pd.DataFrame:
        """Analyze market structure and add required features"""
        try:
            df = data.copy()
            
            # Add technical indicators first using instance method
            df = self.technical_indicators.add_all_indicators(df)  # Fixed method call
            
            # Calculate market structure features
            df['high_point'] = df['high'].rolling(self.lookback_period, center=True).max()
            df['low_point'] = df['low'].rolling(self.lookback_period, center=True).min()
            
            # Calculate trend direction using EMAs
            df['trend_direction'] = 0
            df.loc[(df['ema_fast'] > df['ema_medium']) & (df['ema_medium'] > df['ema_slow']), 'trend_direction'] = 1
            df.loc[(df['ema_fast'] < df['ema_medium']) & (df['ema_medium'] < df['ema_slow']), 'trend_direction'] = -1

            # Calculate support/resistance
            df['support'] = df['low'].rolling(self.lookback_period).min()
            df['resistance'] = df['high'].rolling(self.lookback_period).max()
            
            # Price position relative to support/resistance
            df['price_position'] = (df['close'] - df['support']) / (df['resistance'] - df['support'])
            
            # Initialize signal column
            df['signal'] = 0
            
            logger.info("Market structure analysis completed")
            logger.debug(f"Available columns after analysis: {df.columns.tolist()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in market structure analysis: {str(e)}")
            return data