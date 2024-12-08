from typing import Dict, List
import pandas as pd
import numpy as np
from loguru import logger

class SignalGenerator:
    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            df = data.copy()
            
            # Log initial data statistics
            logger.info("\nMarket Conditions Analysis:")
            logger.info(f"RSI Range: {df['rsi'].min():.2f} to {df['rsi'].max():.2f}")
            logger.info(f"Average volume ratio: {df['volume_ratio'].mean():.2f}")
            logger.info(f"Price position range: {df['price_position'].min():.2f} to {df['price_position'].max():.2f}")
            logger.info(f"Trend strength range: {df['trend_strength'].min():.2f} to {df['trend_strength'].max():.2f}")
            
            # Base conditions with individual logging
            volume_active = df['volume_ratio'] > 1.1
            trend_confirmed = df['trend_strength'] >= 1
            volatility_normal = (df['atr_pct'] > 0.1) & (df['atr_pct'] < 0.5)
            
            logger.info("\nBase Conditions Met:")
            logger.info(f"Volume Active periods: {volume_active.sum()}")
            logger.info(f"Trend Confirmed periods: {trend_confirmed.sum()}")
            logger.info(f"Normal Volatility periods: {volatility_normal.sum()}")
            
            # Long conditions
            price_near_support = df['price_position'] < 0.3
            rsi_oversold = df['rsi'] < 45  # Relaxed condition
            bullish_momentum = df['ema_fast'] > df['ema_medium']
            
            long_base = (
                volume_active &
                trend_confirmed &
                price_near_support &
                rsi_oversold &
                bullish_momentum &
                volatility_normal
            )
            
            # Short conditions
            price_near_resistance = df['price_position'] > 0.7
            rsi_overbought = df['rsi'] > 55  # Relaxed condition
            bearish_momentum = df['ema_fast'] < df['ema_medium']
            
            short_base = (
                volume_active &
                trend_confirmed &
                price_near_resistance &
                rsi_overbought &
                bearish_momentum &
                volatility_normal
            )
            
            # Log conditions breakdown
            logger.info("\nLong Setup Conditions:")
            logger.info(f"Price near support: {price_near_support.sum()} periods")
            logger.info(f"RSI oversold: {rsi_oversold.sum()} periods")
            logger.info(f"Bullish momentum: {bullish_momentum.sum()} periods")
            logger.info(f"Total long setups: {long_base.sum()} periods")
            
            logger.info("\nShort Setup Conditions:")
            logger.info(f"Price near resistance: {price_near_resistance.sum()} periods")
            logger.info(f"RSI overbought: {rsi_overbought.sum()} periods")
            logger.info(f"Bearish momentum: {bearish_momentum.sum()} periods")
            logger.info(f"Total short setups: {short_base.sum()} periods")
            
            # Apply signals with additional filters
            df['signal'] = 0
            
            # Long signals with confirmation
            long_signals = long_base & (df['lower_wick'] > df['body_size'] * 0.3)
            df.loc[long_signals, 'signal'] = 1
            
            # Short signals with confirmation
            short_signals = short_base & (df['upper_wick'] > df['body_size'] * 0.3)
            df.loc[short_signals, 'signal'] = -1
            
            # Add signal strength scoring
            df['signal_strength'] = 0
            if long_signals.any() or short_signals.any():
                df.loc[long_signals, 'signal_strength'] = (
                    df.loc[long_signals, 'volume_ratio'] * 
                    df.loc[long_signals, 'trend_strength']
                )
                df.loc[short_signals, 'signal_strength'] = (
                    df.loc[short_signals, 'volume_ratio'] * 
                    df.loc[short_signals, 'trend_strength']
                )
            
            total_signals = len(df[df['signal'] != 0])
            logger.info(f"\nTotal signals generated: {total_signals}")
            logger.info(f"Long signals: {len(df[df['signal'] == 1])}")
            logger.info(f"Short signals: {len(df[df['signal'] == -1])}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return data