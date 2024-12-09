from typing import Dict, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from stable_baselines3 import PPO
import gym
from gym import spaces
from src.models.trading_env import TradingEnvironment


class PriceActionMLStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.initialize_components()
        self.scaler = StandardScaler()
        
    def initialize_components(self):
        # Initialize support/resistance detector
        self.support_resistance_detector = self._create_support_resistance_detector()
        
        # Initialize breakout detector
        self.breakout_detector = self._create_breakout_detector()
        
        # Initialize market condition classifier
        self.market_classifier = self._create_market_classifier()
        
        # Initialize RL agent
        self.rl_agent = self._create_rl_agent()
        
        # Initialize pullback detector
        self.pullback_detector = self._create_pullback_detector()
        
    def _create_support_resistance_detector(self):
        """Creates an improved neural network for support/resistance detection"""
        input_shape = (self.config.get('lookback', 30), 5)  # Ajustado para 5 features
        
        model = Sequential([
            LSTM(100, input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(2)  # Output: [support_level, resistance_level]
        ])
        
        model.compile(
            optimizer='adam',
            loss='huber_loss',  # Mais robusto que MSE para outliers
            metrics=['mae']  # Métrica adicional para monitoramento
        )
        
        return model
   
    def _create_breakout_detector(self):
        """Creates a Random Forest classifier for breakout detection"""
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    
    def _create_market_classifier(self):
        """Creates a classifier for market condition detection"""
        return RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    
    def _create_rl_agent(self):
        """Creates a Reinforcement Learning agent using PPO"""
        try:
            env = TradingEnvironment()
            
            model = PPO(
                policy='MlpPolicy',
                env=env,
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                verbose=1
            )
            
            return model
        except Exception as e:
            print(f"Error creating RL agent: {str(e)}")
            return None
            
    def _create_pullback_detector(self):
        """Creates a regression model for pullback detection"""
        return RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
    
    def _prepare_sr_features(self, data: pd.DataFrame) -> np.array:
        """Prepara features para detecção de suporte e resistência"""
        df = data.copy()
        sequence_length = self.config.get('lookback', 30)
        
        # Normalizar preços usando retornos logarítmicos
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            df[f'{col}_return'] = np.log(df[col] / df[col].shift(1))
        
        # Normalizar volume
        df['volume_norm'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()
        
        # Selecionar features finais
        features_cols = ['close_return', 'high_return', 'low_return', 'volume_norm', 'body_size']
        
        # Remover NaNs
        df = df.dropna()
        
        # Criar sequências para LSTM
        features = []
        for i in range(len(df) - sequence_length + 1):
            sequence = df[features_cols].iloc[i:i+sequence_length].values
            features.append(sequence)
        
        return np.array(features)
   
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepara todas as features necessárias para a estratégia"""
        df = data.copy()
        
        # Calcular indicadores técnicos básicos
        df['atr'] = self._calculate_atr(df)
        df['rsi'] = self._calculate_rsi(df)
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['price_ma'] = df['close'].rolling(20).mean()
        
        # Calcular features para price action
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        
        return df
        
    def detect_support_resistance(self, data: pd.DataFrame) -> Dict:
        """Detects support and resistance levels"""
        features = self._prepare_sr_features(data)
        predictions = self.support_resistance_detector.predict(features)
        return {
            'support_levels': predictions[:, 0],
            'resistance_levels': predictions[:, 1]
        }
    
    def detect_breakout(self, data: pd.DataFrame) -> pd.Series:
        """Detects breakout patterns"""
        features = self._prepare_breakout_features(data)
        return pd.Series(
            self.breakout_detector.predict_proba(features)[:, 1],
            index=data.index
        )
    
    def classify_market_condition(self, data: pd.DataFrame) -> str:
        """Classifies market conditions (trend/range)"""
        features = self._prepare_market_features(data)
        prediction = self.market_classifier.predict(features)
        return ['uptrend', 'downtrend', 'consolidation'][prediction[0]]
    
    def detect_pullback(self, data: pd.DataFrame) -> pd.Series:
        """Detects pullback opportunities"""
        features = self._prepare_pullback_features(data)
        return pd.Series(
            self.pullback_detector.predict(features),
            index=data.index
        )
    
    def optimize_strategy(self, data: pd.DataFrame) -> Dict:
        """Optimizes strategy parameters using RL"""
        # Train RL agent
        self.rl_agent.learn(total_timesteps=10000)
        
        # Get optimal parameters
        optimal_params = self._extract_optimal_params()
        return optimal_params
    
    def generate_trading_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generates trading signals based on all analyses"""
        df = data.copy()
        
        # Get all analyses
        sr_levels = self.detect_support_resistance(df)
        breakout_proba = self.detect_breakout(df)
        market_condition = self.classify_market_condition(df)
        pullback_levels = self.detect_pullback(df)
        
        # Generate signals
        signals = pd.DataFrame(index=df.index)
        signals['entry_price'] = self._calculate_entry_prices(
            df, sr_levels, breakout_proba, market_condition, pullback_levels
        )
        signals['stop_loss'] = self._calculate_stop_losses(
            df, sr_levels, market_condition
        )
        signals['take_profit'] = self._calculate_take_profits(
            df, sr_levels, market_condition
        )
        signals['position_size'] = self._calculate_position_sizes(
            df, signals['stop_loss']
        )
        
        return signals
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculates Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    def _calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculates Relative Strength Index"""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_entry_prices(self, data: pd.DataFrame, sr_levels: Dict,
                              breakout_proba: pd.Series, market_condition: str,
                              pullback_levels: pd.Series) -> pd.Series:
        """Calculates optimal entry prices based on all analyses"""
        entries = pd.Series(index=data.index)
        
        for i in range(len(data)):
            if breakout_proba[i] > 0.7:  # High probability breakout
                if market_condition == 'uptrend':
                    entries[i] = sr_levels['resistance_levels'][i]
                elif market_condition == 'downtrend':
                    entries[i] = sr_levels['support_levels'][i]
            elif pullback_levels[i] > 0:  # Pullback opportunity
                entries[i] = pullback_levels[i]
                
        return entries
    
    def _calculate_stop_losses(self, data: pd.DataFrame, sr_levels: Dict,
                             market_condition: str) -> pd.Series:
        """Calculates dynamic stop losses"""
        atr = self._calculate_atr(data)
        stops = pd.Series(index=data.index)
        
        for i in range(len(data)):
            if market_condition == 'uptrend':
                stops[i] = min(
                    sr_levels['support_levels'][i],
                    data['low'][i] - 2 * atr[i]
                )
            else:
                stops[i] = max(
                    sr_levels['resistance_levels'][i],
                    data['high'][i] + 2 * atr[i]
                )
                
        return stops
    
    def _calculate_take_profits(self, data: pd.DataFrame, sr_levels: Dict,
                              market_condition: str) -> pd.Series:
        """Calculates take profit levels"""
        atr = self._calculate_atr(data)
        targets = pd.Series(index=data.index)
        
        for i in range(len(data)):
            if market_condition == 'uptrend':
                targets[i] = data['close'][i] + 3 * atr[i]
            else:
                targets[i] = data['close'][i] - 3 * atr[i]
                
        return targets
    
    def _calculate_position_sizes(self, data: pd.DataFrame,
                                stop_losses: pd.Series) -> pd.Series:
        """Calculates position sizes based on risk management"""
        risk_per_trade = self.config.get('risk_per_trade', 0.01)
        account_size = self.config.get('account_size', 100000)
        
        risk_amount = account_size * risk_per_trade
        risk_per_unit = abs(data['close'] - stop_losses)
        
        return risk_amount / risk_per_unit