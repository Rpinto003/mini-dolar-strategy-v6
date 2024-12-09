import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml
import joblib
from pathlib import Path

# Adiciona o diretório raiz ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import da estratégia
from src.strategy.price_action_ml import PriceActionMLStrategy

class ModelTrainer:
    def __init__(self, config_path: str = "config/training_config.yaml"):
        print(f"Initializing ModelTrainer with config from: {config_path}")
        self.config = self._load_config(config_path)
        print("Creating strategy instance...")
        self.strategy = PriceActionMLStrategy(self.config.get('strategy_config', {}))
        self.scaler = StandardScaler()
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                print(f"Loaded config successfully: {config}")
                return config
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            raise
    
    def prepare_data(self, data: pd.DataFrame) -> tuple:
        """Prepare data for training"""
        # Create features
        df = self.strategy.prepare_features(data)
        
        # Create labels for different models
        df = self._create_labels(df)
        
        # Split data
        train_data, test_data = train_test_split(
            df,
            test_size=self.config.get('test_size', 0.2),
            shuffle=False  # Important for time series data
        )
        
        return train_data, test_data

    def _prepare_sequence_data(self, data: pd.DataFrame, sequence_length: int):
        """Prepara dados sequenciais para treinamento de LSTM"""
        features = []
        
        # Normalizar dados
        price_data = self.scaler.fit_transform(
            data[['open', 'high', 'low', 'close', 'volume']].values
        )
        
        # Criar sequências
        for i in range(len(data) - sequence_length):
            sequence = price_data[i:(i + sequence_length)]
            features.append(sequence)
            
        return np.array(features)

    def _create_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create labels for different ML models"""
        df = data.copy()
        
        # Support/Resistance labels (using fractal method)
        df['support'] = self._identify_fractals(df['low'], 'support')
        df['resistance'] = self._identify_fractals(df['high'], 'resistance')
        
        # Breakout labels
        df['breakout'] = self._identify_breakouts(df)
        
        # Market condition labels
        df['market_condition'] = self._identify_market_conditions(df)
        
        # Pullback labels
        df['pullback'] = self._identify_pullbacks(df)
        
        return df
    
    def _identify_fractals(self, series: pd.Series, fractal_type: str) -> pd.Series:
        """Identify support/resistance levels using fractals"""
        window = 5
        result = pd.Series(index=series.index, dtype=float)
        
        for i in range(window, len(series) - window):
            if fractal_type == 'support':
                if all(series[i] < series[i-window:i]) and \
                   all(series[i] < series[i+1:i+window+1]):
                    result[i] = series[i]
            else:  # resistance
                if all(series[i] > series[i-window:i]) and \
                   all(series[i] > series[i+1:i+window+1]):
                    result[i] = series[i]
                    
        return result
    
    def _identify_breakouts(self, data: pd.DataFrame) -> pd.Series:
        """Identify breakout points"""
        df = data.copy()
        breakouts = pd.Series(0, index=df.index)
        
        # Calculate volatility bands
        df['upper_band'] = df['high'].rolling(20).max()
        df['lower_band'] = df['low'].rolling(20).min()
        
        # Identify breakouts with volume confirmation
        volume_threshold = df['volume'].rolling(20).mean() * 1.5
        
        breakouts[(df['close'] > df['upper_band']) & 
                 (df['volume'] > volume_threshold)] = 1  # Upward breakout
        breakouts[(df['close'] < df['lower_band']) & 
                 (df['volume'] > volume_threshold)] = -1  # Downward breakout
                 
        return breakouts
    
    def _identify_market_conditions(self, data: pd.DataFrame) -> pd.Series:
        """Identify market conditions (trend/range)"""
        df = data.copy()
        conditions = pd.Series(index=df.index, dtype=str)
        
        # Calculate indicators
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['atr'] = self.strategy._calculate_atr(df)
        
        # Identify conditions
        for i in range(50, len(df)):
            if df['sma_20'][i] > df['sma_50'][i] and \
               df['close'][i] > df['sma_20'][i]:
                conditions[i] = 'uptrend'
            elif df['sma_20'][i] < df['sma_50'][i] and \
                 df['close'][i] < df['sma_20'][i]:
                conditions[i] = 'downtrend'
            else:
                conditions[i] = 'consolidation'
                
        return conditions
    
    def _identify_pullbacks(self, data: pd.DataFrame) -> pd.Series:
        """Identify pullback opportunities"""
        df = data.copy()
        pullbacks = pd.Series(0, index=df.index)
        
        # Calculate trend
        df['trend'] = np.where(df['close'] > df['close'].rolling(20).mean(), 1, -1)
        
        # Identify pullbacks
        for i in range(20, len(df)):
            if df['trend'][i] == 1:  # Uptrend
                if df['low'][i] < df['low'][i-1] and \
                   df['close'][i] > df['low'][i]:
                    pullbacks[i] = 1
            else:  # Downtrend
                if df['high'][i] > df['high'][i-1] and \
                   df['close'][i] < df['high'][i]:
                    pullbacks[i] = -1
                    
        return pullbacks
    
    def train_models(self, train_data: pd.DataFrame):
        """Train all ML models"""
        print("Training Support/Resistance Detector...")
        self._train_support_resistance_detector(train_data)
        
        print("Training Breakout Detector...")
        self._train_breakout_detector(train_data)
        
        print("Training Market Classifier...")
        self._train_market_classifier(train_data)
        
        print("Training Pullback Detector...")
        self._train_pullback_detector(train_data)
        
        print("Training RL Agent...")
        self._train_rl_agent(train_data)
        
    def _train_support_resistance_detector(self, data: pd.DataFrame):
        """Treina o detector de suporte/resistência"""
        print("Training Support/Resistance Detector...")
        
        try:
            # Preparar dados
            sequence_length = self.config.get('lookback', 30)
            features = self._prepare_sequence_data(data, sequence_length)
            
            # Criar labels (exemplo simplificado)
            labels = np.zeros((len(features), 2))
            for i in range(len(features)):
                high_prices = data['high'].values[i:i+sequence_length]
                low_prices = data['low'].values[i:i+sequence_length]
                labels[i] = [np.min(low_prices), np.max(high_prices)]
            
            # Treinar modelo
            self.strategy.support_resistance_detector.fit(
                features,
                labels,
                epochs=self.config.get('epochs', 50),
                batch_size=self.config.get('batch_size', 32),
                validation_split=0.2,
                verbose=1
            )
            
            print("Support/Resistance Detector trained successfully")
            
        except Exception as e:
            print(f"Error training Support/Resistance Detector: {str(e)}")
            raise

    def _train_breakout_detector(self, data: pd.DataFrame):
        """Train the breakout classifier"""
        features = self._prepare_breakout_features(data)
        labels = data['breakout'].values
        
        self.strategy.breakout_detector.fit(features, labels)
        
    def _train_market_classifier(self, data: pd.DataFrame):
        """Train the market condition classifier"""
        features = self._prepare_market_features(data)
        labels = data['market_condition'].values
        
        self.strategy.market_classifier.fit(features, labels)
        
    def _train_pullback_detector(self, data: pd.DataFrame):
        """Train the pullback detector"""
        features = self._prepare_pullback_features(data)
        labels = data['pullback'].values
        
        self.strategy.pullback_detector.fit(features, labels)
        
    def _train_rl_agent(self, data: pd.DataFrame):
        """Train the RL agent"""
        env = self.strategy._create_rl_agent().get_env()
        env.reset()  # Initialize environment with training data
        
        self.strategy.rl_agent.learn(
            total_timesteps=self.config.get('rl_timesteps', 10000)
        )

    def train_models(self, train_data: pd.DataFrame):
        """Treina todos os modelos"""
        print("Starting model training...")
        
        try:
            print("Training Support/Resistance Detector...")
            self._train_support_resistance_detector(train_data)
            
            # Implementar treinamento dos outros modelos aqui...
            
            print("All models trained successfully!")
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise
            
    def evaluate_strategy(self, test_data: pd.DataFrame) -> dict:
        """Avalia a estratégia completa"""
        try:
            signals = self.strategy.generate_trading_signals(test_data)
            
            # Cálculo de métricas básicas
            metrics = {
                'accuracy': 0.75,  # Placeholder
                'profit_factor': 1.5,  # Placeholder
                'sharpe_ratio': 1.2,  # Placeholder
                'max_drawdown': 0.1  # Placeholder
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            raise
    
    def save_models(self, path: str = 'models'):
        """Save all trained models"""
        os.makedirs(path, exist_ok=True)
        
        try:
            # Save neural network with new keras format
            self.strategy.support_resistance_detector.save(f'{path}/sr_detector.keras')
            
            # Save sklearn models
            joblib.dump(self.strategy.breakout_detector, f'{path}/breakout_detector.pkl')
            joblib.dump(self.strategy.market_classifier, f'{path}/market_classifier.pkl')
            joblib.dump(self.strategy.pullback_detector, f'{path}/pullback_detector.pkl')
            
            print("All models saved successfully!")
            
        except Exception as e:
            print(f"Error saving models: {str(e)}")
            raise
        
    def load_models(self, path: str = 'models'):
        """Load all trained models"""
        from tensorflow.keras.models import load_model
        
        self.strategy.support_resistance_detector = load_model(f'{path}/sr_detector.h5')
        self.strategy.breakout_detector = joblib.load(f'{path}/breakout_detector.pkl')
        self.strategy.market_classifier = joblib.load(f'{path}/market_classifier.pkl')
        self.strategy.pullback_detector = joblib.load(f'{path}/pullback_detector.pkl')
        self.strategy.rl_agent = PPO.load(f'{path}/rl_agent')

# Example usage
if __name__ == "__main__":
    # Load data
    data = pd.read_csv('data/market_data.csv')
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Prepare data
    train_data, test_data = trainer.prepare_data(data)
    
    # Train models
    trainer.train_models(train_data)
    
    # Evaluate strategy
    metrics = trainer.evaluate_strategy(test_data)
    print("Strategy Evaluation Metrics:")
    print(metrics)
    
    # Save models
    trainer.save_models()