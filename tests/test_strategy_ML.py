import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import sys
import os
import joblib

# Adiciona o diretório raiz ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.strategy.price_action_ml import PriceActionMLStrategy
from src.models.trading_env import TradingEnvironment
from train_model_ML import ModelTrainer
import yaml
import unittest


def load_config():
    with open('config/training_config.yaml', 'r') as file:
        return yaml.safe_load(file)

def load_data(config):
    """Load data from SQLite database"""
    conn = sqlite3.connect(config['data']['db_path'])
    
    # Definir data final fixa como 15/11/2024
    end_date = datetime(2024, 11, 15)
    start_date = end_date - timedelta(days=30)  # Pegando 30 dias anteriores
    
    query = f"""
    SELECT time as timestamp, open, high, low, close, volume 
    FROM {config['data']['table_name']}
    WHERE time BETWEEN ? AND ?
    ORDER BY time
    """

    df = pd.read_sql_query(
        query, 
        conn,
        params=(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')),
        parse_dates=['timestamp']
    )
    
    conn.close()
    
    if df.empty:
        raise ValueError("No data found for the specified period")
        
    print(f"Data shape: {df.shape}")
    print("\nPrimeiras linhas dos dados:")
    print(df.head())
    print("\nÚltimas linhas dos dados:")
    print(df.tail())
    
    return df

class TestPriceActionStrategy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Configuração inicial para todos os testes"""
        print("Iniciando ambiente de testes...")
        cls.config = load_config()
        cls.data = load_data(cls.config)
        cls.trainer = ModelTrainer(config_path='config/training_config.yaml')
    
    def test_1_data_loading(self):
        """Teste de carregamento de dados"""
        print("\nTestando carregamento de dados...")
        self.assertIsNotNone(self.data)
        self.assertGreater(len(self.data), 0)
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            self.assertIn(col, self.data.columns)
        print("Teste de carregamento de dados concluído com sucesso!")
    
    def test_2_data_preparation(self):
        """Teste de preparação dos dados"""
        print("\nTestando preparação dos dados...")
        train_data, test_data = self.trainer.prepare_data(self.data)
        self.assertGreater(len(train_data), 0)
        self.assertGreater(len(test_data), 0)
        total_size = len(train_data) + len(test_data)
        self.assertEqual(total_size, len(self.data))
        print("Teste de preparação dos dados concluído com sucesso!")
    
    def test_3_model_training(self):
        """Teste de treinamento dos modelos"""
        print("\nTestando treinamento dos modelos...")
        try:
            train_data, test_data = self.trainer.prepare_data(self.data)
            self.trainer.train_models(train_data)
            
            # Verificar se os modelos foram criados
            self.assertTrue(hasattr(self.trainer.strategy, 'support_resistance_detector'))
            self.assertTrue(hasattr(self.trainer.strategy, 'breakout_detector'))
            self.assertTrue(hasattr(self.trainer.strategy, 'market_classifier'))
            self.assertTrue(hasattr(self.trainer.strategy, 'pullback_detector'))
            print("Teste de treinamento dos modelos concluído com sucesso!")
            
        except Exception as e:
            self.fail(f"Falha no treinamento dos modelos: {str(e)}")
    
    def test_4_strategy_evaluation(self):
        """Teste de avaliação da estratégia"""
        print("\nTestando avaliação da estratégia...")
        try:
            train_data, test_data = self.trainer.prepare_data(self.data)
            self.trainer.train_models(train_data)
            metrics = self.trainer.evaluate_strategy(test_data)
            
            required_metrics = ['accuracy', 'profit_factor', 'sharpe_ratio', 'max_drawdown']
            for metric in required_metrics:
                self.assertIn(metric, metrics)
                self.assertIsInstance(metrics[metric], float)
            print("Teste de avaliação da estratégia concluído com sucesso!")
            
        except Exception as e:
            self.fail(f"Falha na avaliação da estratégia: {str(e)}")
    
    def test_5_model_saving(self):
        """Teste de salvamento dos modelos"""
        print("\nTestando salvamento dos modelos...")
        try:
            train_data, test_data = self.trainer.prepare_data(self.data)
            self.trainer.train_models(train_data)
            self.trainer.save_models('models/')
            
            expected_files = ['sr_detector.h5', 'breakout_detector.pkl', 
                            'market_classifier.pkl', 'pullback_detector.pkl']
            
            for file in expected_files:
                path = os.path.join('models', file)
                self.assertTrue(os.path.exists(path))
            print("Teste de salvamento dos modelos concluído com sucesso!")
            
        except Exception as e:
            self.fail(f"Falha no salvamento dos modelos: {str(e)}")

def main():
    # Load configuration
    print("Loading configuration...")
    config = load_config()
    
    print("Loading data...")
    try:
        data = load_data(config)
        print(f"Loaded {len(data)} rows of data")
        
        # Initialize trainer
        print("Initializing model trainer...")
        trainer = ModelTrainer(config_path='config/training_config.yaml')
        
        # Prepare data
        print("Preparing training and test data...")
        train_data, test_data = trainer.prepare_data(data)
        print(f"Training data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        
        # Train models
        print("Training models...")
        trainer.train_models(train_data)
        
        # Evaluate strategy
        print("Evaluating strategy...")
        metrics = trainer.evaluate_strategy(test_data)
        
        print("\nStrategy Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Save models
        print("\nSaving trained models...")
        trainer.save_models('models/')
        
        print("\nTraining and evaluation complete!")
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # Executar testes
        unittest.main(argv=['first-arg-is-ignored'])
    else:
        # Executar treinamento normal
        main()