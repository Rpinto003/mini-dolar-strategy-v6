import sqlite3
import pandas as pd
from loguru import logger
from typing import Optional

class MarketDataLoader:
    def __init__(self, db_path: str, table_name: str = 'candles'):
        self.db_path = db_path
        self.table_name = table_name
        logger.info(f"Using database: {db_path}, table: {table_name}")

    def get_minute_data(self, 
                       interval: int = 1, 
                       start_date: Optional[str] = None, 
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load minute candle data from SQLite database
        
        Args:
            interval: Candle interval in minutes
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        logger.info(f"Attempting to load data from {self.db_path}, table: {self.table_name}")
        
        try:
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            
            # Build query
            query = f"SELECT * FROM {self.table_name}"
            conditions = []
            
            if start_date:
                conditions.append(f"time >= '{start_date}'")
            if end_date:
                conditions.append(f"time <= '{end_date}'")
                
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
                
            # Load data
            data = pd.read_sql_query(query, conn)
            logger.info(f"Data loaded: {len(data)} records")
            
            # Ensure datetime format
            data['time'] = pd.to_datetime(data['time'])
            
            # Resample if needed
            if interval > 1:
                data = self.resample_data(data, interval)
                
            # Ensure column presence
            self.validate_columns(data)
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return pd.DataFrame()
            
        finally:
            if 'conn' in locals():
                conn.close()

    def resample_data(self, data: pd.DataFrame, interval: int) -> pd.DataFrame:
        """Resample data to specified interval"""
        logger.info(f"Resampling data to {interval}-minute intervals")
        
        # Set time as index for resampling
        data = data.set_index('time')
        
        # Resample rules
        rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Resample
        resampled = data.resample(f'{interval}T').agg(rules)
        resampled = resampled.dropna()
        
        # Reset index
        resampled = resampled.reset_index()
        
        logger.info(f"Data resampled to {interval}-minute intervals: {len(resampled)} records")
        return resampled

    def validate_columns(self, data: pd.DataFrame):
        """Ensure all required columns are present"""
        required_columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        
        # Check if time column exists
        if 'time' not in data.columns:
            data['time'] = data.index
            logger.info("'time' column reset to ensure its presence: {data.columns.tolist()}")
            
        # Verify all required columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")