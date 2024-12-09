import sqlite3
import pandas as pd

def check_db_structure(db_path, table_name):
    """Check the structure of the database table"""
    conn = sqlite3.connect(db_path)
    
    # Get table info
    cursor = conn.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    
    print("\nTable Structure:")
    for col in columns:
        print(f"Column: {col[1]}, Type: {col[2]}")
    
    # Get sample data
    df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 1", conn)
    print("\nSample Data:")
    print(df)
    
    conn.close()

if __name__ == "__main__":
    db_path = "src/data/database/candles.db"
    table_name = "candles"
    check_db_structure(db_path, table_name)