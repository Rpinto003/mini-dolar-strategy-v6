import sqlite3
import pandas as pd
from datetime import datetime, timedelta

def check_data_flow():
    """Verificar dados em diferentes pontos do fluxo"""
    
    # 1. Consulta direta no SQLite
    conn = sqlite3.connect('src/data/database/candles.db')
    cursor = conn.cursor()
    
    # Verificar primeiras linhas direto do banco
    cursor.execute("SELECT * FROM candles LIMIT 5")
    print("\n1. Dados diretos do SQLite:")
    raw_data = cursor.fetchall()
    for row in raw_data:
        print(row)
    
    # 2. Usando pandas read_sql
    query = """
    SELECT *
    FROM candles
    LIMIT 5
    """
    
    df_simple = pd.read_sql(query, conn)
    print("\n2. Dados via pandas read_sql simples:")
    print(df_simple)
    
    # 3. Usando a query completa
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    
    query_full = """
    SELECT time as timestamp, open, high, low, close, volume 
    FROM candles
    WHERE time BETWEEN ? AND ?
    ORDER BY time
    """
    
    df_full = pd.read_sql_query(
        query_full, 
        conn, 
        params=(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')),
        parse_dates=['timestamp']
    )
    
    print("\n3. Dados via query completa:")
    print(df_full)
    
    conn.close()
    
    return df_simple, df_full

if __name__ == "__main__":
    df_simple, df_full = check_data_flow()
    
    # Verificar tipos de dados
    print("\nTipos de dados no DataFrame simples:")
    print(df_simple.dtypes)
    
    print("\nTipos de dados no DataFrame completo:")
    print(df_full.dtypes)
    
    # Verificar valores únicos
    print("\nRange de valores para cada coluna (DataFrame simples):")
    for col in df_simple.select_dtypes(include=['float64', 'int64']).columns:
        print(f"\n{col}:")
        print(f"Min: {df_simple[col].min()}")
        print(f"Max: {df_simple[col].max()}")