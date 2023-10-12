import pandas as pd
import polars as pl


def get_df_size(df: pd.DataFrame | pl.DataFrame) -> str:
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    
    if type(df) == pl.DataFrame:
        df_memory = df.estimated_size('b')
    elif type(df) == pd.DataFrame:
        df_memory = df.memory_usage(index=True, deep=True).sum()
        
    i = 0
    while df_memory >= 1024 and i < len(suffixes)-1:
        df_memory /= 1024.
        i += 1
    
    return f"{round(df_memory, 2)} {suffixes[i]}"


def get_df_info(df: pd.DataFrame | pl.DataFrame, msg: str="") -> pd.DataFrame | pl.DataFrame:
    if msg:
        print(msg)
    print(f"Data Size: {get_df_size(df)}")
    print(f"Number of Columns: {df.shape[1]}")
    print(f"Number of Rows: {df.shape[0]}")
    
    return df