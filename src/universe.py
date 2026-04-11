import numpy as np
import pandas as pd


def get_sp500_changes(file: str, start_date):
    df = pd.read_csv(file)
    df['date'] = df['date'].apply(lambda x: pd.to_datetime(x))
    df['tickers'] = df['tickers'].apply(lambda x: sorted(x.split(',')))
    df = df.set_index('date')
    df = df[df.index > start_date]
    return df

def get_sector_map(df: pd.DataFrame):
    return df.groupby('sector')['tickers'].apply(list).to_dict()

def get_all_sp500_tickers(df: pd.DataFrame):
    unique_tickers = set()
    for tickers in df['tickers']:
        ticks = set(tickers)
        unique_tickers.update(ticks)
    
    return list(unique_tickers)

def get_ticker_data(file: str, sp500_tickers: list[str]):
    df = pd.read_parquet(file)
    df['ticker'] = df['ticker'].apply(lambda x: x.replace(".","-"))
    df = df[(~df['tickers'].isna()) & (df['tickers'].isin(sp500_tickers))]
    df = df[df['table'] == 'SEP']
    return df



