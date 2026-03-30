import pandas as pd
import numpy as np
import nasdaqdatalink as nd
from pathlib import Path
from typing import Iterable
from dotenv import load_dotenv
import os

def price_data_starting_from(df: pd.DataFrame, start):
    df = df[df.index > start]
    return df

def get_ticker_data(file: str,tickers: list[str]):
    df = pd.read_parquet(file)
    df['ticker'] = df['ticker'].apply(lambda x: x.replace(".","-"))
    df = df[(~df['tickers'].isna()) & (df['tickers'].isin(tickers))]
    df = df[df['table'] == 'SEP']
    return df

def load_constituents(path: str) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.iloc[:, 0].str.split(',').apply(lambda x: [t.strip() for t in x])

def get_universe_on_date(date: pd.Timestamp, sp500_daily: pd.Series) -> list:
    valid_dates = sp500_daily.index[sp500_daily.index <= date]
    if len(valid_dates) == 0:
        return []
    return sp500_daily.loc[valid_dates[-1]]

def build_sp500_price_matrix(sep: pd.DataFrame, sp500_daily: pd.Series) -> pd.DataFrame:
    matrix = sep.pivot(index='date', columns='ticker', values='closeadj')
    
    for date in matrix.index:
        universe = set(get_universe_on_date(date, sp500_daily))
        invalid = [t for t in matrix.columns if t not in universe]
        matrix.loc[date, invalid] = np.nan
    
    return matrix.dropna(how='all')
