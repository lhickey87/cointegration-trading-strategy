import numpy as np
import pandas as pd

#issue is we have to be doing this for most of kalman 


#get tickers throughought time period -> from those tickers get sector_map

def get_sp500_changes(file: str, start_date):
    df = pd.read_csv(file)
    df['date'] = df['date'].apply(lambda x: pd.to_datetime(x))
    df['tickers'] = df['tickers'].apply(lambda x: sorted(x.split(',')))
    df = df.set_index('date')
    df = df[df.index > start_date]
    return df

def sector_map(tickers:list, company_df: pd.DataFrame):
    df = company_df[company_df['tickers'].isin(tickers)]
    return df.groupby('sector').apply(list).to_dict()

def sp500_tickers(df: pd.DataFrame):
    unique_tickers = set()
    for tickers in df['tickers']:
        ticks = set(tickers)
        unique_tickers.update(ticks)
    
    return list(unique_tickers)

def get_ticker_data(file: str, sp500_tickers: list[str]):
    df = pd.read_parquet(file)
    df = df[(~df['ticker'].isna()) & (df['ticker'].isin(sp500_tickers))]
    # df['ticker'] = df['ticker'].apply(lambda x: x.replace(".","-"))
    df = df[df['table'] == 'SEP']
    return df



