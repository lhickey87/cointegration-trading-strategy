import pandas as pd
import numpy as np
from datetime import datetime
from price_data import get_price_data

def constituents_df(file: str, start_date):
    df = pd.read_csv(file)
    df['date'] = df['date'].apply(lambda x: pd.to_datetime(x))
    df['tickers'] = df['tickers'].apply(lambda x: sorted(x.split(',')))
    df.set_index('date')
    df = df[df.index > start_date]
    return df

# we would call this after getting constituents_df
def get_constituents(constituents_df: pd.DataFrame):
    mySet = set()
    for _, row in constituents_df.iterrows():
        mySet.update(row["tickers"])
    return list(mySet)

#this gives us the full tickers list 
def get_ticker_data(file: str,tickers: list[str]):
    df = pd.read_parquet(file)
    df['ticker'] = df['ticker'].apply(lambda x: x.replace(".","-"))
    df = df[(~df['tickers'].isna()) & (df['tickers'].isin(tickers))]
    df = df[df['table'] == 'SEP']
    return df


def get_sectors_map(tickers_df: pd.DataFrame):
    groups = tickers_df.groupby('sector').apply(list).to_dict()
    return groups

#df = constituents_df("data/constintuents.csv", start)
# constituents = get_constituents(df)
# get_ticker_data(file, constituents)
# now we can call to get_sectors_map

