'''
This script contains python functions utilised for sentiment analysis in main code
v1.0
'''
import pandas as pd
import numpy as np

def process_sa_data(tweet_old, coy_tweet):
    print('There are %s tweet in total.' %(len(tweet_old)))
    print('Data Read: ')
    display(tweet_old.head(2))
    display(coy_tweet.head(2))
    tweet_old['date'] = pd.to_datetime(tweet_old['post_date'], unit='s').dt.date
    tweet_old.date=pd.to_datetime( tweet_old.date,errors='coerce')
    tweet_1617 = tweet_old.loc[(tweet_old['date'] >= '2016-01-01')
                         & (tweet_old['date'] < '2018-01-01')]
    tweets=tweet_1617.merge(coy_tweet,how='left',on='tweet_id')
    # display(tweets.ticker_symbol.unique())
    # array(['AAPL', 'AMZN', 'MSFT', 'GOOG', 'GOOGL', 'TSLA'], dtype=object)
    tweets['ticker_symbol'] = np.where(tweets['ticker_symbol'] =='GOOGL', 'GOOG', tweets['ticker_symbol'])
    print('From 2016 to 2017, there are %s tweets.' %(len(tweets)))
    display(tweets.head(2))
    return tweets

def get_sentiment(sia, tweets,ticker,start='2016-01-01',end='2017-12-31'):
    df=tweets.loc[((tweets.ticker_symbol==ticker)&(tweets.date>=start)&(tweets.date<=end))]
    # apply the SentimentIntensityAnalyzer
    df.loc[:,('score')]=df.loc[:,'body'].apply(lambda x: sia.polarity_scores(x)['compound'])
    # create label
    df.loc[:,('label')]=pd.cut(np.array(df.loc[:,'score']),bins=[-1, -0.66, 0.32, 1],right=True ,labels=["bad", "neutral", "good"])

    df=df.loc[:,["date","score","label","tweet_id","body"]]
    return df

def preprocess(df, score_col):
    df['date'] = pd.to_datetime(df['date'])
    # group by date and get average sentiment score per day
    daily = df.groupby(df['date'].dt.date)['score'].mean()
    df_daily = pd.DataFrame({'Date': daily.index, score_col: daily.values})
    df_daily['Date'] = pd.to_datetime(df_daily['Date']).dt.date
    df_daily.Date=pd.to_datetime( df_daily.Date,errors='coerce')
    return df_daily

def get_es3_sa(stock_list, stock_col_list, es3ta_1617):
    final = es3ta_1617
    for count, i in enumerate(stock_list):
        temp = preprocess(i, stock_col_list[count])
        final = es3ta_1617.merge(temp, on='Date', how='left')
    datetime_series = pd.to_datetime(final['Date'])
    datetime_index = pd.DatetimeIndex(datetime_series.values)
    final = final.set_index(datetime_index)
    final = final.sort_values(by='Date')
    final = final.drop(columns=['Date','20SD'])
    display(final.head())
    return final
