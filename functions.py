'''
This script contains some of the python functions utilised for analysis and data prep in main code
v1.0
'''
import pandas as pd
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
import seaborn as sns

def chg(df, name):
    df.columns=df.columns.map(lambda x : x+name if x !='Date' else x)

def read_process_stock():
    es3 = pd.read_csv('~/Downloads/proj/data/Stock/ES3.csv')
    amzn = pd.read_csv('~/Downloads/proj/data/Stock/AMZN.csv')
    msft = pd.read_csv('~/Downloads/proj/data/Stock/MSFT.csv')
    aapl = pd.read_csv('~/Downloads/proj/data/Stock/AAPL.csv')
    tsla = pd.read_csv('~/Downloads/proj/data/Stock/TSLA.csv')
    goog = pd.read_csv('~/Downloads/proj/data/Stock/GOOG.csv')
    chg(tsla, '_tsla')
    chg(goog, '_goog')
    chg(es3, '_es3')
    chg(amzn, '_amzn')
    chg(msft, '_msft')
    chg(aapl, '_aapl')
    data = amzn.merge(msft).merge(aapl).merge(tsla).merge(goog)
    data = data.merge(es3, on='Date', how='left')
    data['Date'] = pd.to_datetime(data['Date']).dt.date
    data.Date=pd.to_datetime( data.Date,errors='coerce')
    return data, es3, amzn, msft, aapl, tsla, goog

def plot_ts(data, stock, currency):
    fig, ax = plt.subplots(figsize=(15,4))
    ax.plot(data['Date'], data[f'Close_{stock}'], color='#008B8B')
    ax.set(xlabel="Date", ylabel=f"{currency}", title=f"{stock} Price")
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    plt.show()

def get_corr(data):
    filter_col = [col for col in data if (col.startswith('Close'))]
    df_data = data[filter_col]
    display(df_data.corr())
    display(sns.heatmap(df_data.corr()))
    # sns.jointplot(x='Close_es3', y='Close_spy', data=df_data1, kind='scatter')

def data_2016_2017(data):
    return data[252:755]

# data[776:1219] Feb 2018 to Nov 4 2019
def get_ta(df):
    '''
    include technical analysis to obtain overall trend
    '''
    df['MA7'] = df.iloc[:,4].rolling(window=7).mean() #take close
    df['MA20'] = df.iloc[:,4].rolling(window=20).mean() #take close
    # MACD: subtracting the 26-period exponential moving average (EMA) from the 12-period EMA
    df['MACD'] = df.iloc[:,4].ewm(span=26).mean() - df.iloc[:,1].ewm(span=12,adjust=False).mean()
    # Bollinger Bands
    df['20SD'] = df.iloc[:, 4].rolling(20).std()
    df['upper_band'] = df['MA20'] + (df['20SD'] * 2)
    df['lower_band'] = df['MA20'] - (df['20SD'] * 2)
    return df

def prep_es3_with_technical_analysis(es3):
    ta = get_ta(es3)
    es3ta = ta.iloc[20:,:].reset_index(drop=True)
    es3ta['Date'] = pd.to_datetime(es3ta['Date']).dt.date
    es3ta.Date=pd.to_datetime( es3ta.Date,errors='coerce')
    es3ta_1617 = es3ta.loc[(es3ta['Date'] >= '2016-01-01') & (es3ta['Date'] < '2018-01-01')]
    display(es3ta_1617.head(2))

    #plot graph
    fig,ax = plt.subplots(figsize=(15, 6), dpi = 200)
    x_ = range(3, es3ta_1617.shape[0])
    x_ = list(es3ta.index)

    ax.plot(es3ta_1617['Date'], es3ta_1617['Close_es3'], label='Closing Price', color='#6A5ACD')
    ax.plot(es3ta_1617['Date'], es3ta_1617['MA7'], label='Moving Average (7 days)', color='g', linestyle='--')
    ax.plot(es3ta_1617['Date'], es3ta_1617['MA20'], label='Moving Average (20 days)', color='r', linestyle='-.')
    ax.plot(es3ta_1617['Date'], es3ta_1617['upper_band'], label='Boillinger upper', color='y', linestyle=':')
    ax.plot(es3ta_1617['Date'], es3ta_1617['lower_band'], label='Boillinger lower', color='y', linestyle=':')
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    plt.title('Technical indicators')
    plt.ylabel('Closing Price (SGD)')
    plt.xlabel("Year")
    plt.legend()

    plt.show()
    return es3ta_1617
