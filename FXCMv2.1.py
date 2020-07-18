import fxcmpy
import pandas as pd
import datetime as dt
import time
from pylab import plt
import matplotlib
import mplfinance as mpf
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
import numpy as np
from scipy.signal import find_peaks

class JoeStrat():
    def __init__(self, online = True, save = False, sample_size = 3):
        self.online = online
        self.save = save
        self.sample_size =  sample_size
        self.working_data = pd.DataFrame(np.nan, index=[0], columns=['A'])
        
        if self.online:
            #Connect
            self.con = self.connect_to_fxcm()
            #self.data = self.con.get_candles('EUR/USD', period='m1',columns=['date','asks','tickqty'], number=250)
            #Stream
            #con.subscribe_market_data('EUR/USD', (self.print_data,))
            self.con.subscribe_market_data('USDOLLAR', (self.handle_length,))
            time.sleep(1)#~1.5 Hz
            
            self.con.unsubscribe_market_data('USDOLLAR')

            if self.save:
                data.to_csv('test.csv')
        else:
            self.data = pd.read_csv('test.csv')
            self.data.set_index('date', inplace=True, drop=True)
            self.data.index = pd.to_datetime(data.index)

        #Turning Points
        print(self.get_working_data)
        self.sci_turning_points(self.get_working_data(), "Ask")
    
        #Plot
        self.new_candles_plot(self.get_working_data(), "Ask")

        self.data_diet = self.reduce_data(self.data)
        self.data_diet['deltaAO'] = (data_diet['askopen'] - data_diet['askopen'].shift(1))


    def set_sample_size(self, userinput):
        self.sample_size = userinput
    
    def set_working_data(self, userinput):
        print("Setting")
        print(userinput)
        self.working_data = userinput

    def get_sample_size(self):
        return self.sample_size
    
    def get_working_data(self):
        return self.working_data
    
    def connect_to_fxcm(self):
        print("Attempting to Connect")
        con = fxcmpy.fxcmpy(config_file='fxcm.cfg')
        if con.is_connected()==True:
            print("Connection Successful!")
            ID = con.get_account_ids()
            print("Account ID is", ID)
        else:
            print("Connection Failed")
        return con

    def print_data(self,data, dataframe):
        print(data)
        print(dataframe)
        print('%3d | %s | %s, %6s, %6s, %6s, %6s'
              % (len(dataframe), data['Symbol'],
                 pd.to_datetime(int(data['Updated']), unit='ms'),
                 data['Rates'][0], data['Rates'][1], data['Rates'][2],
                 data['Rates'][3]))
        
    def new_candles_plot(self, data, slide = -1):
        data.index.name = 'Date'
        data = data.rename(columns={"Bid": "Open", "Ask": "Close",
                        "High": "High", "Low": "Low", "tickqty": "Volume"})
        
        #https://github.com/matplotlib/mplfinance/blob/master/examples/addplot.ipynb
        dataPeak = data.copy()
        dataTrough = data.copy()
        dataPeak["Open"].where(dataPeak['TP']==1, np.nan, inplace=True)
        dataTrough["Open"].where(dataTrough['TP']==-1, np.nan, inplace=True)
        
        apds =[mpf.make_addplot(dataPeak["Open"].shift(slide),type='scatter',markersize=200,marker='^'),
               mpf.make_addplot(dataTrough["Open"].shift(slide),type='scatter',markersize=200,marker='v')]
        
        mc = mpf.make_marketcolors(up='g',down='r')
        s  = mpf.make_mpf_style(marketcolors=mc)
        #mpf.plot(data,type='candle',mav=(2),volume=True,style=s, title='EUR/USD',addplot=apds))
        mpf.plot(data,type='line',volume=True,style=s, title='EUR/USD',addplot=apds)
        
        plt.show()

    def old_candles_plot(self, data,slide = -1):
        data.index.name = 'Date'
        data = data.rename(columns={"askopen": "Open", "askclose": "Close",
                        "askhigh": "High", "asklow": "Low", "tickqty": "Volume"})

        data['3T'] = data['Close'].rolling(window=3, min_periods=0).mean()
        data.dropna(inplace=True)
        
        df_ohlc = data['Close'].resample('3T').ohlc() # not a moving avg
        df_volume = data['Volume'].resample('3T').sum()

        dataPeak = data.copy()
        dataTrough = data.copy()
        dataPeak["Open"].where(dataPeak['TP']==1, np.nan, inplace=True)
        dataTrough["Open"].where(dataTrough['TP']==-1, np.nan, inplace=True)

        dataPeak.reset_index(inplace=True)
        dataTrough.reset_index(inplace=True)
        df_ohlc.reset_index(inplace=True)
        dataPeak['Date'] = dataPeak['Date'].map(mdates.date2num)
        dataTrough['Date'] = dataTrough['Date'].map(mdates.date2num)
        df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)

        
        ax1 = plt.subplot2grid((5,1), (0,0), rowspan=3, colspan=1)
        ax2 = plt.subplot2grid((5,1), (4,0), rowspan=1, colspan=1, sharex=ax1)
        ax1.xaxis_date()#display dates as dates
        candlestick_ohlc(ax1, df_ohlc.values, width=0.001, colorup='g')
        ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values,0)

        ax1.plot(data.index, data['3T'])
        ax1.plot(data.index, data['Open'], color="k", linestyle='dashed')
        ax1.scatter(dataPeak["Date"].shift(slide), dataPeak["Open"], marker="^")
        ax1.scatter(dataTrough["Date"].shift(slide), dataTrough["Open"], marker="v")

        ax1.set_ylabel('Price')
        ax2.set_ylabel('Volume')
        ax2.set_xlabel('Date')
        ax1.set_title('EUR/USD')
        
        plt.show()
        
    def turning_points(self, df, column):
        Sag =  ((df[column].shift(-2) < df[column].shift(-1)) & 
                (df[column].shift(-1) < df[column]) &
                (df[column].shift( 1) < df[column]) &
                (df[column].shift( 2) < df[column].shift( 1))).astype(int)
        
        Hog = -((df[column].shift(-2) > df[column].shift(-1)) & 
                (df[column].shift(-1) > df[column]) &
                (df[column].shift( 1) > df[column]) &
                (df[column].shift( 2) > df[column].shift( 1))).astype(int)
        return Sag+Hog

    def sci_turning_points(self, data, column):
        #https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
        prom = 2e-4
        print(column)
        print(data)
        print(data[column].to_numpy())
        print(type(data[column].to_numpy()))
        hog, _ = find_peaks(data[column].to_numpy(), prominence=(prom))
        Hog    = 1*(data.index.isin(data.index[hog].to_numpy()))

        sag, _ = find_peaks(-data[column].to_numpy(), prominence=(prom))
        Sag    = -1*(data.index.isin(data.index[sag].to_numpy()))
        data['TP'] = Sag+Hog
        self.set_working_data(data)
        


    def reduce_data(self, data):
        data.where(data['TP']!=0, np.nan, inplace=True)
        data.dropna(inplace=True)
        return data

    def handle_length(self, data, dataframe):
        if ((len(dataframe)% 10) == 0):
            print(len(dataframe))
        if len(dataframe) > self.get_sample_size():
            dataframe.drop(dataframe.index[0], inplace=True)
        self.set_working_data(dataframe)
        print(dataframe)
        print(self.get_working_data())
        
try1 = JoeStrat(True, False, 4)
print(try1.get_working_data().head())

