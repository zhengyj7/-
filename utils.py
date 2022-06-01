import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline
import statsmodels.regression.rolling as sm
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.stattools import durbin_watson
import itertools



class data_tools(object):
    def __init__(self, table_name, sheet_name, header, index_col, names):
        self.table_name = table_name
        self.sheet_name = sheet_name
        self.header = header
        self.index_col = index_col
        self.names = names
        
        
    def load_data(self, select_date=None):
        """
        - @Target: Load data from excel
        - @Params: 
        - @Return: DataFrame or series
        """
        df = pd.read_excel(io=self.table_name, sheet_name=self.sheet_name, header=self.header, index_col=self.index_col, names=self.names)
        df.sort_index(inplace=True, ascending=True)
        if select_date:
            df = df[df.index >= select_date]
        return df    
    
    
    def is_stationnary(self, df, comment=None):
        """
        - @Target: Function to check if the time series is stationary by Augmented Dickey-Fuller Test
        - @Params: df
        - @Return: None
        """
        # Transform Series objet to DataFrame object
        if not isinstance(df, pd.DataFrame):
            df = df.to_frame()
        # Get all the column names
        column_names = df.columns.values
        for i, column_name in enumerate(column_names):
            if comment:
                print("Augmented Dickey-Fuller Test on {0} {1}".format(column_name, comment))
            else:
                print("Augmented Dickey-Fuller Test on {0}".format(column_name))
        # Calculate ADF value
        results_ADF = adfuller(df.iloc[:,i].dropna(), autolag='AIC')
        print('Null Hypothesis(原假设): Data has unit root. Non-Stationary（不稳定）.')
        print("Test statistic(检验统计量) = {:.3f}".format(results_ADF[0]))
        print("P-value(P值) = {:.3f}".format(results_ADF[1]))
        print("Critical values :")
        for k, v in results_ADF[4].items():
            print("\t当显著性水平为{}: 决断值为{:.3f} ==> The data is {} stationary with {}% confidence".format(k, v, "not" if v<results_ADF[0] else "", 100-int(k[:-1])))
        print('\n')
        return
    
    
    def time_series_analysis(self, df):
        """
        - @Target: Time serires analysis on factor
        - @Params: df
        - @Return: Graph
        """
        # ADF检验: 看原时间序列是否稳定
        self.is_stationnary(df)
        
        # 可视化原时间序列数据
        fig = plt.figure(figsize=(15, 8))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.plot(df)
        ax1.set_title("Raw time series")
        
        # 对原时间序列进行1阶差分并可视化
        df_diff_1 =  df.diff()
        self.is_stationnary(df_diff_1, comment='(一阶差分)')
        ax2.plot(df_diff_1)
        ax2.set_title("Time series with 1 lag")
        plt.show()
        return
    
    
    def acf_pacf_plot(self, df, lags=20, alpha=0.1):
        """
        - @Target: Funtion to visualize the ACF and PACF plot
        - @Params:
            df: DataFrame or Series
            lags: numbers of lags of autocorrelation to be plotted
            alpha: set the width of confidence interval. if aplha=1, hidden blue bound
            acf: is or not to plot acf
        - @Return: None
        """
        # Transform Series objet to DataFrame object
        if not isinstance(df, pd.DataFrame):
            df = df.to_frame()
        fig, axes = plt.subplots(nrows=1, ncols=2)
        # Plot the ACF
        plot_acf(df, lags=lags, alpha=alpha, ax=axes[0])
        # Plot the PACF
        plot_pacf(df, lags=lags, alpha=alpha, ax=axes[1])
        plt.tight_layout()
        plt.show()
        return
    
    
    def grid_search_aic_arimax(self, df, p=range(1,5), d=range(0,2), q=(1,5), return_aic=False):
        """
        - @Target: Function to determine the best orders based on the possible values we found by ACF and PACF plot
        - @Params:
            df: DataFrame or Series
            exog: Array of exogenous regressors
            p: range of p in list, default (0,2)
            d: range of d in list, default (0,2)
            q: range of q in list, default (0,2)
            seasonal_lags: seasonal lags of data
            return_aic: is or not to return AIC evaluation dataframe
        - @Return:
            pdq: trend orders
            seasonal_pdq: seasonal orders
            order_aic: AIC search result
        """
        # Transform Series objet to DataFrame object
        if not isinstance(df, pd.DataFrame):
            df = df.to_frame()
        p, d, q = p, d, q
        # Get all the possible combination of orders
        pdq = list(itertools.product(p, d, q))
        # Initialize the AIC evaluation dict
        order_aic = {}
        for param in pdq:
            try:
                model = ARIMA(endog=df, order=param)
                results = model.fit()
                order_aic['Order'] = [param]
                order_aic['AIC'] = [results.aic] 
            except:
                continue
        # Transform AIC evaluation dict to dataframe
        order_aic = pd.DataFrame.from_dict(order_aic)
        print(order_aic)
        # Affect the column name
        order_aic.columns=['Order', 'AIC']
        # Get the minimum AIC tuple
        order_aic_min = order_aic.loc[np.where(order_aic['AIC'] == order_aic['AIC'].min())]
        # Get the best orders and seasonal orders
        pdq = order_aic_min.iloc[0]['Order']
        if return_aic:
            return pdq,order_aic
        else:
            return pdq
