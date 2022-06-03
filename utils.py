import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
parameters = {'axes.unicode_minus': False,
              'font.sans-serif':'SimHei',
              'axes.facecolor':'0.98',
              'axes.labelsize': 16,
              'axes.titlesize': 16,
              'xtick.labelsize': 16,
              'ytick.labelsize': 16}
plt.rcParams.update(parameters)

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import itertools
import warnings # 不显示 warning
warnings.filterwarnings("ignore")


class data_tools(object):
    def __init__(self, table_name=None, sheet_name=None, header=None, index_col=None, names=None):
        self.table_name = table_name
        self.sheet_name = sheet_name
        self.header = header
        self.index_col = index_col
        self.names = names
        
    def load_data(self, select_date=None):
        """
        - @Target: Load data from excel
        - @Params: select data after select_date
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
    
    def acf_pacf_plot(self, df, lags=20, alpha=0.1, figsize=(16,8)):
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
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        # Plot the ACF
        plot_acf(df, lags=lags, alpha=alpha, ax=axes[0])
        # Plot the PACF
        plot_pacf(df, lags=lags, alpha=alpha, ax=axes[1])
        plt.tight_layout()
        plt.show()
        return
    
    
    def grid_search_aic_arima(self, df, p=range(0,5), d=range(0,2), q=range(0,5), return_aic=False, figsize=(12,6), title='Grid Search of (p,d,q) by AIC'):
        """
        - @Target: Function to determine the best orders based on the possible values we found by ACF and PACF plot
        - @Params:
            df: DataFrame or Series
            exog: Array of exogenous regressors
            p: range of p in list, default (1,5)
            d: range of d in list, default (0,2)
            q: range of q in list, default (1,5)
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
        # Initialize the AIC evaluation list
        aic_list = []
        for param in pdq:
            try:
                model = ARIMA(endog=df, order=param)
                results = model.fit()
                aic_list.append(results.aic)
            except:
                continue
        order_aic = pd.DataFrame(list(zip(pdq, aic_list)), columns=['Order', 'AIC'])
        # Get the minimum AIC tuple
        order_aic_min = order_aic.loc[np.where(order_aic['AIC'] == order_aic['AIC'].min())]
        # Get the best orders and seasonal orders
        pdq = order_aic_min.iloc[0]['Order']
        # Visualize the grid Search result of (p,d,q) by AIC
        plt.figure(figsize=figsize)
        plt.plot(order_aic['AIC'], 'b--')
        plt.title(title, fontsize=15)
        plt.xticks(np.arange(order_aic.shape[0]), list(order_aic['Order']), rotation=90)
        plt.xlabel("(p,d,q)", fontsize=15)
        plt.ylabel("AIC", fontsize=15)
        plt.show()   
    
        if return_aic:
            return pdq, order_aic
        else:
            return pdq
        
    def residual_check(self, df=None, lags=None, figsize=(8,2)):
        """Function to use Ljung Box test to inspect if the residuals are correlated
        - Parameters:
            df: DataFrame or Series
            lags: lags to take
            figsize: size of figure
        - Return value: None
        """
        # Perform the Ljung-Box test
        lb_test = acorr_ljungbox(df, lags=lags)
        p_value = pd.DataFrame(lb_test['lb_pvalue'])
        # Plot the p-values
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(p_value, linewidth=3, linestyle='--')
        ax.set_xlabel('Lags', fontsize=12)
        ax.set_ylabel('P-value', fontsize=12)
        ax.set_title('Ljung Box test - Randomness checking', fontsize=15)
        plt.show()
        return
