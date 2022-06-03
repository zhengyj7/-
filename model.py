import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import warnings # 不显示 warning
warnings.filterwarnings("ignore")
import statsmodels.regression.rolling as sm
from statsmodels.tools.tools import add_constant

class arima(object):    
    def train_predict(self, df, window, order, is_plot=False, columns=None):
        predict_result_list, index_list, previous_list, true_list = [], [], [], []
        for i in range(0, len(df) - window):
            # Truncate data by window length
            x_train = df.loc[df.index[i]:df.index[i+window-1]]
            # Get the previous element
            previous_list.append(df.loc[df.index[i+window-1]])
            # Transform series to dataframe
            if not isinstance(x_train, pd.DataFrame):
                x_train = x_train.to_frame()
            # Train the model by the window length data
            model = ARIMA(endog=x_train, order=order).fit()
            if is_plot:
                print(model.summary())
                model.plot_diagnostics(figsize=(12,8))
                plt.tight_layout()
                plt.show()
            # Predict for the T+1 trade day - "One-step-ahead prediction"
            predict_result = model.forecast(1)
            index_list.append(df.index[i+window])
            predict_result_list.append(predict_result.values[0])
            true_list.append(df.loc[df.index[i+window]])
        predict_result = pd.DataFrame(list(zip(index_list, predict_result_list, previous_list, true_list)), columns=columns).set_index('Index')
        return predict_result

    
class rollingOLS(object):
    def train_predict(self, df, window, columns=None):
        predict_result = pd.DataFrame()
        for i in range(df.shape[0]-window+1):
            # 取预测日期的前window天，用于训练
            df_train = df.iloc[i:i+window, :] 
            # 对前20天数据进行建模
            exog_train, endog_train = pd.DataFrame(df_train[columns[0]]), pd.DataFrame(df_train[columns[1]])
            # 增加偏置项
            exog_train['常数项'] = 1
            # 训练模型
            rollingOLS_model = sm.RollingOLS(endog_train, exog_train, window=window)
            fit_result = rollingOLS_model.fit()
            # 处理回归系数
            params_result = pd.DataFrame(fit_result.params.iloc[-1,:]).T
            params_result.columns = [_ + '_coef' for _ in params_result.columns]
            params = pd.concat([params_result, pd.DataFrame(exog_train.iloc[-1,:]).T], axis = 1)
            predict_result = pd.concat([predict_result,params],axis=0)
            # 计算预测收益率 
            ret_f = predict_result[columns[0] + '_coef'] * predict_result[columns[0]]  \
                                    + predict_result['常数项_coef'] * predict_result['常数项']
            ret_f = ret_f.to_frame() #.shift(-2).fillna(method='ffill')
            ret_f.columns = ['预测收益率']
        return ret_f
