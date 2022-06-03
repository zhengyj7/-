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


# 1. 交易环境初始化
class Broker(object):
    def __init__(self):
        self.commission = 3/10000         # 手续费万3
        self.slipper_rate = 5/10000        # 滑点万5
        self.trades = []               # 交易数据
        self.active_orders = []          # 当前提交的订单
        self.backtest_df = None          # 回测的DataFrame数据
        
    # 设置手续费
    def set_commission(self, commission: float):
        self.commission = commission
    
    # 获取手续费
    def get_commission(self):
        return self.commission

    # 设置滑点率
    def set_slipper_rate(self, slipper_rate: float):
        self.slipper_rate = slipper_rate
    
    # 获取滑点率
    def get_slipper_rate(self):
        return self.slipper_rate
    
    # 设置回测数据
    def set_backtest(self, df):
        if not isinstance(df, pd.DataFrame):
            df = df.to_frame()
        self.backtest_df = df
        
    # 获取回测数据
    def get_backtest(self):
        return self.backtest_df
    
    
# 2. 每日调仓交易
class Trade(object):
    def __init__(self, Broker): 
        self.commission = Broker.get_commission()             # 手续费
        self.slipper_rate = Broker.get_slipper_rate()          # 滑点
        self.backtest_df = Broker.get_backtest()             # 回测数据（净值）
    
     # 获取手续费
    def get_commission(self):
        return self.commission
    
    # 获取滑点率
    def get_slipper_rate(self):
        return self.slipper_rate
    
    # 获取回测数据
    def get_backtest(self):
        return self.backtest_df
    
    # 抽取年份、月份、周
    def extract_year_month_week(self, is_year=False, is_month=False, is_week_of_year=False):
        if is_year:
            year_df = pd.DataFrame([ind.strftime("%Y") for ind in self.backtest_df.index], columns=['年份'], index = self.backtest_df.index)
            self.backtest_df = pd.concat([self.backtest_df, year_df], axis=1)
        if is_month:
            month_df = pd.DataFrame([ind.strftime("%m") for ind in self.backtest_df.index], columns=['月份'], index = self.backtest_df.index)
            self.backtest_df = pd.concat([self.backtest_df, month_df], axis=1)
        if is_week_of_year:
            week_of_year_df = pd.DataFrame([ind.strftime("%V") for ind in self.backtest_df.index], columns=['年周数'], index = self.backtest_df.index)
            self.backtest_df = pd.concat([self.backtest_df, week_df], axis=1)

    def daily_return(self):               # 每日收益率（基于净值）
        daily_return_df = self.backtest_df['净值'].pct_change().to_frame()
        daily_return_df.columns = ['每日收益率']
        daily_return_df.fillna(method='bfill', inplace=True)
        # 将每日收益率附加到原DataFrame
        self.backtest_df = pd.concat([self.backtest_df, daily_return_df], axis=1)
        return daily_return_df
    
    def annualized_return(self):            # 年化收益率（基于净值）
        year_list, annualized_return_list = [], []
        for item in self.backtest_df.groupby(self.backtest_df['年份']):
            # 获取年份
            year = item[0]
            # 获取期初和期末日期
            start_date, end_date = item[1].index[0], item[1].index[-1:] 
            # 累计收益率 = （期末净值 - 期初净值） / 期初净值
            cum_return = (item[1].loc[end_date]['净值'].values[0] - item[1].loc[start_date]['净值']) / item[1].loc[start_date]['净值']
            # 交易频率t = 策略运行的交易日数 / 252
            t = len(item[1]) / 252
            # 年化收益率 = (1 + 累计收益率)^(1/t) - 1
            annualized_return = np.power((1 + cum_return), (1/t)) - 1
            year_list.append(year)
            annualized_return_list.append(annualized_return)
        annualized_return_df = pd.DataFrame(list(zip(year_list, annualized_return_list)), columns=['年份', '年化收益率']).set_index(['年份'])
        return annualized_return_df
    
    def basic_cum_return(self):             # 基准累计收益（基于开盘价）
        basic_cum_return_df = self.backtest_df['甲醇期货主连开盘价'].apply(lambda x: x / self.backtest_df['甲醇期货主连开盘价'][0]).to_frame()
        basic_cum_return_df.columns = ['基准累计收益']
        # 将基准累计收益附加到原DataFrame
        self.backtest_df = pd.concat([self.backtest_df, basic_cum_return_df], axis=1) 
        return basic_cum_return_df
    
    def annualized_volatility(self):          # 年化波动率（基于每日收益率）
        year_list, annualized_volatility_list = [], []
        self.backtest_df.fillna(method='bfill', inplace=True)
        for item in self.backtest_df.groupby(self.backtest_df['年份']):
            # 获取年份
            year = item[0]
            # 算每日收益率标准差
            annualized_volatility = np.std(item[1]['每日收益率'], ddof=1) * np.sqrt(252 / len(item[1]))
            year_list.append(year)
            annualized_volatility_list.append(annualized_volatility)
        annualized_return_df = pd.DataFrame(list(zip(year_list, annualized_volatility_list)), columns=['年份', '年化波动率']).set_index(['年份'])
        return annualized_return_df
    
    def max_drawback(self, is_return_drawback_details=False):       # 最大回撤率（基于净值）
        year_list, max_drawback_list, drawback_by_year_list = [], [], []
        df = self.backtest_df[['年份', '净值']]
        for item in df.groupby(df['年份']):
            # 获取年份
            year = item[0]
            # 算最大回撤率
            drawback_list = []
            for i in range(1, len(item[1])+1):
                new_asset_list = item[1].copy()['净值'].values
                if i > 1:
                    previous_list_max = max(new_asset_list[:i-1])
                else:
                    previous_list_max = max(new_asset_list[:i])
                value_td =  new_asset_list[i-1]
                if value_td < previous_list_max:
                    drawback_list.append((value_td - previous_list_max) / previous_list_max)
                else:
                    drawback_list.append(0)
            year_list.append(year)
            max_drawback_list.append(min(drawback_list))
            drawback_by_year_list.append(drawback_list)
        if is_return_drawback_details:  
            return pd.DataFrame(list(zip(year_list, max_drawback_list, drawback_by_year_list)), columns=['年份', '最大回撤率', '回撤详情']).set_index(['年份'])
        else:
            return pd.DataFrame(list(zip(year_list, max_drawback_list)), columns=['年份', '最大回撤率']).set_index(['年份'])

    def sharpe_ratio(self):                  # 夏普比率 每日收益率.mean()/每日收益率.std()×sqrt(252)
        year_list, sharpe_ratio_list = [], []
        df = self.backtest_df[['年份', '每日收益率']]
        for item in self.backtest_df.groupby(df['年份']):
            # 获取年份
            year = item[0]
            # 算每日收益率均值和标准差
            sharpe_ratio =  (np.mean(item[1]['每日收益率']) / np.std(item[1]['每日收益率'], ddof=1)) * np.sqrt(252 / len(item[1]))
            year_list.append(year)
            sharpe_ratio_list.append(sharpe_ratio)
            sharpe_ratio_df = pd.DataFrame(list(zip(year_list, sharpe_ratio_list)), columns=['年份', '夏普率']).set_index(['年份'])
        return sharpe_ratio_df
        
    def calmar_ratio(self):                  #卡玛比率 = 年化收益率/最大回撤率
        calmar_ratio_list = []
        for ar, md in zip(list(self.annualized_return().values), list(self.max_drawback().values)):
            calmar_ratio_list.append(ar[0]/ md[0])
        calmar_ratio_df = pd.DataFrame(list(zip(list(self.annualized_return().index), calmar_ratio_list)), columns=['年份', '卡玛比率']).set_index(['年份'])
        return calmar_ratio_df
    
    def factor_summary(self):                 # 汇总‘年化收益率’、‘年化波动率’、‘最大回撤率’、‘夏普比率’、‘卡玛比率’
        return pd.concat([self.annualized_return(), self.annualized_volatility(), self.max_drawback(), self.sharpe_ratio(), self.calmar_ratio()], axis=1).T
    
    def backtest_graph(self, figsize=(12,6), title=None):        # 可视化‘回撤详情’（回撤），‘基准累计收益’， ‘净值（策略累计收益）’
        drawback_detail_all_list = []
        for drawback_detail in self.max_drawback(is_return_drawback_details=True)['回撤详情'].values:
            drawback_detail_all_list = drawback_detail_all_list + drawback_detail
        drawback_detail_df = pd.DataFrame(drawback_detail_all_list, columns=['回撤']).set_index(self.backtest_df.index)
        # 初始化画布
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111)
        # 左轴
        self.backtest_df['基准累计收益'].plot(ax=ax1, style='b-',alpha=0.4, label='基准累计收益')
        self.backtest_df['净值'].plot(ax=ax1, style='r-', alpha=0.4 ,label='策略累计收益')
        ax1.set_yticks(np.arange(0.5, 2, 0.2)) # 设置左边纵坐标刻度
        ax1.set_ylabel('累计净值') # 设置左边纵坐标标签
        plt.legend(loc='upper right')
        # 右轴
        ax2 = ax1.twinx()
        ax2.fill_between(x=list(drawback_detail_df.index), y1=[val[0] for val in drawback_detail_df.values], y2=0, data=drawback_detail_df, alpha=0.4, label='回撤')
        ax2.set_yticks(np.arange(0,-1,-0.2)) # 设置右边纵坐标刻度
        ax2.set_ylabel('回撤') # 设置右边纵坐标标签
        plt.legend(loc='upper left')
        plt.tight_layout()
        index_list = np.unique([ind.strftime("%Y-%m") for ind in self.backtest_df.index])
        plt.xticks([index_list[ind] for ind in np.arange(0, len(index_list), 3)])
        if title:
            plt.title(title)
        plt.show()
