## -甲醇期货基本面单因子趋势策略
### 1. 择时因子
- 1.1 基差（日频）
- 1.2 库存（周频）
- 1.3 现货价格（日频）
- 1.4 开工率（周频）
- 数据来源：同花顺、rqdata
### 2. 择时模型
- 2.1 （国泰君安期货）直接择时 + OLS滚动回归择时
- 2.2 （我）基于自己对时间序列模型模型的了解，选择了ARIMA模型
- 可以尝试一下OLS滚动回归模型，将其和ARIMA模型做一个简单的对比（不放入报告，后面有机会再和老师交流）
### 3. 回测
- 以T+1日开盘价开仓、T+2日开盘价平仓，日频调仓
- 为避免主力合约价格的跳空，甲醇期货主力选择的是开盘价前复权
- 警惕期货的交易时间与股票有所不同，回测时要将因子shift(1)，收益率shift(-1)
- 无杠杆、手续费双边万3、滑点万5
### 4. 代码说明
- utils.py：对基本面指标原时间序列进行一系列分析，确定ARIMA(p,d,q)最优参数
- model.py：训练ARIMA模型，同时输出交易信号
- backtest.py：基于交易信号的回测
