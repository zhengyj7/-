## 甲醇期货基本面单因子趋势策略
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
### 5. 反思
- 期货基本面量化的核心是对该品种的供求关系、上下游产业链的情况及品种的内在属性要非常熟悉，这之于数据预处理非常的重要
- 目前还只是单因子趋势策略，未来可拓展至多因子趋势策略、套利策略、……
- 对于能化板块而言，商品价格变动除了受到以需求为主导的供求关系的影响，还会经常受到环保限产、国际贸易政策的影响，短期对板块中期货价格产生较大的波动 ---> 事件驱动策略
- 策略优化空间：开平仓阈值大小的选择、加入止盈止损条件、调仓频率
