# BackTesing




class ToolBox:
    def __init__(self, df):
        import pandas as pd
        import numpy as np
        import pyodbc #Sql-Connector
        
        self.__df = df.copy()
        self.__start = df.index[0]
        self.__end = df.index[-1]
    
    
    def getDataFromSql(self, MarketReturn=True, RiskFreeReturn=None):
        '''
        Connect Sql_database for TWA00 and RiskfreeRate
        '''
        strStartDate = str(self.__start)[:10].replace('-','')
        strEndDate = str(self.__end)[:10].replace('-','')
        
        DB_dt={'ip':'YourIP','db':'YourDataBase','user':'YourUsername','pwd':'YourPassword'}
        conn_DB_dt= pyodbc.connect('DRIVER={SQL Server};SERVER='+DB_dt['ip']+';DATABASE='+DB_dt['db']+';UID='+DB_dt['user']+';PWD='+DB_dt['pwd'])
        
        qrysql_MarketData=  f"SELECT 日期,股票代號,[漲幅(%)] \
                    FROM sysdbasere WHERE 股票代號 = 'TWA00'\
                    AND 日期 BETWEEN '{strStartDate}' AND '{strEndDate}' \
                    order by 股票代號,日期 asc"

        qrysql_RiskFreeData =  f" SELECT 年月, 數值 FROM sysallun WHERE 代號 = 'R001' \
                    AND 年月 BETWEEN '{strStartDate[:-2]}' AND '{strEndDate[:-2]}' \
                    ORDER BY 年月 ASC"
        
        if MarketReturn and RiskFreeReturn:
            return pd.read_sql(qrysql_MarketData,conn_DB_dt), pd.read_sql(qrysql_RiskFreeData,conn_DB_dt)
        
        if MarketReturn:
            return pd.read_sql(qrysql_MarketData,conn_DB_dt)
        
        
    @classmethod
    def getDataFromSqlcls(self, StartDate, EndDate, MarketReturn=True, RiskFreeReturn=None):
        strStartDate = str(StartDate)[:10].replace('-','')
        strEndDate = str(EndDate)[:10].replace('-','')
        
        DB_dt={'ip':'192.168.10.30','db':'northwind','user':'xcmoney','pwd':'7319xyz888'}
        conn_DB_dt= pyodbc.connect('DRIVER={SQL Server};SERVER='+DB_dt['ip']+';DATABASE='+DB_dt['db']+';UID='+DB_dt['user']+';PWD='+DB_dt['pwd'])
        
        qrysql_MarketData=  f"SELECT 日期,股票代號,[漲幅(%)] \
                    FROM sysdbasere WHERE 股票代號 = 'TWA00'\
                    AND 日期 BETWEEN '{strStartDate}' AND '{strEndDate}' \
                    order by 股票代號,日期 asc"

        qrysql_RiskFreeData =  f" SELECT 年月, 數值 FROM sysallun WHERE 代號 = 'R001' \
                    AND 年月 BETWEEN '{strStartDate[:-2]}' AND '{strEndDate[:-2]}' \
                    ORDER BY 年月 ASC"
        
        if MarketReturn and RiskFreeReturn:
            return pd.read_sql(qrysql_MarketData,conn_DB_dt), pd.read_sql(qrysql_RiskFreeData,conn_DB_dt)
        
        if MarketReturn:
            return pd.read_sql(qrysql_MarketData,conn_DB_dt)
        
    
    @classmethod
    def convertYearly(self, cummulativeReturn, period):
        '''
        Convert input cummulativeReturn(%) to YearlyReturn(%)
        '''
        cummulativeReturn_Year = (1 + cummulativeReturn/100) ** (250/period) - 1
        return cummulativeReturn_Year*100
    
    
    @classmethod
    def calculateCummulativeReturn(self, portfolioReturn):
        '''
        Convert Series of Return(%) to Series of CummulativeReturn(%)
        '''
        portfolioReturn['cummulativeReturn'] = portfolioReturn.iloc[:,0] / 100 + 1
        portfolioReturn['cummulativeReturn'] = portfolioReturn['cummulativeReturn'].cumprod() - 1
        return portfolioReturn['cummulativeReturn']*100
    
    
    def calculateBetaAlpha(self):
        '''
        Calculate Beta, Alpha.
        Must fit the DataFrame first.
        '''
        #回歸分析模組(計算beta)----
        import statsmodels.formula.api as smf
        
        marketReturn, riskfreeReturn = self.getDataFromSql(MarketReturn=True, RiskFreeReturn=True)
        
        marketReturn.index = pd.to_datetime(marketReturn['日期'])
        
        RegressionData = pd.concat([self.__df,marketReturn], axis=1)
        RegressionData.rename(columns={'單日報酬(%)':'ret', '漲幅(%)':'MKTRET'}, inplace=True)
        mdl = smf.ols('ret ~ MKTRET', data=RegressionData).fit()
        beta = mdl.params[1]
        
        riskfreeReturn['年月'] = riskfreeReturn['年月'] + '01'
        riskfreeReturn.index = pd.to_datetime(riskfreeReturn['年月'])
        riskfreeReturn.rename(columns={'數值':'RF'}, inplace=True)
        
        cummulativePortfolioReturn = self.calculateCummulativeReturn(self.__df[['單日報酬(%)']])[-1]
        cummulativeMarketReturn = self.calculateCummulativeReturn(marketReturn[['漲幅(%)']])[-1]
        cummulativeRiskfreeReturn = self.calculateCummulativeReturn(riskfreeReturn[['RF']])[-1]
        marketPremium = cummulativeMarketReturn - cummulativeRiskfreeReturn
        
        alpha = cummulativePortfolioReturn - (cummulativeRiskfreeReturn + beta*marketPremium)
        
        return beta, alpha
    

class Backtest:
    def __init__(self, df, fee=0.001425, tax=0.003):
        import pandas as pd
        import numpy as np
        
        self.__df = df.copy()
        self.__fee = fee
        self.__tax = tax
    
    
    def __getHoldingCondition(self):
        def conditionTransformer(condition, hold_period):
            condition = list(condition)
            for i in range(len(condition)):
                if condition[i] == 1:
                    for j in range(1, hold_period):
                        if i+j < len(condition):
                            condition[i+j] = -1
            for i in range(len(condition)):
                if condition[i] == -1:
                    condition[i] = 1
            condition.pop()
            condition.insert(0,0)
            return condition
        hold = self.__df.groupby(by='code')['condition'].apply(conditionTransformer, self.__hold_period)
        return [condition for code in hold for condition in code]
    
    
    def __calculatePortfolioReturn(self):
        time = list(set(self.__df.index))
        time.sort()
        portfolio = self.__df.copy()
        portfolio['是否持有'] = self.__getHoldingCondition()
        portfolio = portfolio[portfolio['是否持有']==1]
        
        portfolioReturn = []
        portfolioNumbers = []
        buyNumbers = []
        sellNumbers = []
        lastdayStocks = portfolio[portfolio['是否持有']==2]['code'] #為了isin() 創造相同格式的空DataFrame
        
        for date in time:
            if date in portfolio.index:
                crossSectionProfolio = portfolio.loc[date]
                portfolioReturn.append(crossSectionProfolio['ret'].mean())
                portfolioNumbers.append(crossSectionProfolio['是否持有'].sum())
                
                buyNumbers.append(crossSectionProfolio[~crossSectionProfolio['code'].isin(lastdayStocks)].shape[0])
                sellNumbers.append(lastdayStocks[~lastdayStocks.isin(crossSectionProfolio['code'])].shape[0])
                lastdayStocks = crossSectionProfolio['code']
            else:
                portfolioReturn.append(0)
                portfolioNumbers.append(0)
                buyNumbers.append(0)
                sellNumbers.append(0)
        
        portfolioNumbers.pop(0)
        portfolioNumbers.append(portfolioNumbers[-1])
        buyNumbers.pop(0)
        buyNumbers.append(0)
        sellNumbers.pop(0)
        sellNumbers.append(portfolioNumbers[-1])
        
        dic = {'個股檔數':portfolioNumbers,'買進':buyNumbers, '賣出':sellNumbers, '單日報酬(%)':portfolioReturn}
        result = pd.DataFrame(dic, index=time)
        result['累計報酬(%)'] = ToolBox.calculateCummulativeReturn(result[['單日報酬(%)']])
        return result[['個股檔數','買進','賣出','單日報酬(%)','累計報酬(%)']]
    
    
    def __calculatePortfolioReturn_Taxed(self):
        portfolioReturn = self.__calculatePortfolioReturn()
        portfolioReturn['買進成本比重'] = portfolioReturn['買進'] / portfolioReturn['個股檔數']
        portfolioReturn['賣出成本比重'] = portfolioReturn['賣出'] / portfolioReturn['個股檔數']
        portfolioReturn['單日報酬(%)'] = (
                                            portfolioReturn['單日報酬(%)']/100 
                                            - self.__fee*portfolioReturn['買進成本比重']
                                            - (self.__fee+self.__tax)*portfolioReturn['賣出成本比重']
                                            )*100
        portfolioReturn['累計報酬(%)'] = ToolBox.calculateCummulativeReturn(portfolioReturn[['單日報酬(%)']])
        return portfolioReturn[['個股檔數','買進','賣出','單日報酬(%)','累計報酬(%)']]
        
        
    def __calculateStatistics(self, Tax):
        if Tax:
            data = self.__calculatePortfolioReturn_Taxed()
        else:
            data = self.__calculatePortfolioReturn()
            
        tool = ToolBox(data)
        cummulativeReturn = data['累計報酬(%)'][-1]
        cummulativeReturn_Year = ToolBox.convertYearly(cummulativeReturn, len(data))
        standard_deviation = data['單日報酬(%)'][1:].std()*np.sqrt(250)
        average_stocks = data['個股檔數'].mean()
        beta, alpha = tool.calculateBetaAlpha()

        return cummulativeReturn, cummulativeReturn_Year, standard_deviation, average_stocks, beta, alpha
    
    
    def __calculateMarketStatistics(self):
        MKTRET = ToolBox.getDataFromSqlcls(self.__df.index[0], self.__df.index[-1], MarketReturn=True)
        MKTRET.index = pd.to_datetime(MKTRET['日期'])
        MKTRET['累計報酬(%)'] = ToolBox.calculateCummulativeReturn(MKTRET[['漲幅(%)']])
        cummulativeReturn = MKTRET['累計報酬(%)'][-1]
        cummulativeReturn_Year = ToolBox.convertYearly(cummulativeReturn, len(MKTRET))
        standard_deviation = MKTRET['漲幅(%)'][1:].std()*np.sqrt(250)
        return cummulativeReturn, cummulativeReturn_Year, standard_deviation, 1, 1, 0
    
    
    def getPortfolioReturn(self, hold_period=5, Tax=True):
        '''
        hold_period: 一旦買入，持有hold_period日後，再決定是否繼續持有
        Tax: 是否計算手續費及稅率
        '''
        self.__hold_period = hold_period #只有重新呼叫才能更改hold_period
        
        if Tax:
            result = self.__calculatePortfolioReturn_Taxed().rename(columns={'單日報酬(%)':'課稅單日報酬(%)','累計報酬(%)':'課稅累計報酬(%)'})
            return result.apply(lambda x : np.round(x, 2))
        else:
            portfolioReturn = self.__calculatePortfolioReturn()
            return portfolioReturn.apply(lambda x : np.round(x, 2))
        
    
    def getStatistics(self, periods, Tax=True):
        index = ['累計報酬(%)', '年化報酬(%)', '標準差', '平均持股', 'Beta', 'Alpha']
        result = pd.DataFrame(index=index)
        result['大盤'] = self.__calculateMarketStatistics()
        if Tax:
            for period in periods:
                self.__hold_period = period
                result[str(period)+'日含稅'] = self.__calculateStatistics(Tax)
        else:
            for period in periods:
                self.__hold_period = period
                result[str(period)+'日'] = self.__calculateStatistics(Tax)
        return result.T.apply(lambda x: np.round(x,2))
        
    def plotResults(self, periods, show_difference=None):
        import matplotlib.pyplot as plt
        # matplotlib 中文顯示問題
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
        plt.rcParams['axes.unicode_minus'] = False
        
        data = self.getPortfolioReturn()
        index = data.index
        tool = ToolBox(data)
        MKTRET = tool.getDataFromSql(MarketReturn=True)
        MKTRET['大盤累計報酬(%)'] = ToolBox.calculateCummulativeReturn(MKTRET[['漲幅(%)']])

        fig, ax = plt.subplots(figsize=(24,12))
        ax.plot(index, MKTRET['大盤累計報酬(%)'], label='TWA00', lw=3)
        
        if show_difference:
            for period in periods:
                ax.plot(index, backtest.getPortfolioReturn(hold_period=period)['課稅累計報酬(%)'], label=str(period)+'日含稅', lw=1.5)
                ax.plot(index, backtest.getPortfolioReturn(hold_period=period, Tax=False)['累計報酬(%)'], label=str(period)+'日', ls='--')    
        else:
            for period in periods:
                ax.plot(index, backtest.getPortfolioReturn(hold_period=period)['課稅累計報酬(%)'], label=str(period)+'日含稅', ls='--', lw=2)
        plt.title('Cummulative Return', fontsize=32)
        plt.legend(fontsize=20)
        plt.ylabel('Return(%)', fontsize=24)
        plt.xlabel('Time', fontsize=24)
            
