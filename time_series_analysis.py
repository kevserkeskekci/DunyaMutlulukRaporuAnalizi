#gereksiz uyarıları gizlemek için
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from pandas.tseries.offsets import DateOffset
import matplotlib.pyplot as plt
%matplotlib inline

data1 = pd.read_csv('../input/world-happiness-report-with-terrorism/WorldHappinessReportwithTerrorism-2015.csv')
data2 = pd.read_csv('../input/world-happiness-report-with-terrorism/WorldHappinessReportwithTerrorism-2016.csv')
data3 = pd.read_csv('../input/world-happiness-report-with-terrorism/WorldHappinessReportwithTerrorism-2017.csv')
data4 = pd.read_csv('../input/world-happiness-report/2018.csv')
data5 = pd.read_csv('../input/world-happiness-report/2019.csv')
data6 = pd.read_csv('../input/world-happiness-report/2020.csv')
data7 = pd.read_csv('../input/world-happiness-report/2021.csv')
data8 = pd.read_csv('../input/world-happiness-report-2022/World Happiness Report 2022.csv')
data1.head(10)

plt.figure(figsize=(10,5))
plt.plot(data1.country,data1.happinessscore)
plt.xlabel("Ülkeler")
plt.ylabel("Mutluluk Puanları")
plt.title("2015 Yılı Mutluluk Sıralaması")

#DURAĞANLIK TESTİ
test_result=adfuller(data1['happinessscore'])

#Ho: It is non stationary
#H1: It is stationary

def adfuller_test(happinessscore):
    result=adfuller(happinessscore)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
adfuller_test(data1['happinessscore']) 

data1['Happiness Score First Difference'] = data1['happinessscore'] - data1['happinessscore'].shift(1)
data1['happinessscore'].shift(1)
data1.head(10)

adfuller_test(data1['Happiness Score First Difference'].dropna()) #NaN değerleri siler.
data1['Happiness Score First Difference'].plot()

autocorrelation_plot(data1['happinessscore'])
plt.show() 

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(data1['Happiness Score First Difference'].iloc[13:],lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(data1['Happiness Score First Difference'].iloc[13:],lags=40,ax=ax2)

#Tahminler için ARIMA’yı uygulayalım:
model=ARIMA(data1['happinessscore'],order=(1,1,1))
model_fit=model.fit() 
model_fit.summary()

#ARIMA kullanarak Tahmini görelim:
data1['tahmin']=model_fit.predict(start=1,end=158,dynamic=True)
data1[['happinessscore','tahmin']].plot(figsize=(10,5))




