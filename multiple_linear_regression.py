!pip install bubbly

import numpy as np
import pandas as pd   
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as grid_spec
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as po
from bubbly.bubbly import bubbleplot
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected = True)


data1 = pd.read_csv('../input/world-happiness-report-with-terrorism/WorldHappinessReportwithTerrorism-2015.csv')
data2 = pd.read_csv('../input/world-happiness-report-with-terrorism/WorldHappinessReportwithTerrorism-2016.csv')
data3 = pd.read_csv('../input/world-happiness-report-with-terrorism/WorldHappinessReportwithTerrorism-2017.csv')
data4 = pd.read_csv('../input/world-happiness-report/2018.csv')
data5 = pd.read_csv('../input/world-happiness-report/2019.csv')
data6 = pd.read_csv('../input/world-happiness-report/2020.csv')
data7 = pd.read_csv('../input/world-happiness-report/2021.csv')
data8 = pd.read_csv('../input/world-happiness-report-2022/World Happiness Report 2022.csv')
data9 = pd.read_csv('../input/global-terrorism-report-for-world-happiness-report/GlobalTerrorismReport-2015.csv')
data10 = pd.read_csv('../input/global-terrorism-report-for-world-happiness-report/GlobalTerrorismReport-2016.csv')
data11 = pd.read_csv('../input/global-terrorism-report-for-world-happiness-report/GlobalTerrorismReport-2017.csv')

#2015 Dünya Mutluluk Raporu
print(data1.columns)
print(data1.info())
print(data1.describe()) 
df1 = data1["country"]
dff1 = df1.value_counts() #Sütundaki null olmayan her bir unique değerin kaç kez kullanıldığını gösteren bir seri döndürür. 

x1 = data1.iloc[:,5:].values #mutluluk puanını etkileyen girdiler
y1 = data1.iloc[:,3:4].values #sonuç(mutluluk puanı sütunu)

from sklearn.model_selection import train_test_split
x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train1 = sc.fit_transform(x_train1)
X_test1 = sc.fit_transform(x_test1)
Y_train1 = sc.fit_transform(y_train1)
Y_test1 = sc.fit_transform(y_test1)

#Makine öğrenmesi modeli oluşturulur ve modele göre ağırlık çarpan değerleri hesaplanır.
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train1, y_train1)
print("b0: ", lr.intercept_)
print("other b: ", lr.coef_)

y_pred1 = lr.predict(x_test1)
prediction1 = lr.predict(np.array([[1.16492,0.87717,0.64718,0.23889,0.12348,0.04707,2.29074,542]]))
print("Prediction is ", prediction1)

y_pred1 = lr.predict(x_test1)
prediction1 = lr.predict(np.array([[1.198274,1.337753,0.637606,0.300741,0.099672,0.046693,1.879278,181]]))
print("Prediction is ", prediction1)

low_c = '#dd4124'
high_c = '#009473'
background_color = '#fbfbfb'
fig = plt.figure(figsize=(12, 10), dpi=150,facecolor=background_color)
gs = fig.add_gridspec(3, 3)
gs.update(wspace=0.2, hspace=0.5)

newdata1 = data1.iloc[:,4:]
categorical = [var for var in newdata1.columns if newdata1[var].dtype=='O']
continuous = [var for var in newdata1.columns if newdata1[var].dtype!='O']

happiness_mean = data1['happinessscore'].mean()

data1['lower_happy'] = data1['happinessscore'].apply(lambda x: 0 if x < happiness_mean else 1)

plot = 0
for row in range(0, 3):
    for col in range(0, 3):
        locals()["ax"+str(plot)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(plot)].set_facecolor(background_color)
        locals()["ax"+str(plot)].tick_params(axis='y', left=False)
        locals()["ax"+str(plot)].get_yaxis().set_visible(False)
        locals()["ax"+str(plot)].set_axisbelow(True)
        for s in ["top","right","left"]:
            locals()["ax"+str(plot)].spines[s].set_visible(False)
        plot += 1

plot = 0

Yes = data1[data1['lower_happy'] == 1]
No = data1[data1['lower_happy'] == 0]

for variable in continuous:
        sns.kdeplot(Yes[variable],ax=locals()["ax"+str(plot)], color=high_c,ec='black', shade=True, linewidth=1.5, alpha=0.9, zorder=3, legend=False)
        sns.kdeplot(No[variable],ax=locals()["ax"+str(plot)], color=low_c, shade=True, ec='black',linewidth=1.5, alpha=0.9, zorder=3, legend=False)
        locals()["ax"+str(plot)].grid(which='major', axis='x', zorder=0, color='gray', linestyle=':', dashes=(1,5))
        locals()["ax"+str(plot)].set_xlabel(variable, fontfamily='monospace')
        plot += 1
        
Xstart, Xend = ax0.get_xlim()
Ystart, Yend = ax0.get_ylim()

ax0.text(Xstart, Yend+(Yend*0.5), 'Mutlu ve Mutsuz Ülkeler Arasındaki Farklar', fontsize=17, fontweight='bold', fontfamily='sansserif',color='#323232')

plt.show()

import statsmodels.regression.linear_model as sm
X1 = np.append(arr = np.ones((158,1)).astype(int), values=x1, axis=1)
r_ols1 = sm.OLS(endog = y1, exog = X1)
r1 = r_ols1.fit()
print(r1.summary())

#2016 Dünya Mutluluk Raporu 
print(data2.columns)
print(data2.info())
print(data2.describe())
df2 = data2["country"]
dff2 = df2.value_counts()

x2 = data2.iloc[:,6:].values
y2 = data2.iloc[:,3:4].values

from sklearn.model_selection import train_test_split
x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train2 = sc.fit_transform(x_train2)
X_test2 = sc.fit_transform(x_test2)
Y_train2 = sc.fit_transform(y_train2)
Y_test2 = sc.fit_transform(y_test2)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train2, y_train2)
print("b0: ", lr.intercept_)
print("other b: ", lr.coef_)

y_pred2 = lr.predict(x_test2)
prediction2 = lr.predict(np.array([[1.06098,0.94632,0.73172,0.22815,0.15746,0.12253,2.08528,422]]))
print("Prediction is ", prediction2)

y_pred2 = lr.predict(x_test2)
prediction2 = lr.predict(np.array([[1.198274,1.337753,0.637606,0.300741,0.099672,0.046693,1.879278,181]]))
print("Prediction is ", prediction2)

figure = bubbleplot(dataset = data2, x_column = 'happinessscore', y_column = 'healthlifeexpectancy', 
    bubble_column = 'country', size_column = 'economysituation', color_column = 'region', 
    x_title = "Happiness Score", y_title = "Health Life Expectancy", title = 'Happiness vs Health Life Expectancy vs Economy',
    x_logscale = False, scale_bubble = 1, height = 650)
​
po.iplot(figure)

import statsmodels.regression.linear_model as sm
X2 = np.append(arr = np.ones((157,1)).astype(int), values=x2, axis=1)
r_ols2 = sm.OLS(endog = y2, exog = X2)
r2 = r_ols2.fit()
print(r2.summary())

#2017 Dünya Mutluluk Raporu
print(data3.columns)
print(data3.info())
print(data3.describe())
df3 = data3["country"]
dff3 = df3.value_counts()

x3 = data3.iloc[:,5:].values
y3 = data3.iloc[:,2:3].values

from sklearn.model_selection import train_test_split
x_train3, x_test3, y_train3, y_test3 = train_test_split(x3, y3, test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train3 = sc.fit_transform(x_train3)
X_test3 = sc.fit_transform(x_test3)
Y_train3 = sc.fit_transform(y_train3)
Y_test3 = sc.fit_transform(y_test3)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train3, y_train3)
print("b0: ", lr.intercept_)
print("other b: ", lr.coef_)

y_pred3 = lr.predict(x_test3)
prediction3 = lr.predict(np.array([[1.06098,0.94632,0.73172,0.22815,0.15746,0.12253,2.08528,422]]))
print("Prediction is ", prediction3)

y_pred3 = lr.predict(x_test3)
prediction3 = lr.predict(np.array([[1.16492,0.87717,0.64718,0.23889,0.12348,0.04707,2.29074,542]]))
print("Prediction is ", prediction3)

trace1 = [go.Choropleth(
               colorscale = 'Electric',
               locationmode = 'country names',
               locations = data3['country'],
               text = data3['country'], 
               z = data3['happinessrank'],
               )]

layout = dict(title = 'Happiness Rank World',
                  geo = dict(
                      showframe = True,
                      showocean = True,
                      showlakes = True,
                      showcoastlines = True,
                      projection = dict(
                          type = 'hammer'
        )))


projections = [ "equirectangular", "mercator", "orthographic", "natural earth","kavrayskiy7", 
               "miller", "robinson", "eckert4", "azimuthal equal area","azimuthal equidistant", 
               "conic equal area", "conic conformal", "conic equidistant", "gnomonic", "stereographic", 
               "mollweide", "hammer", "transverse mercator", "albers usa", "winkel tripel" ]

buttons = [dict(args = ['geo.projection.type', y],
           label = y, method = 'relayout') for y in projections]

annot = list([ dict( x=0.1, y=0.8, text='Projection', yanchor='bottom', 
                    xref='paper', xanchor='right', showarrow=False )])


# Update Layout Object
layout[ 'updatemenus' ] = list([ dict( x=0.1, y=0.8, buttons=buttons, yanchor='top' )])
layout[ 'annotations' ] = annot


fig = go.Figure(data = trace1, layout = layout)
po.iplot(fig)

import statsmodels.regression.linear_model as sm
X3 = np.append(arr = np.ones((155,1)).astype(int), values=x3, axis=1)
r_ols3 = sm.OLS(endog = y3, exog = X3)
r3 = r_ols3.fit()
print(r3.summary())

#2018 Dünya Mutluluk Raporu

#Eksik verilerin 0 olarak doldurulması sağlanmıştır.
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='constant',fill_value= 0)
impdata = data4.iloc[:,2:9].values
imputer = imputer.fit(impdata[:,2:9])
impdata[:,2:9] = imputer.transform(impdata[:,2:9])
data4.iloc[:,2:9] = impdata[:,:]

print(data4.columns)
print(data4.info())
print(data4.describe())
df4 = data4["Country or region"]
dff4 = data4.value_counts()

x4 = data4.iloc[:,3:].values
y4 = data4.iloc[:,2:3].values

from sklearn.model_selection import train_test_split
x_train4, x_test4, y_train4, y_test4 = train_test_split(x4, y4, test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train4 = sc.fit_transform(x_train4)
X_test4 = sc.fit_transform(x_test4)
Y_train4 = sc.fit_transform(y_train4)
Y_test4 = sc.fit_transform(y_test4)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train4, y_train4)
print("b0: ", lr.intercept_)
print("other b: ", lr.coef_)

import statsmodels.regression.linear_model as sm
X4 = np.append(arr = np.ones((156,1)).astype(int), values=x4, axis=1)
r_ols4 = sm.OLS(endog = y4, exog = X4)
r4 = r_ols4.fit()
print(r4.summary())
                            
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='constant',fill_value= 0)
impdata = data5.iloc[:,3:9].values
imputer = imputer.fit(impdata[:,3:9])
impdata[:,3:9] = imputer.transform(impdata[:,3:9])
data5.iloc[:,3:9] = impdata[:,:]

print(data5.columns)
print(data5.info())
print(data5.describe())
df5 = data5["Country or region"]
dff5 = data5.value_counts()

x5 = data5.iloc[:,3:].values
y5 = data5.iloc[:,2:3].values

from sklearn.model_selection import train_test_split
x_train5, x_test5, y_train5, y_test5 = train_test_split(x5, y5, test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train5 = sc.fit_transform(x_train5)
X_test5 = sc.fit_transform(x_test5)
Y_train5 = sc.fit_transform(y_train5)
Y_test5 = sc.fit_transform(y_test5)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train5, y_train5)
print("b0: ", lr.intercept_)
print("other b: ", lr.coef_)

import statsmodels.regression.linear_model as sm
X5 = np.append(arr = np.ones((156,1)).astype(int), values=x5, axis=1)
r_ols5 = sm.OLS(endog = y5, exog = X5)
r5 = r_ols5.fit()
print(r5.summary())
                           
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='constant',fill_value= 0)
impdata = data6.iloc[:,4:20].values
imputer = imputer.fit(impdata[:,4:20])
impdata[:,4:20] = imputer.transform(impdata[:,4:20])
data6.iloc[:,4:20] = impdata[:,:]

print(data6.columns)
print(data6.info())
print(data6.describe())
df6 = data6["Country name"]
dff6 = data6.value_counts() 

x6 = data6.iloc[:,4:].values
y6 = data6.iloc[:,2:3].values

from sklearn.model_selection import train_test_split
x_train6, x_test6, y_train6, y_test6 = train_test_split(x6, y6, test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train6 = sc.fit_transform(x_train6)
X_test6 = sc.fit_transform(x_test6)
Y_train6 = sc.fit_transform(y_train6)
Y_test6 = sc.fit_transform(y_test6)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train6, y_train6)
print("b0: ", lr.intercept_)
print("other b: ", lr.coef_)

import statsmodels.regression.linear_model as sm
X6 = np.append(arr = np.ones((153,1)).astype(int), values=x6, axis=1)
r_ols6 = sm.OLS(endog = y6, exog = X6)
r6 = r_ols6.fit()
print(r6.summary())
                           
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='constant',fill_value= 0)
impdata = data7.iloc[:,4:20].values
imputer = imputer.fit(impdata[:,4:20])
impdata[:,4:20] = imputer.transform(impdata[:,4:20])
data7.iloc[:,4:20] = impdata[:,:]

print(data7.columns)
print(data7.info())
print(data7.describe())
df7 = data7["Country name"]
dff7 = data7.value_counts()

x7 = data7.iloc[:,4:].values 
y7 = data7.iloc[:,2:3].values 

from sklearn.model_selection import train_test_split
x_train7, x_test7, y_train7, y_test7 = train_test_split(x7, y7, test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train7 = sc.fit_transform(x_train7)
X_test7 = sc.fit_transform(x_test7)
Y_train7 = sc.fit_transform(y_train7)
Y_test7 = sc.fit_transform(y_test7)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train7, y_train7)
print("b0: ", lr.intercept_)
print("other b: ", lr.coef_)

import statsmodels.regression.linear_model as sm
X7 = np.append(arr = np.ones((149,1)).astype(int), values=x7, axis=1)
r_ols7 = sm.OLS(endog = y7, exog = X7)
r7 = r_ols7.fit()
print(r7.summary())
                            
#2022 Dünya Mutluluk Raporu
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='constant',fill_value= 0)
impdata = data8.iloc[:,3:12].values
imputer = imputer.fit(impdata[:,3:12])
impdata[:,3:12] = imputer.transform(impdata[:,3:12])
data8.iloc[:,3:12] = impdata[:,:]

print(data8.columns)
print(data8.info())
print(data8.describe())
df8 = data8["Country"]
dff8 = data8.value_counts()

x8 = data8.iloc[:,3:].values 
y8 = data8.iloc[:,2:3].values

from sklearn.model_selection import train_test_split
x_train8, x_test8, y_train8, y_test8 = train_test_split(x8, y8, test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train8 = sc.fit_transform(x_train8)
X_test8 = sc.fit_transform(x_test8)
Y_train8 = sc.fit_transform(y_train8)
Y_test8 = sc.fit_transform(y_test8)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train8, y_train8)
print("b0: ", lr.intercept_)
print("other b: ", lr.coef_)

import statsmodels.regression.linear_model as sm
X8 = np.append(arr = np.ones((146,1)).astype(int), values=x8, axis=1)
r_ols8 = sm.OLS(endog = y8, exog = X8)
r8 = r_ols8.fit()
print(r8.summary())
                            
#2015 Küresel Terörizm Raporu
print(data9.columns)
print(data9.info())
print(data9.describe())
df9 = data9["country"]
dff9 = df9.value_counts()
print(dff9)

#2016 Küresel Terörizm Raporu
print(data10.columns)
print(data10.info())
print(data10.describe())
df10 = data10["country"]
dff10 = df10.value_counts()
print(dff10)

#2017 Küresel Terörizm Raporu
print(data11.columns)
print(data11.info())
print(data11.describe())
df11 = data11["country"]
dff11 = df11.value_counts()
print(dff11)
