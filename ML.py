from yahoo_fin import stock_info
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
import datetime as dt
import datetime


data1 = yf.download("WIPRO.NS", period='7d', interval='1m')
data1.to_csv('file1.csv')

data = pd.read_csv('file1.csv')


print(data)
print(data.head())

data.isnull().sum()

plt.figure(figsize=(15,5))
plt.plot(data['Close'])
plt.title('Close price.', fontsize=15)
plt.ylabel('Price in INR.')
plt.show()


features = ['Open', 'High', 'Low','Close', 'Volume']
 

plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
  plt.subplot(2,3,i+1)
  sb.boxplot(data[col])
plt.show()


data[['Date', 'Time']] = data['Datetime'].str.split(' ', 1, expand=True)
data[['TimeS', 'TimeV']] = data['Time'].str.split('+', 1, expand=True)
data['Date'] =  data['Date']+' '+data['TimeS']


x=pd.to_datetime(data['Date'], format='%Y-%m-%d %H:%M:%S')
y=data['High'].values.reshape(-1, 1)
z=data['Low'].values.reshape(-1, 1)


#ML Model
lm = RandomForestRegressor(n_estimators=20, random_state=0)
lm.fit(x.values.reshape(-1, 1),y)

lmL = RandomForestRegressor(n_estimators=20, random_state=0)
lmL.fit(x.values.reshape(-1, 1),z)

#Predictive data processing
x1=['2021-12-31 15:15:00','2021-12-31 10:30:00','2021-12-31 14:00:00','2021-12-31 14:30:00']
df1 = pd.DataFrame(columns = ['Date'])
df1= df1.append(pd.DataFrame(x1,columns=['Date']),ignore_index = True)
x2=pd.to_datetime(df1['Date'], format='%Y-%m-%d %H:%M:%S')

#ML Predict
predictions = lm.predict(x2.values.astype(float).reshape(-1, 1))
predictions1 = lmL.predict(x2.values.astype(float).reshape(-1, 1))

print(x2)
print(predictions)
print(predictions1)



