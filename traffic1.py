import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
from prophet import Prophet

//CODE START FROM HERE

df=pd.read_csv("C:\\Users\\HP\\Desktop\\Traffic data.csv")

df.head()

df

df.isnull().sum()

df.info()

df['DateTime']=pd.to_datetime(df['DateTime'],format='%d-%m-%Y %H:%M')
df.info()

plt.figure(figsize=(10,7))
plt.plot(df['DateTime'],df['Vehicles'])
plt.show()

df.index=df['DateTime']
df['y']=df['Vehicles']
df.drop(columns=['ID','DateTime','Vehicles'],axis=1,inplace=True)
df=df.resample('D').sum()
df.head()

df['ds']=df.index
df.head()

size=60
from sklearn.model_selection import train_test_split
train,test=train_test_split(df,test_size=size/len(df),shuffle=False)

test.head()

model=Prophet(yearly_seasonality=True, seasonality_prior_scale=0.9)
model.fit(train)

future= model.make_future_dataframe(periods=60)
future

forecast=model.predict(future)
forecast.head()

model.plot_components(forecast)

pred=forecast.iloc[-60:,]

plt.figure(figsize=(10,7))
plt.plot(test['ds'],test['y'])
plt.plot(pred['ds'],pred['yhat'],color='red')
plt.plot(pred['ds'],pred['yhat_lower'],color='green')
plt.plot(pred['ds'],pred['yhat_upper'],color='orange')
plt.show()

plt.plot(df['ds'],df['y'])
plt.show()

plt.plot(df['ds'],forecast['yhat'])
plt.show()

