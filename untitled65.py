# -*- coding: utf-8 -*-
"""Untitled65.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yuytGMUTIX3Hud_NKXSYHcr9CsqmxUS_
"""

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout

from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('deepdata').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])

df = pd.DataFrame(rows[1:], columns=rows[0])
df

df = df.astype({'Input':'float'})
df = df.astype({'Output':'float'})
df.head()

x=df[['Input']].values
y=df[['Output']].values

x
y

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size = 0.4, random_state =35)

Scaler = MinMaxScaler()
Scaler.fit(x_train)

X_train1 = Scaler.transform(x_train)

#Create the model
ai_brain = Sequential([
    Dense(7,activation='relu'),
    Dense(14,activation='relu'),
    Dense(1)
])

#Compile the model
ai_brain.compile(optimizer = 'rmsprop' , loss = 'mse')

# Fit the model
ai_brain.fit(X_train1 , y_train,epochs = 3000)

loss_df = pd.DataFrame(ai_brain.history.history)

loss_df.plot()

X_test1 =Scaler.transform(x_test)
ai_brain.evaluate(X_test1,y_test)

X_n1=[[11]]
X_n1_1=Scaler.transform(X_n1)
ai_brain.predict(X_n1_1)