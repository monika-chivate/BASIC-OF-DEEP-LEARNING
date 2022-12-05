#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


# In[2]:


data=pd.read_csv('Nifty_50.csv')



# In[3]:


data.head(5)


# In[4]:


data.info()



# In[5]:


plt.figure(figsize=(10,7))
plt.plot(data['Date'],data['Price'])
plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
# Define the label for the title of the figure
plt.title("Adjusted Close " , fontsize=16)
# Define the labels for x-axis and y-axis
plt.ylabel('Price', fontsize=14)
plt.xlabel('Year', fontsize=14)
# Show the plot
plt.show()


# In[6]:


plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(),vmin=-1, vmax=1,annot=True)


# In[7]:


sns.jointplot(x='Close_fin', y='Price', data= data, kind='scatter', color='seagreen')


# In[8]:


sns.jointplot(x='Close_IT', y='Price', data= data, kind='scatter', color='seagreen')


# In[9]:


# Set up our figure by naming it returns_fig, call PairPLot on the DataFrame
return_fig = sns.PairGrid(data.dropna())

# Using map_upper we can specify what the upper triangle will look like.
return_fig.map_upper(plt.scatter, color='purple')

# We can also define the lower triangle in the figure, inclufing the plot type (kde) 
# or the color map (BluePurple)
return_fig.map_lower(sns.kdeplot, cmap='cool_d')

# Finally we'll define the diagonal as a series of histogram plots of the daily return
return_fig.map_diag(plt.hist, bins=30)


# In[10]:


data['Date'] =  pd.to_datetime(data['Date'])


# In[11]:


'Date','Price','Close_fin','Close_IT','Close_Energy','Close_FMCG','Close_Auto'


# In[12]:


input_feature = data[['Close_fin','Close_IT','Close_Energy','Close_FMCG','Close_Auto','Price']]
input_data = input_feature.values


# In[ ]:





# In[13]:


scaler = MinMaxScaler(feature_range=(-1,1))
# input_data[:,:] = scaler.fit_transform(input_data[:,:])
input_data = scaler.fit_transform(input_data)



# In[14]:


input_data.shape


# In[15]:


df = pd.DataFrame(input_data)


# In[16]:


df.iloc[120:,-1:]


# In[17]:


lookback = 60
total_size = len(data)


# In[18]:


X=[]
y=[]
for i in range(0, total_size-lookback): # loop data set with margin 50 as we use 50 days data for prediction
    t=[]
    for j in range(0, lookback): # loop for 50 days
        current_index = i+j
        t.append(input_data[current_index, :-1]) # get data margin from 50 days with marging i
    X.append(t)
    y.append(input_data[lookback+i, 5])



# In[19]:


X


# In[20]:


y


# In[21]:


len(X), len(y)



# In[22]:


y


# In[23]:


X, y= np.array(X), np.array(y)
X.shape, y.shape



# In[24]:


test_size = 120 

X_test = X[-test_size:]
Y_test = y[-test_size:]

X_rest = X[: -test_size]
y_rest = y[: -test_size]

X_train, X_valid, y_train, y_valid = train_test_split(X_rest, y_rest, test_size = 0.15, random_state = 101)


# In[25]:


Y_test


# In[26]:


y_train


# In[27]:


X_train = X_train.reshape(X_train.shape[0], lookback, 5)
X_valid = X_valid.reshape(X_valid.shape[0], lookback, 5)
X_test = X_test.reshape(X_test.shape[0], lookback, 5)
print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)



# In[28]:


regressor = Sequential()
#add 1st lstm layer
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 5)))
regressor.add(Dropout(rate = 0.2))


regressor.add(LSTM(units = 35, return_sequences = True))
regressor.add(Dropout(rate = 0.2))


regressor.add(LSTM(units = 25, return_sequences = True))
regressor.add(Dropout(rate = 0.2))


regressor.add(LSTM(units = 15, return_sequences = False))
regressor.add(Dropout(rate = 0.2))

##add output layer
regressor.add(Dense(units = 1))



# In[29]:


callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint('model123.h5', verbose=1, save_best_only=True, save_weights_only=True)
]



# In[30]:


# regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
regressor.compile(optimizer='adam', loss='mean_squared_error')


# In[31]:


regressor.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_valid, y_valid), callbacks=callbacks)



# In[32]:


regressor.load_weights('model1.h5')


# In[33]:


results = regressor.evaluate(X_test, Y_test, batch_size=128)
print("test loss, test acc:", results)


# In[34]:


predicted_value = regressor.predict(X_test)



# In[35]:


Y_test


# In[36]:


T = pd.DataFrame(Y_test,columns = ["True value"])


# In[37]:


Z = pd.DataFrame(predicted_value,columns = ["predicted value"])


# In[38]:


pd.concat([T, Z], axis=1)


# In[39]:


plt.figure(figsize=(18, 8))
plt.plot(predicted_value, color= 'blue')
plt.plot(Y_test, color='green')
plt.title("price of stocks sold")
plt.xlabel("Time (latest ->oldest-> )")
plt.ylabel("Stock Price")
plt.show()


# In[40]:


from sklearn.metrics import mean_squared_error
mean_squared_error(Y_test, predicted_value,squared=False)


# In[41]:


mean_squared_error(Y_test, predicted_value)


# In[42]:


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# In[43]:


r2_score(Y_test, predicted_value)


# In[44]:


mean_absolute_error(Y_test, predicted_value)


# In[ ]:





# In[ ]:





# In[ ]:




