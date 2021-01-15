#!/usr/bin/env python
# coding: utf-8

# In[98]:


# 출처: https://machinelearningmastery.com/how-to-get-started-with-deep-learning-for-time-series-forecasting-7-day-mini-course/


# In[141]:


'''
참조자료
Conv1D 설명: https://soyoung-new-challenge.tistory.com/29
mlp cnn 차이: 
https://www.researchgate.net/figure/The-differences-architecture-between-a-simple-MLP-and-a-CNN_fig1_330510477
https://towardsdatascience.com/simple-introduction-to-convolutional-neural-networks-cdf8d3077bac
'''


# In[1]:


import warnings 
warnings.filterwarnings('ignore')


# In[2]:


import pandas as pd
import numpy as np


# # 전체 설명

# In[4]:


'''
긴 기간, 노이즈 데이터, multi-step 예측, 다중 입/출력 일 때, 시계열 예측은 어려움
딥러닝 방법은 시계열예측을 위해 다양한 promize를 제공함
일시적인 의존의 automatic learning과 트렌드와 시즈널과 같은 일시적인 구조를 자동으로 핸들링함

- 시계열 예측의 기본
- 딥러닝을 위한 기본 파이썬, 넘파이, 케라스 사용 방법

머신러닝 개발자  >  시계열 기반의 예측 프로젝트를 위한 딥러닝 방법을 배울 수 있음

열정과 시간에 따라 7개 강의를 7일 동안 하거나, 7개 강의를 하루에 다 끝낼 수 있음
'''



# ## 01. Promise of Deep Learning

# In[5]:


'''
1. Robust to Noise: input 데이터의 noise에 강건함
2. Nonlinear: 비선형
3. Multivariate Inputs: 다중 입력
4. Multi-step Forcasts
'''


# ## 02: How to Transform Data for Time Series

# In[6]:


'''
1) Series:
1, 2, 3, 4, 5, ...

2) Sliding window transformation:
X,				y
[1, 2, 3]		4
[2, 3, 4]		5

이전 관측값이 모델의 다음을 예측하기 위한 input으로 사용됨
이 케이스의 윈도우 너비는 3 time step임


'''


# In[7]:


df = pd.read_csv('/home/workspace/study/modeling/data/daily-total-female-births.csv')


# In[8]:


df.info()


# In[9]:


print(df.head(4))
print(df.tail(4))


# In[10]:


inPut = []
outPut = []

for i in range(len(df)-3):
    inPut.append(df['Births'][i:i+3])
    outPut.append(df['Births'][i+3])


# In[11]:


X = pd.DataFrame(np.array(inPut), columns =['x1','x2','x3'])
y = pd.DataFrame(np.array(outPut), columns = ['y'])


# In[12]:


data = pd.concat([X, y], axis=1)
data.tail(3)



# ## 03: MLP for Time Series Forecasting

# In[13]:


'''
1. 우리는 모델을 정의 할 수 있음
time_steps의 수(3)는 input_dim으로 첫번째 hidden layer로 정의
확률론적 경사하강인 효율적인 Adam 버전을 사용하고, MSE를 이용해 손실을 줄어 최적화함

'''


# In[14]:


import numpy as np
from keras.models import Sequential
from keras.layers import Dense


# In[15]:


X = np.array([[10, 20 , 30], [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y = np.array([40, 50, 60, 70])

print(X.shape)
print(y.shape)


# In[16]:


y


# In[17]:


model = Sequential()
model.add(Dense(100, activation = 'relu', input_dim = 3))
model.add(Dense(1))
model.compile(optimizer = 'adam', loss ='mse')

model.fit(X, y, epochs = 2000, verbose = 0)


# In[18]:


x_input = np.array([50, 60, 70])
x_input = x_input.reshape(1, 3)


# In[19]:


x_input


# In[20]:


yhat = model.predict(x_input, verbose=0)
print(yhat)




# --------------------------------------------------------------------------

# #### 01. 데이터 불러오기 & time_steps = 3 reshape

# In[21]:


df = pd.read_csv('/home/workspace/study/modeling/data/daily-total-female-births.csv')


# In[22]:


df.info()


# In[23]:


print(df.head(4))
print(df.tail(4))


# In[24]:


inPut = []
outPut = []

for i in range(len(df)-3):
    inPut.append(df['Births'][i:i+3])
    outPut.append(df['Births'][i+3])


# In[25]:


Xd = pd.DataFrame(np.array(inPut), columns =['x1','x2','x3'])
yd = pd.DataFrame(np.array(outPut), columns = ['y'])


# In[26]:


data = pd.concat([Xd, yd], axis=1)
data.tail(3)


# #### 02. 학습 데이터, 정답 추출

# In[27]:


data2 = data[:361]
data2.tail(3)


# In[28]:


x_train = np.array(data2[['x1', 'x2', 'x3']])
y_train = np.array(data2[['y']])


# In[29]:


print(x_train.shape)
print(y_train.shape)


# In[30]:


y_train= y_train.reshape(361,)


# In[31]:


print(x_train.shape)
print(y_train.shape)


# #### 03. 학습

# In[32]:


model = Sequential()
model.add(Dense(100, activation='relu', input_dim = 3))
model.add(Dense(1))
model.compile(optimizer = 'adam', loss='mse')

model.fit(x_train, y_train, epochs = 100, verbose = 0)


# #### 04. 마지막 값 예측 및 정답 확인

# In[33]:


X_input = np.array(data[['x1','x2','x3']].iloc[-1])
print(X_input.shape)

X_input = X_input.reshape((1, 3))
print(X_input.shape)


# In[34]:


X_input


# In[35]:


y_hat = model.predict(X_input, verbose=0)
print(y_hat)


# In[36]:


y_true = data['y'].iloc[-1]
print(y_true)


# In[37]:


print(y_hat - y_true)


# --------------------------------------------------------------------------------------------




# ## 04: CNN for Time Series Forecasting

# In[38]:


'''
일변량 시계열 예측에서 Convolutional Neural Network model 또는 CNN을 어떻게 발전시킬지 발견할 수 있음
integers의 우리는 간단한 일변량 문제를 시퀀스로 정의할 수 있음
3inputs & 1output으로 정의함. 예를 들어 [10, 20, 30] => [40]
MLP모델로부터 주요 차이점은 CNN 모델은 3차원 input을 기대함(samples, timesteps, features)
우리는 [samples, timesteps]로 정의하고 이에 따라 reshape함

첫번째 히든 레이어 input shape인수를 이용해 time_steps를3, 피쳐의 수는 1로 정의함
하나의 convolutional hidden layer와 max pooling layer 사용
Dense 레이어에서 해석되고 예측을 출력하기 전에 flatten됨
경사하강은 Adam, 손실최적화에는 mse 사용
'''


# In[39]:


import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


# In[40]:


X = np.array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y = np.array([40, 50, 60, 70])


# In[41]:


X.shape


# In[42]:


# reshape from [samples, timesteps] > [sampels, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))
X.shape


# In[43]:


model = Sequential()
model.add(Conv1D(filters = 64, kernel_size=2, activation = 'relu', input_shape = (3, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation = 'relu'))
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mse')


# In[44]:


model.fit(X, y, epochs = 1000, verbose = 0)


# In[45]:


x_input = np.array([50, 60, 70])
print(x_input.shape)
print(x_input)


# In[46]:


x_input = x_input.reshape((1, 3, 1))
print(x_input.shape)
print(x_input)


# In[47]:


y_hat = model.predict(x_input, verbose=0)
print(yhat)




# --------------------------------------------------------------------------

# #### 01. 데이터 불러오기 & time_steps = 3 reshape

# In[73]:


df = pd.read_csv('/home/workspace/study/modeling/data/daily-total-female-births.csv')


# In[74]:


df.info()


# In[75]:


print(df.head(4))
print(df.tail(4))


# In[76]:


inPut = []
outPut = []

for i in range(len(df)-3):
    inPut.append(df['Births'][i:i+3])
    outPut.append(df['Births'][i+3])


# In[77]:


Xd = pd.DataFrame(np.array(inPut), columns =['x1','x2','x3'])
yd = pd.DataFrame(np.array(outPut), columns = ['y'])


# In[78]:


data = pd.concat([Xd, yd], axis=1)
data.tail(3)


# #### 02. 학습 데이터, 정답 추출

# In[79]:


data2 = data[:361]
data2.tail(3)


# In[80]:


x_train = np.array(data2[['x1', 'x2', 'x3']])
y_train = np.array(data2[['y']])


# In[81]:


print(x_train.shape)
print(y_train.shape)


# In[82]:


y_train= y_train.reshape(361,)


# In[83]:


print(x_train.shape)
print(y_train.shape)


# ##### 추가부분 [samles, timesteps, features]

# In[84]:


x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_train.shape


# #### 03. 학습

# In[92]:


model = Sequential()
model.add(Conv1D(filters = 64, kernel_size = 2, activation = 'relu', input_shape=(3, 1)))
model.add(MaxPooling1D(pool_size = 2))
model.add(Flatten())          
model.add(Dense(50, activation = 'relu'))
model.add(Dense(1))
model.compile(optimizer = 'adam', loss='mse')

model.fit(x_train, y_train, epochs = 100, verbose = 0)


# #### 04. 마지막 값 예측 및 정답 확인

# In[93]:


X_input = np.array(data[['x1','x2','x3']].iloc[-1])
print(X_input.shape)

X_input = X_input.reshape((1, 3, 1))
print(X_input.shape)


# In[94]:


X_input


# In[95]:


y_hat = model.predict(X_input, verbose=0)
print(y_hat)


# In[96]:


y_true = data['y'].iloc[-1]
print(y_true)


# In[97]:


print(y_hat - y_true)


# --------------------------------------------------------------------------------------------



# ## 05. LSTM for Time Series Forecasting

# In[108]:


'''
3차원 [sampes, timesteps, feautures]
최적화 Adam, 손실 mse
'''


# In[100]:


from keras.layers import LSTM


# In[102]:


X = np.array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y = np.array([40, 50, 60, 70])


# In[103]:


X.shape


# In[104]:


# [samples, timesteps] > [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))


# In[105]:


model = Sequential()
model.add(LSTM(50, activation = 'relu', input_shape = (3, 1)))
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mse')
model.fit(X, y, epochs = 1000, verbose=0)


# In[107]:


x_input = np.array([50, 60, 70])
x_input = x_input.reshape((1, 3, 1))
yhat = model.predict(x_input, verbose=0)
print(yhat)





# --------------------------------------------------------------------------

# #### 01. 데이터 불러오기 & time_steps = 3 reshape

# In[109]:


df = pd.read_csv('/home/workspace/study/modeling/data/daily-total-female-births.csv')


# In[110]:


df.info()


# In[111]:


print(df.head(4))
print(df.tail(4))


# In[112]:


inPut = []
outPut = []

for i in range(len(df)-3):
    inPut.append(df['Births'][i:i+3])
    outPut.append(df['Births'][i+3])


# In[113]:


Xd = pd.DataFrame(np.array(inPut), columns =['x1','x2','x3'])
yd = pd.DataFrame(np.array(outPut), columns = ['y'])


# In[114]:


data = pd.concat([Xd, yd], axis=1)
data.tail(3)


# #### 02. 학습 데이터, 정답 추출

# In[115]:


data2 = data[:361]
data2.tail(3)


# In[116]:


x_train = np.array(data2[['x1', 'x2', 'x3']])
y_train = np.array(data2[['y']])


# In[117]:


print(x_train.shape)
print(y_train.shape)


# In[118]:


y_train= y_train.reshape(361,)


# In[119]:


print(x_train.shape)
print(y_train.shape)


# ##### 추가부분 [samles, timesteps, features]

# In[120]:


x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_train.shape


# #### 03. 학습

# In[135]:


model = Sequential()
model.add(LSTM(50, activation = 'relu', input_shape = (3, 1)))
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mse')
model.fit(x_train, y_train, epochs = 100, verbose = 0)


# #### 04. 마지막 값 예측 및 정답 확인

# In[136]:


X_input = np.array(data[['x1','x2','x3']].iloc[-1])
print(X_input.shape)

X_input = X_input.reshape((1, 3, 1))
print(X_input.shape)


# In[137]:


X_input


# In[138]:


y_hat = model.predict(X_input, verbose=0)
print(y_hat)


# In[139]:


y_true = data['y'].iloc[-1]
print(y_true)


# In[140]:


print(y_hat - y_true)


# --------------------------------------------------------------------------------------------



# ## 06. CNN-LSTM for Time Series Forecasting

'''
hybrid CNN-LSTM model 단변량 시계열 예측 



'''


# In[148]:


from numpy import array
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


# In[149]:


X = array([[10, 20, 30, 40], [20, 30, 40, 50], [30, 40, 50, 60], [40, 50, 60, 70]])
y = array([50, 60, 70, 80])


# In[150]:


print(X.shape)
print(X.shape[0])


# In[151]:


# [samples, timesteps] > [samples, subsequences, timesteps, features]
X = X.reshape((X.shape[0], 2, 2, 1))


# In[153]:


model = Sequential()
model.add(TimeDistributed(Conv1D(filters = 64, kernel_size = 1, activation = 'relu'), input_shape = (None, 2, 1)))
model.add(TimeDistributed(MaxPooling1D(pool_size = 2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(50, activation = 'relu'))
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mse')


# In[154]:


model.fit(X, y, epochs = 500, verbose = 0)


# In[156]:


x_input = np.array([50, 60, 70, 80])
print(x_input.shape)
print(x_input)


# In[158]:


x_input = x_input.reshape((1, 2, 2, 1))


# In[161]:


yhat = model.predict(x_input, verbose=0)
yhat


# --------------------------------------------------------------------------

# #### 01. 데이터 불러오기 & time_steps = 3 reshape

# In[207]:


df = pd.read_csv('/home/workspace/study/modeling/data/daily-total-female-births.csv')


# In[208]:


df.info()


# In[209]:


print(df.head(4))
print(df.tail(4))


# In[210]:


inPut = []
outPut = []
time_steps = 4

for i in range(len(df)-time_steps):
    inPut.append(df['Births'][i:i+time_steps])
    outPut.append(df['Births'][i+time_steps])


# In[211]:


Xd = pd.DataFrame(np.array(inPut), columns =['x1','x2','x3','x4'])
yd = pd.DataFrame(np.array(outPut), columns = ['y'])


# In[212]:


data = pd.concat([Xd, yd], axis=1)
data.tail(3)


# #### 02. 학습 데이터, 정답 추출

# In[213]:


data2 = data[:-1]
data2.tail(3)


# In[214]:


x_train = np.array(data2[['x1', 'x2', 'x3', 'x4']])
y_train = np.array(data2[['y']])


# In[215]:


print(x_train.shape)
print(y_train.shape)


# In[216]:


y_train= y_train.reshape(360,)


# In[217]:


print(x_train.shape)
print(y_train.shape)


# ##### 추가부분 [samples, subsequences, timesteps, features]

# In[218]:


x_train = x_train.reshape((x_train.shape[0], 2, 2, 1))
x_train.shape


# #### 03. 학습

# In[219]:


model = Sequential()
model.add(TimeDistributed(Conv1D(filters = 64, kernel_size = 1, activation ='relu'), input_shape = (None, 2, 1)))
model.add(TimeDistributed(MaxPooling1D(pool_size = 2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(50, activation = 'relu'))
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mse')


# In[221]:


model.fit(x_train, y_train, epochs =500, verbose = 0)


# #### 04. 마지막 값 예측 및 정답 확인

# In[222]:


X_input = np.array(data[['x1','x2','x3','x4']].iloc[-1])
print(X_input.shape)

X_input = X_input.reshape((1, 2, 2, 1))
print(X_input.shape)


# In[223]:


X_input


# In[224]:


y_hat = model.predict(X_input, verbose=0)
print(y_hat)


# In[225]:


y_true = data['y'].iloc[-1]
print(y_true)


# In[226]:


print(y_hat - y_true)


# ------------------------------------------------------------------------------------------------




'''
CNN LSTM 모델 추가 설명
https://machinelearningmastery.com/cnn-long-short-term-memory-networks/


'''

'''
channel = 1 (black and white)
filter = 2x2
input = 10x10
Maxpooling = 2x2(filter크기로 움직이면서 max값으로 특징 추출, 5x5 이미지로 변경)
예측 출력을 위해 25개 요소로 벡터화 함(일렬로 펼침)
'''
# cnn = Sequential()
# cnn.add((Conv2D(1, (2, 2), activation='relu', padding='same', input_shape=(10,10,1))))
# cnn.add(MaxPooling2D(pool_size=(2,2)))
# cnn.add(Flatten())


'''
cnn은 단일 이미지만 핸들링할 수 있으며, 픽셀 인풋을 internal matrix 또는 벡터로 변환함
cnn을 이용해 다수의 이미지 인풋에 lstm을 적용해서 역전파로부터 학습되는것을 바람
TimeDistributed layer는 동일 레이어를 여러번 적용하여 원하는 결과를 얻음(특징을 추출함)
“This wrapper allows to apply a layer to every temporal slice of an input.”
“TimeDistributedDense applies a same Dense (fully-connected) operation to every timestep of a 3D tensor.” 
'''
# model.add(TimeDistributed(...))
# model.add(LSTM(...))
# model.add(Dense(...))

'''
1. CNN layer을 생성함
2. TimeDistributed layer로 감쌈
3. LSTM 적용
'''
# cnn = Sequential()
# cnn.add(Conv2D(...))
# cnn.add(MaxPooling2D(...))
# cnn.add(Flatten())
# # define LSTM model
# model = Sequential()
# model.add(TimeDistributed(cnn, ...))
# model.add(LSTM(..))
# model.add(Dense(...))

'''
이를 더 쉽게 읽기 위해, CNN 모델을 TimeDistributed layer로 감쌈
'''
# model = Sequential()
# # define CNN model
# model.add(TimeDistributed(Conv2D(...))
# model.add(TimeDistributed(MaxPooling2D(...)))
# model.add(TimeDistributed(Flatten()))
# # define LSTM model
# model.add(LSTM(...))
# model.add(Dense(...))



## 07. Encoder-Decoder LSTM Multi-step Forecasting

# multi-step encoder-decoder lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

# define dataset
X = array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y = array([[40,50], [50,60], [60,70], [70,80]])

print(X.shape, y.shape)

# reshape from [samples, timesteps] into [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))
y = y.reshape((y.shape[0], y.shape[1], 1))

print(X.shape, y.shape)

model = Sequential()
model.add(LSTM(100, activation = 'relu', input_shape = (3, 1)))
model.add(RepeatVector(2))
model.add(LSTM(100, activation = 'relu', return_sequences = True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer = 'adam', loss = 'mse')

model.fit(X, y, epochs = 100, verbose =0)

x_input = np.array([50, 60, 70])
x_input = x_input.reshape((1, 3, 1))
yhat = model.predict(x_input, verbose=0)
print(yhat)


# Encoder-Decoder LSTM
# https://machinelearningmastery.com/encoder-decoder-long-short-term-memory-networks/
'''
Encoder - Decoder LSTM 구조
케라스 딥러닝 라이브러리로 이용할 수 있음
encoder - decoder 2 key parts

encoder는 2차원 matrix를 아웃풋으로 생성함. 이는 레이어의 메모리 셀 수로 정해지는 길이임
decoder는 LSTM 레이어로 3차원으로 문제가 발생함
이를 해결하기 위해 RepeatVector layer를 사용함. 
이 레이어는 간단하게 2D input을 반복하여 3D output을 생성함
TimeDistributed로 감싸면 동일 출력레이어로 재사용가능
'''

# model = Sequential()
# model.add(LSTM(..., input_shape=(...)))
# model.add(RepeatVector(...))
# model.add(LSTM(..., return_sequences=True))
# model.add(TimeDistributed(Dense()))
