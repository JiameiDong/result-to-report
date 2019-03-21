'''import numpy as np
from numpy import *
import pandas as pd
import arch  # 条件异方差模型相关的库
from arch.univariate import arch_model
import pylab as mpl #导入中文字体，避免显示乱码
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
#import tensorflow.contrib.rnn as rnn
#from tensorflow.examples.tutorials.mnist import input_data
#from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat
#from sklearn.preprocessing import MinMaxScaler
#from tensorflow.contrib.layers import fully_connected

#set parameter
TEST_EXAMPLES=264    #测试集个数： TEST_EXAMPLES + n_steps
lr = 0.0001          #learning rate，用于梯度下降
training_iters = 150    #训练的循环次数
n_steps = 252          # time steps
u1 = 10                #第一个 LSTM的hidden unit 
u2 = 4                 #第二个 LSTM的hidden unit 
u3 = 2                 #第三个 LSTM的hidden unit 
batch_size = 1         #每一个batch的长度
pt = n_steps               #Garch 模型 与 volatility 的 rolling 长度
youhuaqi = 4         #优化器：1：mse,2:mae,3:hmse,4:hmae
print(pt)
#get data
dateparse = lambda dates:pd.datetime.strptime(dates,'%Y%m%d')  #读取日期格式

data = pd.read_csv("./CSI300daily.csv",
                    sep=',',
                    encoding = "utf-8",
                    parse_dates=['TDATE'],
                    date_parser=dateparse)    #读取数据

table = pd.pivot_table(data,index=['TDATE'],values=['ENDPRC'])  #日期、价格放同一个表中

##get daily return
price = table.ENDPRC.tolist()   #价格 转换格式

ret = np.zeros((len(price)))    #log return 初始定义

ret_mean = np.zeros((len(price))) #mean of log return 初始定义

sum_ret_vol = np.zeros((len(price))) # sum of distance of mean of log return 初始定义

ret_vol = np.zeros((len(price)))     #distance of mean of log return 初始定义

RV = np.zeros((len(price)))          #volatility 初始定义

#计算volatility
for i in range(1,len(price)):
        ret[i] = (math.log(price[i])-math.log(price[i-1])) * 100
        ipt = i-pt
        if ipt >= 0:    
            ret_mean[i] = average(ret[ipt+1:i+1])  
            for j in range(ipt+1,i+1):
                ret_vol[j] = (ret[j] - ret_mean[i]) ** 2
                sum_ret_vol[i] = ret_vol[j] + sum_ret_vol[i]
            RV[i] = sqrt(sum_ret_vol[i] / pt)

print(RV)

real_vol=RV[n_steps:]
train_vol=real_vol[:-n_steps]
test_vol=real_vol[-n_steps:]
yhat=[]

print("real_vol",real_vol)
print("real_vol",len(real_vol))

model = arch_model(train_vol,
                        mean = 'Constant',
                        vol = 'Garch', 
                        p = 1, o = 0, q = 1,
                        dist = 'Normal')  ##Garch(1,1)
model_fit = model.fit()
yhat=model_fit.forecast(horizon=pt)
#yhat=np.append(yhat,yhat1.variance.values[-1,:])
    #pandas.ewma(t)



# the forecast value in position [t, h] is the time-t, h+1 step ahead forecast.
yhat1=yhat.variance.values[-1,:]
print("yhat",yhat1)
mse = mean_squared_error(yhat1,test_vol)        
mae = mean_absolute_error(y_pred=yhat1,y_true=test_vol)    
one = np.ones(shape=(len(yhat1), 1))
ratio = yhat1 / test_vol
hmse=  mean_squared_error(one,ratio)     
hmae = mean_absolute_error(y_pred=one,y_true=ratio)   
print ('mae:',mae,'   mse:',mse)
print ('hmae:',hmae,'   hmse:',hmse)


plt.plot(test_vol,label='actual volitility')
plt.plot(yhat.variance.values[-1,:],label='predict')
plt.show()'''
#-------------------------------------------------------------------------------------------------------------------------
import numpy as np
from numpy import *
import pandas as pd
import arch  # 条件异方差模型相关的库
from arch.univariate import arch_model
import pylab as mpl #导入中文字体，避免显示乱码
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
#import tensorflow.contrib.rnn as rnn
#from tensorflow.examples.tutorials.mnist import input_data
#from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat
#from sklearn.preprocessing import MinMaxScaler
#from tensorflow.contrib.layers import fully_connected

#set parameter
TEST_EXAMPLES=264    #测试集个数： TEST_EXAMPLES + n_steps
lr = 0.0001          #learning rate，用于梯度下降
training_iters = 150    #训练的循环次数
n_steps = 252          # time steps
u1 = 10                #第一个 LSTM的hidden unit 
u2 = 4                 #第二个 LSTM的hidden unit 
u3 = 2                 #第三个 LSTM的hidden unit 
batch_size = 1         #每一个batch的长度
pt = n_steps               #Garch 模型 与 volatility 的 rolling 长度
youhuaqi = 4         #优化器：1：mse,2:mae,3:hmse,4:hmae
print(pt)
#get data
dateparse = lambda dates:pd.datetime.strptime(dates,'%Y%m%d')  #读取日期格式

data = pd.read_csv("./CSI300daily.csv",
                    sep=',',
                    encoding = "utf-8",
                    parse_dates=['TDATE'],
                    date_parser=dateparse)    #读取数据

table = pd.pivot_table(data,index=['TDATE'],values=['ENDPRC'])  #日期、价格放同一个表中

##get daily return
price = table.ENDPRC.tolist()   #价格 转换格式

ret = np.zeros((len(price)))    #log return 初始定义

ret_mean = np.zeros((len(price))) #mean of log return 初始定义

sum_ret_vol = np.zeros((len(price))) # sum of distance of mean of log return 初始定义

ret_vol = np.zeros((len(price)))     #distance of mean of log return 初始定义

RV = np.zeros((len(price)))          #volatility 初始定义

#计算volatility
for i in range(1,len(price)):
        ret[i] = (math.log(price[i])-math.log(price[i-1])) * 100
        ipt = i-pt
        if ipt >= 0:    
            ret_mean[i] = average(ret[ipt+1:i+1])  
            for j in range(ipt+1,i+1):
                ret_vol[j] = (ret[j] - ret_mean[i]) ** 2
                sum_ret_vol[i] = ret_vol[j] + sum_ret_vol[i]
            RV[i] = sqrt(sum_ret_vol[i] / pt)

print(RV)

real_vol=RV[n_steps:]
train_vol=real_vol[:-n_steps]
test_vol=real_vol[-n_steps:]
yhat1=[]

pre=np.zeros(n_steps)
print(real_vol)



#整合数据
rawdata = pd.DataFrame({
                    'vol':RV[n_steps:]}
                    )
print(rawdata)
values = rawdata.values

for j in range(n_steps):
    t = ret[(-2*n_steps+j):(-1*n_steps+j)]

    model = arch_model(     t,
                            mean = 'Constant',
                            vol = 'EGARCH', 
                            p = 1, o = 0, q = 1,
                            dist = 'Normal')  ##Garch(1,1)

    model_fit = model.fit()
    yhat = model_fit.forecast(horizon=1)
    yhat1 =np.append(yhat1, yhat.variance.values[-1,:])




# the forecast value in position [t, h] is the time-t, h+1 step ahead forecast.

print("yhat",yhat1)
mse = mean_squared_error(yhat1,test_vol)        
mae = mean_absolute_error(y_pred=yhat1,y_true=test_vol)    
one = np.ones(shape=(len(yhat1), 1))
ratio = yhat1 / test_vol
hmse=  mean_squared_error(one,ratio)     
hmae = mean_absolute_error(y_pred=one,y_true=ratio)   
print ('mae:',mae,'   mse:',mse)
print ('hmae:',hmae,'   hmse:',hmse)


plt.plot(test_vol,label='actual volitility')
plt.plot(yhat1,label='predict')
plt.show()