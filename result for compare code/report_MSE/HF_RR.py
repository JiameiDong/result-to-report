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
n_steps = 126         # time steps
u1 = 10                #第一个 LSTM的hidden unit 
u2 = 4                 #第二个 LSTM的hidden unit 
u3 = 2                 #第三个 LSTM的hidden unit 
batch_size = 1         #每一个batch的长度
pt = 126               #Garch 模型 与 volatility 的 rolling 长度
youhuaqi = 1         #优化器：1：mse,2:mae,3:hmse,4:hmae

#get data
dateparse = lambda dates:pd.datetime.strptime(dates,'%Y%m%d')  #读取日期格式

data = pd.read_csv("D:/RA/result for teacher/CSI300_daily.csv",
                    sep=',',
                    encoding = "utf-8",
                    parse_dates=['TDATE'],
                    date_parser=dateparse)    #读取数据

table = pd.pivot_table(data,index=['TDATE'],values=['ENDPRC'])  #日期、价格放同一个表中


data1 = pd.read_csv("D:/RA/result for teacher/CSI300_HF_RR_forcs_vol.csv",
                    sep=',',
                    encoding = "ASCII")    #读取数据
data1 = data1['fit2forecast'].tolist()

real_tail = pd.DataFrame({'ENDPRC':table['ENDPRC'][-TEST_EXAMPLES:],
                          'RR_forecs': data1},
                        index=table.index[-TEST_EXAMPLES:])

'''
print(real_tail)
              ENDPRC  RR_forecs
TDATE
2017-09-01  3367.119   0.668256
2017-09-04  3379.583   0.660752
2017-09-05  3384.317   0.631868
'''

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
table['ret']=ret
table['vol']=table['ret'].rolling(pt).std()


yhat1=[0]*pt
for j in range(len(price)-pt):
    t = ret[(j):(pt+j)]

    model = arch_model(     t,
                            mean = 'Constant',
                            vol = 'GARCH', 
                            p = 1, o = 0, q = 1,
                            dist = 'Normal')  ##Garch(1,1)

    model_fit = model.fit()
    yhat = model_fit.forecast(horizon=1)
    yhat1 =np.append(yhat1, np.sqrt(yhat.variance.values[-1,:]))

table['vol_pre']=yhat1
print(table)

yhat1 = table['vol'][-TEST_EXAMPLES:]
test_vol = real_tail['RR_forecs'][-TEST_EXAMPLES:]

mse = mean_squared_error(yhat1,test_vol)        
mae = mean_absolute_error(y_pred=yhat1,y_true=test_vol)    
one = np.ones(shape=(len(yhat1), 1))
ratio = yhat1 / test_vol
hmse=  mean_squared_error(one,ratio)     
hmae = mean_absolute_error(y_pred=one,y_true=ratio)   

print ('mae:',mae,'   mse:',mse)
print ('hmae:',hmae,'   hmse:',hmse)



vol = pd.DataFrame({
                    'vol':table['vol'][pt:]},
                    index=table.index[pt:]
                    )
vol_pre=pd.DataFrame({
                    'vol':real_tail['RR_forecs'][-TEST_EXAMPLES:]},
                    index=table.index[-TEST_EXAMPLES:]
                    )
#print("yhat",yhat1)
plt.plot(vol,label='actual volitility')
plt.plot(vol_pre,label='predict')
plt.legend(['actual volitility','predict'])
plt.show()