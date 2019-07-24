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
n_steps =126       # time steps
u1 = 10                #第一个 LSTM的hidden unit 
u2 = 4                 #第二个 LSTM的hidden unit 
u3 = 2                 #第三个 LSTM的hidden unit 
batch_size = 1         #每一个batch的长度
pt = 252               #Garch 模型 与 volatility 的 rolling 长度
youhuaqi = 4         #优化器：1：mse,2:mae,3:hmse,4:hmae
print(pt)
#get data
dateparse = lambda dates:pd.datetime.strptime(dates,'%Y%m%d')  #读取日期格式

data = pd.read_csv("D:/RA/result for teacher/CSI300daily.csv",
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
table['ret']=ret
table['vol']=table['ret'].rolling(pt).std()
print(table)

'''yhat1=[0]*pt
for j in range(len(price)-pt):
    t = ret[(j):(pt+j)]

    model = arch_model(     t,
                            mean = 'Constant',
                            vol = 'EGARCH', 
                            p = 1, o = 0, q = 1,
                            dist = 'Normal')  ##Garch(1,1)

    model_fit = model.fit()
    yhat = model_fit.forecast(horizon=1)
    yhat1 =np.append(yhat1, np.sqrt(yhat.variance.values[-1,:]))

present = pd.DataFrame({'vol':table['vol'][pt:],
                        'vol_pre':yhat1[pt:]
                        },
                        index= table.index[pt:])

present = present[present.vol_pre<7]
present = present[present.vol_pre>0]
'''
present = pd.DataFrame({'vol':table['vol']
                        },
                        index= table.index)
#present.ewm(span=pt).mean().plot(style='k')
table['yhat1']=present.ewm(span=pt).mean()

#plt.plot(present['vol'],label='actual volitility')
#plt.plot(present['vol_pre'][-TEST_EXAMPLES:],'r.',label='predict')

present= pd.DataFrame({'yhat1':table['yhat1'][-TEST_EXAMPLES:],
                        'test_vol': table['vol'][-TEST_EXAMPLES:]},
                        index=table.index[-TEST_EXAMPLES:])
present =present[present.yhat1>0]

yhat1 = present['yhat1']
test_vol = present['test_vol']
print(yhat1)
print(test_vol)
mse = mean_squared_error(yhat1,test_vol)        
mae = mean_absolute_error(y_pred=yhat1,y_true=test_vol)    
one = np.ones(shape=(len(yhat1), 1))
ratio = yhat1 / test_vol
hmse=  mean_squared_error(one,ratio)     
hmae = mean_absolute_error(y_pred=one,y_true=ratio)   

print ('mae:',mae,'   mse:',mse)
print ('hmae:',hmae,'   hmse:',hmse)
plt.plot(table['vol'][pt:],label='actual volitility')
plt.plot(present['test_vol'],'r.',label='predict')

plt.show()
