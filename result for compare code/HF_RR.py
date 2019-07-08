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
n_steps =126          # time steps
u1 = 10                #第一个 LSTM的hidden unit 
u2 = 4                 #第二个 LSTM的hidden unit 
u3 = 2                 #第三个 LSTM的hidden unit 
batch_size = 1         #每一个batch的长度
pt = 126               #Garch 模型 与 volatility 的 rolling 长度
youhuaqi = 1         #优化器：1：mse,2:mae,3:hmse,4:hmae
print(pt)
#get HF data
dateparse = lambda dates:pd.datetime.strptime(dates,'%Y%m%d %H%M')  #读取日期格式
data_HF = pd.read_csv("D:/RA/result for teacher\SSE_15min.csv",
                    sep=',',
                    encoding = "utf-8",
                    parse_dates={'datetime':['TDATE1','MINTIME']},
                    date_parser=dateparse)    #读取数据

table_HF = pd.pivot_table(data_HF,index=['datetime'],values=['ENDPRC','HIGHPRC','LOWPRC'])  #日期、价格放同一个表中

##get daily return
price_HF = table_HF.ENDPRC.tolist()   #价格 转换格式
high_price_HF = table_HF.HIGHPRC.tolist()  
low_price_HF = table_HF.LOWPRC.tolist()  
ret_HF = np.zeros((len(price_HF)))    #log return 初始定义
RV_HF = np.zeros((len(price_HF)))   

#计算volatility
for i in range(1,len(price_HF)):
        ret_HF[i] = (math.log(price_HF[i])-math.log(price_HF[i-1])) * 100
        RV_HF[i] =  ( (math.log(high_price_HF[i])-math.log(low_price_HF[i])) **2 )
table_HF['ret_HF']=ret_HF
table_HF['r2'] = RV_HF



#get daily data
dateparse = lambda dates:pd.datetime.strptime(dates,'%Y%m%d')  #读取日期格式

data_daily = pd.read_csv("D:/RA/result for teacher/SSE_daily.csv",
                    sep=',',
                    encoding = "utf-8",
                    parse_dates=['TDATE'],
                    date_parser=dateparse)    #读取数据

table_daily = pd.pivot_table(data_daily,index=['TDATE'],values=['ENDPRC','HIGHPRC','LOWPRC'])  #日期、价格放同一个表中

##get daily return
price_daily = table_daily.ENDPRC.tolist()   #价格 转换格式
high_price_daily = table_daily.HIGHPRC.tolist()  
low_price_daily = table_daily.LOWPRC.tolist() 
ret_daily = np.zeros((len(price_daily)))    #log return 初始定义

#计算daily return
for i in range(1,len(price_daily)):
        ret_daily[i] = 100 * (math.log(high_price_daily[i])-math.log(low_price_daily[i])) 
table_daily['ret_daily']=ret_daily
table_daily['vol']=table_daily['ret_daily'].rolling(pt).std()

print('table_daily',table_daily)

'''
table_HF                        
datetime             ENDPRC    ret_HF        r2
2012-09-03 09:30:00  2044.825  0.000000  0.000000e+00
2012-09-03 09:45:00  2048.270  0.168332  2.833577e-02


table_daily               
TDATE         ENDPRC  ret_daily
2012-09-03  2059.147   0.000000
2012-09-04  2043.649  -0.755488
2012-09-05  2037.681  -0.292454'''


'''
HF 每17天，计算一个RV
daily ret 计算vol
对比mse,mae,etc
'''
#计算RV,一天17个数


r2= table_HF['r2'].tolist()
r2 = np.reshape(r2,newshape=[-1,17])
rv = r2.sum(axis=1)
rv = 0.25 * math.log(2) * rv
print('RV',rv)
print(len(rv))

RV  = pd.DataFrame({
                    'RV':rv[pt:],
                    },
                    index=table_daily.index[pt:]
                    )        

print(RV)    #1353行


#计算Garch

yhat1=[0]*pt
for j in range(len(price_daily)-pt):
    t = ret_daily[(j):(pt+j)]

    model = arch_model(     t,
                            mean = 'Constant',
                            vol = 'GARCH', 
                            p = 1, o = 1, q = 1,
                            dist = 'Normal')  ##Garch(1,1)

    model_fit = model.fit()
    yhat = model_fit.forecast(horizon=1)
    yhat1 =np.append(yhat1, np.sqrt(yhat.variance.values[-1,:]))

RV['vol_pre']=yhat1[pt:]
#RV['vol_pre']=table_daily['vol'][pt:]
print(RV)


yhat1 = RV['RV'][-TEST_EXAMPLES:]
test_vol = RV['vol_pre'][-TEST_EXAMPLES:]

mse = mean_squared_error(yhat1,test_vol)        
mae = mean_absolute_error(y_pred=yhat1,y_true=test_vol)    
one = np.ones(shape=(len(yhat1), 1))
ratio = yhat1 / test_vol
hmse=  mean_squared_error(one,ratio)     
hmae = mean_absolute_error(y_pred=one,y_true=ratio)   

print ('mae:',mae,'   mse:',mse)
print ('hmae:',hmae,'   hmse:',hmse)



vol = pd.DataFrame({
                    'vol':RV['RV'][pt:]},
                    index=table_daily.index[pt:]
                    )
vol_pre=pd.DataFrame({
                    'vol':RV['vol_pre'][-TEST_EXAMPLES:]},
                    index=table_daily.index[-TEST_EXAMPLES:]
                    )
#print("yhat",yhat1)
plt.plot(vol,label='actual volitility')
plt.plot(vol_pre,label='predict')
plt.legend(['actual volitility','predict'])
plt.show()