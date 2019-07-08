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





#_____________________________add LSTM____________________
#整合数据
rawdata = pd.DataFrame({
                    'log_return':table_daily['ret_daily'][pt:],
                    'predict':RV['vol_pre'],
                    'vol':RV['RV']},
                    index=table_daily.index[pt:]
                    )



rawdata.to_csv('CSI300daily_testqqqqqqqqqqqqqq.csv', index=True, header=True )  #保存初始LSTM 的 input数据

'''rawdata = pd.read_csv("./CSI300daily_testqqqqqqqqqqqqqq.csv",
                    sep=',',
                    encoding = "utf-8",
                    parse_dates=['TDATE'],
                    date_parser=dateparse)    #读取数据

input_data= pd.pivot_table(rawdata,index=['TDATE'],values=['log_return','omega','alpha','beta','vol'])  #日期、价格放同一个表中
'''
#只取数据，去标题和日期，进行计算
values = rawdata.values
n_inputs = len(rawdata.columns)  #计算列数

#拆分训练集、训练集
train = values[:-TEST_EXAMPLES, :]
test = values[-TEST_EXAMPLES-n_steps:, :]

# 拆分输入、输出
#二维
train_X, train_y = train[:-1,:], train[:, -1]
test_X, test_y = test[:-1,:], test[:, -1]

#初始定义变量 train_X,test_X,test_yand train_y
t ,m = [],[]

#计算train_X
for i in range(len(train_y)-n_steps):
    #m.append(y_train[j+21][:])
    for j in range(i,i+n_steps):
        t.append(train_X[j])

#计算test_X
for i in range(len(test_y)-n_steps):
    #m.append(y_train[j+21][:])
    for j in range(i,i+n_steps):        
        m.append(test_X[j])

#转换形状
#test_X=scaler_X.fit_transform(m)
test_X = np.reshape(m, newshape=[-1, n_steps, n_inputs])

#train_X=scaler_X.fit_transform(t)
train_X = np.reshape(t,newshape=[-1, n_steps, n_inputs])

train_y = np.reshape(train_y[n_steps:],[-1,1])
real_train_y = train_y
#train_y=scaler_y.fit_transform(train_y)

test_y = np.reshape(test_y[n_steps:],[-1,1])
real_test_y = test_y

#test_y=scaler_y.fit_transform(test_y)

#_________________________________________________________
 #||```|n_steps |y|         ```        ||
 #||   ```|n_steps|y|       ```        ||
 #     test:                   |||n_steps|y| ```               ||


def RNN(X):  # X 即 X_p

    #------------------------------------construct LSTM------------------------------------------#
    # cell,activation默认为tanh
        cell1 = tf.nn.rnn_cell.LSTMCell(u1,forget_bias=0.3,state_is_tuple=True)  
        cell2 = tf.nn.rnn_cell.LSTMCell(u2,forget_bias=0.8,state_is_tuple=True) 
        cell3 = tf.nn.rnn_cell.LSTMCell(u3,forget_bias=0.8,state_is_tuple=True) 
        multi_lstm = tf.contrib.rnn.MultiRNNCell(cells = [cell1
                                                        ,cell2
                                                        ,cell3
                                                        ]
                                                ,state_is_tuple=True) 
        # lstm cell is divided into two parts (c_state, h_state)

    #初始定义lstm输入
        init_state = multi_lstm.zero_state(batch_size, dtype=tf.float32)

        # time_major:The shape format of the inputs and outputs
        #time_major=False含义:[batch_size, max_time, cell.output_size]

        outputs, final_state = tf.nn.dynamic_rnn(multi_lstm, 
                                                    X, 
                                                    initial_state = init_state, 
                                                    time_major = False)   

        #则outputs:(batch_size,22,2)    finial_state:len=3,  每层： ('c', 'h')
        h = final_state[-1][1] #-1:（最后一层）第三层的输出，（c为0 ，h为1)  h:shape(1,2)

        #fully connected 1层 output 为5
        FC1 = tf.contrib.layers.fully_connected(h, 5, activation_fn=tf.nn.relu)# 以c为输入，输出5个数据 FC1:shape(1,5)

        #fully connected 2,得出output 1
        results = tf.contrib.layers.fully_connected(FC1, 1, activation_fn=tf.nn.relu)    # 以FC1的5个数据为输入，输出1个数据  shape:(1,1)   

       # results= tf.contrib.layers.fully_connected(h, 1, activation_fn=tf.nn.relu) 
        return results

#设置place holder
X_p = tf.placeholder(dtype = tf.float32,shape = (None, n_steps,n_inputs),name = "input_placeholder") #(?,5，22)
y_p = tf.placeholder(dtype = tf.float32,shape = (None,1),name="pred_placeholder")

#预测结果
pred = RNN(X_p)
#loss function

one = np.ones((pred.shape[0],1))
ratio = pred/y_p

#优化器
if   youhuaqi == 1:
    mse = tf.losses.mean_squared_error(labels = y_p,predictions = pred)   #mse
elif youhuaqi == 2:
    mse = tf.losses.absolute_difference(labels = y_p,predictions = pred)   #mae
elif youhuaqi == 3:
    mse = tf.losses.mean_squared_error(labels = one,predictions = ratio)  #hmse
else:
    mse = tf.losses.absolute_difference(labels = one,predictions = ratio) #hmae
     


#优化
optimizer = tf.train.AdamOptimizer(lr).minimize(loss=mse)

batch_train = train_y.shape[0] // batch_size
batch_test = test_y.shape[0] // batch_size

#-------------------------------------------Define Session---------------------------------------#
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    a_train_loss = np.zeros(shape=(training_iters,1))

    for epoch in range(1,training_iters):

        #初始定义预测结果
        test_results = np.zeros(shape=(test_y.shape[0], 1))
        train_losses = []
        test_losses = []
        
        print("epoch:",epoch) #训练次数
        #----训练------------------------------
       
        #取训练数据，整理shape

        #训练开始
        for k in range(batch_train):
            _,train_loss =sess.run(
                    fetches = (optimizer,mse),
                    feed_dict = { 
                            X_p:train_X[k*batch_size:(k+1)*batch_size],
                            y_p: train_y[k*batch_size:(k+1)*batch_size]
                            }
                )
            train_losses.append(train_loss) #训练时的损失
        print("average training loss:", sum(train_losses) / len(train_losses))
        a_train_loss1 = sum(train_losses) / len(train_losses)
        a_train_loss[epoch] = a_train_loss1

        #----测试-------------------------------

        #进行测试
    for k in range(batch_test):
            result,test_loss = sess.run(
                     fetches = (pred,mse),
                    feed_dict={
                                X_p:test_X[k*batch_size:(k+1)*batch_size]
                                ,y_p:test_y[k*batch_size:(k+1)*batch_size]
                            }
                )
            test_results[k*batch_size:(k+1)*batch_size]=result
            test_losses.append(test_loss)

#test_predict = scaler_y.inverse_transform(test_results)        
#test_y = scaler_y.inverse_transform(test_y)        
#train_y = scaler_y.inverse_transform(train_y)  
test_predict = test_results[:batch_size*batch_test]
real_test_y = real_test_y[:batch_size*batch_test]
mse = mean_squared_error(test_predict,real_test_y)        
mae = mean_absolute_error(y_pred=test_predict,y_true=real_test_y)    
one = np.ones(shape=(len(test_predict), 1))
ratio = test_predict / real_test_y
hmse=  mean_squared_error(one,ratio)     
hmae = mean_absolute_error(y_pred=one,y_true=ratio)   
print ('mae:',mae,'   mse:',mse)
print ('hmae:',hmae,'   hmse:',hmse)
print('Garch(1,1)')
print('youhuaqi:',youhuaqi)

plt.figure(figsize = (18,9))
plt.plot(rawdata.index[-TEST_EXAMPLES-len(real_train_y):-TEST_EXAMPLES],real_train_y,color='g',label='True')
if -TEST_EXAMPLES+batch_size*batch_test<0:

    plt.plot(rawdata.index[-TEST_EXAMPLES:-TEST_EXAMPLES+batch_size*batch_test],real_test_y,color='b',label='True')
    plt.plot(rawdata.index[-TEST_EXAMPLES:-TEST_EXAMPLES+batch_size*batch_test],test_predict,color='orange', label='Prediction')
else:
    plt.plot(rawdata.index[-TEST_EXAMPLES:],real_test_y,color='b',label='True')
    plt.plot(rawdata.index[-TEST_EXAMPLES:],test_predict,color='orange', label='Prediction')




plt.xlabel('Date')
plt.ylabel('Vol')
plt.legend(fontsize=18)
plt.show()

plt.plot(price_daily)
plt.xlabel('Date')
plt.ylabel('price')
plt.legend(fontsize=18)
plt.show()

plt.figure()
plt.plot(range(1,len(a_train_loss)),a_train_loss[1:],color='b',label='Average Loss')
plt.xlabel('Times')
plt.ylabel('Loss')
plt.legend(fontsize=18)
plt.show()