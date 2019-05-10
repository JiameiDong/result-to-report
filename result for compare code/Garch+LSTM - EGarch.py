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
n_steps =252       # time steps
u1 = 10                #第一个 LSTM的hidden unit 
u2 = 4                 #第二个 LSTM的hidden unit 
u3 = 2                 #第三个 LSTM的hidden unit 
batch_size = 1         #每一个batch的长度
pt = 252               #Garch 模型 与 volatility 的 rolling 长度
youhuaqi = 2         #优化器：1：mse,2:mae,3:hmse,4:hmae
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

yhat1=[0]*pt
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


#_____________________________add LSTM____________________
#整合数据
rawdata = pd.DataFrame({
                    'log_return':table['vol'][pt:],
                    'predict':yhat1[pt:],
                    'vol':table['vol'][pt:]},
                    index=table.index[pt:]
                    )

rawdata = rawdata[rawdata.predict<7]
rawdata = rawdata[rawdata.predict>0]

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

plt.plot(price)
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
