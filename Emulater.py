#!/usr/bin/env python
# coding: utf-8

# # $\text{Imported Libraries}$

# In[15]:


import os
from os import listdir
from os.path import isfile, join
import re
from matplotlib import cm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib.ticker import LinearLocator
import matplotlib as mpl
import matplotlib.gridspec as gridspec

import multiprocessing
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping , CSVLogger
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten,Dropout
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)


from rich import print


# In[16]:


from IPython.display import display, HTML
display(HTML("<style>.container { width:97% !important; }</style>"))


# In[ ]:





# In[5]:


get_ipython().run_cell_magic('html', '', "<style type='text/css'>\n.CodeMirror{\n    font-size: 12px;\n}\n\ndiv.output_area pre {\n    font-size: 14px;\n}\n</style>\n\n!jt -t onedork -fs 20 -altp -tfs 20 -ofs 14 -cellw 10%\n")


# In[6]:


print('[bold red] Sanjay')


# # $\boxed{\textbf{Emulater for Power Spectrum}}$

# ### $\textit{jugaad}$ 

# In[ ]:


# my_path = '/home/sanjay/Msc_Project/sanjay_123/Pk_mK2'
# path = os.listdir(my_path)

# with open(r'list_1', 'r') as fp:
#     # read all lines using readline()
#     lines = fp.readlines()
#     i = 0
#     for row in lines:
#         w = row.split("__")
#         w2 = w[2].rstrip("\n")
#         w1 = float(w[1])

#         i +=1
#         for file in path:
#             if file.endswith(w2):
#                 fs = file.split("_")
#                 strng = '{}_{}_{}_{}_{}_{:.3f}_{}_{}_{}'.format(fs[0],fs[1],fs[2],fs[3],fs[4],w1,fs[5],fs[6],fs[7])
#                 os.rename("Pk_mK2/{}".format(file),"Pk_mK2/{}".format(strng))
#                 print(i)
#                 break




# my_path = '/home/sanjay/Msc_Project/sanjay_123/Pk_mK2'
# path = os.listdir(my_path)

# Data_pk = []

# for file in path:
#     f = file.split("_")
#     sk = open("Pk_mK2/{}".format(file))
#     ss = list(sk)
#     ar = []
#     for i in range(1,11):
#         skk = ss[i].split()
#         a = float(skk[1])
#         ar.append(a)
#     Data_pk.append([float(f[5]),ar])


# ## $\textit{Data Reading}$

# In[7]:


my_path = '/home/sanjay/Msc_Project/sanjay_123/Pk_mK2'
path = os.listdir(my_path)


data_pk = []
para_data = []

for file in path:
    f = file.split("_")
    sk1 = np.loadtxt("Pk_mK2/{}".format(file),skiprows = 1)
    para_data.append([float(f[5]),float(f[6]),float(f[7]),float(f[8])])
    data_pk.append(sk1[:,1][2:8])
    
K_vals = sk1[:,0][2:8]
No_sample = sk1[:,3][2:8]
    
data_pk = np.array(data_pk*K_vals**3)
para_data = np.array(para_data)



Data_pk = pd.DataFrame(data = data_pk , 
                        columns = ["K1",
                                   "K2","K3","K4","K5","K6"])

Para_data = pd.DataFrame(data = para_data, 
                        columns = ["X_h",
                                   "M_min","N_ion","R_mfp"])


# In[8]:


xh = Para_data.drop(["M_min","N_ion","R_mfp"],axis = 1)
M = Para_data.drop(["X_h","N_ion","R_mfp"],axis = 1)
N = Para_data.drop(["X_h","M_min","R_mfp"],axis = 1)
R = Para_data.drop(["X_h","M_min","N_ion"],axis = 1)


# ### $\textit{Plots for Power Spectrum}$ 

# In[ ]:


xh = Para_data["X_h"]

num_plots = len(xh)

 


fig, ax = plt.subplots(figsize = (18,9))
colormap = cm.get_cmap('viridis', num_plots)
x = K_vals

for i in range(num_plots):
    y = Data_pk.iloc[i]
    
    ax.loglog(x, y, color=colormap(xh[i]), linewidth=1)
    

sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=xh.min(), vmax=xh.max()))
sm.set_array([])
cbar = plt.colorbar(sm, label='Neutral Fraction $X_h$ $ \longrightarrow$ ',pad = 0.03)

plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.xlabel(r"k (Mp$c^{-1}$) $ \longrightarrow$",fontsize = 15)
plt.ylabel(r" $\Delta ^{2} (mK^2)$ $ \longrightarrow$" ,fontsize = 15)
plt.title(r"PS for different $X_h$ values",fontsize = 15)

# plt.legend(fontsize = 15)
plt.show()


# ## $\textit{Statistics}$

# In[ ]:


# parameters_xh.corr()


# In[ ]:


plt.subplots(figsize=(20,9))
dataplot = sb.heatmap(Para_data.corr(), cmap="YlGnBu", annot=True)


# ## $\textit{Pair Plot}$

# In[ ]:


sb.pairplot(Para_data,height = 3.5,hue = 'X_h',diag_kind="hist")
# sb.pairplot(parameters_xh,height = 3.5 ,kind = 'scatter',corner=True,diag_kind = 'kde')
sns.set_context("paper", rc={"axes.labelsize":20})
sns.set_context("talk", font_scale=1.4)


# In[ ]:


sb.pairplot(Para_data,height = 3.5 ,kind = 'scatter',corner=True,diag_kind = 'hist')
sns.set_context("paper", rc={"axes.labelsize":20})
sns.set_context("talk", font_scale=1.4)


# # $\textit{Model Architecture}$

# #### $\textit{Pre-Processing}$

# In[69]:


# rm_index = []
# for i in range(len(xh_train['neutral fraction'])):
#     if xh_train['neutral fraction'].iloc[i]>0.65:
#         rm_index.append(i)
# # print(er_index)
# print(len(rm_index))

er_index = []
for i in range(len(xh_train)):
    
    if xh_train["X_h"].iloc[i]<0.15:
        er_index.append(i)
# print(er_index)
print(len(er_index))



w = 8



a = w/len(er_index)
b = (10-w)/(len(xh_train)-len(er_index))

sample_weight = np.zeros(len(xh_train))
for i in range(len(xh_train)):

    for j in er_index:
#     if i == any(er_index):
        if i==j              :
#             print(i)
            sample_weight[i] = a
            break
        else:
            sample_weight[i] = b
            
sample_weight


# In[7]:


Parameter = Para_data.drop(["X_h"],axis = 1)

from sklearn.preprocessing import StandardScaler
def scaler(Parameter):
    
    scaler = StandardScaler()


    scaler.fit(Parameter)


    scaled_data = scaler.transform(Parameter)


    scaled_df = pd.DataFrame(scaled_data, columns=Parameter.columns)



    return scaled_df


# In[9]:


zero_nf_rows = Para_data[Para_data['X_h']==0]
removel_data = zero_nf_rows.nsmallest(475, 'N_ion')
rm_indx = removel_data.index
md_data = Para_data.drop(rm_indx)
md_pk = Data_pk.loc[md_data.index]
md_data1 = md_data[md_data['X_h']<=0.55]
pk1 = Data_pk.loc[md_data1.index]
md_data2 = md_data[md_data['X_h']>0.55]
pk2 = Data_pk.loc[md_data2.index]
print(len(md_data1),len(md_data2))
xh1 = md_data1.drop(["M_min","N_ion","R_mfp"],axis = 1)
xh2 = md_data2.drop(["M_min","N_ion","R_mfp"],axis = 1)


# In[ ]:





# In[13]:


# xhh = Para_data["X_h"]
# drop = []
# for i in range(len(xhh)):
#     if max(Data_pk.iloc[i])<=50 and xhh.iloc[i]==0.000 :
#         drop.append(i)
# #         if xh.iloc[i]==0.000:
            
# #             print(xh.iloc[i],Data_pk.iloc[i])
# print(len(drop))



x_train,x_test,y_train,y_test,xh_train,xh_test = train_test_split(md_data,md_pk
                                                                  ,md_data.drop(["M_min","N_ion","R_mfp"],axis = 1),
                                                                  test_size = 0.2,random_state = 100)


# In[14]:


def model_creation(x_train, y_train, num_epochs=1000, batch_size=15, validation_split=0.1,sample_weight = None):
    
    EM_model = Sequential()
    EM_model.add(Dense(512, kernel_initializer='uniform', activation='elu', input_dim=x_train.shape[1]))
    EM_model.add(Dense(1024, activation='elu'))
    EM_model.add(Dense(128, activation='elu'))
    EM_model.add(Dense(64, activation='elu'))
    EM_model.add(Dense(y_train.shape[1]))  

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    EM_model.compile(loss='mean_squared_error', optimizer=optimizer,weighted_metrics=['mae'])

    
    callback_p = EarlyStopping(
        monitor="val_loss",
        min_delta=0.1,
        patience=20,
        verbose=1,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
        start_from_epoch=100
    )

    
    history = EM_model.fit(
        x_train, y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        sample_weight = sample_weight,
        callbacks=[callback_p],
        initial_epoch=0
    )

    return history,EM_model 

import numpy as np

def get_predictions(model, x_test):
    
#     if x_test.shape[1] != model.input_shape[1]:
#         raise ValueError("Input data shape does not match the model's input shape.")

    # Make predictions
    predictions = model.predict(x_test)

    return predictions

# Example usage:
# predictions = make_regression_predictions(EM_model, x_test)




# In[28]:


# x_train,x_test,y_train,y_test,xh_train,xh_test = train_test_split(scaler(md_data1),pk1
#                                                                   ,xh1,
#                                                                   test_size = 0.2,random_state = 120)
# model_pt = model_creation(x_train,y_train)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(512, kernel_initializer='uniform', activation='elu', input_dim=3))
model.add(Dense(1024, activation='elu'))
model.add(Dense(512, activation='elu'))
model.add(Dense(128, activation='elu'))
model.add(Dense(64, activation='elu'))
model.add(Dense(6))


from keras.layers import Concatenate, Input

# Define the input layer
input_layer = Input(shape=(3,))

# Earlier layer (before skip connection)
earlier_layer = Dense(512, kernel_initializer='uniform', activation='elu')(input_layer)

# Later layer (after skip connection)
later_layer = Dense(1024, activation='elu')(earlier_layer)

# Add the skip connection by concatenating the outputs
concatenated_layers = Concatenate()([earlier_layer, later_layer])

# Continue with the rest of your model
x = Dense(512, activation='elu')(concatenated_layers)
x = Dense(128, activation='elu')(x)
x = Dense(64, activation='elu')(x)
output = Dense(6)(x)

# Create the model
model_with_skip = keras.models.Model(inputs=input_layer, outputs=output)

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model_with_skip.compile(loss='mean_squared_error', optimizer=optimizer,weighted_metrics=['mse'])


callback_p = EarlyStopping(
    monitor="val_loss",
    min_delta=0.1,
    patience=20,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
    start_from_epoch=100
)


history = model_with_skip.fit(
    x_train.drop(['X_h'],axis = 1), y_train,
    epochs=1000,
    batch_size=15,
    validation_split=0.1,
    sample_weight = None,
    callbacks=[callback_p],
    initial_epoch=0
)


# In[19]:


from keras.layers import Input, Dense
from keras.models import Model

# Define the input layer
input_layer = Input(shape=(3,))

# Define the hidden layers with the specified number of neurons
hidden_layer1 = Dense(512, activation='relu',kernel_initializer='uniform')(input_layer)
hidden_layer2 = Dense(1024, activation='relu')(hidden_layer1)
hidden_layer3 = Dense(512, activation='relu')(hidden_layer2)
hidden_layer4 = Dense(256, activation='relu')(hidden_layer3)
hidden_layer5 = Dense(128, activation='relu')(hidden_layer4)
hidden_layer6 = Dense(64, activation='relu')(hidden_layer5)

# Define the output layer with 4 neurons (assuming you have 4 output classes)
output_layer = Dense(6)(hidden_layer6)

# Create the model
model_with_skip = Model(inputs=input_layer, outputs=output_layer)
optimizer = keras.optimizers.Adam(learning_rate=0.000859)
model_with_skip.compile(loss='mean_squared_error', optimizer=optimizer,weighted_metrics=['mse'])


callback_p = EarlyStopping(
    monitor="val_loss",
    min_delta=0.1,
    patience=20,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
    start_from_epoch=200
)


history = model_with_skip.fit(
    x_train.drop(['X_h'],axis = 1), y_train,
    epochs=1000,
    batch_size=15,
    validation_split=0.1,
    sample_weight = None,
    callbacks=[callback_p],
    initial_epoch=0
)


# #### $\textit{some jugaad}$ 

# In[ ]:


# x_train,x_test,y_train,y_test,xh_train,xh_test = train_test_split(Parameter,Data_pk
#                                                                   ,xh_vals,
#                                                                   test_size = 0.2,shuffle = False)


# x_train,x_test,y_train,y_test = train_test_split(Parameter,Data_pk,test_size = 0.2,random_state = 20)


# In[ ]:


# EM_xh  = Sequential()
# EM_xh.add(Dense(1024,kernel_initializer = 'uniform', activation = 'elu',input_dim = 3))
# # EM_xh.add(Dense(1024,activation = 'elu'))
# # EM_xh.add(Dense(128,activation = 'elu'))
# # EM_xh.add(Dense(64,activation = 'elu'))
# EM_xh.add(Dense(6))

# optimizer = keras.optimizers.Adam(learning_rate = 0.001,)
# EM_xh.compile(loss='mean_squared_error', optimizer=optimizer)

# callback_p = EarlyStopping(
#     monitor="val_loss",
#     min_delta=0.1,
#     patience=10,
#     verbose=1,
#     mode="auto",
#     baseline=None,
#     restore_best_weights=True,
#     start_from_epoch=200
# )

# # log_csv_p = CSVLogger('my_log_em.csv', separator=",", append=False)


# callback_list_p = [callback_p]

# hist = EM_xh.fit(x_train,xh_train,epochs = 1000,batch_size = 15,
#                           validation_split = 0.1,callbacks = callback_list_p,initial_epoch=0)

# xh_predict = EM_xh.predict(x_test)
# plt.plot(xh_test,'ko',label = 'Actual')
# plt.plot(xh_predict,'go',label = 'Predicted')
# plt.legend()
# plt.show()


# In[ ]:


# import numpy as np
# from tensorflow import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.callbacks import EarlyStopping

# def create_and_train_em_model(x_train, y_train, sample_weights=None):
#     # Define the model
#     EM_model = Sequential()
#     EM_model.add(Dense(512, kernel_initializer='uniform', activation='elu', input_dim=3))
#     EM_model.add(Dense(1024, activation='elu'))
#     EM_model.add(Dense(64, activation='elu'))
#     EM_model.add(Dense(6))

#     # Compile the model
#     optimizer = keras.optimizers.Adam(learning_rate=0.001)
#     EM_model.compile(loss='mean_squared_error', optimizer=optimizer)

#     # Define early stopping callback
#     callback_p = EarlyStopping(
#         monitor="val_loss",
#         min_delta=0.1,
#         patience=20,
#         verbose=1,
#         mode="auto",
#         baseline=None,
#         restore_best_weights=True,
#     )

#     # Train the model
#     hist = EM_model.fit(
#         x_train,
#         y_train,
#         epochs=1000,
#         batch_size=15,
#         validation_split=0.1,
#         callbacks=[callback_p],
#         initial_epoch=0,
#         sample_weight=sample_weights,
#     )

#     return EM_model, hist

# # Example usage:
# # EM_model, hist = create_and_train_em_model(x_train, y_train, sample_weights)
# # print(hist.history)


# In[ ]:


# import numpy as np
# from sklearn.model_selection import train_test_split
# from tensorflow import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.callbacks import EarlyStopping

# class NeuralNetwork:
#     def __init__(self, input_dim, output_dim):
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.model = self._build_model()

#     def _build_model(self):
#         model = Sequential()
#         model.add(Dense(512, kernel_initializer='uniform', activation='elu', input_dim=self.input_dim))
#         model.add(Dense(1024, activation='elu'))
#         model.add(Dense(64, activation='elu'))
#         model.add(Dense(self.output_dim))
#         optimizer = keras.optimizers.Adam(learning_rate=0.001)
#         model.compile(loss='mean_squared_error', optimizer=optimizer)
#         return model

#     def train(self, x_train, y_train, epochs=1000, batch_size=15, validation_split=0.1, sample_weights=None):
#         callback_p = EarlyStopping(
#             monitor="val_loss",
#             min_delta=0.1,
#             patience=20,
#             verbose=1,
#             mode="auto",
#             baseline=None,
#             restore_best_weights=True,
#         )

#         history = self.model.fit(
#             x_train,
#             y_train,
#             epochs=epochs,
#             batch_size=batch_size,
#             validation_split=validation_split,
#             callbacks=[callback_p],
#             sample_weight=sample_weights,
#         )

#         return history

# class Dataset:
#     def __init__(self, data, target):
#         self.data = data
#         self.target = target

#     def split(self, test_size=0.2, random_state=None):
#         return train_test_split(self.data, self.target, test_size=test_size, random_state=random_state)

# # Example usage:
# # dataset = Dataset(x_data, y_data)
# # x_train, x_test, y_train, y_test = dataset.split()

# # nn = NeuralNetwork(input_dim=x_train.shape[1], output_dim=y_train.shape[1])
# # history = nn.train(x_train, y_train)
# # print(history.history)


# In[ ]:


# import numpy as np
# from sklearn.model_selection import train_test_split
# from tensorflow import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.callbacks import EarlyStopping

# class NeuralNetwork:
#     def __init__(self, input_dim, output_dim):
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.model = self._build_model()

#     def _build_model(self):
#         # Define and compile your model architecture here
#         model = Sequential()
#         model.add(Dense(512, kernel_initializer='uniform', activation='elu', input_dim=self.input_dim))
#         model.add(Dense(1024, activation='elu'))
#         model.add(Dense(64, activation='elu'))
#         model.add(Dense(self.output_dim))
#         optimizer = keras.optimizers.Adam(learning_rate=0.001)
#         model.compile(loss='mean_squared_error', optimizer=optimizer)
#         return model

#     def train(self, x_train, y_train, epochs=1000, batch_size=15, validation_split=0.1, sample_weights=None):
#         # Define your training logic here
#         callback_p = EarlyStopping(
#             monitor="val_loss",
#             min_delta=0.1,
#             patience=20,
#             verbose=1,
#             mode="auto",
#             baseline=None,
#             restore_best_weights=True,
#         )

#         history = self.model.fit(
#             x_train,
#             y_train,
#             epochs=epochs,
#             batch_size=batch_size,
#             validation_split=validation_split,
#             callbacks=[callback_p],
#             sample_weight=sample_weights,
#         )

#         return history

# class Dataset:
#     def __init__(self, data, target):
#         self.data = data
#         self.target = target

#     def split(self, test_size=0.2, random_state=None):
#         # Split your dataset into training and testing sets here
#         x_train, x_test, y_train, y_test = train_test_split(self.data, self.target, test_size=test_size, random_state=random_state)
#         return x_train, x_test, y_train, y_test

# # Example usage:

# # Create a Dataset instance
# # dataset = Dataset(scaled_df.drop(drop),Data_pk.drop(drop))

# # # Split the dataset
# # x_train, x_test, y_train, y_test = dataset.split()

# # # Create a NeuralNetwork instance
# # nn = NeuralNetwork(input_dim=x_train.shape[1], output_dim=y_train.shape[1])

# # Train the model
# # histor = nn.train(x_train, y_train)

# # Print training history
# print(histor.history)


# ### $\textit{Architecture}$ 

# In[48]:


y_predict = model_with_skip.predict(x_test.drop(["X_h"],axis = 1))

sanjay_index = np.array(y_test.index.tolist())


 
y_pd_df = pd.DataFrame(data = y_predict, index = sanjay_index,
                        columns = ["K1",
                                   "K2","K3","K4","K5","K6"])


# model_pt1[1].summary()
NF = xh_test

# plt.plot(model_pt1[0].history['loss'])
# plt.plot(model_pt1[0].history['val_loss'])
# plt.title('model loss')

# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper right')
# plt.show()

# plt.plot(hist.history['acc'])
# plt.plot(hist.history['val_acc'])
# plt.title('model loss')

# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper right')
# plt.show()



mae_EM = metrics.mean_absolute_error(y_test,y_pd_df)
mse_EM = metrics.mean_squared_error(y_test,y_pd_df)
rsq_EM = metrics.r2_score(y_test,y_pd_df)
# activation_functions = [layer.activation.__name__ for layer in model_pt1[1].layers]
# optimizer_name = model_pt1[1].optimizer.__class__.__name__
print()
print('Mean absolute error is {0:.4f}'.format(mae_EM))
print()
print('Mean squared error is {0:.4f}'.format(mse_EM))
print()
print('R_square score is {}'.format((round(rsq_EM,5))))
print()
# print("Activation Functions :=> ",activation_functions)
print()
# print("Optimizer :=> ",optimizer_name)

# parameters = pd.read_csv('my_log_em.csv')
# parameter_poly_data = pd.DataFrame(parameters)
# pd.set_option("display.max_columns",None)
# # pd.set_option("display.max_rows",163)

# parameter_poly_data.tail()



# In[53]:


# y_test.loc[78]


# ## $\boxed{\textbf{Testing of Emulater}}$

# #### $\textit{plots}$

# In[58]:


NF = xh_test["X_h"]


# In[63]:


k = 0
for i in range(0,len(xh_test)):
    if NF.iloc[i]<0.1:
        k +=1
        a, b = y_test.iloc[i] , y_pd_df.iloc[i] 
        # y_max = (max(max(a),max(b))) + 
        error_ = abs(a-b)
        rel_error = (error_/a)*100


#         y_max1 = 1 + max(rel_error)
#         y_min = min(rel_error) - 0.5
 
        plt.figure(figsize=(20, 7))
        G = gridspec.GridSpec(1, 3,wspace=0.3,hspace=0.5)
        ax1 = plt.subplot(G[0, :2])
    #         ax2 = plt.subplot(G[0,1])
    #         ax2.set_xticks(())
        ax3 = plt.subplot(G[0, 2])

        plt.suptitle(r"Figure %d :- Power Spectrum at $x_h$ = %.3f" %(k,NF.iloc[i]),
                     fontsize = 22,color = "firebrick",y = 1.1)


        ax1.loglog(K_vals,b,'go--',label = "Predicted PS")
        ax1.loglog(K_vals,a,'ko--',label = "Actual PS")
        ax1.set_title("Power Spectrum",color = 'blue',fontsize = 17)
        ax1.set_xlabel(r"k (Mp$c^{-1}$) $ \longrightarrow$",fontsize = 17,color = 'darkgreen')
        ax1.set_ylabel(r" $\Delta ^{2} (mK^2)$ $ \longrightarrow$" ,fontsize = 17,color = 'darkgreen')
        ax1.legend(fontsize = 15)
        
        p,q,r = md_data.drop(["X_h"],axis = 1).iloc[i]
        ax1.set_title(r'Parameters:  $M_{min}$ = %.3f , $N_{ion}$ = %.3f , $R_{mfp}$ = %.3f'%(p,q,r),
                 fontsize=17,color='purple',bbox={'facecolor': 'white', 'alpha': 0.6, 'pad': 10},loc='right',x= 1.3,y = 1.133)
        ax3.set_box_aspect(11/12)
        ax3.semilogx(K_vals,rel_error,'go--',label = " Relative error in %")

    #         ax3.yaxis.set_major_locator(MultipleLocator(0.5))
        ax1.yaxis.set_major_formatter('{x:.0f}')
        ax3.yaxis.set_minor_locator(MultipleLocator(0.5))


#         ax3.set_ylim(y_max1)
        ax3.set_ylabel(r"Error $ \longrightarrow$" ,fontsize = 15,color = 'darkgreen')
        ax3.set_title("Relative Error in Prediction",color = 'blue',fontsize = 17)
        ax3.set_xlabel(r"k (Mp$c^{-1}$) $ \longrightarrow$",fontsize = 17,color = 'darkgreen')
        ax3.legend(fontsize = 12)
        plt.show()


# In[62]:


k = 0
for i in range(0,len(xh_test)):
    if NF.iloc[i]>=0.1:
        k +=1
        a, b = y_test.iloc[i] , y_pd_df.iloc[i] 
        # y_max = (max(max(a),max(b))) + 
        error_ = abs(a-b)
        rel_error = (error_/a)*100


#         y_max1 = 1 + max(rel_error)
#         y_min = min(rel_error) - 0.5
 
        plt.figure(figsize=(20, 7))
        G = gridspec.GridSpec(1, 3,wspace=0.3,hspace=0.5)
        ax1 = plt.subplot(G[0, :2])
    #         ax2 = plt.subplot(G[0,1])
    #         ax2.set_xticks(())
        ax3 = plt.subplot(G[0, 2])

        plt.suptitle(r"Figure %d :- Power Spectrum at $x_h$ = %.3f" %(k,NF.iloc[i]),
                     fontsize = 22,color = "firebrick",y = 1.1)


        ax1.loglog(K_vals,b,'go--',label = "Predicted PS")
        ax1.loglog(K_vals,a,'ko--',label = "Actual PS")
        ax1.set_title("Power Spectrum",color = 'blue',fontsize = 17)
        ax1.set_xlabel(r"k (Mp$c^{-1}$) $ \longrightarrow$",fontsize = 17,color = 'darkgreen')
        ax1.set_ylabel(r" $\Delta ^{2} (mK^2)$ $ \longrightarrow$" ,fontsize = 17,color = 'darkgreen')
        ax1.legend(fontsize = 15)
        
        p,q,r = md_data.drop(["X_h"],axis = 1).iloc[i]
        ax1.set_title(r'Parameters:  $M_{min}$ = %.3f , $N_{ion}$ = %.3f , $R_{mfp}$ = %.3f'%(p,q,r),
                 fontsize=17,color='purple',bbox={'facecolor': 'white', 'alpha': 0.6, 'pad': 10},loc='right',x= 1.3,y = 1.133)
        ax3.set_box_aspect(11/12)
        ax3.semilogx(K_vals,rel_error,'go--',label = " Relative error in %")

    #         ax3.yaxis.set_major_locator(MultipleLocator(0.5))
        ax1.yaxis.set_major_formatter('{x:.0f}')
        ax3.yaxis.set_minor_locator(MultipleLocator(0.5))


#         ax3.set_ylim(y_max1)
        ax3.set_ylabel(r"Error $ \longrightarrow$" ,fontsize = 15,color = 'darkgreen')
        ax3.set_title("Relative Error in Prediction",color = 'blue',fontsize = 17)
        ax3.set_xlabel(r"k (Mp$c^{-1}$) $ \longrightarrow$",fontsize = 17,color = 'darkgreen')
        ax3.legend(fontsize = 12)
        plt.show()


# ##  $\textit{Interective}$

# #### $\textit{Function for plots}$

# In[54]:


def get_plot(index,xh_val,no_plot):
    
    k = 0
    for i in index:
             
        k = k + 1
        a, b = y_test.loc[i] , y_pd_df.loc[i]

        error_ = abs(a-b)
        rel_error = (error_/a)*100


        y_max1 = 1 + max(rel_error)
        y_min = min(rel_error) - 0.1

        plt.figure(figsize=(20, 7))
        G = gridspec.GridSpec(1, 3,wspace=0.3,hspace=0.5)
        ax1 = plt.subplot(G[0, :2])
   
        ax3 = plt.subplot(G[0, 2])

        plt.suptitle(r"Figure %d :- Power Spectrum at $x_h$ = %.3f" %(k,xh_val),
                     fontsize = 22,color = "firebrick",y = 1.1)

        ax1.loglog(K_vals,b,'go--',label = "Predicted PS")
        ax1.loglog(K_vals,a,'ko',label = "Actual PS")
        ax1.set_title("Power Spectrum",color = 'blue',fontsize = 17)
        ax1.set_xlabel(r"k (Mp$c^{-1}$) $ \longrightarrow$",fontsize = 17,color = 'darkgreen')
        ax1.set_ylabel(r" $\Delta ^{2} (mK^2)$ $ \longrightarrow$" ,fontsize = 17,color = 'darkgreen')
        ax1.legend(fontsize = 15)
        p,q,r = md_data.drop(["X_h"],axis = 1).iloc[i]
        ax1.set_title(r'Parameters:  $M_{min}$ = %.3f , $N_{ion}$ = %.3f , $R_{mfp}$ = %.3f'%(p,q,r),
                 fontsize=17,color='purple',bbox={'facecolor': 'white', 'alpha': 0.6, 'pad': 10},loc='right',x= 1.07,y = 1.134)

        ax3.set_box_aspect(11/12)
        ax3.semilogx(K_vals,rel_error,'go--',label = " Relative error in %")

    #         ax3.yaxis.set_major_locator(MultipleLocator(0.5))
        ax1.yaxis.set_major_formatter('{x:.0f}')
#         ax3.yaxis.set_minor_locator(MultipleLocator(0.5))


#         ax3.set_ylim(y_min,y_max1)
        ax3.set_ylabel(r"Error $ \longrightarrow$" ,fontsize = 15,color = 'darkgreen')
        ax3.set_title("Relative Error in Prediction",color = 'blue',fontsize = 17)
        ax3.set_xlabel(r"k (Mp$c^{-1}$) $ \longrightarrow$",fontsize = 17,color = 'darkgreen')
        ax3.legend(fontsize = 12)
        plt.show()
        if k == no_plot:
            break


# #### $\textit{Function for Interaction to the computer}$

# In[45]:


def interaction():
        
    s = 0
    while s<1:
        print("")
        print("[bold bright_red] Instructions:-")
        print("")
        print("[bright_blue] Please provide me the value of neutral fraction at which you want to see the Power Spectrum.")
        print("[bright_blue italic underline] Note: The value of neutral fraction should be between 0 to 1.")
        print("")
        xh_val = input("Enter a number (or 'q' to quit): ")
        if xh_val.lower() == 'q':
            print("[bold bright_magenta] Bye! We are leaving the loop.")
            break
        else:
            xh_val = float(xh_val)
            if 0<=xh_val<1:

                xx = xh_test.loc[xh_test["X_h"]==xh_val]

                if len(xx) ==0:

                    print(" ")
                    print("[green] ===>  Sorry! I didn't found any Power Spectrum plot at neutral fraction %.3f in testing data set."%(xh_val))
                    print(" ")
                    print("[green] ===>  Provide a range so that I can give you neighbour neutral fraction values around this %.3f value."%xh_val)
                    print("")
                    print("[green] ===>  If you want very close value then please provide very short range like 0.01,0.011 etc.")
                    rang = input("Enter the range (or 'q' to quit): ")
                    if rang.lower() == 'q':
                                 print("[bold bright_magenta] Bye! We are leaving the loop.")
                                 break
                    else:
                        ss = 0
                        while ss<1:
                            
                            rang = float(rang)
                            xh1 =  xh_val - rang
                            xh2 = xh_val + rang
                            new_index = xh_test.query(" %.3f <= X_h <= %.3f"%(xh1,xh2)).index.tolist()
                            NF_array = xh_test.loc[new_index]
                            NF_array = NF_array['X_h'].to_numpy()
                            if len(NF_array)==0:
                                print("[bright_blue] Sorry! there is no any neighbour in this range. Please increase your range.")
                                rang = float(input("Enter new range:  "))

                            else:
                                print()
                                print("[green] ===>  oh! I found {} neutral fraction values which are neighbours of previous value".format(len(new_index)))
                                print(" ")
                                print("[green] ===>  List of these values is here:")
                                print()
                                print(np.sort(NF_array))
                                print()
                                print("[green] ===>  Please select one of the neutral fraction value from above list at which you want to see the power spectrum.")
                                print(" ")
                                print(" ")
                                ss = ss+1

                else:
                    if len(xx)==1:
                        s = s+1
                        print("[green] ===>  I found {a} Power Spectrum plots which is here:-".format(a = len(xx)))
                        index = xx.index
                        get_plot(index,xh_val,len(xx))
                    else:
                        s = s+1
                        print("[green] ===>  I found {a} Power Spectrum plots. How many do you want to see?. Please provide me a number.".format(a = len(xx)))
                        no_plot = input("Enter a number (or 'q' to quit): ")
                        if no_plot.lower()=='q':
                            print("[bold bright_magenta] Bye! We are leaving the loop.")
                            break
                        else:
                            no_plot = float(no_plot)

                            if no_plot<1:


                                print("[green] ===>  oh! I think you made a mistake[/green], [bold cyan] Please provide me a valid number[/bold cyan].")
                                print("[green] ===>  If this is not your mistake then please enter 1 or enter 0 if you want to change number of plot.")
                                choice = float(input("Enter your choice:  "))
                                san = 0
                                while san<1:

                                    if choice==1:
                                        print("[bold cyan italic] ===>  Better luck next time")
                                        san = san +1
                                    elif choice==0:
                                        no_plot = input("Enter a number howmany plots do you want (or 'q' to quit): ")
                                        if no_plot.lower() == 'q':
                                            print("[bold bright_magenta] Bye! We are leaving the loop.")
                                            break
                                        else:
                                            no_plot = float(no_plot)

    #                                         print("[bold cyan italic] ===>  I got your input that you want to see {} plots. If you want to continue with {} plots then type 0 or if you want to change number of plots then type 1".format(no_plot,no_plot))
    #                                         feed = int(input("Enter your choice:  "))
    #                                         if feed ==1:
    #                                             print("[bold cyan italic] ===>  Please re-enter howmany plots do you want.")
    #                                             no_plot = int(input("Enter a number:  "))
    #                                             index = xx.index
    #                                             get_plot(index,xh_val,no_plot)

    #                                         else:
                                            index = xx.index
                                            get_plot(index,xh_val,no_plot)
                                            san = san+1
                                    else:
                                        print("[bold bright_magenta] You are giving invalid input. Try again!")
                                        choice = float(input(" Enter your choice again  "))

                            else:

#                                 print("[bold cyan italic] ===>  I got your input that you want to see {} plots. If you want to continue with {} plots then type 0 or if you want to change number of plots then type 1".format(no_plot,no_plot))
#                                 feed = int(input("Enter your choice: "))
#                                 if feed ==1:
#                                     print("[bold cyan italic] ===>  Please re-enter howmany plots do you want.")
#                                     no_plot = int(input("Enter a number: "))
#                                     index = xx.index
#                                     get_plot(index,xh_val,no_plot)

#                                 else:
                                index = xx.index
                                get_plot(index,xh_val,no_plot)
            else:
                print("[bright_red italic underline dim] Stop! You are giving an invalid input. Please read instructions again.")
    
    print("")

    
    
def Interactive_result():
    
    while True:
        interaction()
        print("")
        print("[bold magenta] ===> Do you want to continue it or quit it? If you want to continue then enter [underline]'c'[/underline] or if you want to quit it then enter [underline]'q'[/underline]")
        print("")
        sk = (input("Enter your choice: "))
        if sk.lower() == 'q':
            break
        
        



# In[ ]:


# def interaction():
        
#     s = 0
#     while s<1:
#         print("")
#         print("[bold bright_red] Instructions:-")
#         print("")
#         print("[bright_blue] Please provide me the value of neutral fraction at which you want to see the Power Spectrum.")
#         print("[bright_blue italic underline] Note: The value of neutral fraction should be between 0 to 1")
#         print("")
#         xh_val = input("Enter a number (or 'q' to quit): ")
#         if xh_val.lower() == 'q':
#             print("[bold bright_magenta] Bye! We are leaving the loop.")
#             break
#         else:
#             xh_val = float(xh_val)
#             if 0<=xh_val<1:

#                 xx = xh_test.loc[xh_test["X_h"]==xh_val]

#                 if len(xx) ==0:

#                     print(" ")
#                     print("[green] ===>  Sorry! I didn't found any Power Spectrum plot at neutral fraction %.3f in testing data set."%(xh_val))
#                     print(" ")
#                     print("[green] ===>  Provide a range so that I can give you neighbour neutral fraction values around this %.3f value."%xh_val)
#                     print("")
#                     print("[green] ===>  If you want very close value then please provide very short range like 0.01,0.011 etc.")
#                     rang = input("Enter the range (or 'q' to quit): ")
#                     if rang.lower() == 'q':
#                                  print("[bold bright_magenta] Bye! We are leaving the loop.")
#                                  break
#                     else:
                        
#                         while True:
                            
#                             rang = float(rang)
#                             xh1 =  xh_val - rang
#                             xh2 = xh_val + rang
#                             new_index = xh_test.query(" %.3f <= X_h <= %.3f"%(xh1,xh2)).index.tolist()
#                             NF_array = xh_test.loc[new_index]
#                             NF_array = NF_array['X_h'].to_numpy()
#                             if len(NF_array)==0:
#                                 print("[bright_blue] Sorry! there is no any neighbour in this range. Please increase your range.")
#                                 rang = float(input("Enter new range:  "))
#                                 continue

#                             else:
#                                 print()
#                                 print("[green] ===>  oh! I found {} neutral fraction values which are neighbours of previous value".format(len(new_index)))
#                                 print(" ")
#                                 print("[green] ===>  List of these values is here:")
#                                 print()
#                                 print(np.sort(NF_array))
#                                 print()
#                                 print("[green] ===>  Please select one of the neutral fraction value from above list at which you want to see the power spectrum.")
#                                 print(" ")
#                                 print(" ")
# #                                 ss = ss+1

#                 else:
#                     if len(xx)==1:
#                         s = s+1
#                         print("[green] ===>  I found {a} Power Spectrum plots which is here:-".format(a = len(xx)))
#                         index = xx.index
#                         get_plot(index,xh_val,len(xx))
#                     else:
#                         s = s+1
#                         print("[green] ===>  I found {a} Power Spectrum plots. How many do you want to see?. Please provide me a number.".format(a = len(xx)))
#                         no_plot = input("Enter a number (or 'q' to quit): ")
#                         if no_plot.lower()=='q':
#                             print("[bold bright_magenta] Bye! We are leaving the loop.")
#                             break
#                         else:
#                             no_plot = float(no_plot)

#                             if no_plot<1:


#                                 print("[green] ===>  oh! I think you made a mistake[/green], [bold cyan] Please provide me a valid number[/bold cyan].")
#                                 print("[green] ===>  If this is not your mistake then please enter 1 or enter 0 if you want to change number of plot.")
#                                 choice = float(input("Enter your choice:  "))
                                
#                                 while True:

#                                     if choice==1:
#                                         print("[bold cyan italic] ===>  Better luck next time")
                                        
#                                     elif choice==0:
#                                         no_plot = input("Enter a number howmany plots do you want (or 'q' to quit): ")
#                                         if no_plot.lower() == 'q':
#                                             print("[bold bright_magenta] Bye! We are leaving the loop.")
#                                             break
#                                         else:
#                                             no_plot = float(no_plot)

#     #                                         print("[bold cyan italic] ===>  I got your input that you want to see {} plots. If you want to continue with {} plots then type 0 or if you want to change number of plots then type 1".format(no_plot,no_plot))
#     #                                         feed = int(input("Enter your choice:  "))
#     #                                         if feed ==1:
#     #                                             print("[bold cyan italic] ===>  Please re-enter howmany plots do you want.")
#     #                                             no_plot = int(input("Enter a number:  "))
#     #                                             index = xx.index
#     #                                             get_plot(index,xh_val,no_plot)

#     #                                         else:
#                                             index = xx.index
#                                             get_plot(index,xh_val,no_plot)
                                            
#                                     else:
#                                         print("[bold bright_magenta] You are giving invalid input. Try again!")
#                                         choice = float(input(" Enter your choice again  "))

#                             else:

# #                                 print("[bold cyan italic] ===>  I got your input that you want to see {} plots. If you want to continue with {} plots then type 0 or if you want to change number of plots then type 1".format(no_plot,no_plot))
# #                                 feed = int(input("Enter your choice: "))
# #                                 if feed ==1:
# #                                     print("[bold cyan italic] ===>  Please re-enter howmany plots do you want.")
# #                                     no_plot = int(input("Enter a number: "))
# #                                     index = xx.index
# #                                     get_plot(index,xh_val,no_plot)

# #                                 else:
#                                 index = xx.index
#                                 get_plot(index,xh_val,no_plot)
#             else:
#                 print("[bright_red italic underline dim] Stop! You are giving invalid input, Please read instructions again.")
    
#     print("")

    
    
# def Interactive_result():
    
#     while True:
#         interaction()
#         print("")
#         print("[bold magenta] ===> Do you want to continue it or quit it? If you want to continue then enter [underline]'c'[/underline] or if you want to quit it then enter [underline]'q'[/underline]")
#         print("")
#         sk = (input("Enter your choice: "))
#         if sk.lower() == 'q':
#             break
        
        



# #### $\textit{Get Results vaya interaction}$

# In[46]:


Interactive_result()


# ## $\textit{Free time things}$

# In[26]:


# from keras.models import Sequential
# from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasRegressor
# from sklearn.ensemble import VotingRegressor
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import numpy as np

# # Define your deep neural network architecture as a function
# def create_model():
#     model = Sequential()
#     model.add(Dense(64, activation='relu', input_dim=3))
#     model.add(Dense(32, activation='relu'))
#     model.add(Dense(6))  # Output layer for regression
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     return model

# # Load your data and split it into training and testing sets
# # X_train, X_test, y_train, y_test = ...

# # Standardize the input features
# # scaler = StandardScaler()
# # X_train = scaler.fit_transform(x_train)
# # X_test = scaler.transform(x_test)

# # Create multiple instances of the deep neural network
# models = [KerasRegressor(build_fn=create_model, epochs=50, batch_size=32, verbose=0) for _ in range(3)]

# # Create the ensemble model using VotingRegressor
# ensemble = VotingRegressor(estimators=[('model1', models[0]), ('model2', models[1]), ('model3', models[2])])

# # Fit the ensemble model on the training data
# ensemble.fit(x_train.drop(["X_h"],axis = 1), y_train)

# # Make predictions with the ensemble
# predictions = ensemble.predict(x_train.drop(["X_h"],axis = 1))

# # Evaluate the ensemble's performance
# mse = mean_squared_error(y_test, predictions)
# print(f"Mean Squared Error (MSE): {mse}")


# In[ ]:


def get_piechart(xh,a,san):    
    data = np.array(xh).reshape(len(xh),)
    data1 = pd.DataFrame(data)
    ssa = data1.groupby(pd.cut(data, a,right = False,precision = 2,include_lowest = True)).count()

    bs = ssa[0]
    label = bs.index
    bs = np.array(bs)

    labels =label
    sizes = bs
    max_index = np.argmax(sizes)
    min_index = np.argmin(sizes)


    explode = [0.3 if i in [max_index, min_index] else 0 for i in range(len(labels))]


    plt.pie(sizes, explode=explode, labels=labels,
    autopct='%1.2f%%', shadow=False, startangle=140)
    plt.axis('equal')
    plt.text(-1,1.5,"Pie chart for {} values in {} different ranges".format(san,a),fontsize = 17,color = "blue")
    plt.show()
get_piechart(xh,5,"Neutral Fraction")
# get_piechart(M,5,"M_min")
get_piechart(N,5,"N_ion")
# get_piechart(R,10,"R_mfp")
    


# In[ ]:


import optuna
import tensorflow as tf
from tensorflow import keras

def objective(trial):
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    
    # Build and compile the neural network with the suggested learning rate
    model = keras.Sequential([
        keras.layers.Dense(512,kernel_initializer = 'uniform', activation = 'elu',input_dim = 3),
        keras.layers.Dense(1024, activation='elu'),
        keras.layers.Dense(128, activation='elu'),
        keras.layers.Dense(64, activation='elu'),
        
        keras.layers.Dense(6)
    ])
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    # Train and evaluate the model
    history = model.fit(x_train, y_train, epochs=200,batch_size = 15, validation_split=0.1, verbose=0)
    return history.history['val_loss'][-1]

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print('Best learning rate:', study.best_params['learning_rate'])
print('Best loss:', study.best_value)

# EM_model.add(Dense(512,kernel_initializer = 'uniform', activation = 'elu',input_dim = 3))
# EM_model.add(Dense(1024,activation = 'elu'))
# EM_model.add(Dense(128,activation = 'elu'))
# EM_model.add(Dense(64,activation = 'elu'))
# EM_model.add(Dense(6))


# In[ ]:


import smtplib
from email.mime.text import MIMEText

# Define your email parameters
email_username = "sanjaykumaryadav10108@gmail.com"
email_password = "kir@njay8o79"
sender_email = 'sanjaykumaryadav10108@gmail.com'
receiver_email = 'sanjay09052001@gmail.com'
message = 'Your code has finished executing.'

# Create and send the email
msg = MIMEText(message)
msg['Subject'] = 'Code Execution Notification'
msg['From'] = sender_email
msg['To'] = receiver_email

try:
    # Connect to the SMTP server and send the email
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(email_username, email_password)
    server.sendmail(sender_email, receiver_email, msg.as_string())
    server.quit()
    print("Email notification sent successfully.")
except Exception as e:
    print(f"Email notification failed: {str(e)}")



# In[ ]:


# import smtplib
# from email.mime.text import MIMEText
import os
import smtplib
from email.mime.text import MIMEText

# Retrieve email credentials from environment variables
email_username = os.environ.get("sanjaykumaryadav10108@gmail.com")
email_password = os.environ.get("kir@njay8o79")

# Check if the credentials exist
if email_username is None or email_password is None:
    print("Email credentials not found.")
else:
    # Rest of your code to send the email using email_username and email_password

# Define your email parameters
    sender_email = 'sanjaykumaryadav10108@gmail.com'
    receiver_email = 'sanjay09052001@gmail.com'
    password = 'kir@njay8o79'  # Use an application-specific password for security
    message = 'Your code has finished executing.'

    # Create and send the email
    msg = MIMEText(message)
    msg['Subject'] = 'Code Execution Notification'
    msg['From'] = sender_email
    msg['To'] = receiver_email

    try:
        # Connect to the SMTP server and send the email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print("Email notification sent successfully.")
    except Exception as e:
        print(f"Email notification failed: {str(e)}")


# In[ ]:


# Suppose you have category labels for your data
category_labels = np.array([0, 1, 1.5, 0, 2.5, 1, 3, 2.5, 2, 3.5])

# Calculate the number of data points in each category
unique_categories, category_counts = np.unique(category_labels, return_counts=True)

# Compute weights based on the reciprocal of category size
# You can adjust this formula to match your specific criterion
category_weights = 1.0 / category_counts

# Create an array of sample weights based on category membership
sample_weights = category_weights[np.searchsorted(unique_categories, category_labels)]

# Now, you can use sample_weights in model.fit()
# model.fit(x_train, y_train, sample_weight=sample_weights, epochs=..., batch_size=...)

sample_weights


# In[ ]:


import numpy as np

# Assuming you have a binary classification problem with 0 and 1 labels
y_train = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 1])

# Create sample weights: assigning higher weight to class 1
sample_weights = np.where(y_train == 1, 2.0, 1.0)

# Now, you can use sample_weights in model.fit()
# model.fit(x_train, y_train, sample_weight=sample_weights, epochs=..., batch_size=...)
sample_weights


# In[ ]:





# In[ ]:


# Get user input as a string
user_input = input("Enter a number: ")

# Convert the input to an integer
try:
    num = int(user_input)
    print("You entered:", num)
except ValueError:
    print("Invalid input. Please enter a valid number.")


# In[ ]:


while True:
    user_input = input("Enter a positive number (or 'q' to quit): ")
    
    if user_input.lower() == 'q':
        print("Exiting the loop.")
        break
    user_input = float(user_input)
    try:
        num = float(user_input)
        if num > 0:
            print("You entered:", num)
            break  # Exit the loop if input is valid
        else:
            print("Please enter a positive number.")
    except ValueError:
        
        print("Invalid input. Please enter a valid number.")

        


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

def get_plot(index, xh_val, no_plot):
    a_values = [y_test.loc[i] for i in index]
    b_values = [y_pd_df.loc[i] for i in index]
    errors = [abs(a - b) for a, b in zip(a_values, b_values)]
    rel_errors = [(error / a) * 100 for error, a in zip(errors, a_values)]

    y_max1 = 1 + max(max(rel_errors))
    y_min = min(min(rel_errors)) - 0.1

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), gridspec_kw={'wspace': 0.3, 'hspace': 0.5})
    plt.suptitle(r"Power Spectrum at $x_h$ = %.3f" % xh_val, fontsize=22, color="firebrick", y=1.1)

    for k, (ax1, ax3, (p, q, r), rel_error) in enumerate(zip(axes[:-1], axes[-1], Parameter.iloc[index], rel_errors), start=1):
        ax1.loglog(K_vals, b_values[k-1], 'go--', label="Predicted PS")
        ax1.loglog(K_vals, a_values[k-1], 'ko', label="Actual PS")
        ax1.set_title("Power Spectrum", color='blue', fontsize=17)
        ax1.set_xlabel(r"k (Mp$c^{-1}$) $ \longrightarrow$", fontsize=17, color='darkgreen')
        ax1.set_ylabel(r" $\Delta ^{2} (mK^2)$ $ \longrightarrow$", fontsize=17, color='darkgreen')
        ax1.legend(fontsize=15)
        ax1.set_title(r'Parameters:  $M_{min}$ = %.3f , $N_{ion}$ = %.3f , $R_{mfp}$ = %.3f' % (p, q, r),
                      fontsize=14, color='purple', bbox={'facecolor': 'white', 'alpha': 0.6, 'pad': 10},
                      loc='right', x=1.06, y=1.135)

        ax3.set_box_aspect(11/12)
        ax3.semilogx(K_vals, rel_error, 'go--', label=" Relative error in %")
        ax3.set_ylabel(r"Error $ \longrightarrow$", fontsize=15, color='darkgreen')
        ax3.set_title("Relative Error in Prediction", color='blue', fontsize=17)
        ax3.set_xlabel(r"k (Mp$c^{-1}$) $ \longrightarrow$", fontsize=17, color='darkgreen')
        ax3.legend(fontsize=12)

        if k == no_plot:
            break

    plt.show()

# Example usage:
# Replace the following with your actual data
# y_test = pd.DataFrame(...)
# y_pd_df = pd.DataFrame(...)
# Parameter = pd.DataFrame(...)
# K_vals = [...]
# index = [...]
# get_plot(index, xh_val, no_plot)


# In[ ]:


import numpy as np

def get_neighbor_values(xh_test, xh_val, range_val):
    xh1 = xh_val - range_val
    xh2 = xh_val + range_val
    new_index = xh_test.query("%.3f <= X_h <= %.3f" % (xh1, xh2)).index.tolist()
    return new_index

def view_power_spectrum_plots(index, xh_val, no_plot):
    
    a_values = [y_test.loc[i] for i in index]
    b_values = [y_pd_df.loc[i] for i in index]
    errors = [abs(a - b) for a, b in zip(a_values, b_values)]
    rel_errors = [(error / a) * 100 for error, a in zip(errors, a_values)]

    y_max1 = 1 + max(max(rel_errors))
    y_min = min(min(rel_errors)) - 0.1

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), gridspec_kw={'wspace': 0.3, 'hspace': 0.5})
    plt.suptitle(r"Power Spectrum at $x_h$ = %.3f" % xh_val, fontsize=22, color="firebrick", y=1.1)

    for k, (ax1, ax3, (p, q, r), rel_error) in enumerate(zip(axes[:-1], axes[-1], Parameter.iloc[index], rel_errors), start=1):
        ax1.loglog(K_vals, b_values[k-1], 'go--', label="Predicted PS")
        ax1.loglog(K_vals, a_values[k-1], 'ko', label="Actual PS")
        ax1.set_title("Power Spectrum", color='blue', fontsize=17)
        ax1.set_xlabel(r"k (Mp$c^{-1}$) $ \longrightarrow$", fontsize=17, color='darkgreen')
        ax1.set_ylabel(r" $\Delta ^{2} (mK^2)$ $ \longrightarrow$", fontsize=17, color='darkgreen')
        ax1.legend(fontsize=15)
        ax1.set_title(r'Parameters:  $M_{min}$ = %.3f , $N_{ion}$ = %.3f , $R_{mfp}$ = %.3f' % (p, q, r),
                      fontsize=14, color='purple', bbox={'facecolor': 'white', 'alpha': 0.6, 'pad': 10},
                      loc='right', x=1.06, y=1.135)

        ax3.set_box_aspect(11/12)
        ax3.semilogx(K_vals, rel_error, 'go--', label=" Relative error in %")
        ax3.set_ylabel(r"Error $ \longrightarrow$", fontsize=15, color='darkgreen')
        ax3.set_title("Relative Error in Prediction", color='blue', fontsize=17)
        ax3.set_xlabel(r"k (Mp$c^{-1}$) $ \longrightarrow$", fontsize=17, color='darkgreen')
        ax3.legend(fontsize=12)

        if k == no_plot:
            break

    plt.show()
#     pass

def interactive():
    while True:
        try:
            xh_val = float(input("Enter the neutral fraction value (0 to 1) or 'q' to quit: "))
            if 0 <= xh_val < 1:
                xx = xh_test.loc[xh_test["X_h"] == xh_val]
                if len(xx) == 0:
                    print("No Power Spectrum plot found at this neutral fraction.")
                    range_val = float(input("Enter a range (0.01, 0.011, etc.) or 'q' to quit: "))
                    if range_val < 0:
                        print("Invalid range value.")
                        continue
                    new_index = get_neighbor_values(xh_test, xh_val, range_val)
                    if len(new_index) == 0:
                        print("No neighbors found in this range. Increase the range.")
                        continue
                    print(f"Found {len(new_index)} neighbor values:")
                    print(np.sort(xh_test.loc[new_index]['X_h'].to_numpy()))
                    xh_val = float(input("Select a neighbor value or 'q' to quit: "))
                    if xh_val == 'q':
                        break
                if len(xx) == 1:
                    get_plot(xx.index, xh_val, len(xx))
                else:
                    no_plot = float(input(f"Found {len(xx)} Power Spectrum plots. Enter the number of plots to view: "))
                    if no_plot < 1:
                        print("Invalid number of plots.")
                        continue
                    get_plot(xx.index, xh_val, no_plot)
            else:
                print("Invalid input. Please enter a number between 0 and 1.")
        except ValueError:
            print("Invalid input. Please enter a valid number or 'q' to quit.")

def interactive_result():
    while True:
        interactive()
        sk = input("Do you want to continue (c) or quit (q)? ")
        if sk.lower() == 'q':
            break

# Example usage:
# xh_test = your DataFrame
# interactive_result()


# In[ ]:


interactive()


# In[ ]:


a = 5


while True:
    c = float(input())
    if c==5:
        print("sanjay")
        break
    continue
        


# In[ ]:


from termcolor import colored

# ...

def interactive():
    while True:
        try:
            xh_val = float(input("Enter the neutral fraction value (0 to 1) or 'q' to quit: "))
            if 0 <= xh_val < 1:
                xx = xh_test.loc[xh_test["X_h"] == xh_val]
                if len(xx) == 0:
                    print(colored("No Power Spectrum plot found at this neutral fraction.", "red"))
                    # ...
                if len(xx) == 1:
                    view_power_spectrum_plots(xx.index, xh_val, len(xx))
                else:
                    no_plot = float(input(f"Found {len(xx)} Power Spectrum plots. Enter the number of plots to view: "))
                    if no_plot < 1:
                        print(colored("Invalid number of plots.", "red"))
                        continue
                    view_power_spectrum_plots(xx.index, xh_val, no_plot)
            else:
                print(colored("Invalid input. Please enter a number between 0 and 1.", "red"))
        except ValueError:
            print(colored("Invalid input. Please enter a valid number or 'q' to quit.", "red"))

# ...



# In[ ]:




