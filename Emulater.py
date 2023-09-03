###########    Impotent Libraries

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


############################################################################################
##    Code for changing the fontsize , fontstyle and color
from rich import print
from IPython.display import display, HTML
display(HTML("<style>.container { width:97% !important; }</style>"))

get_ipython().run_cell_magic('html', '', "<style type='text/css'>\n.CodeMirror{\n    font-size: 12px;\n}\n\ndiv.output_area pre {\n    font-size: 14px;\n}\n</style>\n\n!jt -t onedork -fs 20 -altp -tfs 20 -ofs 14 -cellw 10%\n")







#############################################################################################

##### Code for Data Reading

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


Parameter = Para_data.drop(["X_h"],axis = 1)




#################################################################################

############# Data Pre-processing &  Data spliting for training and testing

zero_nf_rows = Para_data[Para_data['X_h']==0]
removel_data = zero_nf_rows.nsmallest(475, 'N_ion')
rm_indx = removel_data.index
md_data = Para_data.drop(rm_indx)
md_pk = Data_pk.loc[md_data.index]

xh = Para_data.drop(["M_min","N_ion","R_mfp"],axis = 1)
M = Para_data.drop(["X_h","N_ion","R_mfp"],axis = 1)
N = Para_data.drop(["X_h","M_min","R_mfp"],axis = 1)
R = Para_data.drop(["X_h","M_min","N_ion"],axis = 1)


x_train,x_test,y_train,y_test,xh_train,xh_test = train_test_split(md_data,md_pk
                                                                  ,md_data.drop(["M_min","N_ion","R_mfp"],axis = 1),
                                                                  test_size = 0.2,random_state = 100)

################# Function for getting pie-chart to pre-process the data
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

#### Get pie-chart for different feature
get_piechart(xh,5,"Neutral Fraction")
get_piechart(M,5,"M_min")
get_piechart(N,5,"N_ion")
get_piechart(R,10,"R_mfp")






###### Function for Scaling the features
def scaler(Parameter):
    
    scaler = StandardScaler()


    scaler.fit(Parameter)


    scaled_data = scaler.transform(Parameter)


    scaled_df = pd.DataFrame(scaled_data, columns=Parameter.columns)



    return scaled_df









######################################################################

##   Define the sample weight to balanced unbalanced data

er_index = []
for i in range(len(xh_train)):
    
    if xh_train["X_h"].iloc[i]<0.1:
        er_index.append(i)
# print(er_index)
print(len(er_index))



w = 0.8



a = w/len(er_index)
b = (1-w)/(len(xh_train)-len(er_index))

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
            






#####################################################################

###############            Deep Neural Network Architecture in form of function
def model_creation(x_train, y_train, num_epochs=1000, batch_size=15, validation_split=0.1,sample_weight = None):
    
    EM_model = Sequential()
    EM_model.add(Dense(512, kernel_initializer='uniform', activation='elu', input_dim=x_train.shape[1]))
    EM_model.add(Dense(1024, activation='elu'))
    EM_model.add(Dense(128, activation='elu'))
    EM_model.add(Dense(64, activation='elu'))
    EM_model.add(Dense(y_train.shape[1]))  

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    EM_model.compile(loss='mean_squared_error', optimizer=optimizer,weighted_metrics=['mse'])

    
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


####################################################################

###########  function for making pridiction using trained model

def get_predictions(model, x_test):
    
    if x_test.shape[1] != model.input_shape[1]:
        raise ValueError("Input data shape does not match the model's input shape.")

    # Make predictions
    predictions = model.predict(x_test)

    return predictions










####################################################################

##   New model with using skip connection in deep neural network to avoid vanishing gradiant problem


from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(512, kernel_initializer='uniform', activation='elu', input_dim=3))
model.add(Dense(1024, activation='elu'))
model.add(Dense(512, activation='elu'))
model.add(Dense(128, activation='elu'))
model.add(Dense(64, activation='elu'))
model.add(Dense(6))



# Define the input layer
input_layer = Input(shape=(3,))

# Define the hidden layers with the specified number of neurons
hl1 = Dense(512, activation='relu',kernel_initializer='uniform')(input_layer)     # hl stand for hidden layer
hl2 = Dense(1024, activation='relu')(hl1)
hl3 = Dense(512, activation='relu')(hl2)
hl4 = Dense(256, activation='relu')(hl3)
hl5 = Dense(128, activation='relu')(hl4)
hl6 = Dense(64, activation='relu')(hl5)

# Define the output layer with 4 neurons (assuming you have 4 output classes)
output_layer = Dense(6)(hl6)

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







####### Testing the model 

y_predict = model_with_skip.predict(x_test.drop(["X_h"],axis = 1))

sanjay_index = np.array(y_test.index.tolist())

y_pd_df = pd.DataFrame(data = y_predict, index = sanjay_index,
                        columns = ["K1","K2","K3","K4","K5","K6"])





### Errors in prediction
mae_EM = metrics.mean_absolute_error(y_test,y_pd_df)
mse_EM = metrics.mean_squared_error(y_test,y_pd_df)
rsq_EM = metrics.r2_score(y_test,y_pd_df)

print('Mean absolute error is {0:.4f}'.format(mae_EM))
print('Mean squared error is {0:.4f}'.format(mse_EM))
print('R_square score is {}'.format((round(rsq_EM,5))))



##########################################################

#####  Code for getting plots

k = 0
for i in range(0,len(xh_test)):
    k +=1
    a, b = y_test.iloc[i] , y_pd_df.iloc[i] 
    
    error_ = abs(a-b)
    rel_error = (error_/a)*100


    plt.figure(figsize=(20, 7))
    G = gridspec.GridSpec(1, 3,wspace=0.3,hspace=0.5)
    ax1 = plt.subplot(G[0, :2])

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


    ax1.yaxis.set_major_formatter('{x:.0f}')
    ax3.yaxis.set_minor_locator(MultipleLocator(0.5))

    ax3.set_ylabel(r"Error $ \longrightarrow$" ,fontsize = 15,color = 'darkgreen')
    ax3.set_title("Relative Error in Prediction",color = 'blue',fontsize = 17)
    ax3.set_xlabel(r"k (Mp$c^{-1}$) $ \longrightarrow$",fontsize = 17,color = 'darkgreen')
    ax3.legend(fontsize = 12)
    plt.show()




    



###########################################################
##  
#                   Hyper parameter tunning using optuna
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

