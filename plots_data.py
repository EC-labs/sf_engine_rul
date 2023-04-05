

import os
import h5py
import time
import matplotlib
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib import gridspec
import sys
import mathematics as math

#%%
### Set-up - Define file location
filename = 'C:\\Users\\ingeborgdepate\\New C-MAPPS\\N-CMAPSS_DS02-006.h5'

# Load data
with h5py.File(filename, 'r') as hdf:
        # Development set
        W_dev = np.array(hdf.get('W_dev'))             # W
        X_s_dev = np.array(hdf.get('X_s_dev'))         # X_s
        X_v_dev = np.array(hdf.get('X_v_dev'))         # X_v
        T_dev = np.array(hdf.get('T_dev'))             # T
        Y_dev = np.array(hdf.get('Y_dev'))             # RUL  
        A_dev = np.array(hdf.get('A_dev'))             # Auxiliary

        # Test set
        W_test = np.array(hdf.get('W_test'))           # W
        X_s_test = np.array(hdf.get('X_s_test'))       # X_s
        X_v_test = np.array(hdf.get('X_v_test'))       # X_v
        T_test = np.array(hdf.get('T_test'))           # T
        Y_test = np.array(hdf.get('Y_test'))           # RUL  
        A_test = np.array(hdf.get('A_test'))           # Auxiliary
        
        # Varnams
        W_var = np.array(hdf.get('W_var'))
        X_s_var = np.array(hdf.get('X_s_var'))  
        X_v_var = np.array(hdf.get('X_v_var')) 
        T_var = np.array(hdf.get('T_var'))
        A_var = np.array(hdf.get('A_var'))
        
        # from np.array to list dtype U4/U5
        W_var = list(np.array(W_var, dtype='U20'))
        X_s_var = list(np.array(X_s_var, dtype='U20'))  
        X_v_var = list(np.array(X_v_var, dtype='U20')) 
        T_var = list(np.array(T_var, dtype='U20'))
        A_var = list(np.array(A_var, dtype='U20'))
                          
# W = np.concatenate((W_dev, W_test), axis=0)  
# X_s = np.concatenate((X_s_dev, X_s_test), axis=0)
# X_v = np.concatenate((X_v_dev, X_v_test), axis=0)
# T = np.concatenate((T_dev, T_test), axis=0)
# Y = np.concatenate((Y_dev, Y_test), axis=0) 
# A = np.concatenate((A_dev, A_test), axis=0) 

W = W_dev  
X_s = X_s_dev 
X_v = X_v_dev
T = T_dev 
Y = Y_dev 
A = A_dev

print ("W shape: " + str(W.shape))
print ("X_s shape: " + str(X_s.shape))
print ("X_v shape: " + str(X_v.shape))
print ("T shape: " + str(T.shape))
print ("A shape: " + str(A.shape))


#%%
#--------------------------------------------------------------------------------------------------#
#------------------------------------------Auxiliary info------------------------------------------#
#--------------------------------------------------------------------------------------------------#
df_A = DataFrame(data=A_test, columns=A_var)
df_A.describe()

print('Engine units in df: ', np.unique(df_A['unit']))

selected_unit = 2
selected_cycle =1 #75
training = [2,5,10,16,18,20]
test = [11,14,15]

for unit in  np.unique(df_A['unit']):
    print("\nthe unit is " , unit)
    print("the flight class of unit " , unit, " is " , df_A.loc[df_A["unit"] == unit, "Fc"].iloc[0] )
    print("and the number of cycles is " ,  max(np.unique(df_A.loc[df_A["unit"] == unit, "cycle"])) )
    


#%%
#--------------End of failure
training_samples = 0 
for i in np.unique(df_A['unit']):
    max_length = 0 
    min_length = 999999
    mean_length = 0 
    
    print('\n Unit: ' + str(i) + ' - Number of flight cyles (t_{EOF}): ', len(np.unique(df_A.loc[df_A['unit'] == i, 'cycle'])))
    print("and this unit belongs to flight class " , df_A.loc[df_A["unit"] == i, "Fc"].iloc[0])
    print("at its first flight, the health state is ", df_A.loc[(df_A['unit'] == i), 'hs'].iloc[0])
    print(" and it's healthy for ... ", len(np.unique(df_A.loc[(df_A['unit'] == i) & (df_A['hs'] == 1), 'cycle'])) , " flights")
    training_samples = training_samples + len(np.unique(df_A.loc[(df_A['unit'] == i) & (df_A['hs'] == 1), 'cycle']))
    
    data_unit = df_A.loc[df_A["unit"] == i] 
    number_of_flights = len(np.unique(data_unit["cycle"]))

    
    for j in range(1, number_of_flights + 1, 1):    
        #get all the data belonging to this flight
        data_temp = data_unit.loc[data_unit.cycle == j] 
        #print("the length  is " , len(data_temp)  )
        #print("and the dimension is " , data_temp.shape)
        #length.append(len(data_temp))  
        if len(data_temp) > max_length:
            max_length = len(data_temp)
        if len(data_temp) < min_length:
            min_length = len(data_temp)
        mean_length = mean_length +  len(data_temp)
    print("the max length is " , max_length , " minutes")
    print("the min length is " , min_length , " minutes")
    print("and the mean length is " , (mean_length / number_of_flights) )
    
 
print("\n in total, there're " , training_samples, " training samples ")
df_A_dev = DataFrame(data=A_dev, columns=A_var)


#%%
#Normalization 

def min_max_training(training_data, skip = ["cycle", "unit" , "hs"]):
   
    minima = {} 
    maxima = {}
    for column in training_data.columns:
        if column in skip:
            continue
  
        minimum = training_data[[column]].min()[column]
        maximum = training_data[[column]].max()[column]
        minima[column] = minimum
        maxima[column] = maximum

    return minima, maxima 

def normalization(data, minima, maxima, skip = ["cycle", "unit" , "hs"]):    

    for column in data.columns:
        if column in skip:
            continue
        
        # normalize between -1 and 1
        u = 1
        l = -1
        minimum = minima.get(column)
        maximum = maxima.get(column)
        

        data[column] = data[column].apply(lambda x: ((x - minimum) / (maximum - minimum)) * (u-l) + l)
       
    return data

#%%
#training data
df_W_dev =  DataFrame(data=W_dev, columns=W_var)
df_W_dev['unit'] = df_A['unit'].values
df_W_dev['cycle'] = df_A['cycle'].values
minima, maxima = min_max_training(df_W_dev)
all_data_shortened = normalization(df_W_dev, minima, maxima)    

#%%

all_data_shortened = all_data_shortened.loc[(all_data_shortened["unit"] == selected_unit) & (all_data_shortened["cycle"] == selected_cycle) ]


#plot the data
fig, ax = plt.subplots(figsize = (7,5)) 

fontsize = 18 
labelsize = fontsize - 2     
  
for col in all_data_shortened.columns:
    if col == "unit" or col == "cycle": 
        continue 
    #get the data 
    data = np.array(all_data_shortened[col]) 
    
    if col == "alt":
        ax.plot(data, linewidth = 3, label = col, linestyle = "dashed",  color =(0/255,77/255,64/255))
    elif col == "Mach":
        ax.plot(data, linewidth = 3, label = col, linestyle = "dotted",   color = (255/255,193/255,7/255))
    elif col == "TRA":
        ax.plot(data, linewidth = 3, label = col, linestyle = "solid",   color = (30/255,136/255,229/255))
    elif col == "T2":        
        ax.plot(data, linewidth = 3, label = col, linestyle = "dashdot",   color = (216/255,27/255,96/255))
        
      
    
ax.legend(fontsize = fontsize, ncol = 4,  bbox_to_anchor=(1.07, -0.15), loc = "upper right")

ax.set_xlabel("Time during flight (sec.)", fontsize = fontsize)
ax.set_ylabel("Normalized \n operating condition", fontsize = fontsize)
ax.set_yticks(np.arange(-1, 1.1, 0.5)) 
#ax.set_xticks(list(range(0, int(max_true)+1, 5)))
  
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

ax.tick_params(axis='x', labelsize=labelsize)
ax.tick_params(axis='y', labelsize=labelsize)

fig.tight_layout()
name = "operating_conditions"
direction = "C:\\Users\\ingeborgdepate\\fed_learning\\" + name + ".png"
plt.savefig(direction, dpi=400)  

#%%
#take the mean per flight  

all_data_shortened = all_data_shortened.loc[(all_data_shortened["unit"] == selected_unit)  ]

considered_col = ["alt", "Mach", "TRA", "T2" , "cycle"]
means = pd.DataFrame(columns = considered_col) 


for cycle in np.unique(all_data_shortened["cycle"]): 
    #take the means 
    data =  all_data_shortened.loc[(all_data_shortened["cycle"] == cycle)  ]
    mean = data.mean() 
 
    means = means.append(mean, ignore_index = True) 
    
#%%
#plot the mean operaitng condition per flight 
    
#plot the data
fig, ax = plt.subplots(figsize = (7,5)) 

fontsize = 18 
labelsize = fontsize - 2     
  
for col in means.columns:
    if col == "unit" or col == "cycle": 
        continue 
    #get the data 
    data = np.array(means[col]) 
    
    if col == "alt":
        ax.plot(data, linewidth = 3, label = col, linestyle = "dashed",  color =(0/255,77/255,64/255))
    elif col == "Mach":
        ax.plot(data, linewidth = 3, label = col, linestyle = "dotted",   color = (255/255,193/255,7/255))
    elif col == "TRA":
        ax.plot(data, linewidth = 3, label = col, linestyle = "solid",   color = (30/255,136/255,229/255))
    elif col == "T2":        
        ax.plot(data, linewidth = 3, label = col, linestyle = "dashdot",   color = (216/255,27/255,96/255))
        
      
    
ax.legend(fontsize = fontsize, ncol = 4,  bbox_to_anchor=(1.03, -0.15), loc = "upper right")

ax.set_xlabel("Flights", fontsize = fontsize)
ax.set_ylabel("Mean normalized \n operating condition", fontsize = fontsize)
ax.set_yticks(np.arange(-0.8, 0.9, 0.4)) 
#ax.set_xticks(list(range(0, int(max_true)+1, 5)))
  
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

ax.tick_params(axis='x', labelsize=labelsize)
ax.tick_params(axis='y', labelsize=labelsize)

fig.tight_layout()
name = "mean_operating_conditions"
direction = "C:\\Users\\ingeborgdepate\\fed_learning\\" + name + ".png"
plt.savefig(direction, dpi=400)  

#%% sensor data 
sensors = ["SmHPC", "Nf", "T48", "P50", "unit", "cycle"] 

df_X_s =  DataFrame(data=X_s_dev, columns=X_s_var) 
df_X_s['unit'] = df_A['unit'].values
df_X_s['cycle'] = df_A['cycle'].values
df_X_v = DataFrame(data=X_v_dev, columns=X_v_var) 
all_data = [df_X_s, df_X_v]
all_data = pd.concat(all_data, axis=1)
all_data = all_data[sensors]
minima, maxima = min_max_training(all_data)
all_data_shortened = normalization(all_data, minima, maxima)    



#%%

all_data_shortened = all_data_shortened.loc[(all_data_shortened["unit"] == selected_unit) & (all_data_shortened["cycle"] == selected_cycle) ]

#plot the data

fig, ax = plt.subplots(figsize = (7,5)) 

fontsize = 18 
labelsize = fontsize - 2     
  
for col in all_data_shortened.columns:
    if col == "unit" or col == "cycle": 
        continue 
    #get the data 
    data = np.array(all_data_shortened[col]) 
    
    if col == "SmHPC":
        ax.plot(data, linewidth = 3, label = col, linestyle = "dashed",  color =(0/255,77/255,64/255))
    elif col == "Nf":
        ax.plot(data, linewidth = 3, label = col, linestyle = "dotted",   color = (255/255,193/255,7/255))
    elif col == "T48":
        ax.plot(data, linewidth = 3, label = col, linestyle = "solid",   color = (30/255,136/255,229/255))
    elif col == "P50":        
        ax.plot(data, linewidth = 3, label = col, linestyle = "dashdot",   color = (216/255,27/255,96/255))
        
      
    
ax.legend(fontsize = fontsize, ncol = 4,  bbox_to_anchor=(1.07, -0.15), loc = "upper right")

ax.set_xlabel("Time during flight (sec.)", fontsize = fontsize)
ax.set_ylabel("Normalized \n sensor measurement", fontsize = fontsize)
ax.set_yticks(np.arange(-1, 1.1, 0.5)) 
#ax.set_xticks(list(range(0, int(max_true)+1, 5)))
  
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

ax.tick_params(axis='x', labelsize=labelsize)
ax.tick_params(axis='y', labelsize=labelsize)

fig.tight_layout()
name = "sensor_measurements"
direction = "C:\\Users\\ingeborgdepate\\fed_learning\\" + name + ".png"
plt.savefig(direction, dpi=400)   

#%%
#take the mean per flight  

all_data_shortened = all_data_shortened.loc[(all_data_shortened["unit"] == selected_unit)  ]

considered_col =  ["SmHPC", "Nf", "T48", "P50", "cycle"] 
means = pd.DataFrame(columns = considered_col) 


for cycle in np.unique(all_data_shortened["cycle"]): 
    #take the means 
    data =  all_data_shortened.loc[(all_data_shortened["cycle"] == cycle)  ]
    mean = data.mean() 
 
    means = means.append(mean, ignore_index = True) 
    
#%%
#plot the mean operaitng condition per flight 
    
#plot the data
fig, ax = plt.subplots(figsize = (7,5)) 

fontsize = 18 
labelsize = fontsize - 2     
  
for col in means.columns:
    if col == "unit" or col == "cycle": 
        continue 
    #get the data 
    data = np.array(means[col]) 
    
    if col == "SmHPC":
        ax.plot(data, linewidth = 3, label = col, linestyle = "dashed",  color =(0/255,77/255,64/255))
    elif col == "Nf":
        ax.plot(data, linewidth = 3, label = col, linestyle = "dotted",   color = (255/255,193/255,7/255))
    elif col == "T48":
        ax.plot(data, linewidth = 3, label = col, linestyle = "solid",   color = (30/255,136/255,229/255))
    elif col == "P50":        
        ax.plot(data, linewidth = 3, label = col, linestyle = "dashdot",   color = (216/255,27/255,96/255))
        
      
    
ax.legend(fontsize = fontsize, ncol = 4,  bbox_to_anchor=(1.03, -0.15), loc = "upper right")

ax.set_xlabel("Flights", fontsize = fontsize)
ax.set_ylabel("Mean normalized \n snesor measurement", fontsize = fontsize)
ax.set_yticks(np.arange(-0.8, 0.9, 0.4)) 
#ax.set_xticks(list(range(0, int(max_true)+1, 5)))
  
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

ax.tick_params(axis='x', labelsize=labelsize)
ax.tick_params(axis='y', labelsize=labelsize)

fig.tight_layout()
name = "mean_snesor_measurements"
direction = "C:\\Users\\ingeborgdepate\\fed_learning\\" + name + ".png"
plt.savefig(direction, dpi=400)  
 

