# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 14:55:38 2023

@author: ingeborgdepate
"""



#Packages
import json
import matplotlib.pyplot as plt 
import math
import pickle 


#%%


#Function for plotting results
def plot_predictions_vs_true(all_mean_predictions,  all_min_predictions, all_max_predictions):
   
    
    fig, ax = plt.subplots(figsize = (8,5)) #plt.subplots(figsize = (5.5, 4))
    size = 20
    labelsize= 17
    
    #plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    
    max_true = 0 
    
    for unit in all_mean_predictions.keys():
       
        true_RUL = [] 
        predicted_RUL = [] 
        min_RUL = [] 
        max_RUL = [] 
        
        predictions = all_mean_predictions.get(unit)
        #min_predictions = all_min_predictions.get(unit) 
        #max_predictions = all_max_predictions.get(unit) 
        for pred in predictions:
            true_RUL.append(pred.get("RUL"))
            predicted_RUL.append(pred.get("predicted"))             
            if pred.get("RUL") > max_true:
                max_true = pred.get("RUL")
                
                 
            #min_RUL.append(pred.get("predicted") - 1.96 * pred.get("std_dev"))    
        
               
            #max_RUL.append(pred.get("predicted") + 1.96 pred.get("std_dev"))      
            
        if unit == "11.0":
            ax.plot(true_RUL, predicted_RUL, linewidth = 3, label = "Engine " + str(int(float(unit))), marker = "o", markersize = 6, color =(0/255,77/255,64/255))
            # ax.plot(true_RUL, min_RUL, color =(0/255,77/255,64/255), alpha = 0.35, linestyle = "dashed")
            # ax.plot(true_RUL, max_RUL, color = (0/255,77/255,64/255), alpha = 0.35, linestyle = "dashed")
            
            #plt.fill_between(true_RUL,  min_RUL, max_RUL, color = "royalblue", alpha = 0.3)
        elif unit == "14.0": 
            ax.plot(true_RUL, predicted_RUL,  linewidth = 3, label = "Engine " + str(int(float(unit))), marker = "s", markersize = 6, color = (255/255,193/255,7/255)) 
            #x.plot(true_RUL, min_RUL, color = "limegreen", alpha = 0.35, linestyle = "dashed")
            #ax.plot(true_RUL, max_RUL, color = "limegreen", alpha = 0.35, linestyle = "dashed")
            
            #plt.fill_between(true_RUL,  min_RUL, max_RUL, color = "limegreen", alpha = 0.3)
        elif unit == "15.0": 
            ax.plot(true_RUL, predicted_RUL,  linewidth = 3, label = "Engine " + str(int(float(unit))), marker = "X", markersize = 6, color = (30/255,136/255,229/255)) 
            #ax.plot(true_RUL, min_RUL, color = "violet", alpha = 0.35, linestyle = "dashed")
            #ax.plot(true_RUL, max_RUL, color = "violet", alpha = 0.35, linestyle = "dashed")
            
            #plt.fill_between(true_RUL,  min_RUL, max_RUL, color = "violet", alpha = 0.3)
        else:
            ax.plot(true_RUL, predicted_RUL, label = "Engine " + str(int(float(unit))), marker = ".")
            
    #PLot the true RUL as a thick red line 

    true_RUL = list(range(0, int(max_true) + 1,1)) 
    ax.plot(true_RUL, true_RUL, label = "True RUL", color ="red" , linewidth = 5)
    
        
    ax.set_xlabel("True RUL", fontsize = size)
    ax.set_ylabel("RUL prediction", fontsize = size)
    #ax.set_yticks(list(range(0, max_true+1, 5)))
    ax.set_xticks(list(range(0, int(max_true)+1, 5)))
    #ax.set_ylim([-11, 5])
    
    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top_side = ax.spines["top"]
    top_side.set_visible(False)
    
    ax.tick_params(axis='x', labelsize=labelsize)
    ax.tick_params(axis='y', labelsize=labelsize)
  

    ax.legend(fontsize = size - 2,   bbox_to_anchor=(1.01,1.01), loc = "upper right")
    fig.tight_layout()
    name = "federated_predictions"
    direction = "C:\\Users\\ingeborgdepate\\fed_learning\\" + str(name) + ".png"
    plt.savefig(direction, dpi=400)   
   
    
  
    plt.show() 

#Import the results 
file = open("C:\\Users\\ingeborgdepate\\fed_learning\\engine_turobofan_rul.json", "rb")
results = json.load(file)
file.close()

#make a nice plot 
plot_predictions_vs_true(results, {}, {})

#calculate the metrics
rmse_total = 0 
mae_total = 0 
number_pred_total = 0 
for unit in results.keys():
    rmse = 0 
    mae = 0     
    number_pred = 0 
    predictions = results.get(unit)
   
    for pred in predictions:
        true_RUL  = pred[1]
        predicted_RUL = pred[0]
        mae = mae + abs(true_RUL - predicted_RUL)
        rmse = rmse + (true_RUL - predicted_RUL) ** 2 
        number_pred = number_pred + 1 
    
    print("\nThe MAE of unit " , unit, " is " , mae / number_pred)
    print("The RMSE of unit " , unit, " is ", math.sqrt(rmse / number_pred))
    
    rmse_total = rmse_total + rmse 
    mae_total = mae_total + mae 
    number_pred_total = number_pred_total + number_pred
    
print("\n The final MAE of all units is " , mae_total / number_pred_total)
print("The final RMSE of all units is " ,  math.sqrt(rmse_total / number_pred_total))
 
    
    
    
#%%

#Also plot the validation    

#Import the results 
file = open("C:\\Users\\ingeborgdepate\\fed_learning\\distributed_validations.json")
results = json.load(file)
file.close()


fig, ax = plt.subplots(figsize = (8,5)) #plt.subplots(figsize = (5.5, 4))
size = 20
labelsize= 17

ax.plot(results, linewidth = 3, label = "Validation loss", marker = "o", markersize = 6, color =(0/255,77/255,64/255))


    
ax.set_xlabel("Number of epochs", fontsize = size)
ax.set_ylabel("Validation loss", fontsize = size)
#ax.set_yticks(list(range(0, max_true+1, 5)))

#ax.set_ylim([-11, 5])

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

ax.tick_params(axis='x', labelsize=labelsize)
ax.tick_params(axis='y', labelsize=labelsize)
  
  
ax.legend(fontsize = size - 2,   bbox_to_anchor=(1.01,1.01), loc = "upper right")
fig.tight_layout()
name = "federated_validation"
direction = "C:\\Users\\ingeborgdepate\\fed_learning\\" + str(name) + ".png"
plt.savefig(direction, dpi=400)   
   

  
plt.show() 
  









