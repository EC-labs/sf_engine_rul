#####################################################
#-----------------------Packages--------------------#
#####################################################
from torch.utils.data import Dataset, random_split, DataLoader
from torch import nn
import torch

from pandas import DataFrame

import numpy as np
import pandas as pd
import time
import h5py
import  math
import copy 
 
import sys
import matplotlib.pyplot  as plt 

#####################################################
#-----------------------Parameters--------------------#
#####################################################
#THE STEPSIZE WILL PROABBLY BE 1, BUT A LARGER NUMBER IS A LOT FASTER FOR TESTING THE CODE
stepsize_sample = 10 #with what steps do we make our samples (1 is: from 1 to 50, from 2 to 51, etc.)
considered_length = 50 #Length of one sample.

#THESE OARAMETERS PROBABLY HAVE TO CHANGE WITH A FREQUENCY OF 1 AND ALL SENSORS. 
height = 10
number_channels = 10     
num_neurons = 100

#Define the required frequency 
#THE FREQUENCY WILL BE 1, BUT THAT IS VERY SLOW IN TESTING THE CODE-> USING SOME BIGGER VALUE WHEN TESTING THE CODE MIGHT HELP :) 
frequency =  1#1 is much smallerr than I had before, so we might have to change some parameters 

# Define which variables to keep (the ones we will consider)
#WE CAN ALSO KEEP ALL SENSORS HERE
X_v_to_keep =  ["W21", "W50", "SmFan", "SmLPC", "SmHPC"]  
X_s_to_keep = ["Wf", "Nf", "T24", "T30", "T48", "T50", "P2", "P50"] 

#Considered loss function 
loss_function = torch.nn.MSELoss()  
batch_size = 128 # 1024 
num_epochs = 50 #?
learning_rate = 0.001 #0.01 #0.001? 

    

#####################################################
#-----------------------Data------------------------#
#####################################################

def read_in_data(frequency,X_v_to_keep, X_s_to_keep, training_data = True, keep_all_data = True ): #True if you want the training data, False if you want the testing data 
    # Set-up - Define file location
    filename = 'C:\\Users\\ingeborgdepate\\New C-MAPPS\\N-CMAPSS_DS02-006.h5'
    
    # Load data
    with h5py.File(filename, 'r') as hdf:
        # Development set
        W_dev = np.array(hdf.get('W_dev'))             # W
        X_s_dev = np.array(hdf.get('X_s_dev'))         # X_s
        X_v_dev = np.array(hdf.get('X_v_dev'))         # X_v      
        A_dev = np.array(hdf.get('A_dev'))             # Auxiliary

        # Test set
        W_test = np.array(hdf.get('W_test'))           # W
        X_s_test = np.array(hdf.get('X_s_test'))       # X_s
        X_v_test = np.array(hdf.get('X_v_test'))       # X_v   
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

    #-------------------------------------------------------------------------------------------------=-------------#
    #------------------------------------------Get the training data ------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------#
    #Make the flight conditions     
    all_fc = W_var
    
    # step 0. Make the real sensor measurements a dataframe
    if training_data == True:
        df_X_s = DataFrame(data=X_s_dev, columns=X_s_var)
        df_X_v = DataFrame(data=X_v_dev, columns=X_v_var)
        df_A = DataFrame(data=A_dev, columns=A_var)
        df_W = DataFrame(data=W_dev, columns=W_var)
    else:
        df_X_s = DataFrame(data=X_s_test, columns=X_s_var)
        df_X_v = DataFrame(data=X_v_test, columns=X_v_var)
        df_A = DataFrame(data=A_test, columns=A_var)
        df_W = DataFrame(data=W_test, columns=W_var)        

    # step 0.1. Select the relevant sensor measurements
    df_X_s = df_X_s[X_s_to_keep]
    df_X_v = df_X_v[X_v_to_keep]

    # Step 1. Add the unit, the flight cycle and the health status to the real sensor measurements of the training set
    df_X_s["unit"] = df_A["unit"].values  # unit number
    df_X_s["cycle"] = df_A["cycle"].values  # flight cycle
    df_X_s["hs"] = df_A["hs"].values  # health state
    #flight claess irrelevant for us,so we ignore them   

    # Step 2. Add the flight conditions, the virtual measurements and the real measurements together
    all_data = [df_X_s, df_X_v, df_W]
    all_data = pd.concat(all_data, axis=1)

    # Step 1.1 Select only the healthy flights
    #This was only for the autoencoder -> keep all data is standard on true  
    if keep_all_data == False:
        all_data = all_data.loc[all_data["hs"] == 1]  
     
    #Redcue the sampling frequnce 
    #Not for us, but I left it in, might be nice if you want to quickly test your model
    if frequency > 1:
        all_data_shortened = pd.DataFrame(columns = all_data.columns)     
               
        #for all units
        for unit in np.unique(all_data["unit"]) :  
             data_unit = all_data.loc[all_data["unit"] == unit] #Get the data 
    
             #for all flights
             for flight in np.unique(data_unit["cycle"]):              
                data_flight = data_unit.loc[data_unit["cycle"] == flight]
                data_flight.reset_index(inplace = True)
                               
                means = data_flight.groupby(np.arange(len(data_flight))//frequency).mean() #Take the mean per frequency bin
                
                frames = [all_data_shortened, means]
                all_data_shortened = pd.concat(frames, axis = 0)                        

        del all_data #To save some memory 
        all_data_shortened = all_data_shortened.drop(columns = ["index"]) 
        
    else:
        all_data_shortened  = all_data 
    
    return all_data_shortened, all_fc 
    

def min_max_training(training_data, skip = ["cycle", "unit" , "hs"]):
    #This function finds the minimum and the maximum of the training data, for the normalization of the senso measurements 
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
    #THis function normalizes the sensor measurements before inputting them in the neural network (min-max normalization)
    for column in data.columns:
        if column in skip:
            continue        
        # normalize between -1 and 1
        u = 1
        l = -1
        minimum = minima.get(column)
        maximum = maxima.get(column)      

        data.loc[:, column] = data.loc[:, column].apply(lambda x: ((x - minimum) / (maximum - minimum)) * (u-l) + l)       
    return data   
   

class sensor_data(Dataset):
    #THis is the sensor data class, where we make our samples! 
    
    def __init__(self, all_data,  stepsize_sample, all_variables_x, considered_length, considered_flights):
        self.all_data = all_data #The data
        self.all_variables_x = all_variables_x  #The variables for x
        self.considered_length = considered_length #The length of each sample
        
        #storing all samples at once gives a memory problem. Instead, we define for each sample number the engine,
        #the flight, and the beginning and ending time from which we take the measurements in __getitem__ 
        self.sample_index = {} 
        number_sample = 0     
        
        #All sample numbers belonging to a specific engine and flight. Necessary for the testing of the dataset
        self.unit_flight_all_samples = {} 
        
        #we make a dictionary with for each unit, the total number of flights
        #We use this to get the RUL (i.e., the label) in the getitem part 
        self.unit_length = {} 

        # for all units
        for unit in np.unique(all_data["unit"]):           

            data_unit = all_data.loc[all_data["unit"] == unit] #The data of this unit 
            
            #Get the number of flights
            number_flights = len(np.unique(data_unit["cycle"]))
            
            self.unit_length[unit] = number_flights #For getting the RUL later on
            
            #Get the flights we consider for this unit (training or validation)
            flights_unit = considered_flights[unit]
                        
            # for all flights
            for flight in np.unique(data_unit["cycle"]):     
                
                if flight not in flights_unit: #For the training vs validation 
                    continue
                
                all_samples = [] #All samples belonging to this engine and flight. Necessary for the testing part             

                data_flight = data_unit.loc[data_unit["cycle"] == flight] #Data of the flight 
                data_flight.reset_index(inplace=True)

                length_of_flight = data_flight.shape[0]  
                
                start = 0  # Initial index
                while start <= (length_of_flight - self.considered_length): #Loop over the flight
                    # get the final index
                    end = start + self.considered_length  # end is excluded. Get the end point of the sample
                    all_info = (unit, flight, start, end) #The engine, flight, start time and end time of the sample 
                    self.sample_index[number_sample] = all_info #Save! THis is what we use to make the sample in __getitem__
                    
                    all_samples.append(number_sample) #This is again for the testing part 
                    
                    number_sample = number_sample + 1 #update the numer of samples 
                   
                    # update start
                    start = start + stepsize_sample
                
                self.unit_flight_all_samples[(unit, flight)] = all_samples 

    def __len__(self):
        #NUmber of samples in the dataset
        return len(self.sample_index.keys())
    
    def get_all_samples(self, sample):
        #Sample is a (engine, flight) tuple
        #Get all sample numbers belonging to a engine and flight, necessary for testing 
        return self.unit_flight_all_samples.get(sample)
   
    def __getitem__(self, idx):
        #Get the sample belonging to the index :) 
        
        #First, get all the info of the sample
        info = self.sample_index.get(idx)
        unit = info[0] #The engine 
        flight = info[1] #The flight
        start = info[2] #The starting time of the sample
        end = info[3] #The ending time of the sample 

        #Select the relevant data 
        data_considered = self.all_data.loc[(self.all_data["unit"] == unit) & (self.all_data["cycle"] == flight)]
        sample = data_considered.iloc[start:end] 
        
        #Get the sensor measurements and operating conditions 
        sample_x = sample[self.all_variables_x]
        sample_x = sample_x.to_numpy() #Otherwise, we get some weird type error 
        sample_x = np.float32(sample_x) #Otherwise, we get some weird type error 
        
        #Caltulate the RUL (the label) 
        number_flight = self.unit_length.get(unit) #Number of flights of the unit 
        RUL = number_flight - flight #The RUL! 
        RUL = np.float32(RUL) #Otherwise, we get some weird type error 
      
        if RUL < 0:
            print("error: the RUL is " , RUL, " with flight " , flight , " and number flight " , number_flight)
            raise ValueError("The RUL is negative!") 

        return sample_x, RUL
    
def train_one_epoch(neural_network, loss_function, optimizer, data_loader, in_training):
    #This functions trains the neural network for one epoch :)  
    running_loss = 0.#The loss (nice to return in the end)
    
    # Iterate over the batches in the epochs
    for i, (sample_x, RUL) in enumerate(data_loader):  #i is the batch number 

        print("We are at batch " , i)     
            
        if in_training == False: #For the validation set (then, we don't update the gradients, but just calculate the error)
            with torch.no_grad():                
                #Make the predictions 
                predicted = neural_network(sample_x)
                predicted = predicted.squeeze(1) #Necessary to get the right dimensions
                      
                #Calculate the loss 
                loss = loss_function(RUL, predicted)
                running_loss = running_loss + loss.item() 
        else:
            #Make the predictions 
            predicted = neural_network(sample_x)
            predicted = predicted.squeeze(1) #Necessary to get the right dimensions
             
            #Calculate the loss 
            loss = loss_function(RUL, predicted)
            running_loss = running_loss + loss.item() 
      
            #Update the gradients
            optimizer.zero_grad()            
            loss.backward()                
            optimizer.step()
           
    return running_loss

class neural_network(nn.Module):
     #Our neural network :) 

     def __init__(self, input_size, height, number_channels, num_neurons, do = 0):
         
         #WE MIGHT NEED SOME DROPOUT -> TO LOOK INTO! 
       
         #some necessary initialization        
        super(neural_network, self).__init__()        
        
        #(Hyper)Parameters
        self.kernel_size = (height, 1) #height, width of the kernels 
        self.number_maps_first = number_channels #Number of kernels 
        self.input_size = input_size #Size of the single kernel of the last convolutional layer 
        self.num_neurons = num_neurons #Number of neurons in the fully connected layers 
        self.stride = 1 #How we move the kernels over the data 
        
        #Make the neural network
        #Hyperparameter: Number of layers 
        #Convolutional layer
        self.conv_one = nn.Conv2d(in_channels = 1, out_channels = self.number_maps_first, kernel_size = self.kernel_size, stride = self.stride, padding = "same")
        #Activation of this layer (possible hyperparameter)
        self.activation_one = nn.ReLU()
        #Initialization of the weights (this is standard for the ReLU activation function) WITH DISTRIBUTED LEARNING: HAD TO LOOK INTO THIS!
        nn.init.kaiming_normal_(self.conv_one.weight)
        
        self.conv_two = nn.Conv2d(in_channels = self.number_maps_first, out_channels = self.number_maps_first, kernel_size = self.kernel_size, stride = self.stride, padding = "same")
        self.activation_two = nn.ReLU()
        nn.init.kaiming_normal_(self.conv_two.weight)
        
        self.conv_three = nn.Conv2d(in_channels = self.number_maps_first, out_channels = 1, kernel_size = self.kernel_size, stride = self.stride, padding = "same")
        self.activation_three = nn.ReLU()
        nn.init.kaiming_normal_(self.conv_three.weight)
        
         #Flatten the output 
        self.flatten = nn.Flatten() 
        
        #IF WE WOULD DO DROPOUT -> PROBABLY HERE
        #MAYBE NOT NECESSARY: TOO SLOW FOR MANY ITERATIONS? 
        #self.dropout = nn.Dropout(do)
        
        #Add a fully connected layer
        self.fully_connected = nn.Linear(in_features = self.input_size, out_features = self.num_neurons, bias = True) 
        self.act_four = nn.ReLU()
        nn.init.kaiming_normal_(self.fully_connected.weight)
        

        
        #Predict tthe reul
        self.RUL_layer = nn.Linear(in_features = self.num_neurons, out_features = 1, bias = True) 
        nn.init.kaiming_normal_(self.RUL_layer.weight)
        
     def forward(self, sample_x):  
         #WE ONLY HAVE TO DO THIS LINE IF TRAINING! 
         #It ensures the dimensions are correct 
         #sample_x = sample_x.unsqueeze(1)
         
         ##WE ONLY HAVE TO DO THIS LINE IF TESTING
         #IT HAS TO DO SOMETHING WITH THE DIMENSIONALITIES 
         #(WHEN TESTING, WE USE ONE SAMPLE AT THE TIME, AND THEN SOMETHING GOES WRONG WITH THE DIMENSIONS)
         sample_x = sample_x.unsqueeze(0) 
         
         #Put the sample through the neural network 
         x = self.conv_one(sample_x) 
         x = self.activation_one(x) 
         
         x = self.conv_two(x)
         x = self.activation_two(x) 
         
         x = self.conv_three(x) 
         x = self.activation_three(x) 
         
         x  = self.flatten(x) 
         
         #IF WE DO DROPOUT
         #x = self.dropout(x) 
         
         x = self.fully_connected(x) 
         x = self.act_four(x) 
         
         predicted = self.RUL_layer(x) #The rul prediction 
       
         return predicted


def main_function( name_nn, name_loss_file, name_loss_graph, name_console ):
    #The big function for training our neural network!    
    
    
    computer_adress = "C:\\Users\\ingeborgdepate\\fed_learning\\"  
    
    #set the seeds (maybe we should go for one or zero haha)
    #Nice to get the same results all the time 
    np.random.seed(7042018)
    torch.manual_seed(7_04_2018) 
    
    #--------------------------------------------------------------------------------------------------#
    #------------------------------------------Get the data--------------------------------------------#
    #--------------------------------------------------------------------------------------------------#
      
    #Read in the training data 
    all_data_shortened, all_fc = read_in_data(frequency, X_v_to_keep, X_s_to_keep, True, True)   
    all_data_shortened = all_data_shortened.drop(columns = [ "hs"])    
    
    #Split the data over the clients 
    number_clients = 6 
    engine_names = np.unique(all_data_shortened.loc[:, "unit"])
    ENGINE = 2.0 
    
 
    #-------------------------------------------------------------------------------------------------=-------------#
    #------------------------------------------Normalize the training data ------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------#
    #Get the minimum and the maximum of the training data 
    minima, maxima = min_max_training(all_data_shortened.loc[all_data_shortened["unit"] == ENGINE])
    all_data_shortened_engine = normalization(all_data_shortened.loc[all_data_shortened["unit"] == ENGINE], minima, maxima) #normalize the data 
    
    
    #---------------------------------------------------------------------------------------------------------------#
    #------------------------------------------Training and validation with size------------------------------------#
    #---------------------------------------------------------------------------------------------------------------#
    #Split in a training and validation set
    #TO DO: I THOUGHT, MAYBE WE CAN INDEED SPLIT THE FLIGHTS INTO A VALIDATION AND TRAINING SET FOR FINETUNING THE HYPERPARAMETERS
    #THEN, WE CAN TRAIN THE MODEL WITH ALL DATA ONCE WE HAVE DETERMINED THE HUPERPARAMETERS
    
    #Validation and training set on flight level
    validation_size = 0.2 
    #TEST
    #TEST_two
    #Test_three
    
    #Get the name of all units    
    units = np.unique(all_data_shortened.loc[:, "unit"])
    
    #Make a dictionary with for each unit, all validation and training flights
    dict_training_flights = {} 
    dict_validation_flights = {} 
    
    #Loop over all units
    for unit in units:
        #Select the validation flights
        
        #Step 1. make a list with all glight numbers
        last_flight = int(max(all_data_shortened.loc[all_data_shortened["unit"] == unit, "cycle"])) 
        all_flights = list(range(1, last_flight + 1, 1))
        
        #Step 2. Select ... percent of the flights randomly
        validation_flights = np.random.choice(np.array(all_flights), size = math.floor(validation_size * len(all_flights)), replace=False)
        training_flights =  list(set(all_flights) - set(validation_flights)) 
        
        #Step 3. Add the flights to the dictionary
        dict_training_flights[unit] = training_flights 
        dict_validation_flights[unit] = validation_flights

    #-------------------------------------------------------------------------------------------------=-------------#
    #------------------------------------------Make the training data set with Pytorch ------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------#
    # Step 3. Get te samples     
    all_variables_x = X_v_to_keep + X_s_to_keep + all_fc  #all input variables
   
    # #Make the dataset
    # dataset = sensor_data( all_data_shortened,  stepsize_sample, all_variables_x, considered_length)
    # length_data = dataset.__len__() 
           
    # validation_size = math.floor(0.1 * length_data) 
    # train_size = math.ceil(0.9 * length_data) 
    # dataset_train, dataset_valid = random_split(dataset, [train_size, validation_size])
    # dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True) 
    # dataloader_validation  = DataLoader(dataset_valid, batch_size = batch_size, shuffle = False)
    
    # print("There are " , len(dataloader_train), " batches in the training dataloader") 
    
      
    #Dataset with training and validation
    dataset_train = sensor_data( all_data_shortened,  stepsize_sample, all_variables_x, considered_length, dict_training_flights)
    dataset_valid = sensor_data( all_data_shortened,  stepsize_sample, all_variables_x, considered_length, dict_validation_flights)
    
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True) 
    dataloader_validation  = DataLoader(dataset_valid, batch_size = batch_size, shuffle = False)
    
    print("There are " , len(dataloader_train), " batches in the training dataloader") 

    
    #-------------------------------------------------------------------------------------------------=-------------#
    #------------------------------------------Define the hyperparameters ------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------#
    input_size = len(all_variables_x) * considered_length
   
    #Make the neural network 
    neural = neural_network( input_size, height, number_channels, num_neurons, do = 0)   
    
    #Names for saving the weights, the losses and everything on the console.
    nn_path = computer_adress + name_nn +  ".pth"    
    loss_path = computer_adress + name_loss_file + ".txt" 
    console_path = computer_adress + name_console  + ".txt"
    
    #WE SHOULD PROBABLY GO FOR ANOTHER OPTIMIZER, IF WE RESET THE OPTIMIZER AFTER EACH EPOCH. THAT'S NOT GREAT WITH ADAM (SOMETHING WITH INDIVIDUAL LEARNING RATES FOR EACH WEIGHT)
    #https://towardsdatascience.com/optimizers-for-training-neural-network-59450d71caf6
    #Alternative: just Stochastic gradient descent (torch.optim.SGD(momentum = ..., nesterov = ... ))
    #Possibley with momentum or nesterov (with momentum, we do need the previous weight update value -> probably not)
    #Let's see how it is done in the code
    optimizer = torch.optim.Adam(neural.parameters(), lr=learning_rate)
    
    best_validation_loss = 1000000.  #just a big number 
    
    #To save all losses 
    all_train_losses = [] 
    all_validation_losses = [] 
    
  
    #Loop over the epochs 
    for epoch in range(0, num_epochs, 1):
        
        start_time_epochs = time.time() 
        print("We're at epoch " , epoch) 
        
        #Write the info away
        console_file = open(console_path, 'a')
        console_file.write('\n')
        console_file.write('We are at epoch ' + str(epoch))        
        console_file.close()
        
        #The training
        neural.train(True)       
        loss_train = train_one_epoch(neural, loss_function, optimizer, dataloader_train, in_training = True)
        all_train_losses.append(loss_train) 
                       
        #The validaiton 
        neural.train(False) #equiavlent with model.eval()       
        loss_validation =  train_one_epoch(neural, loss_function, optimizer, dataloader_validation, in_training = False)
        all_validation_losses.append(loss_validation)
               
        if loss_validation < best_validation_loss:
            best_validation_loss = loss_validation 
            print("we update the parameters, the best validation loss is " , best_validation_loss)
            
            #Write the info away
            console_file = open(console_path, 'a')
            console_file.write('\n')
            console_file.write("we update the parameters, the best validation loss is "  + str(best_validation_loss))       
            console_file.close()
            
            #Save the model when the validation loss improves
            torch.save(copy.deepcopy(neural.state_dict()), nn_path )                   

            
        time_it_took =  time.time() - start_time_epochs
        print("this took " , time_it_took, " seconds") 
        
        #Write the info away
        console_file = open(console_path, 'a')
        console_file.write('\n')
        console_file.write("this took " + str(time_it_took) +  " seconds")   
        console_file.close()
        
    #Write the losses away
    loss_file = open(loss_path, 'w')
    loss_file.write('\n')
    loss_file.write('Training loss') 
    loss_file.write('\n')
    loss_file.write("[")
    for loss in all_train_losses:
        loss_file.write(str(loss) + ',')
        
    loss_file.write('\n')
    loss_file.write('Validation loss') 
    loss_file.write('\n')
    loss_file.write("[")
    for loss in all_validation_losses:
        loss_file.write(str(loss) + ',')
    loss_file.close()
    
    #MAke a nice plot of the losses 
    fig, ax = plt.subplots() 
    ax.plot(all_train_losses, label = "Training loss" ) 
    ax.plot(all_validation_losses, label = "Validation loss") 
    ax.set_ylabel("Mean loss per sample")
    ax.set_xlabel("Number of epochs") 
    ax.legend() 
    plt.tight_layout()
    
    
    name_plot = computer_adress + name_loss_graph  + str(unit) + ".png"
    plt.savefig(name_plot, dpi = 400)
    plt.close('all') 
        

if __name__ == "__main__":
    #Here, we test our neural network :) 
    
    #--------------------------------------------------------------------------------------------------#
    #------------------------------------------Get the data--------------------------------------------#
    #--------------------------------------------------------------------------------------------------#  
    #Get the test data! 
    all_data_shortened, all_fc = read_in_data(frequency, X_v_to_keep, X_s_to_keep, False, True)
   
    #-------------------------------------------------------------------------------------------------=-------------#
    #------------------------------------------Normalize the data -------------------------------------------------#
    #---------------------------------------------------------------------------------------------------------------#
    #Get the training data to find the minimum and maximum 
    training_data, _ = read_in_data(frequency, X_v_to_keep, X_s_to_keep, True,True)
    minima, maxima = min_max_training(training_data)
    
    #Normalize the test data with the minimum and maximum of the training data 
    all_data_shortened = normalization(all_data_shortened, minima, maxima)
    
    #-------------------------------------------------------------------------------------------------=-------------#
    #------------------------------------------Hyperparameters - general ------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------#
    all_variables_x = X_v_to_keep + X_s_to_keep + all_fc
    
    #This should be the same as in the training loop -> maybe not great coding to define it twich 
    input_size = len(all_variables_x) * considered_length
    
    #-------------------------------------------------------------------------------------------------=-------------#
    #------------------------------------------Make the data set with Pytorch ------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------#
    #We use all flights and all data
    #Get the name of all units    
    units = np.unique(all_data_shortened.loc[:, "unit"])
    
    #Make a dictionary with for each unit, all flights
    dict_test_flights = {} 
   
    #Loop over all units
    for unit in units:
        
        #Step 1. make a list with all glight numbers
        last_flight = int(max(all_data_shortened.loc[all_data_shortened["unit"] == unit, "cycle"])) 
        all_flights = list(range(1, last_flight + 1, 1))
     
        #Step 2. Add the flights to the dictionary
        dict_test_flights[unit] = all_flights 
     
    #Make the dataset
    dataset = sensor_data(all_data_shortened, stepsize_sample, all_variables_x, considered_length, dict_test_flights)
    length_data = dataset.__len__()    
     
    #--------------------------------------------------------------------------------------------------#
    #------------------------------------------Load the Neural network--------------------------------------------#
    #--------------------------------------------------------------------------------------------------#
    neural = neural_network(input_size, height, number_channels, num_neurons, do = 0)
       
    weights = "C:\\Users\\ingeborgdepate\\fed_learning\\cnn_weights"   + ".pth"
    neural.load_state_dict(torch.load(weights))

    neural.train(False)  # equiavlent with model.eval()
   
    #--------------------------------------------------------------------------------------------------#
    #------------------------------------------Predict the RULS-------------------------------------------#
    #--------------------------------------------------------------------------------------------------#
    #Predict the RUL for each unit and each flight 
    MSE = 0 
    MAE = 0 
    points = 0 
    
    for unit in np.unique(all_data_shortened.loc[:, "unit"]):
        print("\n the test unit is " , unit)
        MSE_unit = 0 
        MAE_unit = 0 
        
        last_flight = int(max(all_data_shortened.loc[all_data_shortened["unit"] == unit, "cycle"]))      
           
        for flight in range(1, last_flight + 1, 1):
            
            mean_RUL_prediction = 0 
            
            #Get all samples
            all_samples = dataset.get_all_samples((unit,flight)) 
            num_samples = len(all_samples)
            
            #Get the RUL prediction for each sample
            for s in all_samples:
                data_sample = dataset.__getitem__(s) 
                true_RUL = data_sample[1] 
                inp = data_sample[0]
                inp = torch.tensor(inp)
                with torch.no_grad():
                    predicted = neural(inp)
                
                mean_RUL_prediction = mean_RUL_prediction + predicted[0][0]
            
            #Get the prediction and the metrics
            mean_RUL_prediction = mean_RUL_prediction / num_samples 
            error = mean_RUL_prediction - true_RUL 
            MAE_unit = MAE_unit + abs(error)
            MSE_unit = MSE_unit + error**2 
      
        #Update the grand total
        MAE = MAE + MAE_unit 
        MSE = MSE + MSE_unit 
        points = points +( last_flight  )   
        
        #Get the total MSE/MAE for the unit
        MAE_unit = MAE_unit / (last_flight )
        MSE_unit = math.sqrt(MSE_unit/(last_flight ))         
        print("For unit " , unit, " the MAE is " , MAE_unit, " and the RMSE is " , MSE_unit)
       
               
    #Get the grand total
    MAE = MAE / points
    MSE = math.sqrt(MSE/points) 
    print("In total, the MAE is " , MAE, " and the RMSE is " , MSE)
   
                
                
                
                
            
            
            
            
            
            
            
        
        
    
    
    