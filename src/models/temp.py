#####################################################
#-----------------------Packages--------------------#
#####################################################
import tqdm
import torch
import numpy as np
import pandas as pd
import time
import h5py
import math
import copy 
import os

from typing import List, Dict
from torch.utils.data import Dataset, random_split, DataLoader
from pandas import DataFrame
from torch import nn
 
import config

#####################################################
#-----------------------Parameters--------------------#
#####################################################
#THE STEPSIZE WILL PROABBLY BE 1, BUT A LARGER NUMBER IS A LOT FASTER FOR TESTING THE CODE
stepsize_sample = 10 #with what steps do we make our samples (1 is: from 1 to 50, from 2 to 51, etc.)
considered_length = 50 #Length of one sample.

#THESE OARAMETERS PROBABLY HAVE TO CHANGE WITH A FREQUENCY OF 1 AND ALL SENSORS. 
height = 9
number_channels = 10     
num_neurons = 100

#Define the required frequency 
#THE FREQUENCY WILL BE 1, BUT THAT IS VERY SLOW IN TESTING THE CODE-> USING SOME BIGGER VALUE WHEN TESTING THE CODE MIGHT HELP :) 
frequency = 60#1 is much smallerr than I had before, so we might have to change some parameters 

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

def read_in_data(frequency, X_v_to_keep, X_s_to_keep, training_data=True, keep_all_data=True): #True if you want the training data, False if you want the testing data 
    # Set-up - Define file location
    filename = "data/raw/turbofan_simulation/data_set2/N-CMAPSS_DS02-006.h5"
    
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

def normalization(data, minima, maxima, skip = ["in_training", "cycle", "unit" , "hs"]):        
    #THis function normalizes the sensor measurements b
    for column in data.columns:
        if column in skip:
            continue
        # normalize between -1 and 1
        u = 1
        l = -1
        minimum = minima.get(column)
        maximum = maxima.get(column)

        data.loc[:, column] = (
            (data.loc[:, column] - minimum)*(u-l)/(maximum-minimum)+l
        )
    return data
   

class sensor_data(Dataset):
    """Turbofan Engine Simulation dataset class. 

    Due to the size of the simulation, this class reduces the sampling
    granularity, and augments the resulting data. The input data is considered
    to be a fixed size sample of a flight for a unit, and the output is the RUL
    (in number of flights). 

    """
    

    def __init__(
        self, 
        all_data: pd.DataFrame, 
        stepsize_sample: int, 
        all_variables_x: List[str], 
        considered_length: int, 
        considered_flights: Dict[int, List[int]],
    ):
        """Initialize the TurbofanSimulation Dataset class. 

        Args:
            all_data: Dataframe with the flight samples at the
                rate of 1 sample/second
            stepsize_sample: size of the step between 2 consecutive sample
                streams. E.g., if a sample stream has a length of 50 samples,
                and the stepsize is of 10 samples, the first stream would
                consider the samples 0-49, and the second stream, the samples
                10-59.
            all_variables_x: A list of sensor identifiers (virtual +
                physical) which are to be considered. Having performed a
                statistical analysis as to which sensors correlate to the RUL,
                this parameter specifies the list of these sensors.
            considered_length: The number of samples to be taken into
                consideration in a single sample stream. E.g., if this value is
                50, then the first stream of a given flight will hold the first
                50 samples (0-49).
            considered_flights: Dictionary holding for each unit, the list of
                flights which are to be taken into consideration from the whole
                DataFrame in the initialized Dataset. 

        """
        self.all_data = all_data
        self.all_variables_x = all_variables_x
        self.considered_length = considered_length

        self.sample_index = {} 
        self.unit_flight_all_samples = {} 
        self.unit_length = {} 
        number_sample = 0     

        for unit in np.unique(all_data["unit"]):           
            data_unit = all_data.loc[all_data["unit"] == unit]
            number_flights = len(np.unique(data_unit["cycle"]))
            self.unit_length[unit] = number_flights
            flights_unit = considered_flights[unit]

            for flight in np.unique(data_unit["cycle"]):     
                if flight not in flights_unit:
                    continue
                all_samples = []
                data_flight = data_unit.loc[data_unit["cycle"] == flight]
                data_flight.reset_index(inplace=True)
                length_of_flight = data_flight.shape[0]

                start = 0
                while start <= (length_of_flight - self.considered_length):
                    end = start + self.considered_length
                    all_info = (unit, flight, start, end)
                    self.sample_index[number_sample] = all_info
                    all_samples.append(number_sample)
                    number_sample = number_sample + 1
                    start = start + stepsize_sample
                
                self.unit_flight_all_samples[(unit, flight)] = all_samples 

    def __len__(self):
        """Length of the initialized Dataset.

        This dunder function is called by the Dataloader class to get the size
        of the dataset.
        """

        return len(self.sample_index)
    
    def get_all_samples(self, sample):
        """Get all measurement streams for a unit, flight tuple."""

        return self.unit_flight_all_samples.get(sample)
   
    def __getitem__(self, idx):
        """Get a single entry from the dataset. 

        Given `idx`, this function should return the entry at the position
        specified by `idx`. 
        """

        info = self.sample_index.get(idx)
        unit = info[0]
        flight = info[1]
        start = info[2]
        end = info[3]

        data_considered = self.all_data.loc[
            (self.all_data["unit"] == unit) 
            & (self.all_data["cycle"] == flight)
        ]
        sample = data_considered.iloc[start:end] 
        
        sample_x = sample[self.all_variables_x]
        sample_x = sample_x.to_numpy()
        sample_x = np.float32(sample_x)
        
        number_flight = self.unit_length.get(unit)
        RUL = number_flight - flight
        RUL = np.float32(RUL)
        if RUL < 0:
            print("error: the RUL is " , RUL, " with flight " , flight , " and number flight " , number_flight)
            raise ValueError("The RUL is negative!") 

        return (
            torch.from_numpy(sample_x).unsqueeze(0),
            torch.from_numpy(np.array(np.float32(RUL))).unsqueeze(0)
        )

    
def train_one_epoch(neural_network, loss_function, optimizer, data_loader, in_training):
    #This functions trains the neural network for one epoch :)  
    running_loss = 0.#The loss (nice to return in the end)
    
    # Iterate over the batches in the epochs
    loss_sum = 0
    len_dataset = len(data_loader.dataset)
    for sample_x, RUL in tqdm.tqdm(
            data_loader,
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}'
        ):  #i is the batch number 
            
        if in_training == False: #For the validation set (then, we don't update the gradients, but just calculate the error)
            with torch.no_grad():                
                predicted = neural_network(sample_x)
                loss_sum += torch.sum((RUL - predicted)**2).item() 
        else:
            #Make the predictions 
            optimizer.zero_grad()            
            predicted = neural_network(sample_x)
            #Calculate the loss 
            loss = loss_function(predicted, RUL)
      
            #Update the gradients
            loss.backward()                
            optimizer.step()
           
    if in_training == False:
        MSE = loss_sum/len_dataset
        RMSE = math.sqrt(MSE)
        print(f"MSE: {MSE} \t RMSE: {RMSE}")
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
        self.activation_one = nn.ReLU(inplace=True)
        #Initialization of the weights (this is standard for the ReLU activation function) WITH DISTRIBUTED LEARNING: HAD TO LOOK INTO THIS!
        
        self.conv_two = nn.Conv2d(in_channels = self.number_maps_first, out_channels = self.number_maps_first, kernel_size = self.kernel_size, stride = self.stride, padding = "same")
        self.activation_two = nn.ReLU(inplace=True)
        
        self.conv_three = nn.Conv2d(in_channels = self.number_maps_first, out_channels = 1, kernel_size = self.kernel_size, stride = self.stride, padding = "same")
        self.activation_three = nn.ReLU(inplace=True)
        
         #Flatten the output 
        self.flatten = nn.Flatten() 
        
        #IF WE WOULD DO DROPOUT -> PROBABLY HERE
        #MAYBE NOT NECESSARY: TOO SLOW FOR MANY ITERATIONS? 
        #self.dropout = nn.Dropout(do)
        
        #Add a fully connected layer
        self.fully_connected = nn.Linear(in_features = self.input_size, out_features = self.num_neurons, bias = True) 
        self.act_four = nn.ReLU(inplace=True)
        
        #Predict tthe reul
        self.RUL_layer = nn.Linear(in_features = self.num_neurons, out_features = 1, bias = True) 
        
        for m in self.modules(): 
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity = "relu")
                assert m.bias != None # mypy
                nn.init.constant_(m.bias, 0)

     def forward(self, sample_x):  
         #WE ONLY HAVE TO DO THIS LINE IF TRAINING! 
         #It ensures the dimensions are correct 
         #sample_x = sample_x.unsqueeze(1)
         
         ##WE ONLY HAVE TO DO THIS LINE IF TESTING
         #IT HAS TO DO SOMETHING WITH THE DIMENSIONALITIES 
         #(WHEN TESTING, WE USE ONE SAMPLE AT THE TIME, AND THEN SOMETHING GOES WRONG WITH THE DIMENSIONS)
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


def main_function(name_nn, name_loss_file, name_loss_graph, name_console):
    #set the seeds (maybe we should go for one or zero haha)
    #Nice to get the same results all the time 
    np.random.seed(7042018)
    torch.manual_seed(7_04_2018) 

    #Read in the training data 
    all_data_shortened, all_fc = read_in_data(frequency, X_v_to_keep, X_s_to_keep, True, True)   
    all_data_shortened = all_data_shortened.drop(columns = ["hs"])    
    
    #Split the data over the clients 
    number_clients = 6 
    engine_names = np.unique(all_data_shortened.loc[:, "unit"])
    ENGINE = int(os.getenv("ENGINE", "0"))

    #-------------------------------------------------------------------------------------------------=-------------#
    #------------------------------------------Normalize the training data ------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------#
    #Get the minimum and the maximum of the training data 
    
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

    # all_data_shortened["in_training"] = False    
    # for unit, flights  in dict_validation_flights.items():
        # for flight in flights: 
            # all_data_shortened.loc[
                # (all_data_shortened["unit"]==unit) & (all_data_shortened["cycle"]==flight), 
                # "in_training"
            # ] = True

    train_minima, train_maxima = min_max_training(
        all_data_shortened # .loc[all_data_shortened["in_training"]==True, :]
    )
    train_all_data_shortened = normalization(all_data_shortened, train_minima, train_maxima) #normalize the data 
    valid_minima, valid_maxima = min_max_training(
        all_data_shortened # .loc[all_data_shortened["in_training"]==False, :]
    )
    valid_all_data_shortened = normalization(all_data_shortened, valid_minima, valid_maxima) #normalize the data 
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
    dataset_train = sensor_data(train_all_data_shortened, stepsize_sample, all_variables_x, considered_length, dict_training_flights)
    dataset_valid = sensor_data(valid_all_data_shortened, stepsize_sample, all_variables_x, considered_length, dict_validation_flights)
    
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True) 
    dataloader_validation  = DataLoader(dataset_valid, batch_size = batch_size, shuffle=True)
    
    print("There are " , len(dataloader_train), " batches in the training dataloader") 

    
    #-------------------------------------------------------------------------------------------------=-------------#
    #------------------------------------------Define the hyperparameters ------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------#
    input_size = len(all_variables_x) * considered_length
   
    #Make the neural network 
    neural = neural_network( input_size, height, number_channels, num_neurons, do = 0)   
    
    #Names for saving the weights, the losses and everything on the console.
    nn_path = "models/nn.pth"
    loss_path = "logs/loss.log"
    console_path = "logs/console.log"
    
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
        loss_validation = train_one_epoch(neural, loss_function, optimizer, dataloader_validation, in_training = False)
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
    
main_function("cnn_weights", "Losses", "Graph", "console")
