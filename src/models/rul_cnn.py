#####################################################
#-----------------------Packages--------------------#
#####################################################
import torch
import numpy as np
import pandas as pd
import time
import h5py
import math
import copy 
import os
import logging
import sys

from typing import List, Dict, Tuple
from torch.utils.data import Dataset, random_split, DataLoader
from pandas import DataFrame
from torch import nn
 
import config

logger_console = logging.getLogger(__name__)
logger_console.propagate = False 

logger_loss = logging.getLogger(f"{__name__}.loss")
logger_loss.propagate = False 

handler_stream_console = logging.StreamHandler(sys.stdout)
handler_file_console = logging.FileHandler('logs/console.log', mode='a')
handler_file_loss = logging.FileHandler('logs/loss.log', mode='a')

handler_stream_console.setLevel(logging.INFO)
handler_file_console.setLevel(logging.INFO)
handler_file_loss.setLevel(logging.INFO)

format_console = logging.Formatter('[%(levelname)s]: %(name)s : %(message)s')
format_file = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

handler_stream_console.setFormatter(format_console)
handler_file_console.setFormatter(format_file)
handler_file_loss.setFormatter(format_file)

logger_console.addHandler(handler_stream_console)
logger_console.addHandler(handler_file_console)
logger_loss.addHandler(handler_stream_console)
logger_loss.addHandler(handler_file_loss)

logger_console.info("=== New Execution ===")
logger_loss.info("=== New Execution ===")

stepsize_sample = 10
considered_length = 50

height = 9
number_channels = 10     
num_neurons = 100
frequency =  1

X_v_to_keep =  ["W21", "W50", "SmFan", "SmLPC", "SmHPC"]  
X_s_to_keep = ["Wf", "Nf", "T24", "T30", "T48", "T50", "P2", "P50"] 

loss_function = torch.nn.MSELoss()  
batch_size = 128
num_epochs = 50
learning_rate = 0.001

def read_in_data(
    filename: str,
    frequency: int, 
    X_v_to_keep: List[str], 
    X_s_to_keep: List [str], 
    training_data: bool = True, 
    keep_all_data: bool = True
) -> Tuple[pd.DataFrame, List[str]]: 
    """Read data from a raw data h5 file. 

    Args: 
        frequency: by how many samples the original dataset is to be aggregated. 
        X_v_to_keep: list of virutal sensor names to be kept from the dataset.
        X_s_to_keep: list of real sensor names to be kept from the dataset.
        training_data: boolean indicating whether the training or the testing
            dataset is to be read in.
        keep_all_data: boolean indicating whether to include only the healthy
            flights from the dataset.
    """

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

        # column names
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

    df_X_s = df_X_s[X_s_to_keep]
    df_X_v = df_X_v[X_v_to_keep]

    df_X_s["unit"] = df_A["unit"].values
    df_X_s["cycle"] = df_A["cycle"].values
    df_X_s["hs"] = df_A["hs"].values

    all_data = [df_X_s, df_X_v, df_W]
    all_data = pd.concat(all_data, axis=1)

    if keep_all_data == False:
        all_data = all_data.loc[all_data["hs"] == 1]  
     
    if frequency > 1:
        all_data_shortened = pd.DataFrame(columns = all_data.columns)     
        for unit in np.unique(all_data["unit"]) :  
             data_unit = all_data.loc[all_data["unit"] == unit]
             for flight in np.unique(data_unit["cycle"]):              
                data_flight = data_unit.loc[data_unit["cycle"] == flight]
                data_flight.reset_index(inplace = True)
                               
                means = (
                    data_flight
                    .groupby(
                        np.arange(len(data_flight)) // frequency
                    )
                    .mean()
                )
                all_data_shortened = pd.concat([all_data_shortened, means], axis = 0)                        
        del all_data
        all_data_shortened = all_data_shortened.drop(columns = ["index"]) 
        return all_data_shortened, W_var

    return all_data, W_var 
    

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

        data.loc[:, column] = (
            (data.loc[:, column] - minimum)*(u-l)/(maximum-minimum)+l
        )
    return data   
   

class TurbofanSimulationDataset(Dataset):
    """Turbofan Engine Simulation dataset class. 

    Due to the size of the simulation, this class reduces the sampling
    granularity, and augments the resulting data. For a single row of the
    dataset, the input data is considered to be a fixed size sample of a flight
    for a unit, and the label is the RUL (in number of flights). 
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

        self.sample_index = []
        self.unit_flight_all_samples = {}

        for unit in np.unique(all_data["unit"]):
            data_engine = all_data.loc[all_data["unit"] == unit, :]
            flights = np.unique(data_engine["cycle"])
            nr_flights = len(flights)

            self.unit_flight_all_samples[unit] = {}
            for flight in flights:
                if flight not in considered_flights[unit]:
                    continue
                data_flight = data_engine.loc[data_engine["cycle"] == flight]
                nr_samples = data_flight.shape[0]
                nr_streams = (nr_samples - considered_length)//stepsize_sample + 1
                for i in range(nr_streams):
                    self.sample_index.append((
                        data_flight.index[0]+i*stepsize_sample, 
                        nr_flights - flight,
                    ))
                self.unit_flight_all_samples[unit][flight] = [
                    len(self.sample_index)-nr_streams,
                    len(self.sample_index),
                ]
        self.pre_processing()

    def pre_processing(self): 
        minimum, maximum = min_max_training(self.all_data)
        self.all_data = normalization(self.all_data, minimum, maximum)

    def __len__(self):
        """Length of the initialized Dataset.

        This dunder function is called by the Dataloader class to get the size
        of the dataset.
        """

        return len(self.sample_index)
    
    def get_all_samples(self, sample):
        """Get all measurement streams for a (unit, flight) tuple."""

        u, f = sample
        return range(*self.unit_flight_all_samples[u][f])
   
    def __getitem__(self, idx):
        """Get a single entry from the dataset. 

        Given `idx`, this function should return the entry at the position
        specified by `idx`. 
        """

        start, RUL = self.sample_index[idx]
        sample_x = self.all_data.loc[
            (start):(start + self.considered_length-1), 
            self.all_variables_x
        ]
        sample_x = sample_x.to_numpy()
        sample_x = np.float32(sample_x)
        return (
            torch.from_numpy(sample_x).unsqueeze(0), 
            np.float32(RUL)
        )


class EngineSimulationDataset(TurbofanSimulationDataset): 


    def __init__(self, engine, *args, **kwargs): 
        self.engine = engine
        super().__init__(*args, **kwargs)

    def pre_processing(self): 
        start_idx = min([start 
            for [start, _] in self.unit_flight_all_samples[self.engine].values()
        ])
        end_idx = max([end 
            for [_, end] in self.unit_flight_all_samples[self.engine].values()
        ])
        self.index_range = (start_idx, end_idx) 
        self.all_data = self.all_data.loc[self.all_data["unit"] == self.engine, :].copy()
        super().pre_processing()

    def get_all_samples(self, flight): 
        [flight_start, flight_end] = self.unit_flight_all_samples[self.engine][flight]
        return range(
            flight_start - self.index_range[0], 
            flight_end - self.index_range[0],
        )

    def __getitem__(self, idx): 
        return super().__getitem__(self.index_range[0]+idx)

    def __len__(self): 
        return self.index_range[1] - self.index_range[0]


def train_one_epoch(neural_network, loss_function, optimizer, data_loader, in_training):
    running_loss = 0.
    num_batches = len(data_loader)
    for i, (sample_x, RUL) in enumerate(data_loader):
        logger_console.info(f"Batch {i}/{num_batches}")     
        if in_training == False:
            with torch.no_grad():                
                predicted = neural_network(sample_x)
                predicted = predicted.squeeze(1)
                loss = loss_function(RUL, predicted)
                running_loss = running_loss + loss.item() 
        else:
            predicted = neural_network(sample_x)
            predicted = predicted.squeeze(1)
            loss = loss_function(RUL, predicted)
            running_loss = running_loss + loss.item() 
            optimizer.zero_grad()            
            loss.backward()                
            optimizer.step()
    return running_loss


class neural_network(nn.Module):


    def __init__(self, input_size, height, number_channels, num_neurons, do = 0):
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
        self.conv_one = nn.Conv2d(
            in_channels=1, 
            out_channels=self.number_maps_first,
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            padding="same",
        )
        #Activation of this layer (possible hyperparameter)
        self.activation_one = nn.ReLU()
        #Initialization of the weights (this is standard for the ReLU activation function) WITH DISTRIBUTED LEARNING: HAD TO LOOK INTO THIS!
        nn.init.kaiming_normal_(self.conv_one.weight)
        
        self.conv_two = nn.Conv2d(
            in_channels=self.number_maps_first,
            out_channels=self.number_maps_first, 
            kernel_size=self.kernel_size,
            stride=self.stride, 
            padding="same",
        )
        self.activation_two = nn.ReLU()
        nn.init.kaiming_normal_(self.conv_two.weight)
        
        self.conv_three = nn.Conv2d(
            in_channels=self.number_maps_first, 
            out_channels=1, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            padding="same",
        )
        self.activation_three = nn.ReLU()
        nn.init.kaiming_normal_(self.conv_three.weight)
        
         #Flatten the output 
        self.flatten = nn.Flatten() 
        
        #IF WE WOULD DO DROPOUT -> PROBABLY HERE
        #MAYBE NOT NECESSARY: TOO SLOW FOR MANY ITERATIONS? 
        #self.dropout = nn.Dropout(do)
        
        #Add a fully connected layer
        self.fully_connected = nn.Linear(
            in_features=self.input_size, 
            out_features=self.num_neurons, 
            bias=True,
        ) 
        self.act_four = nn.ReLU()
        nn.init.kaiming_normal_(self.fully_connected.weight)

        #Predict tthe reul
        self.RUL_layer = nn.Linear(in_features = self.num_neurons, out_features = 1, bias = True) 
        nn.init.kaiming_normal_(self.RUL_layer.weight)
        
    def forward(self, sample_x):  
        x = self.conv_one(sample_x) 
        x = self.activation_one(x) 

        x = self.conv_two(x)
        x = self.activation_two(x) 

        x = self.conv_three(x) 
        x = self.activation_three(x) 

        x  = self.flatten(x) 

        x = self.fully_connected(x) 
        x = self.act_four(x) 

        predicted = self.RUL_layer(x) #The rul prediction 

        return predicted


def main_function(name_nn, name_loss_file, name_loss_graph, name_console):
    np.random.seed(7042018)
    torch.manual_seed(7_04_2018) 

    all_data_shortened, all_fc = read_in_data(
        "data/raw/turbofan_simulation/data_set2/N-CMAPSS_DS02-006.h5", 1, X_v_to_keep, X_s_to_keep, True, True
    )   
    all_data_shortened = all_data_shortened.drop(columns = ["hs"])    
    
    ENGINE = int(os.getenv("ENGINE", "2.0"))
    
    validation_size = 0.2 
    units = np.unique(all_data_shortened.loc[:, "unit"])
    
    dict_training_flights = {} 
    dict_validation_flights = {} 

    for unit in units:
        last_flight = int(max(all_data_shortened.loc[all_data_shortened["unit"] == unit, "cycle"])) 
        all_flights = list(range(1, last_flight + 1, 1))
        
        validation_flights = np.random.choice(
            np.array(all_flights), size=math.floor(validation_size * len(all_flights)), replace=False
        )
        training_flights =  list(set(all_flights) - set(validation_flights)) 
        
        dict_training_flights[unit] = training_flights 
        dict_validation_flights[unit] = validation_flights

    all_variables_x = X_v_to_keep + X_s_to_keep + all_fc
   
    dataset_train = EngineSimulationDataset(ENGINE, all_data_shortened, stepsize_sample, all_variables_x, considered_length, dict_training_flights)
    dataset_valid = EngineSimulationDataset(ENGINE, all_data_shortened, stepsize_sample, all_variables_x, considered_length, dict_validation_flights)
    
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True) 
    dataloader_validation  = DataLoader(dataset_valid, batch_size = batch_size, shuffle = False)
    
    input_size = len(all_variables_x) * considered_length
   
    neural = neural_network(input_size, height, number_channels, num_neurons, do = 0)   
    
    nn_path = "trained/" + name_nn +  ".pth"    
    
    optimizer = torch.optim.Adam(neural.parameters(), lr=learning_rate)
    
    best_validation_loss = None
    
    all_train_losses = [] 
    all_validation_losses = [] 

    for epoch in range(0, num_epochs, 1):

        start_time_epochs = time.time() 
        logger_console.info(f"Epoch {epoch}/{num_epochs}")

        neural.train(True)       
        loss_train = train_one_epoch(
            neural, loss_function, optimizer, dataloader_train, in_training=True
        )
        all_train_losses.append(loss_train) 
                       
        neural.train(False)
        loss_validation = train_one_epoch(
            neural, loss_function, optimizer, dataloader_validation, in_training=False
        )
        all_validation_losses.append(loss_validation)

        if (best_validation_loss == None) or (loss_validation < best_validation_loss):
            best_validation_loss = loss_validation 
            logger_console.info(f"Update parameters. "
                        f"Best validation loss: {best_validation_loss}")
            torch.save(copy.deepcopy(neural.state_dict()), nn_path )                   

        time_it_took =  time.time() - start_time_epochs
        logger_console.info(f"Epoch duration: {time_it_took}") 

    logger_loss.info('Training loss') 
    logger_loss.info(all_train_losses)
    logger_loss.info('Validation loss') 
    logger_loss.info(all_validation_losses)
    
main_function("cnn_weights", "Losses", "Graph", "console")
