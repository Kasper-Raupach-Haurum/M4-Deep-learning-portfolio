# M4-Deep-learning-portfolio

This is the first assignment for the module 4: Deep learning and artificial intelligence 

For this assignment we are a group composed of Raiyan Alam, and Kasper Haurum. 
This assignment is focused on constructing a neural network, more precisely a neural network with 5 different variations whereas these variations are different epochs, learning rates, amount of hidden layers used, and inputs.

For this assignment we decided to pick the HR-data from the 1st semester, where upon the selected variables used from the HR-Dataset was the following:

 1   Age                  
 2   Attrition           
 3   MonthlyIncome      
 4   TotalWorkingYears  
 5   YearsAtCompany     

Before we went ahead, we checked for NaN/Null values, which we then removed as a part of the preliminary cleanup of the data. Likewise we also converted all values to float as we saw some of them were configured as objects. As "Attrition" was a binary value, thereby a yes/no, we turned it into a numerical [0-1] value instead.

Moving on, we ran the two selected input variables through a scaler ("MinMaxScaler"), and afterwards used the panda function pd.concat to set the new dataframe.

We detected some NaN/Null values after this was accomplished, so we once again removed these values to do a further cleanup.

Setting the tensors to be 'Age', and 'YearsAtCompany' with the inputs of data_x.size, and data_y.size being 'Attrition', we moved ahead to start setting up the neural network.

For the first ANN network, we picked a single neuron and 2 inputs to see if everything ran according to plan, which it did. This lead to the attached illustration showing the epochs, and the MSE (Mean squared error value). 

Afterwards, the next network had 2 inputs and 1 hidden layer, where we after picked a higher learning rate and epoch threshold to see it could lead to a even lower MSE score, which it did not as the lowest of the first operation had a loss.score of Loss: 0.1270068883895874, compared to Loss: 0.13029886782169342. It would thus seem to be counterintutive to make the epochs longer, as well as the learning rate.

The next operation we ran was a neural network with 2 hidden layers and 2 inputs. 

The last neural network we tested was the same configuration of 2 hidden layers, and 2 inputs, however with a learning rate of 0.01, compared to 0.001 previously, and a epoch amount of 100, compared to the previously epoch amount of 50.

This final neural network lead to a final loss score of 0.12653271853923798, but however made a sudden spike of MSE when it arrived to the 3 epoch, which could be due to the learning rate being too high.

