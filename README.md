# M4-Deep-learning-portfolio


# Assignment 1: Build, train, and evaluate a neural network with Pytorch.

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


# Assignment 2: Build, train, and evaluate 2 special types of networks neural network (CNN & LSTM) with Pytorch

In this second assignment we will be creating two different neural networks, first a CNN network for spatial prediction, and secondly a LSTM network for sequential data handling

In the first network we selected pol_tweet data from our first semester when we worked on NLP (Natural langauge processing) where we had a dataset on tweets made by US Ctizens largely about the topic of their domestic politics, with one camp being Republican and the other Democrat. In this neural network we want to make a network that can handle the spatial prediction based on the data of the tweets made, where one variation will focus on the republican tweets, and the other the tweets made by democrats.

To accomplish this we used the Wandb AI performance visualisation tool to aid in doing this, whereupon we selected the tensors and the train/test data.

For the republican tweets we selected the word classes being 'trump', 'forthepeople', 'climate', 'black', 'crisis', 'democracy', 'heroesact', 'china', 'democrats', 'inittogether'

The overall prediction accuracy for the Republicans was 56.09 %, with some words being predicted at a higher rating (Biden: 69.6 %), and others at a lower rating (Paycheckprotectionprogram: 28.9 %). This may be due to the fact some words are more prominent in the data sample which will as a result increase the accuracy rating as it leads to the train/test size being likewise more filled with said word(s).

For the democrat tweets we selected the word classes being 'biden', 'inittogether', 'dems', 'paycheckprotectionprogram', 'obama', 'congress', 'latinx', 'wellfare', 'diversity', 'lgbtq'

The overall prediction accuracy for the Democrats was 56.09 %, with some words being predicted at a higher rating (Biden: 69.6 %), and others at a lower rating (Paycheckprotectionprogram: 28.9 %). This may be due to the fact some words are more prominent in the data sample which will as a result increase the accuracy rating as it leads to the train/test size being likewise more filled with said word(s).

The overall prediction accuracy for the Republicans was 54.58 %, with some words being predicted at a higher rating (Trump: 74.5%), and others at a lower rating (Democracy: 28.9 %). Again the potential reason for the differance in the rating can be the higher frequency of some words, and visa versa with a lower frequency of others.

For the second task which is making a RNN & LSTM sequential network we decided to select Tesla stock trading data as our dataset, with said dataset tracking the stock activity of Tesla in the period of 2015-2017.

From the variables in the dataset, we select the "Date" and "Volume" as our chosen variables we want to base our X/Y data on, as we want to see if we can make a accurate neural network on sequential data, and nothing is more sequential then stocks as it is a continuous fluctuation of ups and downs. 

In the first run of the LSTM network we made it with the following model parameters: input_size = 5, hidden_layer = 16, output_size = 1, and num_epochs= 10

Running the model with 1000 epochs, we created a illustration displaying the MSE loss measured against the number of epochs. In the case of this model, the optimal amount of epochs used would be somewhere around 20 as it goes linear afterwards with a very horizontal line.

When measuring it against the backdrop of the actual trading volume it tracked it somewhat accurately, however it is evident that it likely would not be as accurate enough to predict the real motions taking place on the trading floor as far Tesla stocks goes.

For the next model run we decided to set the output size of the model being, as compared to 1 previously, and let the number of epochs that the model used itself be changed to 50, as compared to 10 previously

Running the model this time it had a much more curved line when displaying the MSE score lore against the epochs, with the optimal amount of epochs being 600 in this case. 

When tracking it against the real trading of Tesla stock this time, it would seem it followed the trading more accurately, which may due to the fact it had a Test MSE 0.0067035122723343495 score as compared against the former model's Test MSE score of 0.00702314528987687, thus having a slight improvement over the other model.






