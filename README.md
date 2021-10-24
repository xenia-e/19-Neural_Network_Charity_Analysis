# Neural_Network_Charity_Analysis

# Purpouse

The purpose of this project is to create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

### Methods
We use the following methods for the analysis:

- preprocessing the data for the neural network model,
- compile, train and evaluate the model,
- optimize the model.


## Deliverables

__**Deliverable 1:**__ Preprocessing Data for a Neural Network Model [(Jupyter notebook)](https://github.com/xenia-e/19-Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity.ipynb)

__**Deliverable 2:**__ Compile, Train, and Evaluate the Model [(Jupyter notebook)](https://github.com/xenia-e/19-Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity.ipynb)

__**Deliverable 3:**__ Optimize the Model [(Jupyter notebook)](https://github.com/xenia-e/19-Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity_Optimzation.ipynb)

__**Deliverable 4:**__ A Written Report on the Neural Network Model [(README.md)](https://github.com/xenia-e/19-Neural_Network_Charity_Analysis/blob/main/README.md)



# Overview

## Data Preprocessing

We considered IS_SUCCESSFUL as the target for the model

APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT are considered to be the features for the model.

"EIN" and "NAME" are neither targets nor features and were removed from the input data.

Encoding of the categorical variables, splitting into training and testing datasets, and standardization have been applied to the features.

## Compiling, Training, and Evaluating the Model

The neural network model is made of two hidden layers with 80 and 30 neurons respectively. We are using ReLU as an activation function.

The output layer is made of a unique neuron with Sigmoid as activation function as it is a binary classification

Adam was used as an optimizer and the loss function is binary_crossentropy.

![model structure](https://github.com/xenia-e/19-Neural_Network_Charity_Analysis/blob/main/Analysis/model_structure.png)

>Figure 1 - Model Structure

>Results
```
--------------------------------------------------------------
268/268 - 0s - loss: 0.5567 - accuracy: 0.7255
Loss: 0.5567498803138733, Accuracy: 0.7254810333251953
--------------------------------------------------------------
```
The model accuracy is under 75%. This is not a satisfying model performance.

To increase the performance of the model, we applied the next steps:

1. Bucketing to the feature ASK_AMT and organizing the different values by intervals.
2. Combine the rare values of INCOME_AMT in the 'Other' category.
3. Add a new hidden layer.
4. Increase the number of neurons on one of the hidden layers

These steps led to an insignificant increase in model performance and loss.

```
--------------------------------------------------------------
268/268 - 1s - loss: 0.5708 - accuracy: 0.7279
Loss: 0.5708379745483398, Accuracy: 0.7279300093650818
--------------------------------------------------------------
```

We also tried to drop STATUS and SPECIAL_CONSIDERATIONS columns, used different activation functions (tanh). None of these steps helped improve the model's performance and some lower performance.


# Summary

It is safe to assume that the model is not outperforming, considering that the target level of accuracy (75%) was not reached.

We tried using a supervised machine learning model (Random Forest Classifier) and evaluated its performance against our deep learning model. The model takes less time and gives almost the same predictive accuracy: - 72,6%

![Random forest evaluation](https://github.com/xenia-e/19-Neural_Network_Charity_Analysis/blob/main/Analysis/randomforest.png)

>Figure 1 - Random forest model

