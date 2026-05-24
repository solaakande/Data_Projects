#!/usr/bin/env python
# coding: utf-8

# #NEURAL NETWORKS

# In[71]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # MATLAB-like way of plotting
from matplotlib.colors import Normalize
from matplotlib.colors import ListedColormap

# sklearn package for machine learning in python:
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# In[72]:


# read data
df = pd.read_csv("./datasets/nba_rookie_data.csv")

print(df.head(),'\n') # print first 5 rows of data


# In[73]:


# Calculate the correlation matrix to help us select features
# First select only numeric columns
numeric_df = df.select_dtypes(include=['number'])

# Move 'TARGET_5Yrs' to the first position
columns = ['TARGET_5Yrs'] + [col for col in numeric_df.columns if col != 'TARGET_5Yrs']
numeric_df = numeric_df[columns]

# Calculate and display the correlation matrix
correlation_matrix = numeric_df.corr()
correlation_matrix.style.background_gradient(cmap='coolwarm')


# The features with the highest correlations to TARGET_5Yrs (in order of correlation strength) and their corresponding index numbers are:
# 
# Games Played (Index: 1, Correlation: 0.397)
# Minutes Played (Index: 2, Correlation: 0.318)
# Field Goals Made (Index: 4, Correlation: 0.318)
# Points Per Game (Index: 3, Correlation: 0.316)
# Rebounds (Index: 15, Correlation: 0.299)
# Free Throw Made (Index: 10, Correlation: 0.297)
# Free Throw Attempts (Index: 11, Correlation: 0.296)
# Offensive Rebounds (Index: 13, Correlation: 0.293)
# Field Goal Attempts (Index: 5, Correlation: 0.293)
# Defensive Rebounds (Index: 14, Correlation: 0.285)

# In[ ]:





# LET'S BEGIN BUILDING OUR MODEL.
# Starting with One Feature

# In[74]:


X = df.iloc[:,[1]].values #Using column [Games Played] - The feature with the highest correlation
y = df.iloc[:,20].values #y is ['TARGET_5Yrs']


# In[75]:


# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=5)


# In[76]:


# setup the neural network architecture
mlp = MLPClassifier(hidden_layer_sizes=(50,30,20), activation="tanh" ,random_state=5, max_iter=2000)
mlp.fit(X_train, y_train)


# In[77]:


# performance metrics
print('Our Accuracy is %.2f' % mlp.score(X_test, y_test))
print('Out of a total %d points : %d mislabeled' % (X_test.shape[0], (y_test != mlp.predict(X_test)).sum()))


# In[78]:


#Visualize Results
fig0, ax0 = plt.subplots()

# Scatter plot of the actual target values for comparison
ax0.scatter(X_test, y_test, color = 'blue', label='Actual')

# Scatter plot of the model's binary class predictions
ax0.scatter(X_test, mlp.predict(X_test), color = 'red', marker = '*', label='Predicted')

# Scatter plot of the model's predicted probabilities for the positive class (lasting 5 years)
ax0.scatter(X_test, mlp.predict_proba(X_test)[:,1], color='green', marker='.', label='Predicted Probabilities')

ax0.set_xlabel('X')
ax0.set_ylabel('y')

# Adding labels and title
ax0.set_xlabel('Games Played')
ax0.set_ylabel('Career Longevity (0 = <5 Years, 1 = 5+ Years)')
ax0.set_title('Neural Network Predictions for NBA Rookie Career Longevity Using Games Played')
ax0.legend(loc='best')


# In[ ]:





# EXPLORING WITH TWO FEATURES TO SEE IF ACCURACY INCREASES
# We will use 'X1' and 'y1' to avoid overwriting our previous data inorder to compare models

# In[79]:


X1 = df.iloc[:,[1, 2]].values #Using columns [Games Played and Minutes Played] Top 2 features with highest correlation
y1 = df.iloc[:,20].values #y is ['TARGET_5Yrs']

# split the data
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=1/3, random_state=5)


# In[80]:


# setup the neural network architecture
mlp1 = MLPClassifier(hidden_layer_sizes=(50,30,20), activation="tanh" ,random_state=5, max_iter=2000)
mlp1.fit(X1_train, y1_train)


# In[81]:


# performance metrics
print('Our Accuracy is %.2f' % mlp1.score(X1_test, y1_test))
print('Out of a total %d points : %d mislabeled' % (X1_test.shape[0], (y1_test != mlp1.predict(X1_test)).sum()))


# In[83]:


# Visualize the model
fig, ax = plt.subplots()

# Set up a mesh to plot the contour of the model
x_min, x_max = X1[:, 0].min() - 0.05, X1[:, 0].max() + 0.05
y_min, y_max = X1[:, 1].min() - 0.05, X1[:, 1].max() + 0.05

h = 0.05  # Step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Model predicts every point in the mesh and reshapes the array for plotting
Z = mlp1.predict(np.column_stack([xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

# Set up the color and symbol encoding
nm = Normalize(vmin=0, vmax=1)
cm = ListedColormap(['blue', 'red'])
m = {'o', 's'}

# Contour plot of the model
ax.contourf(xx, yy, Z, cmap=cm, norm=nm, alpha=0.5)

# Plot the data
for i in range(len(X_test)):
    ax.scatter(X1_test[i,0], X1_test[i,1], marker = 'o', c = y1_test[i], cmap = cm, norm = nm, s = 10)

# Find the misclassified points
mis_ind = np.where(y1_test != mlp1.predict(X1_test))[0]
#print('Misclassified Points:\n', X_test[mis_ind], y_test[mis_ind])

# Plot the misclassified points
ax.scatter(X1_test[mis_ind, 0], X1_test[mis_ind, 1], marker='*', c='white', s=2)  # Increased size for visibility

# Create a legend from the scatter plot
legend0 = ax.legend(*scatter0.legend_elements(), loc="center left", title="TARGET_5Yrs")

ax.add_artist(legend0)

# Set labels
ax.set_xlabel('Games Played')
ax.set_ylabel('Minutes Played')
ax.set_title("Neural Network Prediction of NBA Rookie Career Longevity Using Games and Minutes Played")


# Save the figure
fig.savefig('ANN_test_plot.png', bbox_inches='tight', pad_inches=0.1)


# In[ ]:





# EXPLORING WITH MORE FEATURES TO SEE IF ACCURACY INCREASES

# In[84]:


# Select features with the highest correlations to 'TARGET_5Yrs'
X2 = df.iloc[:, [1, 2, 4, 3, 15, 10, 11, 13, 5, 14]].values  # Features based on their column indexes
y2 = df.iloc[:, 20].values  # Target variable 'TARGET_5Yrs'

# split the data
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=1/3, random_state=5)


# In[85]:


# setup the neural network architecture
mlp2 = MLPClassifier(hidden_layer_sizes=(50,30,20), activation="tanh" ,random_state=5, max_iter=2000)
mlp2.fit(X2_train, y2_train)


# In[86]:


# performance metrics
print('Our Accuracy is %.2f' % mlp2.score(X2_test, y2_test))
print('Out of a total %d points : %d mislabeled' % (X2_test.shape[0], (y2_test != mlp2.predict(X2_test)).sum()))


# In[ ]:





# In[87]:


# Using the feature set [Games Played, Minutes Played], which previously achieved the highest accuracy of 0.68.
# Running the model without a fixed random state over 100 iterations to calculate the average accuracy,
# aiming to evaluate the model’s stability and generalization performance across varied data splits.
# Store accuracies across iterations
accuracies = []

# Run the model 100 times
for _ in range(100):
    # Split the data without a fixed random state
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=1/3)
    
    # Initialize the MLP model (without setting random_state)
    mlp1 = MLPClassifier(hidden_layer_sizes=(50, 30, 20), activation="tanh", max_iter=2000)
    
    # Fit the model
    mlp1.fit(X1_train, y1_train)
    
    # Calculate accuracy and store it
    accuracy = mlp1.score(X1_test, y1_test)
    accuracies.append(accuracy)

# Calculate the average accuracy over 100 runs
average_accuracy = np.mean(accuracies)
print(f'Average Accuracy over 100 runs: {average_accuracy:.2f}')


# In[ ]:


#Convert to .py file
get_ipython().system('jupyter nbconvert --to script "Task 3_Neural Networks__2417206.ipynb"')


# In[ ]:




