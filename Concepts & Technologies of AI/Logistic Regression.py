#!/usr/bin/env python
# coding: utf-8

# LOGISTIC REGRESSION

# In[172]:


# Import necessary libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # MATLAB-like way of plotting

# sklearn package for machine learning in python:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.naive_bayes import GaussianNB


# In[173]:


# read data
df = pd.read_csv("./datasets/nba_rookie_data.csv")

print(df.head(),'\n') # print first 5 rows of data


# In[174]:


# Calculate the correlation matrix to help us select features
# First select only numeric columns
numeric_df = df.select_dtypes(include=['number'])

# Move 'TARGET_5Yrs' to the first position
columns = ['TARGET_5Yrs'] + [col for col in numeric_df.columns if col != 'TARGET_5Yrs']
numeric_df = numeric_df[columns]

# Calculate and display the correlation matrix
correlation_matrix = numeric_df.corr()
correlation_matrix.style.background_gradient(cmap='coolwarm')


# In[175]:


# Select feature with high correlation to player performance
# We will start with one feature
X = df[['Games Played']].values

# Set the target variable to predict if a player's career length is 5 years or less (1 = Yes, 0 = No)
y = df['TARGET_5Yrs'].values


# In[176]:


# Feature scaling: normalize each feature to have a mean of 0 and standard deviation of 1
# Standardizing helps logistic regression by ensuring features contribute equally to the model

scaler = StandardScaler()
X = scaler.fit_transform(X)  # Apply scaling to X; each feature now has mean 0 and std deviation 1


# In[177]:


# Split the data into training and test sets for evaluating model performance
# test_size=1/3 indicates that 1/3 of the data will be used for testing

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=5)


# In[178]:


# Initialize the Logistic Regression model and fit it to the training data
# Logistic regression is used here for binary classification (TARGET_5Yrs: 1 or 0)

logre = LogisticRegression()
logre.fit(X_train, y_train)  # Train the model on the scaled data


# In[179]:


# Output the accuracy of the model on the test data
# This score represents the percentage of correct predictions out of total predictions
accuracy = logre.score(X_test, y_test)
print(f"Our Accuracy is {accuracy:.2f}")

# Calculate the number of mislabeled points (incorrect predictions)
mislabeled_points = (y_test != logre.predict(X_test)).sum()
print(f"Number of mislabeled points out of a total {X_test.shape[0]} points : {mislabeled_points}")


# In[180]:


# Visualization: plot the predictions, probabilities, and actual values for insight into model performance
fig, ax = plt.subplots(figsize=(10, 6))

# Scatter plot of true labels
ax.scatter(X_test, y_test, color='blue')

# Scatter plot of model's binary class predictions
ax.scatter(X_test, logre.predict(X_test), color='red', marker='*')

# Scatter plot of model's predicted probabilities for class "lasting 5 years"
ax.scatter(X_test, logre.predict_proba(X_test)[:,1], color='green', marker='.')

# Labels
ax.set_xlabel('Standardized Games Played (X)')
ax.set_ylabel('TARGET_5Yrs (y)')

# Save the plot
fig.savefig('Class_plot.png')


# Show plot
plt.show()


# We will now extend to two features to see how our model performs

# In[181]:


# Extend to two features: "Games Played" and the second-best feature, "Minutes Played"
# Select the two features for input data
#using 'X1' and 'y1' to avoid overwriting x in codes above
X1 = df[['Games Played', 'Minutes Played']].values
X1 = StandardScaler().fit_transform(X1)


# In[182]:


# Split the data into training and testing sets
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y, test_size=1/3, random_state=5)


# In[183]:


# Train logistic regression model with two features
logre1 = LogisticRegression()
logre1.fit(X1_train, y1_train)


# In[184]:


# Output the accuracy of the model on the test data for two features
accuracy1 = logre1.score(X1_test, y1_test)
print(f"Two-Feature Model Accuracy: {accuracy1:.2f}")

# Calculate the number of mislabeled points (incorrect predictions) for two features
mislabeled_points1 = (y1_test != logre1.predict(X1_test)).sum()
print(f"Number of mislabeled points out of a total {X1_test.shape[0]} points : {mislabeled_points1}")


# In[185]:


# 3D visualization for the two features
# Set up 3D plot
fig1 = plt.figure(figsize=(10, 8))
ax1 = fig1.add_subplot(111, projection='3d')

# Plot actual values in 3D space
ax1.scatter(X1_test[:, 0], X1_test[:, 1], y1_test, color='blue', label="Actual (y1_test)", alpha=0.5)

# Plot model's binary class predictions in 3D space
ax1.scatter(X1_test[:, 0], X1_test[:, 1], logre1.predict(X1_test), color='red', marker='*', label="Predicted (logre1.predict)", alpha=0.5)

# Plot model's predicted probabilities in 3D space
ax1.scatter(X1_test[:, 0], X1_test[:, 1], logre1.predict_proba(X1_test)[:, 1], color='green', marker='.', label="Predicted Probabilities", alpha=0.5)

# Set labels and title
ax1.set_xlabel('Standardized Games Played')
ax1.set_ylabel('Standardized Minutes Played')
ax1.set_zlabel('TARGET_5Yrs')
ax1.set_title('3D Logistic Regression Model Predictions using "Games Played" and "Minutes Played"')
ax1.legend(loc="upper left")

# Save the plot
fig1.savefig('Class_plot1.png')

# Show plot
plt.show()


#  We will now chnage features to see if model improves, we will use 3 features here

# In[186]:


# Extend to three features: "Games Played", "Points Per Game" and "Rebounds"
#using 'x2', 'y2' to avoid overwriting x in codes above

X2 = df[['Games Played', 'Points Per Game', 'Rebounds']].values
X2 = StandardScaler().fit_transform(X2)


# In[187]:


# Split the data into training and testing sets
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=1/3, random_state=5)


# In[188]:


logre2 = LogisticRegression()
logre2.fit(X2_train, y2_train)


# In[189]:


# Output the accuracy of the model on the test data for three features
accuracy2 = logre2.score(X2_test, y2_test)
print(f"Three-Feature Model Accuracy: {accuracy2:.2f}")

# Calculate the number of mislabeled points (incorrect predictions) for three features
mislabeled_points2 = (y2_test != logre2.predict(X2_test)).sum()
print(f"Number of mislabeled points out of a total {X2_test.shape[0]} points : {mislabeled_points2}")


# In[190]:


# This code evaluates the robustness of the logistic regression model by running it 100 times without a fixed random state
# Each run splits the data randomly, trains the model on three features ("Games Played", "Points Per Game", "Rebounds"), 
# and calculates the accuracy on the test set. The average accuracy across these runs gives an indication of the model's 
# stability and performance over multiple data splits.

# Select and scale the features: "Games Played", "Points Per Game", "Rebounds"
X2 = df[['Games Played', 'Points Per Game', 'Rebounds']].values
X2 = StandardScaler().fit_transform(X2)
y = df['TARGET_5Yrs'].values  # Assuming the target variable is labeled as 'Target'

# Initialize list to store accuracy scores for each run
accuracy_scores = []

# Run the model training and evaluation process 100 times with different data splits
for _ in range(100):
    # Randomly split the dataset into training and testing sets (no random_state set for varying splits)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=1/3)
    
    # Initialize and train the logistic regression model on the current training set
    logre2 = LogisticRegression()
    logre2.fit(X2_train, y2_train)
    
    # Calculate accuracy on the test set and add to the list of scores
    accuracy = logre2.score(X2_test, y2_test)
    accuracy_scores.append(accuracy)

# Calculate the average accuracy across the 100 runs
average_accuracy = np.mean(accuracy_scores)

# Output the average accuracy and an example of the model's stability over multiple runs
print(f"Average Model Accuracy over 100 runs: {average_accuracy:.2f}")


# In[ ]:





# In[191]:


#Convert to .py file
get_ipython().system('jupyter nbconvert --to script "Task 3_Logistic__2417206.ipynb"')


# In[ ]:




