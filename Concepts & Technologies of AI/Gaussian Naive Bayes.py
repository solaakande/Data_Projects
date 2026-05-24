#!/usr/bin/env python
# coding: utf-8

# # GAUSSIAN NAIVE BAYES

# In[150]:


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


# In[151]:


# read data
df = pd.read_csv("./datasets/nba_rookie_data.csv")

print(df.head(),'\n') # print first 5 rows of data


# In[152]:


# Calculate the correlation matrix to help us select features
# First select only numeric columns
numeric_df = df.select_dtypes(include=['number'])

# Move 'TARGET_5Yrs' to the first position
columns = ['TARGET_5Yrs'] + [col for col in numeric_df.columns if col != 'TARGET_5Yrs']
numeric_df = numeric_df[columns]

# Calculate and display the correlation matrix
correlation_matrix = numeric_df.corr()
correlation_matrix.style.background_gradient(cmap='coolwarm')


# We will start with one feature and work up to more features

# In[153]:


# Select feature and target for Gaussian Naive Bayes
X = df[['Field Goals Made']].values
y = df['TARGET_5Yrs'].values


# In[154]:


# Scale the features for improved model performance
scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[155]:


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=5)


# In[156]:


# Initialize and fit the Gaussian Naive Bayes model
gnb = GaussianNB()
gnb.fit(X_train, y_train)


# In[157]:


# Output the number of mislabeled points and accuracy
print('Number of mislabeled points out of a total of %d points: %d'
      % (X_test.shape[0], (y_test != gnb.predict(X_test)).sum()))
print('Our accuracy is %.2f:' % gnb.score(X_test, y_test))

# Predict the outcome for a sample input:  [60 field goals made]
scaled_input = scaler.transform([[60]])  # Scale the input value of 60
y_pred = gnb.predict(scaled_input)
print('Predict a value for 60 field goals made:', y_pred)


# In[158]:


# Visualise the model
fig, ax = plt.subplots(figsize=(10, 6))

# Scatter plot of the actual target values for comparison
ax.scatter(X_test, y_test, color='blue', label='Actual (y_test)')

# Scatter plot of the model's binary class predictions
ax.scatter(X_test, gnb.predict(X_test), color='red', marker='*', label='Predicted (gnb.predict(X_test))')

# Scatter plot of the model's predicted probabilities for the positive class (lasting 5 years)
ax.scatter(X_test, gnb.predict_proba(X_test)[:,1], color='green', marker='.', label='Predicted Probabilities')

# Labels and title
ax.set_xlabel('Field Goals Made (X)')
ax.set_ylabel('TARGET_5Yrs (y)')
ax.set_title('Gaussian Naive Bayes Model Predictions with "Field Goals Made"')

# Save the plot
fig.savefig('Class_plot.png')

# Show plot
plt.show()


# In[169]:


#Trying more features to see improvement
# Select features and target for Gaussian Naive Bayes
#using 'X1' and 'y1' to avoid overwriting x in codes above
X1 = df[['Games Played', 'Minutes Played']].values
y1 = df['TARGET_5Yrs'].values


# In[170]:


# Scale the features for improved model performance
scaler1 = StandardScaler()
X1 = scaler1.fit_transform(X1)


# In[171]:


# Split the data into training and test sets
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=1/3, random_state=5)


# In[172]:


# Initialize and fit the Gaussian Naive Bayes model
gnb1 = GaussianNB()
gnb1.fit(X1_train, y1_train)


# In[173]:


# Output the number of mislabeled points and accuracy in the requested format
print('Number of mislabeled points out of a total of %d points: %d'
      % (X1_test.shape[0], (y1_test != gnb1.predict(X1_test)).sum()))
print('Our accuracy is %.2f:' % gnb1.score(X1_test, y1_test))

# Predict the outcome for a sample input:  [60 Games Played, 20 Minutes Played]
y1_pred = gnb1.predict([[60, 20]])
print('Predict a value:', y1_pred)


# In[174]:


# Set up 3D plot
fig1 = plt.figure(figsize=(12, 8))
ax1 = fig1.add_subplot(111, projection='3d')

# Plot actual values in 3D space
ax1.scatter(X1_test[:, 0], X1_test[:, 1], y1_test, color='blue', label="Actual (y1_test)", alpha=0.5)

# Plot model's binary class predictions in 3D space
ax1.scatter(X1_test[:, 0], X1_test[:, 1], gnb1.predict(X1_test), color='red', marker='*', label="Predicted (gnb_model)", alpha=0.5)

# Plot model's predicted probabilities in 3D space
ax1.scatter(X1_test[:, 0], X1_test[:, 1], gnb1.predict_proba(X1_test)[:, 1], color='green', marker='.', label="Predicted Probabilities", alpha=0.5)

# Set labels and title
ax1.set_xlabel('Games Played')
ax1.set_ylabel('Minutes Played')
ax1.set_zlabel('TARGET_5Yrs')
ax1.set_title('3D Gaussian Naive Bayes Model Predictions using "Games Played", "Minutes Played"')
ax1.legend(loc="upper left")

#save the plot
fig1.savefig('GNB_test_plot.png', bbox_inches='tight', pad_inches=0.1)

# Show plot
plt.show()


# In[175]:


# Using multiple features to see if there will be improvements
# Select features and target for Gaussian Naive Bayes
#using 'X2' and 'y2' to avoid overwriting x in codes above
X2 = df[['Games Played', 'Minutes Played', 'Points Per Game', 'Field Goals Made', 'Field Goal Attempts']].values
y2 = df['TARGET_5Yrs'].values


# In[176]:


# Scale the features for improved model performance
scaler2 = StandardScaler()
X2 = scaler2.fit_transform(X2)


# In[177]:


# Split the data into training and test sets
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=1/3, random_state=5)


# In[178]:


# Initialize and fit the Gaussian Naive Bayes model
gnb2 = GaussianNB()
gnb2.fit(X2_train, y2_train)


# In[179]:


# Output the number of mislabeled points and accuracy in the requested format
print('Number of mislabeled points out of a total of %d points: %d'
      % (X2_test.shape[0], (y2_test != gnb2.predict(X2_test)).sum()))
print('Our accuracy is %.2f:' % gnb2.score(X2_test, y2_test))

# Predict the outcome for a sample input:  [60 Games Played, 40 Minutes Played, 20 Pointes Per Game, 10 Field Goals Made, 5 Field Goal Attemps]
y2_pred = gnb2.predict([[60, 40, 20, 10, 5]])
print('Predict a value:', y2_pred)


# In[ ]:





# In[ ]:


#Convert to .py file
get_ipython().system('jupyter nbconvert --to script "Task 3_GNB__2417206.ipynb"')


# In[ ]:




