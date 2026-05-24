#!/usr/bin/env python
# coding: utf-8

# # LINEAR REGRESSION

# In[102]:


#Import neccessary libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # MATLAB-like way of plotting
from mpl_toolkits.mplot3d import Axes3D

# sklearn package for machine learning in python:
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, mean_absolute_percentage_error, r2_score


# In[103]:


df = pd.read_csv("./datasets/houseprice_data.csv") # read data


# In[104]:


# Calculate the correlation matrix to help us select features

# Calculate and display the correlation matrix
correlation_matrix = df.corr()
correlation_matrix.style.background_gradient(cmap='coolwarm')


# The features with the highest correlations to Price (in order of correlation strength) and their corresponding index numbers are:
# 
# sqft_living	(Index: 3, Correlation: 0.702044)
# grade (Index: 9, Correlation: 0.667463)
# sqft_above (Index: 10, Correlation: 0.605566)
# sqft_living15 (Index: 17, Correlation: 0.585374)

# In[105]:


print(df.head(),'\n') # print first 5 rows of data


# In[ ]:





# In[106]:


# Select features and target variable
X = df.iloc[:, [3]].values # inputs(we are using sqft_living which is at column 3)
y = df.iloc[:, 0].values # target (price, which is at column 0)


# In[107]:


# split the data into training and test sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/5, random_state = 5)


# In[108]:


# fit the linear least-squares regression line to the training data:
regr = LinearRegression()
regr.fit(X_train, y_train)


# In[109]:


# The coefficients
print('Coefficients: ', regr.coef_)

# The coefficients
print('Intercept: ', regr.intercept_)

# The mean squared error
print('Mean squared error: %.8f'
% mean_squared_error(y_test, regr.predict(X_test)))

# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
% r2_score(y_test, regr.predict(X_test)))


# In[110]:


# Scatter plot of the training data
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', alpha=0.5, label='Training Data')

# Scatter plot of the test data
plt.scatter(X_test, y_test, color='green', alpha=0.5, label='Test Data')

# Plot the regression line
plt.plot(X_train, regr.predict(X_train), color='red', linewidth=2, label='Regression Line')

# Add titles and labels
plt.title('Linear Regression on Square Footage vs. Price')
plt.xlabel('Square Footage of Living Area')
plt.ylabel('Price ($)')
plt.legend()

# Show the plot
plt.show()


# In[ ]:





# In[111]:


# we will explore different features to evaluate their potential impact on model performance.
# The new input and target variables will be named X1 and y1
# to reflect that they are a different feature set.

# Select features: Using 'sqft_living' (column 3) and 'grade' (column 9) as inputs
X1 = df.iloc[:, [3, 9]].values  # Inputs (sqft_living and grade)

# Select target variable: Using 'price' (column 0)
y1 = df.iloc[:, 0].values  # Target (price)


# In[112]:


# Feature set for the new model
# split the data into training and test sets:
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 1/5, random_state = 5)

# The commented code below increases the training data to see if we get a more accurate model. 
# X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.1, random_state=5)
# The result was less accurate wih a Coefficient of Determination (R^2) of 0.52


# In[113]:


# fit the linear least-squares regression line to the training data:
# Fit the new model (using new features)
regr1 = LinearRegression()
regr1.fit(X1_train, y1_train)


# In[114]:


# The coefficients
print('Coefficients: ', regr1.coef_)

# The coefficients
print('Intercept: ', regr1.intercept_)


mse = mean_squared_error(y1_test, regr1.predict(X1_test))
print('Mean Squared Error: %.8f' % mse)


# Mean Absolute Error (MAE)
mae = mean_absolute_error(y1_test, regr1.predict(X1_test))
print('Mean Absolute Error (MAE): %.2f' % mae)


# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print('Root Mean Squared Error (RMSE): %.2f' % rmse)

# Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(y1_test, regr1.predict(X1_test))
print('Mean Absolute Percentage Error (MAPE): %.2f%%' % (mape * 100))



# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'% r2_score(y1_test, regr1.predict(X1_test)))


# In[115]:


# Visualize initial data set for new features in 3D
fig1 = plt.figure(figsize=(12, 10))  # Descriptive name for the figure
ax1 = fig1.add_subplot(111, projection='3d')  # Descriptive name for the axes

# Scatter plot: Use the new features and target variable
ax1.scatter(X1[:, 0], X1[:, 1], y1, color='blue', alpha=0.5, label='Data Points')

# Adjusting the viewing angle
ax1.azim = -60
ax1.dist = 10
ax1.elev = 20

# Set axis limits based on the full dataset
ax1.set_xlim(X1[:, 0].min(), X1[:, 0].max())  # X axis limit (square footage)
ax1.set_ylim(X1[:, 1].min(), X1[:, 1].max())  # Y axis limit (grade)
ax1.set_zlim(y1.min(), y1.max())              # Z axis limit (price)

# Set the title and labels
ax1.set_title('3D Scatter Plot of Initial Dataset (New Feature Set)')
ax1.set_xlabel('Square Footage of Living Area')
ax1.set_ylabel('Grade')
ax1.set_zlabel('Price')

# Add legend
ax1.legend()

# Adjust layout using subplots_adjust
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

# Save the figure
fig1.savefig('LR1_initial_plot.png')  # Save with a descriptive name

# Show the plot
plt.show()


# In[116]:


# Visualize training set results in 3D
fig2 = plt.figure(figsize=(12, 10))
ax2 = fig2.add_subplot(111, projection='3d')

# Plot the training data
ax2.scatter(X1_train[:, 0], X1_train[:, 1], y1_train, color='blue', label='Training Data')

# Create a meshgrid for the regression plane based on the new features
X1, X2 = np.meshgrid(
    np.linspace(X1_train[:, 0].min(), X1_train[:, 0].max(), 100),
    np.linspace(X1_train[:, 1].min(), X1_train[:, 1].max(), 20)
)

# Calculate Z values for the regression plane based on the regr1 model
Z = regr1.coef_[0] * X1 + regr1.coef_[1] * X2 + regr1.intercept_

# Plot the regression plane
ax2.plot_surface(X1, X2, Z, alpha=0.5, color='red')

# Set view angles and axis limits based on the training data
ax2.azim = -60
ax2.dist = 10
ax2.elev = 20
ax2.set_xlim(X1_train[:, 0].min(), X1_train[:, 0].max())
ax2.set_ylim(X1_train[:, 1].min(), X1_train[:, 1].max())
ax2.set_zlim(y1_train.min(), y1_train.max())

# Set titles and labels
ax2.set_title('3D Scatter Plot of Training Set Results')
ax2.set_xlabel('Square Footage of Living Area (sqft)')
ax2.set_ylabel('Grade')
ax2.set_zlabel('Price ($)')

# Adjust layout, add legend, and save the figure
fig2.tight_layout(pad=-2.0)
ax2.legend()
fig2.savefig('LR_train_plot.png')

# Show the plot
plt.show()


# In[117]:


# Visualize test set results in 3D
fig3 = plt.figure(figsize=(12, 10))
ax3 = fig3.add_subplot(111, projection='3d')

# Plot the test data for the original features
ax3.scatter(X1_test[:, 0], X1_test[:, 1], y1_test, color='blue', label='Test Data')

# Create a meshgrid for the regression plane based on the test data
X1_, X2_ = np.meshgrid(
    np.linspace(X1_test[:, 0].min(), X1_test[:, 0].max(), 100),  # For sqft_living
    np.linspace(X1_test[:, 1].min(), X1_test[:, 1].max(), 20)    # For grade
)

# Calculate Z values based on the regression coefficients from the regr1 model
Z = regr1.coef_[0] * X1_ + regr1.coef_[1] * X2_ + regr1.intercept_

# Plot the regression plane
ax3.plot_surface(X1_, X2_, Z, alpha=0.5, color='red')

# Set view angles and limits
ax3.azim = -60
ax3.dist = 10
ax3.elev = 20

# Set axis limits based on the test data
ax3.set_xlim(X1_test[:, 0].min(), X1_test[:, 0].max())  # X1 limits (sqft_living)
ax3.set_ylim(X1_test[:, 1].min(), X1_test[:, 1].max())  # X2 limits (grade)
ax3.set_zlim(y1_test.min(), y1_test.max())              # Z limits (price)

# Set titles and labels
ax3.set_title('Test Set Results with Regression Plane')
ax3.set_xlabel('Square Footage of Living Area (sqft)')
ax3.set_ylabel('Grade')
ax3.set_zlabel('Price ($)')

# Adjust layout and save the figure
fig3.tight_layout(pad=-2.0)
ax3.legend()
fig3.savefig('LR_test_plot.png', bbox_inches='tight', pad_inches=0.1)

# Show the plot
plt.show()


# # Purpose of model below: To evaluate the model's performance over multiple random data splits (100 runs) without a fixed random state.
# # By averaging the accuracy metrics over these runs, we aim to get a more reliable estimate of the model's performance, 
# # minimizing the effect of any single train-test split's variance on the final accuracy metrics.

# In[118]:


# Select features and target variable from the dataset
X1 = df.iloc[:, [3, 9]].values  # Inputs: 'sqft_living' and 'grade'
y1 = df.iloc[:, 0].values       # Target: 'price'

# Initialize lists to store performance metrics for each run
mse_list = []
mae_list = []
rmse_list = []
r2_list = []

# Run the model training and evaluation process 100 times with different data splits
for _ in range(100):
    # Randomly split the dataset into training and testing sets (no random_state set for varying splits)
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2)
    
    # Initialize and train the linear regression model on the current training set
    regr1 = LinearRegression()
    regr1.fit(X1_train, y1_train)
    
    # Make predictions on the test set
    y1_pred = regr1.predict(X1_test)
    
    # Calculate regression metrics for the current model
    mse = mean_squared_error(y1_test, y1_pred)
    mae = mean_absolute_error(y1_test, y1_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y1_test, y1_pred)
    
    # Store each metric in the respective list for averaging later
    mse_list.append(mse)
    mae_list.append(mae)
    rmse_list.append(rmse)
    r2_list.append(r2)

# Calculate the average of each metric across the 100 runs
average_mse = np.mean(mse_list)
average_mae = np.mean(mae_list)
average_rmse = np.mean(rmse_list)
average_r2 = np.mean(r2_list)

# Print out the average performance metrics to assess the model's general accuracy
print(f'Average Mean Squared Error (MSE): {average_mse:.8f}')
print(f'Average Mean Absolute Error (MAE): {average_mae:.2f}')
print(f'Average Root Mean Squared Error (RMSE): {average_rmse:.2f}')
print(f'Average Coefficient of Determination (R^2): {average_r2:.2f}')


# In[122]:


#Convert to .py file
get_ipython().system('jupyter nbconvert --to script "Task 1__2417206.ipynb"')


# In[ ]:




