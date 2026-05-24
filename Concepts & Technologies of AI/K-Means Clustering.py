#!/usr/bin/env python
# coding: utf-8

# #CLUSTERING

# In[38]:


#import neccessary libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # MATLAB-like way of plotting
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D

# sklearn package for machine learning in python:
from sklearn.cluster import KMeans, MeanShift


# In[39]:


# Load the dataset
df = pd.read_csv("./datasets/country_data.csv") # read data 
print(df.head(), '\n') # print first 5 rows of data


# In[40]:


# Dropping rows with NaN values (if any)
df = df.dropna(axis=0)


# In[41]:


# Selecting columns for clustering
# Selecting two specific features: income and child mortality
X = df[['income', 'child_mort']].values


# In[42]:


# Construct the KMeans model with 4 clusters
# 4 clusters is an initial estimate to explore potential groupings in the data
model = KMeans(n_clusters=4, n_init=10, random_state=5)
model.fit(X)


# In[43]:


# Visualize the clustering result
fig, ax = plt.subplots(figsize=(10, 6))

# Normalize color encoding based on the number of clusters
nm = Normalize(vmin=0, vmax=3)

# plot the clustered data
scatter = ax.scatter(X[:, 0], X[:, 1], c=model.predict(X), s=50, cmap='plasma', norm=nm)

# Plot the centroids
centers = model.cluster_centers_
for i in range(centers.shape[0]):
    ax.text(centers[i, 0], centers[i, 1], str(i), c='black',
            bbox=dict(boxstyle="round", facecolor='white', edgecolor='black'))

# Set axis labels
ax.set_xlabel('Income')
ax.set_ylabel('Child Mortality')
ax.set_title('K-means Clustering on Income vs Child Mortality')

# Add legend
legend = ax.legend(*scatter.legend_elements(), loc="upper right", title="Clusters")
ax.add_artist(legend)

fig.savefig('Cluster_2D_Plot.png', bbox_inches='tight', pad_inches=0.1)

# Display the plot
plt.show()


# In[ ]:





# In the following code, we will include more features to see how the clustering changes

# In[44]:


# Selecting three specific features to assess how clustering varies with different indicators
# Features: life expectancy, health spending and GDP Per Capita
#using 'X1' and 'model1' to avoid overwriting x in codes above
X1 = df[['life_expec', 'health', 'gdpp']].values


# In[45]:


# Construct the KMeans model with 3 clusters to observe changes in clustering with additional features
model1 = KMeans(n_clusters=3, n_init=10, random_state=5)
model1.fit(X1)


# In[46]:


# Visualize the clustering result using a 3D plot to observe the distribution in three dimensions
fig1 = plt.figure(figsize=(10, 8))
ax1 = fig1.add_subplot(111, projection='3d')

# plot the clustered data
scatter1 = ax1.scatter(X1[:, 0], X1[:, 1], X1[:, 2], 
                     c=model1.predict(X1), s=50, cmap='plasma')


# Plot the centroids
centers1 = model1.cluster_centers_
for i in range(centers1.shape[0]):
    ax1.text(centers1[i, 0], centers1[i, 1], centers1[i, 2], 
            str(i), c='black', bbox=dict(boxstyle="round", facecolor='white', edgecolor='black'))


# Set axis labels
ax1.set_xlabel('Life Expectancy')
ax1.set_ylabel('Health Spending')
ax1.set_zlabel('GDP Per Capita')
ax1.set_title('K-means Clustering on Life Expectancy, Health Expenditure and GDP Per Capita')


# Adjusting 3D view for better clarity
ax1.azim = -60  # Azimuthal angle
ax1.dist = 8   # Distance from plot
ax1.elev = 10   # Elevation angle


# Adding a legend for cluster colors
legend1 = ax1.legend(*scatter1.legend_elements(), loc="center left", title="Clusters")
ax1.add_artist(legend1)


# Adjust layout and save the plot
fig1.tight_layout(pad=2.0)
fig1.savefig('cluster_3Dplot.png')

plt.show()


# In[ ]:


#Convert to .py file
get_ipython().system('jupyter nbconvert --to script "Task 2__2417206.ipynb"')


# In[ ]:




