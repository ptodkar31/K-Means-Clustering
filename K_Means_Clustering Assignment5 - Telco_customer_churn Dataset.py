# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 21:56:23 2023

@author: Priyanka
"""

"""
1)Business Statement:
A telecommunications company is experiencing a 
high churn rate among its customers. Churn refers 
to the situation where customers switch to a 
different service provider or terminate their 
subscription with the company.

2)Objective:
The primary objective of this project is to 
perform clustering analysis on the telecom dataset, 
which contains both categorical and numerical data

3)Constraints:Data: The analysis relies on the availability and quality
of data in the "Telco_customer_churn.xlsx" dataset.

4)Time: The project should be completed within a specified time frame.

5)Minimize:
Churn Rate: The primary objective is to minimize the churn rate, 

6)Maximize:
By understanding customer behavior and preferences,
 the company can work to maximize customer 
 satisfaction, thereby reducing churn.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

data = pd.read_excel("C:\Data Set\Telco_customer_churn.xlsx")
data
# Drop non-numeric columns and columns that are not needed for clustering
data1 = data.drop(['Offer'], axis=1)

# Apply normalization to the data
def norm_func(i):
    x = (i - i.min()) / (i.max() - i.min())
    return x

df_norm = norm_func(data1.iloc[:, 1:])

# Calculate the linkage matrix for hierarchical clustering
linkage_matrix = linkage(df_norm, method='ward')

# Plot the dendrogram
plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, p=5, truncate_mode='level')
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.show()


###k means
from sklearn.cluster import KMeans

# Specify the number of clusters
n_clusters = 3

# Create a KMeans object
kmeans = KMeans(n_clusters=n_clusters)

# Fit the KMeans object to the data
kmeans.fit(df_norm)
cluster_labels = kmeans.labels_

data['cluster'] = cluster_labels

churn_rate_by_cluster = data.groupby('cluster')['Churn'].mean()
print(churn_rate_by_cluster)

plt.scatter(data['Tenure'], data['MonthlyCharges'], c=cluster_labels, s=10)
plt.xlabel('Tenure')
plt.ylabel('MonthlyCharges')
plt.title('Customer Clustering')
plt.show()

