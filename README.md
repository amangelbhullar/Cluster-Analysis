# Clustering
#import the libraries numpy, pandas, sklearn, matplotlib, kneed

import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np

data = pd.read_csv('data.csv')
pd.set_option("display.max_columns", 500)  # Make sure we can see all of the columns
pd.set_option("display.max_rows", 10)  # Keep the output on one page
data.head()

data.shape
data.describe()
data.info()

#data_num = data._get_numeric_data()
#print(data_num)

data_fill = data.fillna(0)
print(data_fill)

X = data_fill.iloc[:, :].values

X.shape

print(X)

#scaler = StandardScaler()
#scaled_features = scaler.fit_transform(X)

kmeans = KMeans(init="random", n_clusters=3, n_init=10, max_iter=300, random_state=42)

kmeans.fit(scaled_features)

#The lowest Sum of Square Error value
kmeans.inertia_

#Final locations of the centroid
kmeans.cluster_centers_

#The number of iterations required to converge
kmeans.n_iter_

kmeans.labels_[:5]

kmeans_kwargs = {"init": "random","n_init": 10,"max_iter": 300,"random_state": 42,}

#A list holds the Sum of Square Error values for each k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_features)
    sse.append(kmeans.inertia_)
    
   
plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.title("Linear graph")
plt.xlabel("Number of Clusters")
plt.ylabel("Sum of Square Error")
plt.show()


#elbow method to find number of clusters
kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")

kl.elbow

#kmeans clustering
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(scaled_features)

#list holds the silhouette coefficients for each k
silhouette_coefficients = []
#start at 2 clusters for silhouette coefficient
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_features)
    score = silhouette_score(scaled_features, kmeans.labels_)
    silhouette_coefficients.append(score)
    
    
plt.style.use("fivethirtyeight")
plt.plot(range(2, 11), silhouette_coefficients)
plt.xticks(range(2, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()
