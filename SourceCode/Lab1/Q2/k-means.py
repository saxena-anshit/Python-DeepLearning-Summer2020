# Our goal is to cluster our customers into buying groups based off of their Annual Income and Spending Scores

# Report which K is the best using the elbow method.

# Evaluate  with  silhouette  score  or  other  scores  relevant  for  unsupervised  approaches
# (before applying clustering clean the data set if needed)

# Can you interpret the clustering result that you have visualized?

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics

# read in data
data = pd.read_csv('Customers.csv')

# checked for null values, none are found
nulls = pd.DataFrame(data.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls, '\n')

# data to be focused on
X = data[['Annual Income (k$)', 'Spending Score (1-100)']].iloc[:, :]
print(X.head(5), '\n')

# look at data, easy to see 5 groupings already
sns.FacetGrid(X, height=6).map(plt.scatter, 'Annual Income (k$)', 'Spending Score (1-100)').add_legend()
plt.title("LOOK AT DATA")
plt.show()

# elbow graph to determine k
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, max_iter=300, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Graph')
plt.xlabel('Clusters')
plt.ylabel('Inertia')
plt.show()

# 5 clusters to be found
nclusters = 5
seed = 0

# training data
km = KMeans(n_clusters=nclusters)
km.fit(X)

# evaluating data -> 0.5 indicates healthy spread of clusters
# closer to 1 means data in clusters is tightly dense
# close to -1 indicates incorrect spreading
y_cluster_kmeans = km.predict(X)
score = metrics.silhouette_score(X, y_cluster_kmeans)
print('The silhouette score is -> ', score)

# place results into data to see which row belongs to which cluster
X['results'] = y_cluster_kmeans
sns.FacetGrid(X, hue="results", height=4).map(plt.scatter, 'Annual Income (k$)', 'Spending Score (1-100)').add_legend()
plt.title("X CLUSTERED")
plt.show()

# based on the result it appears that clustering was an efficient method of grouping the data in
# an unsupervised approach as the silhouette score showed positive results and the resulting graph
# showed clear differentiation between the groups correlated to annual income and spending score
