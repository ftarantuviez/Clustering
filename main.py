
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title='Clustering', page_icon="./f.png")
st.title('Clustering')
st.subheader('By [Francisco Tarantuviez](https://www.linkedin.com/in/francisco-tarantuviez-54a2881ab/) -- [Other Projects](https://franciscot.dev/portfolio)')
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.write('---')
st.write("""
The purpose of this app is use different clustering techniques to predict if a patient has cancer or not. Usually you use clustering algorithms when you don't have data previously labeled, but in this case the data has target labels. However, the idea of the project is not precisely build the cancer predictor, but understand how the different clustering techniques works.

The project consist in predict if the patient has cancer or if does not have cancer.The data that we are gonna use is provided by Scikit-learn in their package called: "load_breast_cancer". \n
**WARNING**: The idea of this post is not explain deeply how clustering works. However, in case you are interested on know more about these magnificent mathematical advances, I'll give you some resources to visit at the bottom of this page.
""")
st.sidebar.header("Custom models")

# Importing Libraries
st.write(""" 
## Importing Libraries

First we have to import the libraries and packages that we are gonna use for this project. Numpy and Pandas to basic data manipulation, sklearn for our dataset and another useful machine learning algorithms and finally from scipy different functions for hierarchical clustering.
""")
st.code(""" 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
""")

# / Importing Libraries

# Loading Data
st.write(""" 
## Loading Data
We have to load our dataset and separate features from the target. In this case we have 30 different features and the target variable is a binary column, which means when target = 1, the patient has cancer. In the other hand, if target = 0, patient has not cancer.
""")
st.code("""
data = load_breast_cancer()
X = data.data
y = data.target
""")
data = load_breast_cancer()
X = data.data
y = data.target

st.dataframe(pd.DataFrame(X, columns=data.feature_names))
# / Loading Data

# KMeans
st.write(""" 
## K-Means

K-Means is a very common clustering algorithm in Machine Learning. How it works is looking for the $k$ nearest datapoints according to a previously specify distance measure. Having that distance of the $k$ datapoints, it re-compute the location of the centroid to measure the next $k$ datapoints distance.
This is why K-Means is a centroid-based clustering algorithm.
In the last section I provided you some links to understand better this kind of algorithms :)

We are gonna initialize the algorithm setting $k$ = 2 because we know the classification is binary: 1 or 0. You can change different parameters of the model in the left sidebar of this application.
Also we will use Principal Component Analysis (PCA) to dimensionality reduction, with the only purpose of visualize the cluster in a chart.
""")

st.code(""" 
km = KMeans(n_clusters=2)
km.fit(X)
labels = km.labels_
centers = km.cluster_centers_

pca = PCA(n_components=2)
bc_pca = pca.fit_transform(X)
""")

st.sidebar.write("""
---
### K-Means

**This may take a while**
""")
init_k = st.sidebar.selectbox("Init", ["k-means++", "random"])
n_init = st.sidebar.slider("Number of inits", 2, 20, value=8)


km = KMeans(n_clusters=2, init=init_k, n_init=n_init, random_state=0)
km.fit(X)
labels = km.labels_
centers = km.cluster_centers_
pca = PCA(n_components=2)
bc_pca = pca.fit_transform(X)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
fig.suptitle("Visualizing breas cancer clusters")
fig.subplots_adjust(top=.85, wspace=.5)
ax1.set_title("Actual Labels")
ax2.set_title("Clustered Labels")

for i in range(len(y)):
  if y[i] == 0:
    c1 = ax1.scatter(bc_pca[i, 0], bc_pca[i, 1], c="g", marker=".")
  if y[i] == 1:
    c2 = ax1.scatter(bc_pca[i, 0], bc_pca[i, 1], c="r", marker=".")
  if labels[i] == 0:
    c3 = ax2.scatter(bc_pca[i, 0], bc_pca[i, 1], c="g", marker=".")
  if labels[i] == 1:
    c4 = ax2.scatter(bc_pca[i, 0], bc_pca[i, 1], c="r", marker=".")

l1 = ax1.legend([c1, c2], ["0", "1"])
l2 = ax2.legend([c3, c4], ["0", "1"])
st.pyplot(fig)

st.dataframe(pd.DataFrame(pd.Series(accuracy_score(y_pred=labels, y_true=y)), columns=["Accuracy"]))

# / KMeans

# Linkage

st.write(""" 
## Linkage

Single-linkage clustering is one of several methods of hierarchical clustering. It is based on grouping clusters in bottom-up fashion (agglomerative clustering), at each step combining two clusters that contain the closest pair of elements not yet belonging to the same cluster as each other.

We initialize as follow (you can change parameters from left sidebar):
""")
st.code(""" 
Z = linkage(X, method="ward", metric='euclidean')
""")
st.sidebar.write("""
---
### Linkage
""")
method = st.sidebar.selectbox("Method", ["ward", "single", "complete", "average", "centroid"])
metric = st.sidebar.selectbox("Label", ["euclidean", "minkowski", "cityblock", "cosine"])

Z = linkage(X, method=method, metric=metric)
plt.figure(figsize=(8, 3))
plt.title("Hierarchical Clustering Dendogram")
plt.xlabel("Data Point")
plt.ylabel("Distance")
dendrogram(Z)
plt.axhline(y=10000, c="k", ls="--", lw=.5)
plt.xticks([])
st.pyplot()

st.write(""" 
And then we 'parse' the results of Linkage into labels (predictions), setting a max distance of 1000:
""")
st.code(""" 
max_dist = 10000
hc_labels = fcluster(Z, max_dist, criterion="distance")
hc_labels_converted = [0 if x == 1 else 1 for x in hc_labels]
metrics.accuracy_score(hc_labels_converted, y)
""")

max_dist = 10000
hc_labels = fcluster(Z, max_dist, criterion="distance")
hc_labels_converted = [0 if x == 1 else 1 for x in hc_labels]
st.dataframe(pd.DataFrame(pd.Series(accuracy_score(y_pred=hc_labels_converted, y_true=y)), columns=["Accuracy"]))
# / Linkage

st.write(""" 
## Resources of Clustering Algorithms:
* [sklearn.cluster.KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
* [Clustering | Types Of Clustering | Clustering Applications](https://www.analyticsvidhya.com/blog/2016/11/an-introduction-to-clustering-and-different-methods-of-clustering/#:~:text=Clustering%20is%20the%20task%20of,and%20assign%20them%20into%20clusters.)
* [StatQuest: K-means clustering](https://www.youtube.com/watch?v=4b5d3muPQmA)
* [12. Clustering](https://www.youtube.com/watch?v=esmzYhuFnds)

And on Twitter, [svpino](https://twitter.com/svpino) can guide you better than anyone to understand how to start with Machine Learning and AI.
""")

# This app repository

st.write("""
## App repository

[Github](https://github.com/ftarantuviez/Clustering)
""")
# / This app repository