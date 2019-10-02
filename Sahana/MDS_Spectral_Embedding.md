### Manifold learning-
Approach to non-linear dimensionality reduction.
As visualisation is hard for higher dimensional data, the need for dimensionality reduction for visual purposes is identified. 
The simplest way to accomplish this dimensionality reduction is by taking a random projection of the data
In a random projection, it is likely that the more interesting structure within the data will be lost.
Manifold Learning can be thought of as an attempt to generalize linear frameworks like PCA to be sensitive to non-linear structure in data.

Isomap can be viewed as an extension of Multi-dimensional Scaling (MDS) or Kernel PCA. 
Isomap seeks a lower-dimensional embedding which maintains geodesic distances between all points. 


### MDS
 the goal of the analysis is to detect meaningful underlying dimensions that allow the researcher to explain observed similarities or dissimilarities (distances) between the investigated object
With MDS, you can analyze any kind of similarity or dissimilarity matrix, in addition to correlation matrices.
In general then, MDS attempts to arrange "objects" in a space with a particular number of dimensions so as to reproduce the observed distances. As a result, we can "explain" the distances in terms of underlying dimensions;
MDS arrives at a configuration that best approximates the observed distances


### Spectral Embedding
Spectral Embedding is an approach to calculating a non-linear embedding. Scikit-learn implements Laplacian Eigenmaps, which finds a low dimensional representation of the data using a spectral decomposition of the graph Laplacian. The graph generated can be considered as a discrete approximation of the low dimensional manifold in the high dimensional space. Minimization of a cost function based on the graph ensures that points close to each other on the manifold are mapped close to each other in the low dimensional space, preserving local distances

Manifold learning on noisy and/or incomplete data is an active area of research.
