# Intro:

Operating on the nearest neighbour might not always be desirable.
Measurement of error modelling – assuming that x’s are noisy data that are true but unobserved. Better to assume the noise than a noise free measurement. Since using kernel regression on noise-free measurement is useless it promotes to understand the latent structure of the data. Performance increases when only learning the structure of x explicit of the label y.
(Geodesic distance – shortest path considering a curved surface)
(Geodesic learning is removing this geodesic distance from data corpus (being only text information?))
Space partitioning trees uses binary and recursive splits with hyperplanes -> these are optimized for relative proximities of noisy measurements.
Decision trees are linked to kernel learning (what is kernel learning?)
When we’re talking about high dimensional data and the noise associated with it, what does it really mean with regards to the features/dataset?
URerF was developed for linear space and time complexity which approximates the latent geodesic distances between all pairs of points. 
Does not compute the geodesic distances between all points but instead looks at the latent structure and makes clusters in subspaces recursively.
Randomer forest-> enables URerF to separate meaningful data from noise.
Splitting criteria is introduced – Fast BIC -> computation of BIC for gaussian mixture model in ONE dimension.
(gaussian mixture model?)(Embedding?)
URerF find nearest neighbour in low dimensions spaces even amongst the noise better than most algorithms.

# Related work:
Steps for preserving and estimation geodesic distances : 
1) Estimate geodesic distance in the original manifold 2)
All-pair shortest path is computed 
3)points are embedded into a lower dimensional space to preserve the distances.
UMAP – new algorithm for dimensionality reduction. The process is weighted k-nearest neighbors and embed into a lower dimension through force directed layout algo.
In comparison to random projection forest, the splits are optimized and uses a forest of many trees.

# Unsupervised randomer forests:
Distinction b/w random forests – new fast BIC splitting criterion, “ramdomer” because the splitting methods are based on random sparse linear combinations of the features used to strengthen each tree.
Proximity matrix from random forests?

