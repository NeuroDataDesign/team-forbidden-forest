
# Note 9_23_19

## Random Forest
---
### Boothstrap Method
Source: [intro to bootstrap, machinelearning mastery](https://machinelearningmastery.com/a-gentle-introduction-to-the-bootstrap-method/)
- estimating quantities about a population by averaging estimates from multiple small data samples
- Method
    1. get a data set $N$: 6 observation \[1, 2, .., 6\] or $\mathcal{N}(\mu,\sigma^2)$
    2. choose the size of a sample $n$, randomly choose the observation __independently__ $n$ times: sample with size 4, get \[2, 1, 2, 6\]
    3. calculate statistic data from a sample
---

### Understanding Random Forest
Source: [understanding RF, towards data science](https://towardsdatascience.com/understanding-random-forest-58381e0602d2)<br>[visualize RF, towards data science](https://towardsdatascience.com/how-to-visualize-a-decision-tree-from-a-random-forest-in-python-using-scikit-learn-38ad2d75f21c)
- $T$ # of individual classification trees (binary maps)
- each tree $t$ has its own sample set $\mathcal{X}^n, n<N$ and class prediction, using the lost function $\sum f()$
- The most vote becomes our model's prediciton
- __Strength__: large number of __uncorrelated__ models $T$ work together as a voter prevent individual errors.

#### Ensure that the models diversity each other
- Bagging(Bootstrap Aggression): making the decision trees very __sensitive__ to the training data $\mathcal{X}^n, n<N$. Different data leads to a different tree.
- Feature Randomess: can split a node from the given subset of features $d<D$. the goal is to pick a node that have most separation between left and right.

- Q) Why do we need to go through these pain?
- A) The decision tree will be too sensitive (over-fitting) if we only fit 1 time from the whole data set. The most vote tree represents the most accurate, adaptable model. 
---



## Outliner Detection
---
### Outlier Detection with Isolation Forest
Source: [IsolationForest, towards data science](https://towardsdatascience.com/outlier-detection-with-isolation-forest-3d190448d45e)
- Outliners observation is unprobable data. They lies further away from regular observations
- To detect the outliners $s(x,n)=2^{-\frac{E(h(X))}{c(n)}}$. The closer $s(x,n)$ to $1$, the more probabailty for outliners
    - $h(x)$: path length of observation x
    - $c(n)$: path length of unsuccesfull search in a Binary Tree
    - $n$: number of external node
---

### Novelty Detection
source: [Novelty and Outlier Detection
, sckitlearn](https://towardsdatascience.com/outlier-detection-with-isolation-forest-3d190448d45e)

To be continue...

---
### Gaussian Mixture Model
Source: [GaussianMixture, sckitlearn](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)<br>[Mixture_model, wiki](https://en.wikipedia.org/wiki/Mixture_model)<br> [brilliant, gaussian-mixture-model](https://brilliant.org/wiki/gaussian-mixture-model/)

To be continue...


```python

```
