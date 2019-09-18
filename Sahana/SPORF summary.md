## 1)Introduction:
SPORF at each node of a tree chooses splits of the sample space to be very sparse random projections. This helps maintain the properties of axis aligned decision trees.
Yields better performance than decision trees as by maintaining interpretability.
## 2)	Background: 
- Statistical learning – classification : goal of the classifier is predicting the right class of an unlabeled testing sample y, from the prediction model created by previously modelled systems.
- Random Forests – builds T decision tress via recurring binary splits on randomized data. Daughter node splits – maximizes information gain. Splits on the nodes occur until purity is reached, maximum depth or min no. of observations in that node is reached resulting in split and leaf nodes.
- Oblique extensions to random forest – improve RF by varying the notion of feature splits having to be along the coordinate axes. (BSP?) 
- Random projections – random projection matrix is constructed.
- Gradient Boosted Trees - another tree ensemble method  for regression and classification. learned through mimizing a cost function via gradient descent
 Random Search for Splits 
 Flexible Sparsity 
 Ease of Tuning 
 Data Insight 
 Expediency and Scalability

Axis aligned decision trees – split along the feature dimension only.
Very sparse random projections – Linear combinations of a small subset of features.
Ensemble methods – combine multiple machine learning algos to solve an issue.
Identically and independently distributed? Probability distribution function?
Classifier – to come up with a predictor function for which any new sample is associated with a label from the subset of Y.
The goal would be to reduce the misclassification error.
Check how Baye’s classifier works on reducing the misclassification error?

Random forests – Build T no. of decision trees which are constructed as recursive binary splits on the training sets. The nodes are split based on the information gain (Entropy? Reduction in impurity index?)
Commonly a reduction in gini impurity will give information on the node splits in a decision tree.
WRITE THE OPTIMIZATION FOR THE THETA* (Splitting the parent to left and right nodes).
Nodes are recursively split until stopping criterion is reached.
