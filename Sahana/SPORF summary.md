## 1)Introduction:
SPORF at each node of a tree chooses splits of the sample space to be very sparse random projections. This helps maintain the properties of axis aligned decision trees.
Yields better performance than decision trees as by maintaining interpretability.
## 2)	Background: 
- Statistical learning – classification : goal of the classifier is predicting the right class of an unlabeled testing sample y, from the prediction model created by previously modelled systems.
- Random Forests – builds T decision tress via recurring binary splits on randomized data. Daughter node splits – maximizes information gain. Splits on the nodes occur until purity is reached, maximum depth or min no. of observations in that node is reached resulting in split and leaf nodes.
- Oblique extensions to random forest – improve RF by varying the notion of feature splits having to be along the coordinate axes. (BSP?) 
- Random projections – random projection matrix is constructed.
- Gradient Boosted Trees - another tree ensemble method  for regression and classification. learned through mimizing a cost function via gradient descent
• Random Search for Splits 
• Flexible Sparsity 
• Ease of Tuning 
• Data Insight 
• Expediency and Scalability
