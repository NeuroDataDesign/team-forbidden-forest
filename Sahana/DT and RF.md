# Decision Trees: 
Uses Divide and Conquer Rule
-	Taking an attribute of the data set, use it to divide the training set into different subsets. 
-	The data is split until homogeneity, i.e purity in the data set is reached.
-	Tree with the labels at the leaf nodes. Does not need to be balanced, just that it needs to be pure.
-	ID3 Algorithm: 
    o	Find best attribute A for splitting the training set
    o	A is now the decision attribute to the node
    o	For each value of A create a child node
    o	Split the training data to the child nodes.
    o	For each child node/subset – if pure, then stop else repeat the steps.
-	We essentially need to pick an attribute which splits the data such that it is heavily biased either to the positive or negative.
-	Pick best attribute by computing gain of the attribute.
-	Can always classify training sets perfectly – singleton = pure
-	Doesn’t work on new data? Overfitting. (Stop splitting when not statistically significant)
-	DTs pick thresholding for continuous attributes.
-	Cons: 
    o	Only axis aligned splits
    o	Greedy(may not find best tree)

# Random Forest:
Grow K different trees on your training set.
-	Randomize the training examples(input)- Sr.
-	No pruning, grow full ID3 data.
-	Computes again based on the random subset Sr and not the full set.
-	Repeat this for each decision tree.
-	Use majority vote.
-	Random forests uses the simplicity of decision trees but improves the accuracy.
-	BAGGING – Average noisy and unbiased models to create a model with low variance
