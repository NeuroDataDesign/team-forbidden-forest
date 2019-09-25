## Summary
### Algorithm 1 
Building an unsupervised random decision tree using 1D projections of sparse linear combinations of the features.

We provide an input to build the three as BuildTree(X,d,Θ) 
Where X is a pxd dimensional input which is a subspace of the original data.
d provides the dimension
Θ gives split eligibility criteria

-Step 1: if the split elegibility criteria (Θ) not satisfied then we create a Leafnode constructed on X[pxd]
If the criteria is met then, we sample the subset of X for some random set of features [a1,..ap] with a distribution of fa -> A matrix stores these sampled features.

-Step2: Projection on these new sampled points onto a new dimension is as follows:
X ̃ = A^T * X
min_t* ← ∞
At this point we have 1D data?

-Step3 : test all the dimensions for an optimal split
i.e  i ∈ {1, ..., d}
X ̃(i) ←X ̃[:,i]
(midpt, t*) = ChooseSplit(X ̃(i)) - Here either pick the TwoMeans or FastBIC as the splitting algorithm.
The output of this gives the midpoint and the BIC score estimated for the best partitions in that sampled subspace.
Note: the splitting occurs for data that is nx1, i.e 1D data.

-Step4: Find the dimension rendering the optimal split and the point at which the spit occurs & store it.
if (t* <min_t*) then
bestDim = i - we store the dimension 'i' along which we get the best split, lowest BIC score?
splitPoint = midpt - we store the point at which the associated dimension of the split occured.

-Step5: Check the points to see which side of the tree model it belongs to and split it accordingly
Xleft = {x ∈ X|x(bestDim) < splitPoint} - Checks if the new point along the best dimension is of lesser value that the split point, if that;s true then it gets allocated to the left leaf node else it checks with the right leaf node poit
Xright = {x ∈ X|x(bestDim) ≥ splitPoint} - Same as above

If the above criteria is checked then you recursively apply the same procedure to the daughter nodes (the above was the splitting for the parent node)
Daughters.Left = BuildTree(Xleft, d, Θ) 
Daughters.Right = BuildTree(Xright, d, Θ) 


 
