# Classification 

## Outline
* Introduction
* Evaluation of classifiers
* Decision Trees
* Bayesian Classification
* Logistic Regression
* Nearest-neighbour Classification
* Support Vector Machine
* Regression Analysis
* Ensemble Classifiers

## Introduction
### The Classification Problem
* Description
	* Objects (o_1, o_2, ..., O_d) 
	* Attributes A_i, 1 <= i <= d
	* Class label c {c_1, c_2, ..., c_k}
* Difference to clustering
	* Classification: set of classes C know apriori
	* Clustering: classes are output
* Related problem: regression analysis
	* Predict the value of a numerical attribute

## Evaluation of Classifiers 
### Introduction
* Given a sample of labeled data O
* Want to learn a classifier that labels the entire population
	* In particular, the unseen data D / O
	* Can only estimate the performance of the classifier on unseen data
* Split labeled data into two disjoint subsets
	* Training data
		* For training the classifier (model learning)
	* Test data
		* To evaluate the trained classifier
* Do not use the test set to tune the parameters of the classification algorithm or make other choices about classifier design
	* This would overestimate the true accuracy because knowledge of the test set has been implicitly used in the training process
* To avoid this problem, split the training set further into training and validation set. Use validation set for parameter tuning etc.
* Approaches for creating training and test datasets
	* Hold out
	* Cross-validation

### Hold-out
* Partition set O randomly into two (disjoint) subsets: training data and test data
* Drawback: only a subset of the available labeled data is used for training
* Full power of the labeled data is not reflected in the error estimate
	* Not recommended for small O
* If class distribution is very skewed, take a stratified sample
	* Sample the same percentage of each class separately

### Cross-validation
* Partition set O randomly into m same size subsets
* Train m different classifiers using a different one of these m subsets as test data the other m-1 subsets for training
* Average the evaluation result of the m classifiers
* Typically, m = 5 or m = 10
* Obtains representative estimate of classification performance
* Appropriate also for small O

## Evaluation of classifiers
### Evaluation Criteria
* Classification accuracy
* Interpretability
	* Set of a decision tree
	* Insight gained by the user
* Efficiency
	* of model learning
	* of model application
* Scalability for large datasets
	* for secondary storage data
* Robustness
	* w.r.t. noise and unknown attribute values

### Classification Accuracy
* Let K be a classifier
	* TR is subset of O the training data
	* TE is subset of O the test data
* Classification accuracy of K on TE
* Classification error

### Confusion Matrix
* Let c_1 in C be the target (positive) class, the union of all other classes the contrasting (negative) class
* For the target class, comparing the predicted and the actual class labels, we can distinguish four different cases

||Predicted as positive| Predicted as negative|
|:--|:--|:--|
|Actually positive|True Positive|False Negative|
|Actually negative|False Positive|True Negative|

### Precision and Recall
* Precision(K) = |TP| / (|TP| + |FP|)
* Recall(K) = |TP| / (|TP| + |FN|)
* There is a trade-off between precision and recall
* Therefore, we also define a measure combining precision and recall
	* F-Measure(K) = 2 * Precision(K) * Recall(K) / (Precision(K) + Recall(K))
	* F-Measure captures only one of the possible trade-offs between precision and recall

### ROC Curves
* Receiver Operating Characteristic Curve
	* Curve: plots true positive rate vs. false positive rate
		* True positive rate: percentage of positive data correctly predicted
		* False positive rate: percentage of negative data falsely predicted as positive
	* Area under ROC as quantitative measure ideally = 1

### Model Selection
* Which of classifier is really better?
	* Naive approach: just take the one with higher mean classification accuracy
		* But classification accuracy may vary greatly among the m folds
		* Differences in classification accuracies may be insignificant due only to chance
* We measure the classification on a small test dataset O belongs to D
	* Questions: 
		* How to estimate the true classification error on the whole data space D?
		* How does the deviation from the observed classification error depend on the size of the test set.
* Random experiment to determine the classification error on test set (of size n): repeat n times
	* Draw random object from D
	* Compare predicted vs. actual class label for this object
* Classification error is percentage of misclassified objects
	* Observed classification error follows a Binomial distribution with mean = true classification error.

## Decision Trees
* Disjunction of conjunction of attribute constraints
* Hierarchical structure
* A decision tree is a tree with the following properties:
	* An inner node represents an attribute
	* An edge represents a test on the attribute of the parent node
	* A leaf represents one of the classes of C
* Construction of a decision tree
	* Based on the training data
	* Top-Down strategy
* Application of a decision tree
	* Traversal of decision tree from the root to one of the leaves
	* Unique path
		* Assignment of the object to class of the resulting leaf

### Base Algorithm
* Initially, all training data objects belongs to the root
* Next attribute is selected and split
* Training data objects are partitioned according to the chosen split
* Method is applied recursively to each partition
	* Local optimization method (greedy)
* Termination conditions
	* No more split attributes
	* All (or most) training data objects of the node belong to the same class

### Types of Splits
* Categorical attributes
	* Conditions of the form "attribute = a" or "attribute in set"
* Numerical attributes
	* Conditions of the form "attribute < a"
	* Many possible split points

### Quality Measures for Splits
* Given
	* Set T of training data objects
* Wanted
	* Measure of the impurity of any set S of labeled data w.r.t. class labels based on the p_i, the relative frequency of class c_i in S
	* Split of T in T_1, T_2, ..., T_m minimizing this impurity measure. Split is a partitioning, i.e. a disjoint cover of T.
		* Information gain, gini-index

### Information Gain
* Entropy: minimal number of bits to encode a message to transmit the class of a random training data object.
* Entropy for a set T of training objects:
	* entropy(T) = sum_{i=1}^k p_i * log_2(p_i);
		* entropy(T) = 0, if p_i = 1 for some i
		* entropy(T) = 1 for k = 2 classes with p_i = 1/2
* Let attribute A produce the partitioning T1, T2, ..., Tm of T
* The information gain of attribute A w.r.t. is defined as 
	* InformationGain(T, A) = entropy(T) - sum (|Ti|/|T|)*entropy(T_i)

### Gini-Index
* Gini index for a set T of training data objects
	* low gini index -> low impurity
	* high gini index -> high impurity
* Let attribute A produce the partitioning T_1, T_2, ..., Tm of T
* Gini index of attribute A w.r.t. T is defined as 
	* gini_A(T) = sum |T_i| / |T| * gini(T_i)

### Overfitting
### Approaches to Avoid Overfitting
#### Choice of appropriate minimum confidence
#### Subsequent pruning of the decision tree
### Error Reduction-Pruning
### Minimal Cost Complexity Pruning

## Logistic Regression
