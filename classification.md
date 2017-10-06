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
	* F-Measure(K) = 2 * Precision(K) * Recall(K) / (Precision(K)+Recall(K))