# Data Preprocessing

## Outline
* Introduction
* Feature extraction
* Data transformation
* Data cleaning
* Data integration
* Data reduction

## Introduction

### Why do need data preprocessing?
* Data mining is based on existing data, which in the real word is dirty:
	* incomplete
	* noisy
	* inconsistent
* Quality of data mining results crucially depends on quality of input data

## Feature Extraction
Derive meaning features from the data

### Goal
* Extract meaningful features that are relevant for the given data mining task
* Extract meaningful features that lead to interpretable results

### Document data
* Choose releveant terms in document set
* Calcualte term frequencies in a document
* Map document to vector in term space

### Image Data
* To extract features, histograms of colors/textures can be used.

### Time Series Data
* Raw data is variable-length sequence of numerical or categorical data, which is associated with time stamp
* Naive approach creates one feature for each pre-defined time window, aggregating all values within the window.

## Data Transformation

### Normalization
* To make different records comparable
	* Min-max scaling
	* Z-score standardization
	* Percentile rank

### Convert data types
To allow application of data mining methods for other data type

#### Discretization
numerical -> ordinal (categorical)

##### Goal
* Reduce the number of values for a given numerical feature by partitoning the range of the feature into intervals
* Interval labels replace actual feature values

##### Methods
* Binning
	* Equal-width binning
	* Equal-depth binning
* Entropy-based discretization
		
#### Binarization
categorical -> numberical
* Because binary data is special form of both numberical and categorical data, it is possible to convert categorical attributes to binary form.

## Data Cleaning
Deal with missing values and noisy data

### Handling Missing Data
Data is not always available

* Ignore the record
* Inpute missing values
	* Use a default to fill in the missing value
	* Use the attribute mean to fill in the missing value
	* Use the most probable value to fill in the missing value
		* inference-based such as Bayesian formula or regression

### Handing Noisy Data
Random erro or variance in a measured attribute

* Binning
	* Sort data and partition into bins
	* Smooth by bin means, bin median, bin boundaries, etc.
* Regression
	* Smooth by fitting a regression function
* Clustering
	* Detect and remove outliers
* Combined computer and human inspection
	* Detect suspicious values automatically and check by human

## Data Integration
Integration of multiple datasets, resolve inconsistencies

### Purpose
Combine datasets from multiple sources into a coherent dataset (database)

### Schema intergration
* Integrate metadata from different sources
* Attribute identification problem
	* "same" attributes from multiple data sources may have different names
* Instance integration
	* Integrate instances from differnt sources
	* For the same real world entity, attribute values from different sources maybe different
	* Possible reason:
		* Different representations
		* Different conventions
		* Different scales, errors
* Approach
	* Identification
		* Detect corresponding tables from different sources may use corrlelation analysis
			* e.g. A.cust-id = B.cust-#
		* Detect duplicate records from different sources involves approximate matching of attribute values
			* e.g. 3.14283 = 3.1, Schwartz = Schwarz
	* Treatment
		* Merge correspoding tables
		* Use attribute values as synonyms
		* Remove duplicate records

## Data Reduction
Reduce number of records or attributes. Reduced dataset should be representative.

### Motivation
* Improve efficiency
	* Runtime of data mining algorithms is linear w.r.t. number of records and number of attributes
* Improve quality
	* Removal of irrelevant attributes and /or records avoids overfitting and improves the quality of the discovered patterns.

#### Feature Selection

##### Goal
* Select as relevent features
* For classification:
	* Select a set of features such that the probability distribution of classes given the values for selected attributes is as close as possible to the class distribution given the values of all attributes.

##### Problem
* 2^d possible subsets of set of d attributes
* Need heurisic feature selection methods

##### Methods
* Feature independence assumption
	* Choose features independently by their relevance
* Greedy bottom-up feature selection
	* The best single-feature is picked first
	* Then next best feature conditioned on the first
* Greedy top-down feature elimination
	* Repeatly eliminate the worst feature
* Set-oriended feature selection
	* Consider trade-off between relevance of individual features and the redundancy of feature set

##### Feature Selection Criteria
* Mutual information
	* For categorical features
	* Measure the information that feature X and class Y share
	* How much does knowing one of the attributes reduce uncertainty about the other?


```
// When x, y are indenpendent, p(x, y) = P(x) * p (y), MI(x, y) = 0
MI(x, y) = sum_x sum_y p(x, y) * log (p(x, y) / (p(x) * p(y)))
```

* Fisher score
* Pricinpal Component Analysis (PCA)
	* Task
		* Given N data vectors from d-dinmensional space, find c << d oethogonal vectrors that can best prepresent the data
		* Data representation by projection onto the c resulting vectors
		* Best fit: minimal squared error
			* error = difference between original and transformed vectors
	* Properties
		* Resulting c vectors are the directions of the maximum variance of original data
		* These vectors are linear combinations of the original attributes
		* Works for numerical data only

### Sampling
#### Goal
Choose a representative subset of the data records

* Sampling without replacement
	* No duplication
* Sampling with replacement
	* Might have duplication

Random sampling may overlook small but important groups, thuw we might need some advanced sampling methods:

* Biased sampling
	* Oversample more important records, e.g. more receent ones
* Stratified sampling
	* Draw random samples independently from each given stratum
* Cluster sampling
	* Draw random samples independently from each given cluster