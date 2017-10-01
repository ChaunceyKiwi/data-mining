# Cluster Analysis

## Outline
* Introduction
* Representative-based clustering
* Probabilistic model-based clustering
* Hierarchical clustering
* Density-based clustering
* Non-negative matrix factorization
* Cluster ensembles
* High-dimensional clustering
* Semi-supervised clustering

## Introduction

### Goal of cluster analysis
* Identification of a finite set of categories, classes or groups (clusters) in the dataset
* Objects within the same cluster shall be as similar as possbile
* Objects of different clusters shall be dissimilar as possible

## Clustering as Optimization Problem
### Steps
1. Choice of model category. Partitioning, hierarchical, density-based, etc.
2. Definition of score function. Typically based on distance function.
3. Choice of model structure. Feature selection / number of clusters.
4. Search for model parameters. Clusters / cluster represenetatives

### Distance Functions
#### Basiscs
##### Formalizing similarity
* Sometimes: similarity function
* Typically: distance function dist(o1, o2) for pairs of objects o1 and o2
* Small distance -> similar objects
* Large distance -> dissimilar objects

##### Requirements for distance functions
1. dist(o1, o2) >= 0
2. dist(o1, o2) = 0 iff o1=o2
3. dist(o1, o2) = dist(o2, o1)
4. dist(o, o3) <= dist(o1, o2) + dist(o2, o3)

##### Distance Functions for Numerical Attributes
* L_p-Metric (Minkowski-Distance)
* Euclidean Distance
* Manhattan Distance
* Maximum Metric
* Correlation Coefficient ?

### Typical Clustering Applications
#### Overview
* Market segmentation
	* Clustering the set of customer transactions
* Determing user groups on the WWW
	* Clusteing web-logs
* Structuring large sets of text documents
	* Hierarchical cluserting of the text documents
* Generating thematic maps from satellite images
	* Clustering sets of raster images of the same area (feature vectors)

#### Types of Clusteing Methods
* Representative-based (Partitioning) Clustering
	* Parameters: number k of clusters, distance function
	* Determines a "flat" clustering into k clusters (with minimal costs)
* Probablistic Model-Based Clustering
	* Parameters: number k of clusters
	* Determines a "flat" clustering into k clusters (with maximum data likelihood)
* Hierarchical Clustering
	* Parameters: distance function for objects and for clusters
	* Determines a hierarchy of clusterings, merge always the most similar clusters
* Density-Based Clustring
	* Parameters: minimum density within a cluster, distance function
	* Extends cluster by neighboring objects as long as the density is large enough

#### Cluster Validation
* Internal validation criteria
* External validation criteria

## Representative-based clustering

### Basics
* Goal
	* A disjoint partitioning into k clusters with minimal costs.
* Local optimization method
	* Choose k initial cluster representatives
	* Optimize these representatives iteratively
	* Assign each object to its most similar cluster representative
* Type of Cluster Representatives
	* Mean of a cluster 
		* Construction of central points
	* Medoid of a cluster
		* Selection of representative points

### Construction of Central Points
* Objects are points in an Euclidean vector space
* Using Euclidean distance
* Centroid: mean vector of all objects in cluster C
* Measure for the cost (compactness) of a cluster C
	* TD^2(C) = sum_{p in C}dist(p, miu_c)^2
* Measure for the cost (compactness) of a clustering
	* TD^2 = sum{i=1}{k} TD^2(C_i)

#### Variants
* k-means [MacQueen 67]
* IOSDATA

#### Discussion
* Advantage
	* Efficiency
	* Simple imeplementation
* Disadvantage
	* Sensitivity to noise and outliers
	* All clusters have a convex shape
	* The number k of clusters is often hard to determine
	* Highly dependent on the initial partioning, including clusterinhg

### Selection of Representative Points
* Assumes only distance function of pairs of objects, no Euclidean vector space
* Less sensitive to outliers
* Medoid:a representative element of the cluster (representative point)
* Measure for the cost (compactness) of a cluster C
	* TD(C) = sum_{p in C}dist(p, m_c)
* Measure for the cost (compactness) of a clustering
	* TD = sum{i=1}{k} TD(C_i)
* Search space for the clustering algorithm
	* all subsets of cardinality k of the dataset D with |D| = n
		* Runtime complexity O(n^k)

#### Algorithms
* PAM
* CLARANS

### Choice of Initial Clusterings
* Standard approach
	* Draw a random sample of k objects from D and take them as initial cluster representatives
* Better approach
	* Draw a sample S of >> k objects from D, cluster S with k-means, and take the resutling cluster representatives as initial cluster
		* To initialize the clustering of S, draw k objects randomly from S
		* To make the initialization even more robust to outliers, repeat this procedure m times and choose the best set of k cluster representatives.
* Idea
	* In general, clustering of a small sample yields good initial clusters
	* But some samples may have a significantly different distribution
* Method
	* Draw independently m different samples
	* Cluster each of these samples
	* Cluster the dataset DB = A union B union C union ... [How to use this database? ]
	* From the m clusterings obtained, choose the one with the highest clustering quality as initial clustering for the whole dataset

### Choice of Parameter k
* Method
	* For k = 2, 3, ..., n-1, determine one clustering each
		* Enough to try k = 2, 3, ..., sqrt(n)
* Choose the clustering with the highest clustering quality

* Measure of clustering quality
	* Independent from k
	* For k-means and k-medoid
		* TD^2 and TD decrease monotonically with increasing k
	* For EM (probabilistic)
		* E increases monotonically with increasing k

* Silhouette-Coefficient
	* Measure of clustering quality for k-means and k-medoid-methods
	* a(o): distance of object o to its cluster representive
	* b(o): distance of object o to the representative of the "second-best" cluster
	* s(o) = (b(o) - a(o)) / max{a(o), b(o)}
		* s(o) = -1/0/1: bad / indifferent / good assignment
	* Silhouette coefficient S_c of clustering C
		* Average silhouette over all objects
	* Interpretation of silhouette coefficient
		* S_c > 0.7: strong cluster structure
		* S_c > 0.5: reasonable cluster structure 


## Probablistic model-based clustering
### Basics
* Objects are points in an Euclidean vector space
* A cluster is described by a probability density distribution
* Typically: Gaussian distribution (Normal distribution)
* Representation of a cluster C
	* mean of all cluster points
	* d by d covariance matrix for the points of cluster C
* Probablity density function of cluster C
### Discussion
* EM algorithm converges to a local maximum
* Runtime complexity O(n * k * #iterations)
* Clustering result and runtime strongly depend on
	* initial clustering
	* "correct" choice of parameter k
* Modification for determining k disjoint clusters
	* assign each object x only to cluster C_i with maximum P(C_i|x)

## Hierarchical clustering
* Goal
	* Construction of a hierarchy of clusters (dendrogram)
* Dendrogram
	* A binary tree of nodes representing clusters, with the following properties
		* Root represents the whole dataset
		* Leaf node represents singleton clusters containing a single object
		* Inner node represents the union of all objects contained in its corresponding subtree

### Types of hierarchical methods
* Bottom-up construction of dendrogram (agglomerative)
* Top-down construction of dendrogram (divisive)

### Agglomerative Hierarchichal Clustering
1. Form initial clusters consisting of a singleton object, and compute the distance between each pair of clusters
2. Merge the two clusters having minimum distance
3. Calculate the distance between the new cluster and all other clusters
4. Stop if there is only one cluster containing all objects. Otherwise go to step 2.

### Distance function for clusters
* Let dist(x, y) be a distance function for pairs of objects x, y
* Let X, Y be clusters
	* Single-Link
	* Complete-Link
	* Average-Link

### Discussion
* Advantages
	* Does not require knowledge of the number k of clusters
	* Finds not only a flat clustering, but a hierarchy of clusters (dendrogram)
	* A single clustering can be obtained from the dendrogram (e.g. by performing a horizontal cut)
* Disadvantages'
	* Decisions (merge/splits) cannot be undone
	* Sensitive to noise (Single-Link)
		* a link of objects can connect two clusters
	* Inefficient
		* runtime complexity at least O(n^2) for n objects

### CURE
* Representation of a cluster
	* partitioning methods: one object
	* hierarchical method: all objects
	* CURE: representation of a cluster by c representatives

## Density-based Clustering
### Idea
Clusters as dense areas in a d-dimensional dataspace, separated by areas of low density

### Requirements for density-based clusters
* For each cluster object, the local density exceeds some threshold
* The set of objects of one cluster must be spatially connected

### Strenghts of density-based clustering
* Clusters of arbitary shape
* RObust to noise
* Effieiency

### Concept
* core object
* directly density-reachable
* density-reachable
* border object
* Density reachability is not a symmetric relationship. But symmetry is desirable to make the clustering independent from the order of visiting objects
* density-connected
	* symmertic
* Cluster C w.r.t. epsilon and MinPts
	* Maximality
	* Connectivity
* Clustering
	* A density-based clustering CL of a dataset D w.r.t. epsilon and MinPts is the set of all density based clusters w.r.t. epsilon and MinPts in D
* The set Noise_CL is defined as the set of all objects in D which do not belong to any of the clusters
* Property
	* Let C be a density-based cluster and p in C a core object. Then:
		* C = {o in D | o density-reachable from p w.r.t. eplison and MinPts}

### Choice of Parameters
* Cluster: density above the "minimum density" defined by epsilon and MinPts
* Wanted: the density of the cluster with the lowest density. All objects with higher density should belong to one of the clusters
* Heuristic method: consider the distances to the k-nearest neighbors
* Function k-distance: distance of an object to its k-nearest neighbor
* k-distance-diagram: k-distances in descending order
* Heuristic Method
	* User specifies a value for k, MinPts := k + 1
	* System calculates the k-distance-diagram for the dataset and visualizes it
	* User chooses a threshold object o from the k-distance-diagram,
		* k := k-distance(o)

#### Problems with choosing the Parameters
* Hierarchical clusters
* Significantly differing densities in different areas of the dataspace
* Clusters and noise are not well-separated

#### Hierarchical Density-Based Clustering
* For constant MinPts-value, density-based clusters w.r.t. a smaller episilon are completely contained within density-based clusters w.r.t. a larger episilon
* The clusterings for different density parameters can be determined simultaneously in a single scan
	* first dense sub-cluster, then less dense rest-cluster
* Does not generate a dendrogramm, but a graphical visualization of the hierarchical cluster structure 

## Non-negative matrix factorization
### Introduction
* Nonnegative matrix facorization (NMF) is a dimensionality reduction method that is tailored to clustering.
* Suitable for matrices that are non-negative and sparse. E.g. document-term matrix of term frequencies
		* Most term frequencies are 0
* Embeds the data into a latent, low-dimensional space that makes it more amenable to clustering. 
* The basis system of vectors and the coordinates of the data objects in this system are nonnegative, which makes the solution interpretable.
* D = UV^T
	* D: n by d matrix of n objects with d attributes
	* V: d by k matrix of the k basis vectors in terms of the original lexicon
	* U: n by k matrix of k-dimensional cordinates of the rows oof D in the transformed basis system
	* k << d
	* Basis vectors represent topics / clusters of attributes

### Alternating Non-negative Least Squares
Temporarily skipped.

### Discussion
* To avoid overfitting, add a regularizer to the object function
	* lamba * (||U||^2 + ||V||^2)
* Compared to SVD, NMF is harder to optimize because of the non-negativity constraint.
	* SVD (Singular-value decomposition)
* But SCD may produce negative entries for U, V which makes it less useful for clustering ?

## Cluster ensembles
* Different clustering algorithms produce many different clusterings
* Cluster validation is typically hard
* None of the many alternative clusterings is the "true" clustering, but they all capture some aspects of the cluster structure
* The goal of cluster ensembles is to combine multiple clusterings to create a more robust clustering
* Do not consider the attributes/features of the objects, but only the structure of the clusterings
* Also called consensus clustering or multi-view clustering.

### Selecting Ensemble Components
* Model-based ensembles
	* clusterings obtained from differnt clustering algorithms
	* clustering obtained from same clustering algorithm with different parameter settings
* Data-based ensembles
	* select different subsets of the data
	* select different subsets of the set of dimensions

### Combining Ensemble Components
* Graph partioning
	* vertex = object, edge = connect objects appearing togrther in a cluster, edge weight = number of shared clusters
		* apply graph partitioning algorithm such as Min-cut
* Hypergraph partioning
	* vertex = object, hyperedge = cluster
	* apply hypergraph partitioning algorithm such as HMETIS
* Distance-based clustering

## High-dimensional clustering

### Curse of Dimensionality
* In high-dimensional data, the (average) pairwise distances are big and rather uniformly distributed
* Clusters only in lower-dimensional subspaces

### Subspace Clustering
* Cluster: "dense area" in dataspace
* Density-threshold tau
	* Region is dense if it contains more than tau
* Grid-based approach
	* Each dimension is divided into xi intervals
	* Cluster is union of connected dense regions
* Phases
	* Identification of subspaces with clusters
	* Identification of clusters
	* Generation of cluster description 

### Identification of Subspace with Clusters
* Task: detect dense base regions
* Naive approach
	* Calculate histigrams for all subsets of the set of dimensions
		* Infeasible for high-dimentional dataset (O(2^d) for d dimensions)
* Greedy algorithm
	* Start with the empty set
	* Add one more dimension at a time
* Monotonnicity Property
	* If a region R in k-dimensional space is dense, then each projection of R in (k-1)-dimension subspace is dense as well (more than tau objects)
		* If monotonicity property violated, prune candidate region

### Identification of Clusters
* Task: find maximal sets of connexted dense base regions
* Given: all dense base region in a k-dimensional subspace
* DFS search of the following graph (search space)
	* nodes: dense base regions
	* edges: joint edges / dimensions of the two base regions

### Generation of Cluster Descriptions
* Given: a cluster i.e. a set of connected dense base regions
* Task: find optimal cover of this cluster by a set of hyperrectangles
* Standard method:
	* Infeasible for large value of d. The problem is NP-complete
* Heuristic method:
	* Cover thhe cluster by maximal regions
	* Remove redundant regions

### Discussion
### Advantages
* Automatic detection of subspace with clusters
* No assumptions on the data distribution and number of clusters
* Scalable w.r.t. the number n of data object
### Disadvantages
* Accuracy crucially depends on paramaters tau and xi
	* Sample parameter value for all dimensionalities is problematic
* Needs a heuristics to reduce the search space
	* Method is not complete
* Typically finds many clusters

### Pattern-Based Subspace Clusters
* Shifting pattern
* Scaling pattern
* Such pattern cannot be found using existing subspace clustering methods since
	* These method are distance-based
	* The above points are not close enough
 