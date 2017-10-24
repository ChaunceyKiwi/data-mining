library(cluster) 
library(fossil)
setwd("~/Desktop/Assignment1") # set working directory 

# Task 1
wine = read.csv("wine.csv") # Read in the Wine dataset from .csv file
wine_normalized <- scale(wine[-1]) # Drop the attribute Wine and all attributes

# Task2
kmeans_normalized_fit <- kmeans(wine_normalized, 4)
dissE <- daisy(wine_normalized)
sk <- silhouette(kmeans_normalized_fit$cl, dissE)
print(summary(sk)$avg.width)

# Task3
wine.filtered <- scale(wine[, c("Alcohol", "Malic.acid")])
kmeans.filtered.fit <- kmeans(wine.filtered, 4)
clusplot(wine.filtered,
         kmeans.filtered.fit$cl,
         main="Scatter Plot",
         color = TRUE,
         shade = TRUE,
         lines = 0)

# Task4
kmeans_normalized_fit <- kmeans(wine_normalized, 3)
dissE <- daisy(wine_normalized)
sk <- silhouette(kmeans_normalized_fit$cl, dissE)
# plot(sk)

# Task5
d <- dist(wine_normalized, method = "euclidean")
dendogram_complete <- hclust(d, method = "complete")
dendogram_average <- hclust(d, method = "average")
dendogram_single <- hclust(d, method = "single")
plot(dendogram_complete, main = "Complete Link Dendogram")
plot(dendogram_average, main = "Average Link Dendogram")
plot(dendogram_single, main = "Single Link Dendogram")

# Task6
cut_complete <- cutree(dendogram_complete, k=3)
cut_average <- cutree(dendogram_average, k=3)
cut_single <- cutree(dendogram_single, k=3)

plot(cut_complete, main="Clustering Result For Complete Link");
plot(cut_average, main="Clustering Result For Average Link")
plot(cut_single, main="Clustering Result For Single Link")

sil_complete <- silhouette(cut_complete, dissE)
sil_average <- silhouette(cut_average, dissE)
sil_single <- silhouette(cut_single, dissE)

print(summary(sil_complete)$avg.width)
print(summary(sil_average)$avg.width)
print(summary(sil_single)$avg.width)

# Task7
print(rand.index(wine[['Wine']], kmeans_normalized_fit$cluster))
print(rand.index(wine[['Wine']], cut_average))