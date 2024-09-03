#-----------------------UNSUPERVISED LEARNING: CLUSTERING--------------------
library(dplyr)
library(ggplot2)
library(scales)
library(reshape2)
library(cluster)  # For silhouette calculation
library(dbscan)   # For DBSCAN clustering

#----------------1. Data Collection-------------------
# Load the dataset,Customer Segmentation Dataset from kaggle.
data <- read.csv("./Dataset/Online-Retail.csv")
# View the first few rows of the dataset
head(data)

#----------------2. Data Preprocessing-------------------
# Check for missing values
sum(is.na(data))

# Remove rows with missing CustomerID
data <- data[!is.na(data$CustomerID), ]

# Check the structure of the dataset
str(data)

# Check for duplicates
sum(duplicated(data))

# Remove duplicates
data <- data[!duplicated(data), ]

# Convert InvoiceDate to DateTime format
data$InvoiceDate <- as.POSIXct(data$InvoiceDate, format="%m/%d/%Y %H:%M")

# Convert CustomerID and Country to factors
data$CustomerID <- as.factor(data$CustomerID)
data$Country <- as.factor(data$Country)



#----------------3. Feature Engineering-------------------
# Feature engineering: Calculate Total Price (Quantity * UnitPrice)
data$TotalPrice <- data$Quantity * data$UnitPrice

# Aggregate total spending and number of orders per customer
customer_data <- data %>%
  group_by(CustomerID) %>%
  summarise(Recency = as.numeric(difftime(max(InvoiceDate), min(InvoiceDate), units = "days")),
            Frequency = n_distinct(InvoiceNo),
            MonetaryValue = sum(TotalPrice))



#----------------4. Data Transformation and Visualization-------------------
# RMF Variables Distribution
nf <- c("Recency", "Frequency", "MonetaryValue")
n <- 3

# Set theme for white background
custom_theme <- theme_bw() + 
                theme(panel.background = element_rect(fill = "white", color = "white"),
                      plot.background = element_rect(fill = "white", color = "white"),
                      panel.grid.major = element_line(color = "gray90"),
                      panel.grid.minor = element_line(color = "gray95"))

# Original Distributions
ggplot_melt <- melt(customer_data[, nf])
ggplot_original <- ggplot(ggplot_melt, aes(x = value)) + 
  geom_histogram(aes(y = ..density..), bins = 30, fill = "blue", color = "black") + 
  facet_wrap(~variable, scales = "free", ncol = 3) + 
  labs(title = "RMF Variables Distribution - Original") + 
  custom_theme
ggsave("./plots/original_distribution.png", plot = ggplot_original)

# Log Transformation
customer_data_logT <- customer_data
customer_data_logT[nf] <- log1p(customer_data[nf])

ggplot_melt_log <- melt(customer_data_logT[, nf])
ggplot_logT <- ggplot(ggplot_melt_log, aes(x = value)) + 
  geom_histogram(aes(y = ..density..), bins = 30, fill = "green", color = "black") + 
  facet_wrap(~variable, scales = "free", ncol = 3) + 
  labs(title = "RMF Variables Distribution - Log Transformed") + 
  custom_theme
ggsave("./plots/log_transformed_distribution.png", plot = ggplot_logT)

# Sqrt Transformation
customer_data_sqrtT <- customer_data
customer_data_sqrtT[nf] <- sqrt(customer_data[nf])

ggplot_melt_sqrt <- melt(customer_data_sqrtT[, nf])
ggplot_sqrtT <- ggplot(ggplot_melt_sqrt, aes(x = value)) + 
  geom_histogram(aes(y = ..density..), bins = 30, fill = "purple", color = "black") + 
  facet_wrap(~variable, scales = "free", ncol = 3) + 
  labs(title = "RMF Variables Distribution - Sqrt Transformed") + 
  custom_theme
ggsave("./plots/sqrt_transformed_distribution.png", plot = ggplot_sqrtT)

# Cube Root Transformation
customer_data_cbrtT <- customer_data
customer_data_cbrtT[nf] <- sign(customer_data[nf]) * abs(customer_data[nf])^(1/3)

ggplot_melt_cbrt <- melt(customer_data_cbrtT[, nf])
ggplot_cbrtT <- ggplot(ggplot_melt_cbrt, aes(x = value)) + 
  geom_histogram(aes(y = ..density..), bins = 30, fill = "orange", color = "black") + 
  facet_wrap(~variable, scales = "free", ncol = 3) + 
  labs(title = "RMF Variables Distribution - Cube Root Transformed") + 
  custom_theme
ggsave("./plots/cbrt_transformed_distribution.png", plot = ggplot_cbrtT)



#----------------5. Sampling the Data-------------------
# Set seed for reproducibility
set.seed(123)

# Check the number of available rows in the dataset
num_rows <- nrow(customer_data)

# Determine the sample size (min between 100,000 and available rows)
sample_size <- min(100000, num_rows)

# Sample rows from the dataset
sampled_data <- customer_data[sample(num_rows, sample_size), ]

# View the sampled data
head(sampled_data)

#----------------6. Normalizing the Data-------------------
scaled_data <- scale(sampled_data[, c("Recency", "Frequency", "MonetaryValue")])



#----------------7. Optimal Number of Clusters-------------------
# Optimal Number of Clusters
wss <- numeric(15)
for (i in 1:15) {
  kmeans_result <- kmeans(scaled_data, centers = i, nstart = 25)
  wss[i] <- kmeans_result$tot.withinss
}

elbow_plot <- ggplot(data.frame(Clusters = 1:15, WSS = wss), aes(x = Clusters, y = WSS)) + 
  geom_line() +
  geom_point() +
  labs(title = "Elbow Method for Optimal Clusters", x = "Number of Clusters", y = "Total Within Sum of Squares")
ggsave("./plots/elbow_method.png", plot = elbow_plot)

# Silhouette Method
sil_width <- numeric(15)
for (i in 2:15) {
  km <- kmeans(scaled_data, centers = i, nstart = 25)
  sil <- silhouette(km$cluster, dist(scaled_data))
  sil_width[i] <- mean(sil[, 3])
}

silhouette_plot <- ggplot(data.frame(Clusters = 2:15, Silhouette = sil_width[2:15]), aes(x = Clusters, y = Silhouette)) + 
  geom_line() +
  geom_point() +
  labs(title = "Silhouette Method for Optimal Clusters", x = "Number of Clusters", y = "Average Silhouette Width")
ggsave("./plots/silhouette_method.png", plot = silhouette_plot)






#----------------8. Implement Clustering Models-------------------

# Choose the optimal number of clusters based on the above plots
optimal_k <- which.max(sil_width)

# K-Means Clustering
set.seed(123)
kmeans_result <- kmeans(scaled_data, centers = optimal_k, nstart = 25)

# Add the cluster information to the dataset
sampled_data$Cluster_KMeans <- kmeans_result$cluster

# Hierarchical Clustering
dist_matrix <- dist(scaled_data)
hclust_result <- hclust(dist_matrix, method = "ward.D2")

# Cut the dendrogram into clusters
sampled_data$Cluster_Hierarchical <- cutree(hclust_result, k = optimal_k)


# DBSCAN Clustering
dbscan_result <- dbscan(scaled_data, eps = 0.5, minPts = 5)

# Add the cluster information to the dataset
sampled_data$Cluster_DBSCAN <- dbscan_result$cluster


#----------------9. Clustering on PCA Data (With PCA)------------------

pca_result <- prcomp(scaled_data, scale. = TRUE)
sampled_data$PC1 <- pca_result$x[, 1]
sampled_data$PC2 <- pca_result$x[, 2]
pca_data <- data.frame(PC1 = sampled_data$PC1, PC2 = sampled_data$PC2)


# K-Means Clustering on PCA Data
set.seed(123)
kmeans_result_pca <- kmeans(pca_data, centers = optimal_k, nstart = 25)

# Add the cluster information to the dataset
sampled_data$Cluster_KMeans_PCA <- kmeans_result_pca$cluster


# Hierarchical Clustering on PCA Data
dist_matrix_pca <- dist(pca_data)
hclust_result_pca <- hclust(dist_matrix_pca, method = "ward.D2")

# Cut the dendrogram into clusters
sampled_data$Cluster_Hierarchical_PCA <- cutree(hclust_result_pca, k = optimal_k)


# DBSCAN Clustering on PCA Data
dbscan_result_pca <- dbscan(pca_data, eps = 0.5, minPts = 5)

# Add the cluster information to the dataset
sampled_data$Cluster_DBSCAN_PCA <- dbscan_result_pca$cluster


#----------------9. Analyze the Clusters-------------------
# Plot K-Means Clustering Result (Without PCA)
kmeans_plot <- ggplot(sampled_data, aes(x = PC1, y = PC2, color = as.factor(Cluster_KMeans))) +
  geom_point(size = 2) +
  labs(title = "K-Means Clustering (Without PCA)", x = "Principal Component 1", y = "Principal Component 2", color = "Cluster")
ggsave("./plots/kmeans_clusters.png", plot = kmeans_plot)

# Plot K-Means Clustering Result (With PCA)
kmeans_plot_pca <- ggplot(sampled_data, aes(x = PC1, y = PC2, color = as.factor(Cluster_KMeans_PCA))) +
  geom_point(size = 2) +
  labs(title = "K-Means Clustering (With PCA)", x = "Principal Component 1", y = "Principal Component 2", color = "Cluster")
ggsave("./plots/kmeans_clusters_pca.png", plot = kmeans_plot_pca)


# Plot Hierarchical Clustering Result (Without PCA)
hclust_plot <- ggplot(sampled_data, aes(x = PC1, y = PC2, color = as.factor(Cluster_Hierarchical))) +
  geom_point(size = 2) +
  labs(title = "Hierarchical Clustering (Without PCA)", x = "Principal Component 1", y = "Principal Component 2", color = "Cluster")
ggsave("./plots/hclust_clusters.png", plot = hclust_plot)

# Plot Hierarchical Clustering Result (With PCA)
hclust_plot_pca <- ggplot(sampled_data, aes(x = PC1, y = PC2, color = as.factor(Cluster_Hierarchical_PCA))) +
  geom_point(size = 2) +
  labs(title = "Hierarchical Clustering (With PCA)", x = "Principal Component 1", y = "Principal Component 2", color = "Cluster")
ggsave("./plots/hclust_clusters_pca.png", plot = hclust_plot_pca)


# Plot DBSCAN Clustering Result (Without PCA)
dbscan_plot <- ggplot(sampled_data, aes(x = PC1, y = PC2, color = as.factor(Cluster_DBSCAN))) +
  geom_point(size = 2) +
  labs(title = "DBSCAN Clustering (Without PCA)", x = "Principal Component 1", y = "Principal Component 2", color = "Cluster")
ggsave("./plots/dbscan_clusters.png", plot = dbscan_plot)

# Plot DBSCAN Clustering Result (With PCA)
dbscan_plot_pca <- ggplot(sampled_data, aes(x = PC1, y = PC2, color = as.factor(Cluster_DBSCAN_PCA))) +
  geom_point(size = 2) +
  labs(title = "DBSCAN Clustering (With PCA)", x = "Principal Component 1", y = "Principal Component 2", color = "Cluster")
ggsave("./plots/dbscan_clusters_pca.png", plot = dbscan_plot_pca)
