library(dplyr)
library(readr)
library(tidyr)
library(caret)
library(FNN)


data <- read.csv("algforestfires.csv")

# Clean column names by removing leading/trailing spaces
names(data) <- trimws(names(data))

# Convert 'DC' column to numeric, forcing errors to NA
data$DC <- as.numeric(data$DC)

# Remove rows with NA values in 'DC' column
data_cleaned <- data %>% drop_na(DC)

# Exclude 'Classes' and 'Region' columns for standardization
covariates <- data_cleaned %>% select(-Classes, -Region)

# Standardize the covariates
preprocess_params <- preProcess(covariates, method = c("center", "scale"))
standardized_covariates <- predict(preprocess_params, covariates)

df_1 <- data_cleaned
df_2 <- data_cleaned
df_2_bs <- data_cleaned
df_3 <- data_cleaned









#### 1. VARYING THE RESPONSE

# Number of neighbors to consider for each observation, varying from 3 to 10
n_neighbors_list <- seq(3, 10)

# Calculate the response variable with varying number of neighbors
calculate_response_varying_neighbors <- function(standardized_covariates, classes, n_neighbors_list) {
  response_probabilities <- rep(0, nrow(standardized_covariates))
  for (n_neighbors in n_neighbors_list) {
    nn <- get.knnx(standardized_covariates, standardized_covariates, k = n_neighbors)$nn.index
    probabilities <- apply(nn, 1, function(neighbors) {
      fire_count <- sum(classes[neighbors] == 0)
      probability <- fire_count / n_neighbors
      return(probability)
    })
    response_probabilities <- response_probabilities + probabilities
  }
  response_probabilities <- response_probabilities / length(n_neighbors_list)
  return(response_probabilities)
}

resp_prob1 <- calculate_response_varying_neighbors(standardized_covariates, df_2$Classes, n_neighbors_list)

# Add the new response variable to the cleaned dataframe
df_1 <- df_1 %>% mutate(Response = resp_prob1)

hist(df_1$Response)











### 2. WEIGHTED PROBABILITIES (BEST)

n_neighbors <- 10

# Calculate the response variable with weighted probabilities using
# Gaussian kernel to weight the neighbors
calculate_response_smooth <- function(standardized_covariates, classes, n_neighbors, sigma) {
  nn <- get.knnx(standardized_covariates, standardized_covariates, k = n_neighbors)
  distances <- nn$nn.dist
  indices <- nn$nn.index
  response_probabilities <- sapply(1:nrow(standardized_covariates), function(i) {
    # Gaussian kernel weights
    weights <- exp(-distances[i, ]^2 / (2 * sigma^2))
    weights <- weights / sum(weights)
    fire_count <- sum(weights * (classes[indices[i, ]] == 0))
    return(fire_count)
  })
  return(response_probabilities)
}

# Experiment with different sigma values
sigma_values <- c(0.1, 0.5, 0.75, 1)
response_probabilities_list <- lapply(sigma_values, function(sigma) {
  calculate_response_smooth(standardized_covariates, data_cleaned$Classes, n_neighbors = 5, sigma)
})

for (i in seq_along(sigma_values)) {
  df_2[[paste0("Response_sigma_", sigma_values[i])]] <- response_probabilities_list[[i]]
}

# Plot histograms of the new response variables for each sigma value
par(mfrow = c(2, 2))  # Create a 2x2 plot layout
for (i in seq_along(sigma_values)) {
  hist(df_2[[paste0("Response_sigma_", sigma_values[i])]], breaks = 20, col = "grey", 
       main = paste("Histogram of Response (sigma =", sigma_values[i], ")"), 
       xlab = "Response")
}

# Plot empirical CDF of the new response variables for each sigma value
par(mfrow = c(2, 2))  # Create a 2x2 plot layout
for (i in seq_along(sigma_values)) {
  response_var <- df_2[[paste0("Response_sigma_", sigma_values[i])]]
  plot(ecdf(response_var), main = paste("Empirical CDF (sigma =", sigma_values[i], ")"), 
       xlab = "Response", ylab = "CDF", verticals = TRUE, do.points = FALSE, col = "blue")
}

###### bootstrapping

# Number of bootstrap samples
n_bootstrap <- 100

# Calculate the response variable with bootstrapping
calculate_response_bootstrap <- function(standardized_covariates, classes, n_neighbors, sigma, n_bootstrap) {
  n <- nrow(standardized_covariates)
  response_matrix <- matrix(0, nrow = n, ncol = n_bootstrap)
  
  for (b in 1:n_bootstrap) {
    # Create a bootstrap sample
    bootstrap_indices <- sample(1:n, replace = TRUE)
    bootstrap_covariates <- standardized_covariates[bootstrap_indices, ]
    bootstrap_classes <- classes[bootstrap_indices]
    
    # Calculate the response for the bootstrap sample
    nn <- get.knnx(bootstrap_covariates, standardized_covariates, k = n_neighbors)
    distances <- nn$nn.dist
    indices <- nn$nn.index
    
    response_probabilities <- sapply(1:n, function(i) {
      weights <- exp(-distances[i, ]^2 / (2 * sigma^2))
      weights <- weights / sum(weights)
      fire_count <- sum(weights * (bootstrap_classes[indices[i, ]] == 0))
      return(fire_count)
    })
    
    response_matrix[, b] <- response_probabilities
  }
  
  # Average the response variables from all bootstrap samples
  final_response <- rowMeans(response_matrix)
  return(final_response)
}


# Different Sigma Values
response_probabilities_list <- lapply(sigma_values, function(sigma) {
  calculate_response_bootstrap(standardized_covariates, df_2_bs$Classes, n_neighbors, sigma, n_bootstrap)
})

# Add the new response variable to the cleaned dataframe for each sigma value
for (i in seq_along(sigma_values)) {
  df_2_bs[[paste0("Response_sigma_", sigma_values[i])]] <- response_probabilities_list[[i]]
}

# Plot 2x2 histograms of the new response variables for each sigma value
par(mfrow = c(2, 2))  # Create a 2x2 plot layout
for (i in seq_along(sigma_values)) {
  response_var <- df_2_bs[[paste0("Response_sigma_", sigma_values[i])]]
  hist(response_var, breaks = 20, col = "grey", 
       main = paste("Histogram of Response (sigma =", sigma_values[i], ")"), 
       xlab = "Response")
}

# Plot 2x2 empirical CDFs of the new response variables for each sigma value
par(mfrow = c(2, 2))  # Create a 2x2 plot layout
for (i in seq_along(sigma_values)) {
  response_var <- df_2_bs[[paste0("Response_sigma_", sigma_values[i])]]
  plot(ecdf(response_var), main = paste("Empirical CDF (sigma =", sigma_values[i], ")"), 
       xlab = "Response", ylab = "CDF", verticals = TRUE, do.points = FALSE, col = "blue")
}











### 3. RANDOM SAMPLING

n_samples <- 10

# Calculate the response variable with random sampling of neighbors
calculate_response_random <- function(standardized_covariates, classes, n_neighbors, n_samples) {
  response_probabilities <- rep(0, nrow(standardized_covariates))
  for (i in seq_len(n_samples)) {
    nn <- get.knnx(standardized_covariates, standardized_covariates, k = n_neighbors)$nn.index
    probabilities <- apply(nn, 1, function(neighbors) {
      sampled_neighbors <- sample(neighbors, n_neighbors, replace = TRUE)
      fire_count <- sum(classes[sampled_neighbors] == 0)
      probability <- fire_count / n_neighbors
      return(probability)
    })
    response_probabilities <- response_probabilities + probabilities
  }
  response_probabilities <- response_probabilities / n_samples
  return(response_probabilities)
}

resp_prob3 <- calculate_response_random(standardized_covariates, df_3$Classes,
                                        n_neighbors = n_neighbors, 
                                        n_samples = n_samples)

# Add the new response variable to the cleaned dataframe
df_3 <- df_3 %>% mutate(Response = resp_prob3)

hist(df_3$Response)
