library(dplyr)
library(readr)
library(tidyr)
library(caret)
library(FNN)
library(mgcv)


library(GGally)
library(caret)
library(ggplot2)
#library(glmnet)


data <- read.csv("algforestfires.csv")

# Clean column names by removing leading/trailing spaces
names(data) <- trimws(names(data))

# Convert 'DC' column to numeric, forcing errors to NA
data$DC <- as.numeric(data$DC)

# Checking distribution of rain
#boxcox_rain <- powerTransform(data$Rain)

#data <- data %>% mutate(log_rain = log(data$Rain))

# Remove rows with NA values in 'DC' column
data_cleaned <- data %>% drop_na(DC)
data_cleaned <- data_cleaned %>% drop_na(Classes)
eda_data <- data_cleaned
data_cleaned <- data_cleaned[, -which(names(data) == "Rain")]


# Exclude 'Classes' and 'Region' columns for standardization
covariates <- data_cleaned %>% select(Temperature, RH, Ws, FFMC, DMC, DC, ISI, BUI, FWI, day, month, Region)

# Standardize the covariates
preprocess_params <- preProcess(covariates, method = c("center", "scale"))
standardized_covariates <- predict(preprocess_params, covariates)

df_2 <- data_cleaned
df_2_bs <- data_cleaned




##### EDA
selected_data <- eda_data[, c("Temperature", "RH", "Ws", "Rain", "FFMC", "DMC", "DC", "ISI", "BUI", "FWI", "Classes")]
selected_data$Classes <- as.factor(selected_data$Classes)
# Create the pairs plot
ggpairs(selected_data, 
        columns = 1:10,    # Columns to plot
        aes(color = Classes, alpha = 0.5)) +
  theme_bw() +
  labs(title = "Pairs plot of features distinguished by fire occurence")



### PREPROCESSING: WEIGHTED PROBABILITIES (BEST)

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
sigma_values <- c(0.5, 0.75, 1, 2)
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

df_final <- df_2_bs[which(df_2_bs$Response_sigma_1 < 1.0 & df_2_bs$Response_sigma_1 > 0.0),]


### MLS:

## ALL:
mls_fit1 <- lm(Response_sigma_1 ~ day 
              + month 
              + Temperature 
              + RH
              + Ws
              + FFMC
              + DMC
              + DC
              + ISI
              + BUI
              + FWI
              + as.factor(Region), data = df_final)

summary(mls_fit1)
plot(residuals(mls_fit1) ~ fitted(mls_fit1),
     xlab = "Fitted values", ylab = "Residuals")

mls_fit2 <- lm(Response_sigma_1 ~ Temperature 
               + RH
               + Ws, data = df_final)

summary(mls_fit2)
plot(residuals(mls_fit2) ~ fitted(mls_fit2),
     xlab = "Fitted values", ylab = "Residuals")

mls_fit3 <- lm(Response_sigma_1 ~ FFMC
               + DMC
               + DC
               + ISI
               + BUI
               + FWI, data = df_final)

summary(mls_fit3)
plot(residuals(mls_fit3) ~ fitted(mls_fit3),
     xlab = "Fitted values", ylab = "Residuals")


### RIDGE:
df_final$Region <- as.factor(df_final$Region)
x1 <- as.matrix(df_final[, c('day', 'month','Temperature', 'RH', 'Ws','FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI')])
x2 <- as.matrix(df_final[, c('Temperature', 'RH', 'Ws')])
x3 <- as.matrix(df_final[, c('FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI')])
y <- as.matrix(df_final[,'Response_sigma_1'])

cv_ridge1 <- cv.glmnet(x = x1, y = y, alpha = 0, family = "gaussian")
plot(cv_ridge1)
cv_r1 <- cv_ridge1$cvm[cv_ridge1$lambda == r1_lambda_min]

cv_ridge2 <- cv.glmnet(x = x2, y = y, alpha = 0, family = "gaussian")
plot(cv_ridge2)
cv_r2 <- cv_ridge2$cvm[cv_ridge2$lambda == r2_lambda_min]

cv_ridge3 <- cv.glmnet(x = x3, y = y, alpha = 0, family = "gaussian")
cv_r3 <- cv_ridge3$cvm[cv_ridge3$lambda == r3_lambda_min]
plot(cv_ridge3)

r1_lambda_min <- cv_ridge1$lambda.min
r1_lambda_1se <- cv_ridge1$lambda.1se

r2_lambda_min <- cv_ridge2$lambda.min
r2_lambda_1se <- cv_ridge2$lambda.1se

r3_lambda_min <- cv_ridge3$lambda.min
r3_lambda_1se <- cv_ridge3$lambda.1se

ridge1_min <- glmnet(x = x1, y = y, alpha = 0, lambda = r1_lambda_min,
                                 family = "gaussian")
ridge1_1se <- glmnet(x = x1, y = y, alpha = 0, lambda = r1_lambda_1se,
                     family = "gaussian")

plot(ridge1_min, label = TRUE)

plot(ridge1_1se, label = TRUE)


ridge2_min <- glmnet(x = x2, y = y, alpha = 0, lambda = r2_lambda_min,
                     family = "gaussian")
ridge2_1se <- glmnet(x = x2, y = y, alpha = 0, lambda = r2_lambda_1se,
                     family = "gaussian")

ridge3_min <- glmnet(x = x3, y = y, alpha = 0, lambda = r2_lambda_min,
                     family = "gaussian")
ridge3_1se <- glmnet(x = x3, y = y, alpha = 0, lambda = r2_lambda_1se,
                     family = "gaussian")

plot(ridge2_min, label = TRUE)
plot(ridge2_1se, label = TRUE)

min_y_pred <- predict(ridge3_min, newx = x3)
se_y_pred <- predict(ridge3_1se, newx = x3)

residuals_min <- y - min_y_pred
residuals_1se <- y - se_y_pred
plot(residuals_min ~ min_y_pred)
plot(residuals_1se ~ se_y_pred)


### GAM:
knots <- 10
mod1 <- gam(Response_sigma_1 ~ s(day, bs = "bs", k = knots)
            + s(month, bs = "bs", k = knots)
            + s(Temperature, bs = "bs", k = knots)
            + s(Ws, bs = "bs", k = knots)
            + s(RH, bs = "bs", k = knots)
            + s(FFMC, bs = "bs", k = knots)
            + s(DMC, bs = "bs", k = knots)
            + s(DC, bs = "bs", k = knots)
            + s(ISI, bs = "bs", k = knots)
            + s(BUI, bs = "bs", k = knots)
            + s(FWI, bs = "bs", k = knots), data = df_final)
summary(mod1)
plot(mod1)
plot(residuals(mod1) ~ fitted(mod1),
     xlab = "Fitted values", ylab = "Residuals")

qqnorm(residuals(mod1))

mod2 <- gam(Response_sigma_1 ~ s(Temperature, bs = "bs", k = knots)
            + s(Ws, bs = "bs", k = knots)
            + s(RH, bs = "bs", k = knots), data = df_final)
plot(residuals(mod1) ~ fitted(mod1),
     xlab = "Fitted values", ylab = "Residuals")



summary(mod2)

plot(residuals(mod2) ~ fitted(mod2),
     xlab = "Fitted values", ylab = "Residuals")
qqnorm(residuals(mod2))

mod3 <- gam(Response_sigma_1 ~ s(FFMC, bs = "bs", k = knots)
            + s(DMC, bs = "bs", k = knots)
            + s(DC, bs = "bs", k = knots)
            + s(ISI, bs = "bs", k = knots)
            + s(BUI, bs = "bs", k = knots)
            + s(FWI, bs = "bs", k = knots), data = df_final)
summary(mod3)
plot(mod3)

plot(residuals(mod3) ~ fitted(mod3),
     xlab = "Fitted values", ylab = "Residuals")

#qqnorm(residuals(mod1))

mod4 <- gam(Response_sigma_1 ~ s(FFMC, bs = "bs", k = knots)
            + s(DMC, bs = "bs", k = knots)
            + s(DC, bs = "bs", k = knots)
            + s(ISI, bs = "bs", k = knots)
            + s(BUI, bs = "bs", k = knots)
            + s(FWI, bs = "bs", k = knots)
            + Region, data = df_final)
summary(mod4)
plot(mod4)

plot(residuals(mod4) ~ fitted(mod4),
     xlab = "Fitted values", ylab = "Residuals")



### trying ridge reg

x <- as.matrix(df_final[, c('FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI')])
y <- as.matrix(df_final[,'Response_sigma_1'])


lallfit <- glmnet(x = x, y = y, alpha = 0, family = "gaussian")
plot(lallfit, label = TRUE)
cvlall <- cv.glmnet(x = x, y = y, alpha = 0, family = "gaussian")
plot(cvlall)
sqrt(min(cvlall$cvm))

minlambda <- cvlall$lambda.min
lambda1se <- cvlall$lambda.1se


# test 2
min_lallfit <- lallfit <- glmnet(x = x, y = y, alpha = 0, lambda = minlambda,
                                 family = "gaussian")
onese_lallfit <- lallfit <- glmnet(x = x, y = y, alpha = 0, lambda = minlambda,
                                   family = "gaussian")

min_y_pred <- predict(min_lallfit, newx = x)

residuals <- y - min_y_pred

plot(residuals ~ y_pred)




