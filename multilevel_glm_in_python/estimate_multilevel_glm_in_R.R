# Title     : Estimate Multilevel GLM in R
# Created by: josef
# Created on: 13.11.20

# Prerequisite for this script is running generate_data.py

library(lme4)
library(blmeco)

get_random_intercepts <- function(model) {
  random_intercepts = coef(model)$group_index[["(Intercept)"]]
  random_intercepts = random_intercepts - mean(random_intercepts)
  return(random_intercepts)
}

gaussian_data <- read.csv("/tmp/multilevel-glm-in-python/Gaussian_identity_data.csv")
gamma_data <- read.csv("/tmp/multilevel-glm-in-python/Gamma_log_data.csv")

gaussian_model = glmer("y ~ x1 + x2 + x3 + (1 | group_index)", family = gaussian(link = "identity"), data = gaussian_data)
gamma_model = glmer("y ~ x1 + x2 + x3 + (1 | group_index)", family = Gamma(link = "log"), data = gamma_data)

# Fixed effects
summary(gamma_model)
summary(gaussian_model)

# Random effects
get_random_intercepts(gamma_model)
get_random_intercepts(gaussian_model)

# Dispersion parameter phi
dispersion_glmer(gamma_model)^2
dispersion_glmer(gaussian_model)^2