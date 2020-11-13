# How to estimate multilevel GLM in statsmodels package
# Prerequisite for this script is running generate_data.py

# We only show example for gaussian model, because gamma model is not implemented in the package
# see: https://www.statsmodels.org/devel/mixed_glm.html

from pandas import read_csv
from statsmodels.regression.mixed_linear_model import MixedLM

gaussian_data = read_csv("../data/Gaussian_identity_data.csv")

model = MixedLM.from_formula("y ~ x1 + x2 + x3", data=gaussian_data, groups=gaussian_data["group_index"])
model_result = model.fit()

# Fixed effects
print(model_result)

# Random effects
print(model_result.random_effects)

# Dispersion parameter
print(model_result.scale)
