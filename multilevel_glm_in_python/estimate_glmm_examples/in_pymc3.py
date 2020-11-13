from typing import Any, Dict

import numpy as np
import pymc3 as pm
from pandas import read_csv, DataFrame
from pymc3 import Model

from multilevel_glm_in_python.exponential_family import link_str_to_link_inverse, family_to_pm_distribution, \
    family_to_variance_function


def _print_map(maximum_aposteriori_estimate: Dict[str, Any]):
    for coeff_name, value in maximum_aposteriori_estimate.items():
        if not "__" in coeff_name:
            print(f"{coeff_name}: {' ' * (20 - len(coeff_name))} {value}")


def create_model(formula: str, group_name: str, data: DataFrame,
                 family: str = "Gaussian", link: str = "identity") -> Model:
    n_groups = len(set(data[group_name]))
    with pm.Model() as model:
        # Prior on random effects
        sd_b_group = pm.HalfNormal("sd_b_group", sigma=100)
        b_group = pm.Normal("b_group", mu=0, sigma=sd_b_group, shape=n_groups)

        # Prior on phi (= sigma^2 in this case)
        phi = pm.HalfNormal("phi", sigma=100)

        # Prior on fixed effects
        lm = pm.glm.LinearComponent.from_formula(formula=formula, data=data)

        # Likelihood
        eta = lm.y_est + b_group[data[group_name]]
        g = link_str_to_link_inverse[link]
        V = family_to_variance_function[family]
        mu = g(eta)
        sd = np.sqrt(phi * V(mu))
        pm_distribution = family_to_pm_distribution[family]
        y = pm_distribution("y", mu=mu, sigma=sd, observed=data["y"])

    return model


if __name__ == '__main__':
    # Requires to set the working directory to the project directory
    gaussian_data = read_csv("./data/Gaussian_identity_data.csv")
    gamma_data = read_csv("./data/Gamma_log_data.csv")

    model = create_model(formula="y ~ x1 + x2 + x3", group_name="group_index", data=gaussian_data,
                         family="Gaussian", link="identity")
    map_gaussian = pm.find_MAP(model=model)
    _print_map(map_gaussian)

    model = create_model(formula="y ~ x1 + x2 + x3", group_name="group_index", data=gamma_data,
                         family="Gamma", link="log")
    map_gamma = pm.find_MAP(model=model)
    _print_map(map_gamma)
