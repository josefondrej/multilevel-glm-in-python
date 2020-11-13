import json
import os
from typing import Tuple, Dict, Any

import numpy as np
from pandas import DataFrame

from multilevel_glm_in_python.exponential_family import family_to_variance_function, family_to_random_sampler, \
    link_str_to_link_inverse


def parse_formula(formula: str, intercept_name: str = "Intercept") -> Tuple[np.ndarray, np.ndarray, str]:
    y_name, x_formula = formula.split("~")
    y_name = y_name.strip()

    x_terms = x_formula.split("+")
    [intercept_coeff], *x_coeff_x_name = [x_term.split("*") for x_term in x_terms]
    intercept_coeff = float(intercept_coeff)
    x_coeffs = [intercept_coeff] + [float(x_coeff) for x_coeff, _ in x_coeff_x_name]
    x_names = [intercept_name] + [x_name.strip() for _, x_name in x_coeff_x_name]

    return np.array(x_coeffs), np.array(x_names), y_name


def generate_x_data(formula: str, n_obs: int = 1000, intercept_name: str = "Intercept",
                    linear_predictor_name: str = "eta") -> DataFrame:
    x_coeffs, x_names, y_name = parse_formula(formula, intercept_name=intercept_name)
    data = DataFrame()
    for x_name in x_names:
        if x_name == intercept_name:
            data[x_name] = [1.0] * n_obs
        else:
            data[x_name] = np.random.randn(n_obs)
        # Linear predictor
    data[linear_predictor_name] = np.matmul(np.array(data), x_coeffs.reshape(-1, 1))
    return data


def add_random_intercept(data: DataFrame, n_groups: int = 10, group_sd: float = 1.0,
                         linear_predictor_name: str = "eta", group_index_name: str = "group_index",
                         center_sample: bool = True) -> Dict[int, float]:
    group_index_to_random_intercept = {group_index: np.random.randn() * group_sd for group_index in range(n_groups)}
    if center_sample:
        mu_b = np.mean(list(group_index_to_random_intercept.values()))
        group_index_to_random_intercept = {grp_ix: b - mu_b for grp_ix, b in group_index_to_random_intercept.items()}

    data[group_index_name] = np.random.choice(range(n_groups), len(data))
    data[linear_predictor_name] += data[group_index_name].apply(group_index_to_random_intercept.get)
    return group_index_to_random_intercept


def generate_glm_response(eta: np.ndarray, family: str = "Gaussian", link: str = "identity", phi: str = 1.0):
    """
    Generate observations of random variables Y_1, ... Y_n which follow generalized linear model
    f(y, theta, phi) = exp((y * theta - b(theta)) / phi  + c(y, phi))
    theta = (b')^{-1}(g^{-1}(eta)), where g ... link function

    More helpful characterization is this Y is observation of random variable with distribution from
    exponential family with mean and variance defined as:
    mu := E[Y] = g^{-1}(eta)
    var := var[Y] = phi * V(mu)
    """
    link_inverse = link_str_to_link_inverse[link]
    variance_fn = family_to_variance_function[family]
    random_sampler = family_to_random_sampler[family]

    mu = link_inverse(eta)
    var = phi * variance_fn(mu)
    sd = np.sqrt(var)

    y = random_sampler(mu=mu, sd=sd)
    return y


def generate_data(formula: str, n_obs: int = 1000, n_groups: int = 10, group_sd: float = 1.0,
                  family: str = "Gaussian", link: str = "identity", phi: float = 1.0):
    data = generate_x_data(formula=formula, n_obs=n_obs, linear_predictor_name="eta")
    group_index_to_random_intercept = add_random_intercept(data=data, n_groups=n_groups, group_sd=group_sd)
    data["y"] = generate_glm_response(eta=data["eta"], family=family, link=link, phi=phi)
    return data, group_index_to_random_intercept


def _save_data(name: str, data: DataFrame, metadata: Dict[str, Any], out_dir: str = "./data/"):
    os.makedirs(out_dir, exist_ok=True)
    data.to_csv(f"{out_dir}{name}_data.csv")
    with open(f"{out_dir}{name}_metadata.json", "w") as metadata_file:
        json.dump(metadata, metadata_file)


if __name__ == '__main__':
    np.random.seed(123456789)

    formula = "y ~ 1.4 + 3.15 * x1 + 2.5 * x2 + 0.6 * x3"
    n_obs = 10000
    n_groups = 8
    group_sd = 3.0
    phi = 0.6

    for family, link in [("Gaussian", "identity"), ("Gamma", "log")]:
        data, grp_to_b = generate_data(formula=formula,
                                       n_obs=n_obs,
                                       n_groups=n_groups,
                                       group_sd=group_sd,
                                       family=family,
                                       link=link,
                                       phi=phi)

        metadata = dict(formula=formula, n_obs=n_obs, n_groups=n_groups, group_sd=group_sd, phi=phi, grp_to_b=grp_to_b)

        name = f"{family}_{link}"
        _save_data(name, data, metadata)
