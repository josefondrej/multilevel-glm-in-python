import numpy as np

__families__ = ["Gaussian", "Gamma"]


def sample_gaussian(mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    n = len(mu)
    return mu + sd * np.random.randn(n)


def sample_gamma(mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """
    See https://en.wikipedia.org/wiki/Gamma_distribution and
    https://numpy.org/doc/stable/reference/random/generated/numpy.random.gamma.html
    """
    shape = mu ** 2 / sd ** 2
    scale = sd ** 2 / mu
    y = np.random.gamma(shape=shape, scale=scale)
    return y


family_to_variance_function = {
    "Gaussian": lambda mu: np.ones_like(mu),
    "Gamma": lambda mu: mu ** 2
}

family_to_random_sampler = {
    "Gaussian": sample_gaussian,
    "Gamma": sample_gamma
}

if __name__ == '__main__':
    n = 100000
    mu = 0.6
    sd = 1.3
    y = sample_gamma(mu=np.ones(n) * mu, sd=np.ones(n) * sd)
    print(np.mean(y))
    print(np.sqrt(np.var(y)))
