# Multilevel Generalized Linear Models in Python

This is a collection of scripts that show different possibilities how to fit 
multilevel generalized linear models in Python (and R). 

## Theory

Mostly taken from these [course notes](https://www2.karlin.mff.cuni.cz/~kulich/vyuka/pokreg/doc/advreg_notes_200522.pdf).

**Definition** *Exponential Family*: If density of random variable $Y$ can be written as 
$$f(y, \theta, \phi) = \exp(\frac{y\theta - b(\theta)}{\phi} + c(y, phi))$$
where 
- $\theta$ is called the *canonical parameter*; 
- $\phi \in (0, \inf)$ is called the *dispersion parameter*;
- $b$ and $c$ are some real functions;

then the distribution belongs to *exponential family* of distributions. The expression ^^ is called canonical form of the density.

**Theorem**: If $b$ is twice continuously differentiable then $Y$ has finite first two moments.
- $\mu := E[Y] = b'(\theta)$
- $var[Y] = \phi b''(\theta)$

**Corollary**: There exists a function $V(\mu)$ such that $var[Y] = \phi V(\mu)$, which is called 
a variance function.
 
**Note**: Each distribution belonging to the exponential family has a different variance 
function. Within the exponential family, the variance function determines the distribution. 

**Definition** *Generalized Linear Model*: The data $(Y_i, X_i)$ satisfy the generalized linear 
model (GLM) if:
1. $Y_1, \dots, Y_n$ are independent and the distribution of $Y_i$ depends on $X_i$ through 
regression parameters $\beta$. 
2. The conditional density of $Y_i$ given $X_i$ has the form $f(y, \theta_i, \phi)$ 
(see definition of Exponential Family), where $b(.)$ is known twice continuously differentiable 
function, $\theta_i$ depends on $X_i$ and $\beta$ and $\phi$ is known or unknown constant. 
3. There exists a strictly monotone, twice continuously differentiable *link function* $g$ 
such that $\mu_i = g^{-1}(\eta_i)$ where $\eta_i = X_i^{\top}\beta$ is the *linear predictor*. 

 
