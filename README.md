# Multilevel Generalized Linear Models in Python

This is a collection of scripts that show different possibilities how to fit 
multilevel generalized linear models in Python (and R). 

## Theory

Mostly taken from these [course notes](https://www2.karlin.mff.cuni.cz/~kulich/vyuka/pokreg/doc/advreg_notes_200522.pdf).

**Definition** *Exponential Family*: If density of random variable <img src="https://render.githubusercontent.com/render/math?math=\large Y"> can be written as
 
<img src="https://render.githubusercontent.com/render/math?math=\large f(y, \theta, \varphi) = \exp(\frac{y\theta - b(\theta)}{\varphi} %2B c(y, \varphi))">

where 
- <img src="https://render.githubusercontent.com/render/math?math=\large \theta"> is called the <i>canonical parameter</i>; 
- <img src="https://render.githubusercontent.com/render/math?math=\large \varphi \in (0, \infty)"> is called the <i>dispersion parameter</i>;
- <img src="https://render.githubusercontent.com/render/math?math=\large b"> and <img src="https://render.githubusercontent.com/render/math?math=\large c"> are some real functions;

then the distribution belongs to *exponential family* of distributions. The expression ^^ is called canonical form of the density.

**Theorem**: If <img src="https://render.githubusercontent.com/render/math?math=\large b"> is twice continuously differentiable then <img src="https://render.githubusercontent.com/render/math?math=\large Y"> has finite first two moments.
- <img src="https://render.githubusercontent.com/render/math?math=\large \mu := E[Y] = b'(\theta)">
- <img src="https://render.githubusercontent.com/render/math?math=\large var[Y] = \varphi b''(\theta)">

**Corollary**: There exists a function <img src="https://render.githubusercontent.com/render/math?math=\large V(\mu)"> such that <img src="https://render.githubusercontent.com/render/math?math=\large var[Y] = \varphi V(\mu)">, which is called 
a variance function.
 
**Note**: Each distribution belonging to the exponential family has a different variance 
function. Within the exponential family, the variance function determines the distribution. 

**Definition** *Generalized Linear Model*: The data <img src="https://render.githubusercontent.com/render/math?math=\large (Y_i, X_i)"> satisfy the generalized linear 
model (GLM) if:

1. <img src="https://render.githubusercontent.com/render/math?math=\large Y_1, \dots, Y_n"> are independent and the distribution of <img src="https://render.githubusercontent.com/render/math?math=\large Y_i"> depends on <img src="https://render.githubusercontent.com/render/math?math=\large X_i"> through 
regression parameters <img src="https://render.githubusercontent.com/render/math?math=\large \beta">.
 
2. The conditional density of <img src="https://render.githubusercontent.com/render/math?math=\large Y_i"> given <img src="https://render.githubusercontent.com/render/math?math=\large X_i"> has the form <img src="https://render.githubusercontent.com/render/math?math=\large f(y, \theta_i, \varphi)"> 
(see definition of Exponential Family), where <img src="https://render.githubusercontent.com/render/math?math=\large b(.)"> is known twice continuously differentiable 
function, <img src="https://render.githubusercontent.com/render/math?math=\large \theta_i"> depends on <img src="https://render.githubusercontent.com/render/math?math=\large X_i"> and <img src="https://render.githubusercontent.com/render/math?math=\large \beta"> and <img src="https://render.githubusercontent.com/render/math?math=\large \varphi"> is known or unknown constant.
 
3. There exists a strictly monotone, twice continuously differentiable *link function* <img src="https://render.githubusercontent.com/render/math?math=\large g"> 
such that <img src="https://render.githubusercontent.com/render/math?math=\large \mu_i := E[Y_i | X_i] = g^{-1}(\eta_i)"> where <img src="https://render.githubusercontent.com/render/math?math=\large \eta_i = X_i^{\top}\beta"> is the *linear predictor*. 

 
