#!/usr/bin/env python
# coding: utf-8

# ### Librerias

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ### Librerias para interactuar con R

# In[2]:


import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects import r


from rpy2.robjects.conversion import localconverter


# ### Convertidor DataFrame R a Pandas

# In[3]:


def dset(datasetr):
    x=r(datasetr)
    with localconverter(ro.default_converter + pandas2ri.converter):
        pd_from_r_df = ro.conversion.rpy2py(x)
        return pd_from_r_df


# ### Dataset

# In[4]:


r('library(aplore3)')


# ## 2. Regresión Logística Multiple
# 
# Consideramos una coleción de $p$ variables denotada $X^T = (X_1,X_2,\cdots, X_p)$. Denotamos la probabilidad condicional $\mathbb{P}(Y=1|X) = \pi(X)$
# 
# $$\pi(X) = \frac{e^{\beta_0+\beta_1x_1 + \beta_2x_2+\cdots+\beta_px_p}}{1+e^{\beta_0+\beta_1x_1 + \beta_2x_2+\cdots+\beta_px_p}}$$

# Si tenemos una muestra de n observaciones independientes $(x_i,y_i)$ con $i=1,2,\cdots,n$. $x_i$ un vector con $p$ valores e $y_i$ toma el valor 0 o 1.
# 
# Las ecuaciones del likelihood son las mismas que para el caso univariado. 
# 
# $$\sum_{i=1}^n y_i - \pi(x_i) = 0$$
# 
# $$\sum_{i=1}^n x_{ij}(y_i - \pi(x_i)) = 0$$
# 
# para $j=1,2,\cdots,p$
# 
# Para la estimación de las varianzas y covarianzas de los coeficientes necesitamos las segundas derivadas del loglikelihood

# $$\frac{\partial^2 L}{\partial \beta_j^2} = -\sum_{i=1}^nx_{ij}^2\pi(x_i)(1-\pi(x_i))$$
# 
# $$\frac{\partial^2 L}{\partial \beta_j\beta_k} = -\sum_{i=1}^nx_{ij}x_{ik}\pi(x_i)(1-\pi(x_i))$$
# 
# $j,k = 0,1,2,\cdots,p$

# El estimador de la varianza está dado por la inversa de la matriz de información observada $I$
# 
# $$\hat{Var(\beta)} = \begin{pmatrix} -\frac{\partial^2L}{\partial \beta_1^2} & -\frac{\partial^2L}{\partial \beta_1\beta_2} & \cdots & -\frac{\partial^2L}{\partial \beta_1\beta_{p+1}}\\
# -\frac{\partial^2L}{\partial \beta_2\beta_{1}} & -\frac{\partial^2L}{\partial \beta_2^2} & \cdots & -\frac{\partial^2L}{\partial \beta_2\beta_{p+1}}\\
# \vdots & \ddots & \ddots & \vdots\\
# -\frac{\partial^2L}{\partial \beta_{p+1}\beta_{1}} &\cdots & \cdots & -\frac{\partial^2L}{\partial \beta_{p+1}^2}
# \end{pmatrix}^{-1}$$
# 
# Podemos definir 
# 
# $$X=\begin{pmatrix} 1 & x_{11} & x_{12}&\cdots&x_{1p}\\
# 1& x_{21} & x_{22} &\cdots &x_{2p}\\
# \vdots & \vdots &\vdots&\ddots&\vdots\\
# 1 & x_{n1}&x_{n2} & \cdots & x_{np}\end{pmatrix}$$
# 
# y la matriz 
# 
# $$\hat{V} = \begin{pmatrix} \hat{\pi}_1(1-\hat{\pi}_1) & 0 &\cdots &0\\
# 0 &  \hat{\pi}_2(1-\hat{\pi}_2)&\cdots & 0\\
# \vdots  &0 &\ddots & \vdots\\
# 0 &\cdots & 0 & \hat{\pi}_n(1-\hat{\pi}_n)\end{pmatrix}$$
# 
# donde $\pi_i = \pi(x_i)$
# 
# Luego podemos estimar la matriz de información observada como 
# 
# $$\hat{I}(\hat{\beta}) = X^T\hat{V}X$$

# ## Ejemplo

# In[5]:


GLOW = dset('glow500')
GLOW.head(5)


# In[6]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# In[7]:


labelencoder = LabelEncoder()
GLOW["priorfrac"]=labelencoder.fit_transform(GLOW["priorfrac"])
GLOW["premeno"]=labelencoder.fit_transform(GLOW["premeno"])
GLOW["momfrac"]=labelencoder.fit_transform(GLOW["momfrac"])
GLOW["armassist"]=labelencoder.fit_transform(GLOW["armassist"])   
GLOW["smoke"]=labelencoder.fit_transform(GLOW["smoke"])
GLOW["fracture"]=labelencoder.fit_transform(GLOW["fracture"])


# In[8]:


onehotencoder = OneHotEncoder()
ratk = onehotencoder.fit_transform(GLOW["raterisk"].to_numpy().reshape(-1,1)).toarray()

GLOW["raterisk_1"] = ratk[:,0]
GLOW["raterisk_2"] = ratk[:,2]


# In[9]:


GLOW.head(5)


# In[10]:


import statsmodels.api as sm
from statsmodels.tools.tools import add_constant

log_reg = sm.Logit(GLOW["fracture"],add_constant(GLOW[["age","weight","priorfrac","premeno","raterisk_1","raterisk_2"]])).fit()
print(log_reg.summary())


# ## Testeo de significancia de las variables

# El test de significancia sigue la misma logica del caso univariado. Solo que ahora la $G$ tiene distribución chi cuadrado con $p$ grados de libertad. El test ratio test ahora compara el modelo con todos los parametros con el modelo con solo la constante.
# 
# ## Continuación ejemplo 
# 
# Mediante la tabla anterior obtenemos
# 
# $$LikelihoodRatioTest= -2[-281.17-(-259.04)] = 44.2598$$
# 
# con un p-value $6.565\cdot 10^{-8}$, luego las variables son significantes. Lo que nos dice que al menos una de las variables no es cero. 
# 
# Si bien esto no es muy util, ahora podemos usar el wald test parámetro por parámetro
# 
# $$W_j = \frac{\hat{\beta}_j}{\hat{SE}(\hat{\beta})_j}$$
# 
# Estos valores pueden ser obtenidos directo desde la tabla que retorna statsmodel junto a los p-values.
# 
# Siguiendo con el ejemplo podemos notar que $WEIGHT$ y $PREMENO$ no son significantes (tomando un nivel de significancia de 0.05)

# Ahora ajustaremos el modelo sin las variables no significantes

# In[11]:


log_reg_2 = sm.Logit(GLOW["fracture"],add_constant(GLOW[["age","priorfrac","raterisk_1","raterisk_2"]])).fit()
print(log_reg_2.summary())


# Si hacemos un likelihood ratio test entre los dos modelos (chi cuadrado con 2 grados de libertad) obtenemos
# 
# $$G = -2[-259.45 - (-259.04)] = 0.82$$

# In[32]:


from scipy.stats import chi2

1-chi2.cdf(0.82,2)


# Ya que este valor es mayor a 0.05 concluimos que el modelo con todos los parametros no es mejor que el modelo sin los parametros

# ## Wald test multivariado
# 
# El Wald-test multivariado está dado por 
# 
# $$W = \hat{\beta}^T[\hat{\mathbb{V}}(\hat{\beta})]^{-1}\hat{\beta}$$
# 
# este test distribuye chi cuadrado con $p+1$ grados de libertad, bajo la hipotesis de que los $p+1$ coeficientes son ceros. 
# 
# Es equivalente al likelihood ratio test si eliminamos la constante, pero es mucho mas costoso ya que hay que invertir una matriz

# ## Intervalos de confianza
# 
# El intervalo de confianza basado en el Wald test es el mismo visto anteriormente.

# ## Intervalo de confianza para el logit
# 
# El estimador del logit es 
# 
# $$\hat{g}(x) = \hat{\beta}_0 + \hat{\beta}_1 x_1+\cdots+\hat{\beta}_px_p$$
# 
# Reescribimos 
# 
# $$\hat{g}(x) = x^T\hat{\beta}$$
# 
# Luego el estimador de la varianza es 
# 
# $$\hat{\mathbb{V}}(\hat{g}(x))=\sum_{i=0}^p x_i^2\hat{\mathbb{V}}(\hat{\beta}_i)+\sum_{j=0}^p\sum_{i=j+1}^p 2x_{i}x_{j}\hat{cov}(\hat{\beta}_j,\hat{\beta}_i)$$
# 
# En forma matricial queda 
# 
# $$\hat{\mathbb{V}}(\hat{g}(x)) = x^T\hat{\mathbb{V}}(\hat{\beta})x = x^T(X^T\hat{V}X)^{-1}x$$

# Mediante statsmodels podemos obtener la matriz de covarianza 

# In[78]:


round(log_reg_2.cov_params(),5)


# ## Ejemplo
# 
# intervalo de confianza de tener Edad = 65, Priorfrac = 1, Raterisk_1=1, Raterisk_2=0

# In[79]:


np.array([1,65,1,1,0]).T.dot(round(log_reg_2.cov_params(),5)).dot(np.array([1,65,1,1,0]))


# In[75]:


0.81487+(65)**2*0.00015+0.05816 + 0.07563 + 2*65*(-0.01089) + 2*0.04450 + 2*(-0.06039) + 2*65*(-0.00083) + 2*65*0.00022 + 2*(-0.00313)


# In[ ]:




