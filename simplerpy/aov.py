#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Michelle Hsieh
# Created Date: 3/5/2022
# =============================================================================

"""

The class 'AOV' contains an ordinary linear model object (aov) from R 
through the bridge package rpy2. This class mimics how most machine learning 
models in Python work and provide extra information retrieved from the R object.

"""


import pandas as pd
import rpy2
import numpy as np
import math
from rpy2 import robjects as ro
from rpy2.robjects import Formula
from rpy2.robjects import pandas2ri
pandas2ri.activate()
R=ro.r

#class WIP
class AOV:
  def __init__(self):
    """
    Initialize the AOV object, fit() must be called before calling all other 
    methods of this object.

    """
    self._model=None
  
  def fit(self, formula):
    """
    Fits the aov model with input formula using aov() in R
    through the package rpy2.
    :param formula: formula used for aov() i.e. "y~x"
    :return: None, assign the R model object to self._model
    """

  


#finished part
def simpAOV(text, df):
  """
  Arguments: text, df
    text: the formula you want to pass into aov
    df: the data frame 
  """
  model=R.aov(Formula(text), df)
  vars=list(pd.DataFrame(model.rx2('model')).columns) #extracting headers
  sumOut=R.summary(model)[0]
  SSR=list(sumOut['Sum Sq'])
  output="Terms"+"\n" + "              " + "\t"
  maxLength=len("{:e}".format(SSR[0]))+1  
  for i in range(1, len(vars)):
    x=vars[i]
    output+= f"{x:>{maxLength}}"
  output+=f"{'Residuals':>{maxLength}}"+"\n" +"Sum of Squares"+"\t"
  SSResid=(float(SSR[len(SSR)-1]))
  for i in SSR: #extracting SSR values
    x=SSR.pop(0)
    xText="{:e}".format(x)
    SSR.append(xText)
    output+=f"{xText:>{maxLength}}"
  output+="\n"+"Deg. of Freedom"+"\t"
  degFree=list(sumOut['Df'])
  for i in range(len(degFree)):
    output+=f"{str(int(degFree[i])):>{maxLength}}"
  RSE=math.sqrt(SSResid/degFree[len(degFree)-1]) 
  #calculating residual standard error
  output+= "\n"+"\n"+"Residual standard error: "+str(round(RSE, 1)) +"\n"+\
  "Estimated effects may be unbalanced"
  return output
