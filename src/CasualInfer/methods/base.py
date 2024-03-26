import pandas as pd 
import numpy as np
from scipy.stats import ttest_ind
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


#  4 different propensity score methods: 
# 1. matching on the propensity score, 
# 2. stratification on the propensity score, i
# 3. inverse probability of treatment weighting using the propensity score, and 
# 4. covariate adjustment using the propensity score
## Overview of sections(1-6): 
## Sec 1, I briefly describe the potential outcomes framework, causal treatment effects, RCTs, and observational studies
## Sec 2, I introduce the concept of the propensity score and describe four different methods in which it can be used to estimate treatment effects. 
## Sec 3, I describe methods to assess whether the propensity score model has been adequately specified. 
## Sec 4, I discuss variable selection for the propensity score model. 
## Sec 5, I compare the use of propensity score-based approaches with that of regression analyses in observational studies. 
## Sec 6, I summarize our discussion in the final section

## Definitions: For each subject, the effect of treatment is defined to be Yi(1) − Yi(0). The average treatment effect (ATE) is defined to be is E[Yi(1) − Yi(0)] (Imbens, 2004). 
## The ATE is the average effect, at the population level, of moving an entire population from untreated to treated. 
## A related measure of treatment effect is the average treatment effect for the treated (ATT; Imbens, 2004). The ATT is defined as E[Y(1)− 7(0)|Z = 1].
## When to use ATE vs ATT?
## 1. ATE is used when the goal is to estimate the effect of treatment on the entire population.
## 2. ATT is used when the goal is to estimate the effect of treatment on the treated population.
## Examples:
## 1. ATE: In contrast, when estimating the effect on smoking cessation of an information brochure given by family physicians to patients who are current smokers, the ATE may be of greater interest than the ATT
## 2. ATT: In estimating the effectiveness of an intensive, structured smoking cessation program, the ATT may be of greater interest than the ATE.
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3144483/


##### Other propensity score methods:
# Although logistic regression appears to be the most commonly used method for estimating the propensity score, 
# the use of bagging or boosting (Lee, Lessler, & Stuart, 2010; McCaffrey, Ridgeway, & Morral, 2004), 
# recursive partitioning or tree-based methods (Lee et al., 2010; Setoguchi, Schneeweiss, Brookhart, Glynn, & Cook, 2008), 
# random forests (Lee et al., 2010), and neural networks (Setoguchi et al., 2008) for estimating the propensity score have been examined.
class ScoreMethod:
    def __init__(self) -> None:
        self.data = None
        self.name = None 
        
    def calc_ps_scores(self):
        raise NotImplementedError("This method is not implemented")

