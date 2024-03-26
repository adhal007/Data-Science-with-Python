import pandas as pd 
import numpy as np
from scipy.stats import ttest_ind
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


class Estimator:
    def __init__(self) -> None:
        self.data = None
        self.name = None 
        
    def calc_ps_scores(self):
        raise NotImplementedError("This method is not implemented")

