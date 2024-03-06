import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import math
from scipy.stats import ttest_ind
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

class PropensityBase:
    def __init__(self, data:pd.DataFrame, features:list, caliper:float) -> None:
        self.data = data 
        self.features = features
        self.caliper = caliper
        self._X = None
        self._y = None
        self._coeffs = None
        if self.caliper is None or self.caliper <= 0:
            raise ValueError('Caliper must be a positive number')
    
    @property
    def control_data(self):
        return self.data[self.data['treatment'] == 0]
    
    @property
    def treatment_data(self):
        return self.data[self.data['treatment'] == 1]
    
    @property
    def X(self):
        if self._X is None:
            self._X = self.data[self.features]
        return self._X
    
    @property
    def y(self):
        if self._y is None:
            self._y = self.data['treatment']
        return self._y
        
    @property
    def coeffs(self):
        if self._coeffs is None:
            lr = LogisticRegression()
            lr.fit(self.X, self.y)
            self._coeffs = pd.DataFrame({
                'column':self.X.columns.to_numpy(),
                'coeff':lr.coef_.ravel(),
            })
            return self._coeffs
    
    def check_features(self):
        if len(self.features) > 0:
            return True
        else:
            return False
        
    def check_columns(self):
        if 'treatment' in self.data.columns:
            return True
        else:
            return False
    
    def check_treatment_types(self):
        if self.data['treatment'].nunique() == 2:
            return True
        else:
            return False
    
    def check_students_t_test(self, df_control, df_treatment, column):
        print(df_control[column].mean(), df_treatment[column].mean())

        # compare samples
        _, p = ttest_ind(df_control[column], df_treatment[column])
        print(f'p={p:.3f}')

        # interpret
        alpha = 0.05  # significance level
        if p > alpha:
            print('same distributions/same group mean (fail to reject H0 - we do not have enough evidence to reject H0)')
        else:
            print('different distributions/different group mean (reject H0)')
    
    def visualize_ps_overlap(self):
        sns.histplot(data=self.df, x='ps', hue='treatment')
        plt.show()
        return
     
    @staticmethod
    def logit(p):
        logit_value = math.log(p / (1-p))
        return logit_value

    def calc_ps(self):
        model = LogisticRegression()
        model.fit(self.X, self.y)
        self.data['ps'] = model.predict_proba(self.X)[:,1]
        self.data['ps_logit'] = self.data.ps.apply(lambda x: self.logit(x))
        return self.data

    def optimal_k(self):
        accuracy_rate = []
        error_rate = []
        for i in range(1,40):
            
            knn = NearestNeighbors(n_neighbors=i, radius=self.caliper)
            score = cross_val_score(knn, self.X, self.y, cv=10)
            accuracy_rate.append(score.mean())
            error_rate.append(1-score.mean())     
        best_k = np.argmin(error_rate) + 1
        return best_k
    
    def fit_knn(self):
        best_k = self.optimal_k()
        knn = NearestNeighbors(n_neighbors=best_k, radius=caliper)
        ps = self.data[['ps']]
        knn.fit(ps)
        return knn

# n_neighbors = 10

# # setup knn
# knn = NearestNeighbors(n_neighbors=n_neighbors, radius=caliper)

# ps = df[['ps']]  # double brackets as a dataframe
# knn.fit(ps) 
            