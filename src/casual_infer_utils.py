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

### Possible variations 
### 1. Propensity score(PS) without bootstrapping for large sample populations 
### 2. Propensity score(PS) with bootstrapping for small sample populations
class PropensityBase:
    def __init__(self, data:pd.DataFrame, index:str, features:list) -> None:
        self.data = data 
        self.features = features
        self.index = index

        self._bootstrap = None
        self._caliper = None
        self._X = None
        self._y = None
        self._coeffs = None

#### All properties are read-only    
    @property
    def bootstrap(self):
        if self._bootstrap is None:
            if self.X.shape[0] > 100:
                self._bootstrap = False
            else:
                self._bootstrap = True
            return self._bootstrap
    ## Can modify algorithm to use bootstrapping for small sample populations in the future 
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

    @property
    def caliper(self):
        if self._caliper is None:
            df = self.logistic_ps(self.data)
            calip = np.std(df.ps) * 0.25
            self._caliper = calip
        return self._caliper
       
## Checks for running algorithm    
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
        return 


    
## Calculation functions 
    def logit(self, p):
        logit_value = math.log(p / (1-p))
        return logit_value

    def logistic_ps(self, df):
        model = LogisticRegression()
        model.fit(self.X, self.y)
        df['ps'] = model.predict_proba(self.X)[:,1]
        df['ps_logit'] = df.ps.apply(lambda x: self.logit(x))
        return df

### Logic for defining optimal K in the KNN algorithm 
### Variables: n_neighbors, caliper (standard deviation of the propensity score)
### Cross-validation is used to determine the optimal number of neighbors
### The optimal number of neighbors is the one that minimizes the error rate
### The error rate is calculated as 1 - accuracy rate

    def optimal_k(self):
        ## calculate radius for Nearest Neighbors 
        accuracy_rate = []
        error_rate = []
        for i in range(1,40):
            knn = NearestNeighbors(n_neighbors=i, radius=self.caliper)
            score = cross_val_score(knn, self.X, self.y, cv=10, scoring='accuracy')
            accuracy_rate.append(score.mean())
            error_rate.append(1-score.mean())     
        best_k = np.argmin(error_rate) + 1
        return best_k
    
### K-Means vs KNN:
# https://towardsdatascience.com/k-means-vs-knn-which-one-is-better-c8f1b8c10e9f
# K-Means is used for clustering and KNN is used for classification.
# Optimal number of clusters/Neighbors is determined using the elbow method or silhouette score for K-Means and using cross-validation for KNN.
    
# ### optimal K using silhouette score
# silhouette_avg = []
# for num_clusters in range_n_clusters:
 
#  # initialise kmeans
#  kmeans = KMeans(n_clusters=num_clusters)
#  kmeans.fit(data_frame)
#  cluster_labels = kmeans.labels_
 
#  # silhouette score
#  silhouette_avg.append(silhouette_score(data_frame, cluster_labels))plt.plot(range_n_clusters,silhouette_avg,’bx-’)
# plt.xlabel(‘Values of K’) 
# plt.ylabel(‘Silhouette score’) 
# plt.title(‘Silhouette analysis For Optimal k’)
# plt.show()
    
    def knn_matched(self, k=None):
        if k is None:
            k = self.optimal_k()
        
        knn = NearestNeighbors(n_neighbors=k, radius=self.caliper)
        df = self.logistic_ps(self.data)
        ps = df[['ps']]
        knn.fit(ps)

        distances, neighbor_indexes = knn.kneighbors(ps)

        matched_control = []  # keep track of the matched observations in control

        for current_index, row in df.iterrows():  # iterate over the dataframe
            if row.treatment == 0:  # the current row is in the control group
                df.loc[current_index, 'matched'] = np.nan  # set matched to nan
            else: 
                for idx in neighbor_indexes[current_index, :]: # for each row in treatment, find the k neighbors
                    # make sure the current row is not the idx - don't match to itself
                    # and the neighbor is in the control 
                    if (current_index != idx) and (df.loc[idx].treatment == 0):
                        if idx not in matched_control:  # this control has not been matched yet
                            df.loc[current_index, 'matched'] = idx  # record the matching
                            matched_control.append(idx)  # add the matched to the list
                            break 
        # try to increase the number of neighbors and/or caliper to get more matches
        print('total observations in treatment:', len(df[df.treatment==1]))
        print('total matched observations in control:', len(matched_control))
        treatment_matched = df.dropna(subset=['matched'])  # drop not matched

        # matched control observation indexes
        control_matched_idx = treatment_matched.matched
        control_matched_idx = control_matched_idx.astype(int)  # change to int
        control_matched = df.loc[control_matched_idx, :]  # select matched control observations

        # combine the matched treatment and control
        df_matched = pd.concat([treatment_matched, control_matched])             
        return df_matched 


    def make_matched_data(self):
        df_matched = self.knn_matched(k=5)
        return df_matched
    
### Can Probably put this into its own plotting class (Too lazy to do it)
    def visualize_ps_overlap(self):
        sns.histplot(data=self.data, x='ps', hue='treatment')
        plt.show()
        return
# df_matched.treatment.value_counts()
# n_neighbors = 10
# # setup knn
# knn = NearestNeighbors(n_neighbors=n_neighbors, radius=caliper)

# ps = df[['ps']]  # double brackets as a dataframe
# knn.fit(ps)             
    
    # def fit_knn(self):
    #     best_k = self.optimal_k()
    #     knn = NearestNeighbors(n_neighbors=best_k, radius=self.caliper)
    #     ps = self.data[['ps']]
    #     knn.fit(ps)
    #     return knn
    
    # def get_matched_control(self, df):
    #     knn_obj = self.fit_knn()
    #     ps = self.data[['ps']]
    #     distances, neighbor_indexes = knn_obj.kneighbors(ps)

    #     matched_control = []  # keep track of the matched observations in control

    #     for current_index, row in df.iterrows():  # iterate over the dataframe
    #         if df.treatment == 0:  # the current row is in the control group
    #             self.data.loc[current_index, 'matched'] = np.nan  # set matched to nan
    #         else: 
    #             for idx in neighbor_indexes[current_index, :]: # for each row in treatment, find the k neighbors
    #                 # make sure the current row is not the idx - don't match to itself
    #                 # and the neighbor is in the control 
    #                 if (current_index != idx) and (df.loc[idx].treatment == 0):
    #                     if idx not in matched_control:  # this control has not been matched yet
    #                         df.loc[current_index, 'matched'] = idx  # record the matching
    #                         matched_control.append(idx)  # add the matched to the list
    #                         break
    #     return matched_control, df 

    # def get_matched_data(self):
    #     # control have no match
    #     treatment_matched = self.data.dropna(subset=['matched'])  # drop not matched

    #     # matched control observation indexes
    #     control_matched_idx = treatment_matched.matched
    #     control_matched_idx = control_matched_idx.astype(int)  # change to int
    #     control_matched = self.data.loc[control_matched_idx, :]  # select matched control observations

    #     # combine the matched treatment and control
    #     df_matched = pd.concat([treatment_matched, control_matched])
    #     return df_matched