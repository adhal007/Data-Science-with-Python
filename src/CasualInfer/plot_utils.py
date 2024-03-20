import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class GenericPlotter:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def plot_distribution(self, col: str): 
        ax = sns.distplot(self.data[col])
        iqr = np.percentile(self.data[col], 75) - np.percentile(self.data[col], 25)
        upper_bound = np.percentile(self.data[col], 75) + 3.0 * iqr
        lower_bound = np.percentile(self.data[col], 75) + 1.0 * iqr
        ax.axvline(x=np.mean(self.data[col]), color='r', linestyle='--', label='mean')
        ax.axvline(x=upper_bound, color='g', linestyle='--', label='tukey upper bound')
        ax.axvline(x=lower_bound, color='g', linestyle='--', label='tukey lower bound')
        ax.legend()
        plt.show()

    def plot_bar(self, col: str, x: str, y: str, hue: str, orient: str = 'v'):
        fig, ax = plt.subplots(figsize=(15, 5))
        ax = sns.barplot(data=self.data, x=x, y=y, hue=hue, orient=orient)
        ax.set_xlabel('{}'.format(x))
        # ax.axvline(x=-np.log10(0.05), color='r', linestyle='--', label='alpha = -np.log10(0.05)')
        ax.legend()
        plt.show()
    
    