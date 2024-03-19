```python
# load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
sns.set()  # set the style
```


```python
# read data
df = pd.read_csv('./Data-Science-with-Python/data/groupon.csv')
df.info()

```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 710 entries, 0 to 709
    Data columns (total 13 columns):
     #   Column           Non-Null Count  Dtype 
    ---  ------           --------------  ----- 
     0   deal_id          710 non-null    object
     1   start_date       710 non-null    object
     2   min_req          710 non-null    int64 
     3   treatment        710 non-null    int64 
     4   prom_length      710 non-null    int64 
     5   price            710 non-null    int64 
     6   discount_pct     710 non-null    int64 
     7   coupon_duration  710 non-null    int64 
     8   featured         710 non-null    int64 
     9   limited_supply   710 non-null    int64 
     10  fb_likes         710 non-null    int64 
     11  quantity_sold    710 non-null    int64 
     12  revenue          710 non-null    int64 
    dtypes: int64(11), object(2)
    memory usage: 72.2+ KB



```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>deal_id</th>
      <th>start_date</th>
      <th>min_req</th>
      <th>treatment</th>
      <th>prom_length</th>
      <th>price</th>
      <th>discount_pct</th>
      <th>coupon_duration</th>
      <th>featured</th>
      <th>limited_supply</th>
      <th>fb_likes</th>
      <th>quantity_sold</th>
      <th>revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>heli-flights</td>
      <td>9/23/2011</td>
      <td>10</td>
      <td>1</td>
      <td>4</td>
      <td>99</td>
      <td>51</td>
      <td>185</td>
      <td>1</td>
      <td>1</td>
      <td>290</td>
      <td>540</td>
      <td>53460</td>
    </tr>
    <tr>
      <th>1</th>
      <td>gl-champion-series-tennis-electric-factory</td>
      <td>9/23/2011</td>
      <td>20</td>
      <td>1</td>
      <td>2</td>
      <td>95</td>
      <td>41</td>
      <td>9</td>
      <td>0</td>
      <td>1</td>
      <td>43</td>
      <td>190</td>
      <td>18050</td>
    </tr>
    <tr>
      <th>2</th>
      <td>realm-of-terror-horror-experience</td>
      <td>9/23/2011</td>
      <td>50</td>
      <td>1</td>
      <td>3</td>
      <td>25</td>
      <td>50</td>
      <td>38</td>
      <td>0</td>
      <td>1</td>
      <td>208</td>
      <td>380</td>
      <td>9500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>newport-gourmet</td>
      <td>9/23/2011</td>
      <td>15</td>
      <td>1</td>
      <td>3</td>
      <td>50</td>
      <td>50</td>
      <td>369</td>
      <td>0</td>
      <td>1</td>
      <td>16</td>
      <td>90</td>
      <td>4500</td>
    </tr>
    <tr>
      <th>4</th>
      <td>the-clayroom</td>
      <td>9/23/2011</td>
      <td>20</td>
      <td>1</td>
      <td>4</td>
      <td>25</td>
      <td>52</td>
      <td>185</td>
      <td>0</td>
      <td>1</td>
      <td>85</td>
      <td>580</td>
      <td>14500</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>705</th>
      <td>whitewater-challengers-9</td>
      <td>5/2/2012</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>45</td>
      <td>54</td>
      <td>164</td>
      <td>0</td>
      <td>1</td>
      <td>110</td>
      <td>170</td>
      <td>7650</td>
    </tr>
    <tr>
      <th>706</th>
      <td>world-domination-events</td>
      <td>6/23/2012</td>
      <td>1</td>
      <td>0</td>
      <td>8</td>
      <td>149</td>
      <td>67</td>
      <td>52</td>
      <td>0</td>
      <td>1</td>
      <td>116</td>
      <td>150</td>
      <td>22350</td>
    </tr>
    <tr>
      <th>707</th>
      <td>xtreme-xperience-chicago</td>
      <td>7/27/2012</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>159</td>
      <td>60</td>
      <td>29</td>
      <td>0</td>
      <td>1</td>
      <td>104</td>
      <td>130</td>
      <td>20670</td>
    </tr>
    <tr>
      <th>708</th>
      <td>your-neighborhood-theatre</td>
      <td>4/12/2012</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>19</td>
      <td>51</td>
      <td>187</td>
      <td>0</td>
      <td>1</td>
      <td>93</td>
      <td>1000</td>
      <td>19000</td>
    </tr>
    <tr>
      <th>709</th>
      <td>yu-kids-island</td>
      <td>5/1/2012</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>28</td>
      <td>53</td>
      <td>99</td>
      <td>0</td>
      <td>1</td>
      <td>214</td>
      <td>820</td>
      <td>22960</td>
    </tr>
  </tbody>
</table>
<p>710 rows × 13 columns</p>
</div>




```python
df.iloc[:, 2:].groupby(['treatment']).mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>min_req</th>
      <th>prom_length</th>
      <th>price</th>
      <th>discount_pct</th>
      <th>coupon_duration</th>
      <th>featured</th>
      <th>limited_supply</th>
      <th>fb_likes</th>
      <th>quantity_sold</th>
      <th>revenue</th>
    </tr>
    <tr>
      <th>treatment</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.00000</td>
      <td>3.809717</td>
      <td>37.870445</td>
      <td>52.973684</td>
      <td>95.941296</td>
      <td>0.105263</td>
      <td>0.852227</td>
      <td>77.941296</td>
      <td>333.002024</td>
      <td>9720.987854</td>
    </tr>
    <tr>
      <th>1</th>
      <td>26.50463</td>
      <td>3.379630</td>
      <td>29.421296</td>
      <td>53.263889</td>
      <td>131.842593</td>
      <td>0.143519</td>
      <td>0.777778</td>
      <td>113.203704</td>
      <td>509.351852</td>
      <td>12750.694444</td>
    </tr>
  </tbody>
</table>
</div>




```python
# separate control and treatment for t-test
df_control = df[df.treatment==0]
df_treatment = df[df.treatment==1]
```


```python
# student's t-test for revenue (dependent variable)
from scipy.stats import ttest_ind

print(df_control.revenue.mean(), df_treatment.revenue.mean())

# compare samples
_, p = ttest_ind(df_control.revenue, df_treatment.revenue)
print(f'p={p:.3f}')

# interpret
alpha = 0.05  # significance level
if p > alpha:
    print('same distributions/same group mean (fail to reject H0 - we do not have enough evidence to reject H0)')
else:
    print('different distributions/different group mean (reject H0)')
```

    9720.987854251012 12750.694444444445
    p=0.040
    different distributions/different group mean (reject H0)



```python
# student's t-test for facebook likes (dependent variable)
from scipy.stats import ttest_ind

print(df_control.fb_likes.mean(), df_treatment.fb_likes.mean())

# compare samples
_, p = ttest_ind(df_control.fb_likes, df_treatment.fb_likes)
print(f'p={p:.3f}')

# interpret
alpha = 0.05  # significance level
if p > alpha:
    print('same distributions/same group mean (fail to reject H0 - we do not have enough evidence to reject H0)')
else:
    print('different distributions/different group mean (reject H0)')

```

    77.9412955465587 113.20370370370371
    p=0.004
    different distributions/different group mean (reject H0)



```python
# choose features for propensity score calculation
X = df[['prom_length', 'price', 'discount_pct', 'coupon_duration', 'featured', 'limited_supply']]
y = df['treatment']

X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prom_length</th>
      <th>price</th>
      <th>discount_pct</th>
      <th>coupon_duration</th>
      <th>featured</th>
      <th>limited_supply</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>99</td>
      <td>51</td>
      <td>185</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>95</td>
      <td>41</td>
      <td>9</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>25</td>
      <td>50</td>
      <td>38</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>50</td>
      <td>50</td>
      <td>369</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>25</td>
      <td>52</td>
      <td>185</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# use logistic regression to calculate the propensity scores
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X, y)

```




<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression()</pre></div></div></div></div></div>




```python
# get the coefficients 
lr.coef_.ravel()
```




    array([-0.32837139, -0.0085971 , -0.00794036,  0.00352025,  0.30799496,
           -0.3664981 ])




```python
# get the feature names
X.columns.to_numpy()
```




    array(['prom_length', 'price', 'discount_pct', 'coupon_duration',
           'featured', 'limited_supply'], dtype=object)




```python
# combine features and coefficients into a dataframe
coeffs = pd.DataFrame({
    'column':X.columns.to_numpy(),
    'coeff':lr.coef_.ravel(),
})
coeffs
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>column</th>
      <th>coeff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>prom_length</td>
      <td>-0.328371</td>
    </tr>
    <tr>
      <th>1</th>
      <td>price</td>
      <td>-0.008597</td>
    </tr>
    <tr>
      <th>2</th>
      <td>discount_pct</td>
      <td>-0.007940</td>
    </tr>
    <tr>
      <th>3</th>
      <td>coupon_duration</td>
      <td>0.003520</td>
    </tr>
    <tr>
      <th>4</th>
      <td>featured</td>
      <td>0.307995</td>
    </tr>
    <tr>
      <th>5</th>
      <td>limited_supply</td>
      <td>-0.366498</td>
    </tr>
  </tbody>
</table>
</div>




```python
# prediction
pred_binary = lr.predict(X)  # binary 0 control, 1, treatment
pred_prob = lr.predict_proba(X)  # probabilities for classes

print('the binary prediction is:', pred_binary[0])
print('the corresponding probabilities are:', pred_prob[0])
```

    the binary prediction is: 0
    the corresponding probabilities are: [0.7408075 0.2591925]



```python
# the propensity score (ps) is the probability of being 1 (i.e., in the treatment group)
df['ps'] = pred_prob[:, 1]

# calculate the logit of the propensity score for matching if needed
# I just use the propensity score to match in this tutorial
def logit(p):
    logit_value = math.log(p / (1-p))
    return logit_value

df['ps_logit'] = df.ps.apply(lambda x: logit(x))

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>deal_id</th>
      <th>start_date</th>
      <th>min_req</th>
      <th>treatment</th>
      <th>prom_length</th>
      <th>price</th>
      <th>discount_pct</th>
      <th>coupon_duration</th>
      <th>featured</th>
      <th>limited_supply</th>
      <th>fb_likes</th>
      <th>quantity_sold</th>
      <th>revenue</th>
      <th>ps</th>
      <th>ps_logit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>heli-flights</td>
      <td>9/23/2011</td>
      <td>10</td>
      <td>1</td>
      <td>4</td>
      <td>99</td>
      <td>51</td>
      <td>185</td>
      <td>1</td>
      <td>1</td>
      <td>290</td>
      <td>540</td>
      <td>53460</td>
      <td>0.259192</td>
      <td>-1.050170</td>
    </tr>
    <tr>
      <th>1</th>
      <td>gl-champion-series-tennis-electric-factory</td>
      <td>9/23/2011</td>
      <td>20</td>
      <td>1</td>
      <td>2</td>
      <td>95</td>
      <td>41</td>
      <td>9</td>
      <td>0</td>
      <td>1</td>
      <td>43</td>
      <td>190</td>
      <td>18050</td>
      <td>0.230198</td>
      <td>-1.207194</td>
    </tr>
    <tr>
      <th>2</th>
      <td>realm-of-terror-horror-experience</td>
      <td>9/23/2011</td>
      <td>50</td>
      <td>1</td>
      <td>3</td>
      <td>25</td>
      <td>50</td>
      <td>38</td>
      <td>0</td>
      <td>1</td>
      <td>208</td>
      <td>380</td>
      <td>9500</td>
      <td>0.288405</td>
      <td>-0.903144</td>
    </tr>
    <tr>
      <th>3</th>
      <td>newport-gourmet</td>
      <td>9/23/2011</td>
      <td>15</td>
      <td>1</td>
      <td>3</td>
      <td>50</td>
      <td>50</td>
      <td>369</td>
      <td>0</td>
      <td>1</td>
      <td>16</td>
      <td>90</td>
      <td>4500</td>
      <td>0.511781</td>
      <td>0.047131</td>
    </tr>
    <tr>
      <th>4</th>
      <td>the-clayroom</td>
      <td>9/23/2011</td>
      <td>20</td>
      <td>1</td>
      <td>4</td>
      <td>25</td>
      <td>52</td>
      <td>185</td>
      <td>0</td>
      <td>1</td>
      <td>85</td>
      <td>580</td>
      <td>14500</td>
      <td>0.325212</td>
      <td>-0.729919</td>
    </tr>
  </tbody>
</table>
</div>




```python
# check the overlap of ps for control and treatment using histogram
# if not much overlap, the matching won't work
sns.histplot(data=df, x='ps', hue='treatment')  # multiple="dodge" for 
```

    /opt/homebrew/anaconda3/envs/sklearn-env/lib/python3.12/site-packages/seaborn/_base.py:948: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /opt/homebrew/anaconda3/envs/sklearn-env/lib/python3.12/site-packages/seaborn/_base.py:948: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)





    <Axes: xlabel='ps', ylabel='Count'>




    
![png](propensity_score_matching_files/propensity_score_matching_14_2.png)
    



```python
#  adding 'min_req' here makes matching not working - because treatment is derived from min_req
# there is no overlap and thus matching will not work
X1 = df[['min_req', 'prom_length', 'price', 'discount_pct', 'coupon_duration', 'featured','limited_supply']]
y = df['treatment']

# use logistic regression to calculate the propensity scores
lr1 = LogisticRegression(max_iter=1000)
lr1.fit(X1, y)

pred_prob1 = lr1.predict_proba(X1)  # probabilities for classes
df['ps1'] = pred_prob1[:, 1]

sns.histplot(data=df, x='ps1', hue='treatment')
```

    /opt/homebrew/anaconda3/envs/sklearn-env/lib/python3.12/site-packages/seaborn/_base.py:948: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /opt/homebrew/anaconda3/envs/sklearn-env/lib/python3.12/site-packages/seaborn/_base.py:948: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)





    <Axes: xlabel='ps1', ylabel='Count'>




    
![png](propensity_score_matching_files/propensity_score_matching_15_2.png)
    



```python
# stating date can also determine treatment value for most cases
# so we do not include it in the propensity score calculation
df.start_date = pd.to_datetime(df.start_date)
fig, ax = plt.subplots(figsize=(20, 10))
sns.scatterplot(data=df, x='start_date', y='revenue', hue='treatment')
```




    <Axes: xlabel='start_date', ylabel='revenue'>




    
![png](propensity_score_matching_files/propensity_score_matching_16_1.png)
    



```python
# adding 'starting_date' here via a `recency` feature
# there is only little overlap resulting in not enough matched observations

last_date = df.start_date.max()
df['recency'] = (last_date - df.start_date).dt.days

X2 = df[['recency', 'prom_length', 'price', 'discount_pct', 'coupon_duration', 'featured','limited_supply']]
y = df['treatment']

# use logistic regression to calculate the propensity scores
lr2 = LogisticRegression(max_iter=1000)
lr2.fit(X2, y)

pred_prob2 = lr2.predict_proba(X2)  # probabilities for classes
df['ps2'] = pred_prob2[:, 1]

sns.histplot(data=df, x='ps2', hue='treatment')
```

    /opt/homebrew/anaconda3/envs/sklearn-env/lib/python3.12/site-packages/seaborn/_base.py:948: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /opt/homebrew/anaconda3/envs/sklearn-env/lib/python3.12/site-packages/seaborn/_base.py:948: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)





    <Axes: xlabel='ps2', ylabel='Count'>




    
![png](propensity_score_matching_files/propensity_score_matching_17_2.png)
    



```python
# use 25% of standard deviation of the propensity score as the caliper/radius
# get the k closest neighbors for each observations
# relax caliper and increase k can provide more matches

from sklearn.neighbors import NearestNeighbors

caliper = np.std(df.ps) * 0.25
print(f'caliper (radius) is: {caliper:.4f}')

n_neighbors = 10

# setup knn
knn = NearestNeighbors(n_neighbors=n_neighbors, radius=caliper)

ps = df[['ps']]  # double brackets as a dataframe
knn.fit(ps)

```

    caliper (radius) is: 0.0304





<style>#sk-container-id-2 {color: black;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>NearestNeighbors(n_neighbors=10, radius=0.030379121102554488)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">NearestNeighbors</label><div class="sk-toggleable__content"><pre>NearestNeighbors(n_neighbors=10, radius=0.030379121102554488)</pre></div></div></div></div></div>




```python

```
