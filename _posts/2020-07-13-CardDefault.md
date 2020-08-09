---
title: "Credit Card Default"
date: 2020-07-13 08:00:30 -0400
classes: wide
toc: true
categories: DataScience
---



# Introduction

## Dataset

This dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from __April 2005__ to __September 2005.__

## Variables

__ID__: Customer ID

__LIMIT_BAL__: Amount of given credit in NT dollars (includes individual and family/supplementary credit)

-----
 __SEX__: Gender (1=male, 2=female)

 __EDUCATION__: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)

 __MARRIAGE__: Marital Status (1=married, 2=single, 3=others)

 __AGE__: Age
 
-----
 __PAY_0__: Repayment status in September, 2005  
 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months,.. 8=payment delay for eight months, 9=payment delay for nine months and above)

 __PAY_2__: Repayment status in August, 2005

 __PAY_3__: Repayment status in July, 2005

 __PAY_4__: Repayment status in June, 2005

 __PAY_5__: Repayment status in May, 2005

 __PAY_6__: Repayment status in April, 2005

-----
 __BILL_AMT1__: Amount of bill statement in September, 2005 (NT dollar)

 __BILL_AMT2__: Amount of bill statement in August, 2005 (NT dollar)

 __BILL_AMT3__: Amount of bill statement in July, 2005 (NT dollar)

 __BILL_AMT4__: Amount of bill statement in June, 2005 (NT dollar)

 __BILL_AMT5__: Amount of bill statement in May, 2005 (NT dollar)

 __BILL_AMT6__: Amount of bill statement in April, 2005 (NT dollar)

-----
 __PAY_AMT1__: Amount of previous payment in September, 2005 (NT dollar)

 __PAY_AMT2__: Amount of previous payment in August, 2005 (NT dollar)

 __PAY_AMT3__: Amount of previous payment in July, 2005 (NT dollar)

 __PAY_AMT4__: Amount of previous payment in June, 2005 (NT dollar)

 __PAY_AMT5__: Amount of previous payment in May, 2005 (NT dollar)

 __PAY_AMT6__: Amount of previous payment in April, 2005 (NT dollar)

-----
 __default.payment.next.month__: Default payment (1=yes, 0=no)

# Load Packages


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

import warnings
```


```python
%matplotlib inline
```


```python
warnings.filterwarnings('ignore')
```

# Read Data


```python
df = pd.read_csv('UCI_Credit_Card.csv')
```

## Check Data (Find Missing Data, etc.)


```python
print(df.shape)
df.head()
```

    (30000, 25)
    




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
      <th>ID</th>
      <th>LIMIT_BAL</th>
      <th>SEX</th>
      <th>EDUCATION</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>PAY_0</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>...</th>
      <th>BILL_AMT4</th>
      <th>BILL_AMT5</th>
      <th>BILL_AMT6</th>
      <th>PAY_AMT1</th>
      <th>PAY_AMT2</th>
      <th>PAY_AMT3</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
      <th>default.payment.next.month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>20000.0</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>24</td>
      <td>2</td>
      <td>2</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>689.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>120000.0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>26</td>
      <td>-1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3272.0</td>
      <td>3455.0</td>
      <td>3261.0</td>
      <td>0.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>0.0</td>
      <td>2000.0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>90000.0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>14331.0</td>
      <td>14948.0</td>
      <td>15549.0</td>
      <td>1518.0</td>
      <td>1500.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>5000.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>50000.0</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>28314.0</td>
      <td>28959.0</td>
      <td>29547.0</td>
      <td>2000.0</td>
      <td>2019.0</td>
      <td>1200.0</td>
      <td>1100.0</td>
      <td>1069.0</td>
      <td>1000.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>50000.0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>57</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>...</td>
      <td>20940.0</td>
      <td>19146.0</td>
      <td>19131.0</td>
      <td>2000.0</td>
      <td>36681.0</td>
      <td>10000.0</td>
      <td>9000.0</td>
      <td>689.0</td>
      <td>679.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>




```python
df.describe()
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
      <th>ID</th>
      <th>LIMIT_BAL</th>
      <th>SEX</th>
      <th>EDUCATION</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>PAY_0</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>...</th>
      <th>BILL_AMT4</th>
      <th>BILL_AMT5</th>
      <th>BILL_AMT6</th>
      <th>PAY_AMT1</th>
      <th>PAY_AMT2</th>
      <th>PAY_AMT3</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
      <th>default.payment.next.month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>...</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>3.000000e+04</td>
      <td>30000.00000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>15000.500000</td>
      <td>167484.322667</td>
      <td>1.603733</td>
      <td>1.853133</td>
      <td>1.551867</td>
      <td>35.485500</td>
      <td>-0.016700</td>
      <td>-0.133767</td>
      <td>-0.166200</td>
      <td>-0.220667</td>
      <td>...</td>
      <td>43262.948967</td>
      <td>40311.400967</td>
      <td>38871.760400</td>
      <td>5663.580500</td>
      <td>5.921163e+03</td>
      <td>5225.68150</td>
      <td>4826.076867</td>
      <td>4799.387633</td>
      <td>5215.502567</td>
      <td>0.221200</td>
    </tr>
    <tr>
      <td>std</td>
      <td>8660.398374</td>
      <td>129747.661567</td>
      <td>0.489129</td>
      <td>0.790349</td>
      <td>0.521970</td>
      <td>9.217904</td>
      <td>1.123802</td>
      <td>1.197186</td>
      <td>1.196868</td>
      <td>1.169139</td>
      <td>...</td>
      <td>64332.856134</td>
      <td>60797.155770</td>
      <td>59554.107537</td>
      <td>16563.280354</td>
      <td>2.304087e+04</td>
      <td>17606.96147</td>
      <td>15666.159744</td>
      <td>15278.305679</td>
      <td>17777.465775</td>
      <td>0.415062</td>
    </tr>
    <tr>
      <td>min</td>
      <td>1.000000</td>
      <td>10000.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>21.000000</td>
      <td>-2.000000</td>
      <td>-2.000000</td>
      <td>-2.000000</td>
      <td>-2.000000</td>
      <td>...</td>
      <td>-170000.000000</td>
      <td>-81334.000000</td>
      <td>-339603.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>7500.750000</td>
      <td>50000.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>28.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>...</td>
      <td>2326.750000</td>
      <td>1763.000000</td>
      <td>1256.000000</td>
      <td>1000.000000</td>
      <td>8.330000e+02</td>
      <td>390.00000</td>
      <td>296.000000</td>
      <td>252.500000</td>
      <td>117.750000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>15000.500000</td>
      <td>140000.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>34.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>19052.000000</td>
      <td>18104.500000</td>
      <td>17071.000000</td>
      <td>2100.000000</td>
      <td>2.009000e+03</td>
      <td>1800.00000</td>
      <td>1500.000000</td>
      <td>1500.000000</td>
      <td>1500.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>22500.250000</td>
      <td>240000.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>41.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>54506.000000</td>
      <td>50190.500000</td>
      <td>49198.250000</td>
      <td>5006.000000</td>
      <td>5.000000e+03</td>
      <td>4505.00000</td>
      <td>4013.250000</td>
      <td>4031.500000</td>
      <td>4000.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>30000.000000</td>
      <td>1000000.000000</td>
      <td>2.000000</td>
      <td>6.000000</td>
      <td>3.000000</td>
      <td>79.000000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>...</td>
      <td>891586.000000</td>
      <td>927171.000000</td>
      <td>961664.000000</td>
      <td>873552.000000</td>
      <td>1.684259e+06</td>
      <td>896040.00000</td>
      <td>621000.000000</td>
      <td>426529.000000</td>
      <td>528666.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 25 columns</p>
</div>




```python
print('default amount: ', len(df[df['default.payment.next.month'] == 1]))
print('default ratio: ', len(df[df['default.payment.next.month'] == 1]) / len(df))
```

    default amount:  6636
    default ratio:  0.2212
    

- __Implication__

    There are 30,000 distinct credit card customers.

    The average amount of given credit, as implied in "LIMIT_BAL" columns, is 167,484 but we have to note that the amount varies from minimum 10,000 to maximum 1,000,000.   

    The amounts of credit card default are 6636, which takes arround 22% of total credit contracts.

    The average age of clients are 35.5 and most of the clients have received university & graduate school-level education. 




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 30000 entries, 0 to 29999
    Data columns (total 25 columns):
    ID                            30000 non-null int64
    LIMIT_BAL                     30000 non-null float64
    SEX                           30000 non-null int64
    EDUCATION                     30000 non-null int64
    MARRIAGE                      30000 non-null int64
    AGE                           30000 non-null int64
    PAY_0                         30000 non-null int64
    PAY_2                         30000 non-null int64
    PAY_3                         30000 non-null int64
    PAY_4                         30000 non-null int64
    PAY_5                         30000 non-null int64
    PAY_6                         30000 non-null int64
    BILL_AMT1                     30000 non-null float64
    BILL_AMT2                     30000 non-null float64
    BILL_AMT3                     30000 non-null float64
    BILL_AMT4                     30000 non-null float64
    BILL_AMT5                     30000 non-null float64
    BILL_AMT6                     30000 non-null float64
    PAY_AMT1                      30000 non-null float64
    PAY_AMT2                      30000 non-null float64
    PAY_AMT3                      30000 non-null float64
    PAY_AMT4                      30000 non-null float64
    PAY_AMT5                      30000 non-null float64
    PAY_AMT6                      30000 non-null float64
    default.payment.next.month    30000 non-null int64
    dtypes: float64(13), int64(12)
    memory usage: 5.7 MB
    

There are no missing data in the entire dataset

# EDA & Preprocessing


```python
df.columns
```




    Index(['ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',
           'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
           'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
           'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
           'default.payment.next.month'],
          dtype='object')




```python
features = ['ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 
            'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 
            'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 
            'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
```


```python
# Customer Characteristics
# Education (1,2,3,4,5,6) -> 0 should not be included
# Marriage (1,2,3) -> 0 should not be included

df[['SEX', 'EDUCATION', 'MARRIAGE', 'AGE']].describe()
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
      <th>SEX</th>
      <th>EDUCATION</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>1.603733</td>
      <td>1.853133</td>
      <td>1.551867</td>
      <td>35.485500</td>
    </tr>
    <tr>
      <td>std</td>
      <td>0.489129</td>
      <td>0.790349</td>
      <td>0.521970</td>
      <td>9.217904</td>
    </tr>
    <tr>
      <td>min</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>21.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>28.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>34.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>41.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>2.000000</td>
      <td>6.000000</td>
      <td>3.000000</td>
      <td>79.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Rapayment Status Characteristics 
# pay (-1,1,2,3,4,5,6,7,8) -> -2? 

df[['PAY_0','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']].describe()
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
      <th>PAY_0</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>PAY_5</th>
      <th>PAY_6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>-0.016700</td>
      <td>-0.133767</td>
      <td>-0.166200</td>
      <td>-0.220667</td>
      <td>-0.266200</td>
      <td>-0.291100</td>
    </tr>
    <tr>
      <td>std</td>
      <td>1.123802</td>
      <td>1.197186</td>
      <td>1.196868</td>
      <td>1.169139</td>
      <td>1.133187</td>
      <td>1.149988</td>
    </tr>
    <tr>
      <td>min</td>
      <td>-2.000000</td>
      <td>-2.000000</td>
      <td>-2.000000</td>
      <td>-2.000000</td>
      <td>-2.000000</td>
      <td>-2.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>8.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Bill Statement Characteristics
# 사용한 금액 아닌가? (-)가 왜 나오지?

df[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].describe()
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
      <th>BILL_AMT1</th>
      <th>BILL_AMT2</th>
      <th>BILL_AMT3</th>
      <th>BILL_AMT4</th>
      <th>BILL_AMT5</th>
      <th>BILL_AMT6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>3.000000e+04</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>51223.330900</td>
      <td>49179.075167</td>
      <td>4.701315e+04</td>
      <td>43262.948967</td>
      <td>40311.400967</td>
      <td>38871.760400</td>
    </tr>
    <tr>
      <td>std</td>
      <td>73635.860576</td>
      <td>71173.768783</td>
      <td>6.934939e+04</td>
      <td>64332.856134</td>
      <td>60797.155770</td>
      <td>59554.107537</td>
    </tr>
    <tr>
      <td>min</td>
      <td>-165580.000000</td>
      <td>-69777.000000</td>
      <td>-1.572640e+05</td>
      <td>-170000.000000</td>
      <td>-81334.000000</td>
      <td>-339603.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>3558.750000</td>
      <td>2984.750000</td>
      <td>2.666250e+03</td>
      <td>2326.750000</td>
      <td>1763.000000</td>
      <td>1256.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>22381.500000</td>
      <td>21200.000000</td>
      <td>2.008850e+04</td>
      <td>19052.000000</td>
      <td>18104.500000</td>
      <td>17071.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>67091.000000</td>
      <td>64006.250000</td>
      <td>6.016475e+04</td>
      <td>54506.000000</td>
      <td>50190.500000</td>
      <td>49198.250000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>964511.000000</td>
      <td>983931.000000</td>
      <td>1.664089e+06</td>
      <td>891586.000000</td>
      <td>927171.000000</td>
      <td>961664.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# previous payment Characteristics

df[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']].describe()
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
      <th>PAY_AMT1</th>
      <th>PAY_AMT2</th>
      <th>PAY_AMT3</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>30000.000000</td>
      <td>3.000000e+04</td>
      <td>30000.00000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>5663.580500</td>
      <td>5.921163e+03</td>
      <td>5225.68150</td>
      <td>4826.076867</td>
      <td>4799.387633</td>
      <td>5215.502567</td>
    </tr>
    <tr>
      <td>std</td>
      <td>16563.280354</td>
      <td>2.304087e+04</td>
      <td>17606.96147</td>
      <td>15666.159744</td>
      <td>15278.305679</td>
      <td>17777.465775</td>
    </tr>
    <tr>
      <td>min</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>1000.000000</td>
      <td>8.330000e+02</td>
      <td>390.00000</td>
      <td>296.000000</td>
      <td>252.500000</td>
      <td>117.750000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>2100.000000</td>
      <td>2.009000e+03</td>
      <td>1800.00000</td>
      <td>1500.000000</td>
      <td>1500.000000</td>
      <td>1500.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>5006.000000</td>
      <td>5.000000e+03</td>
      <td>4505.00000</td>
      <td>4013.250000</td>
      <td>4031.500000</td>
      <td>4000.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>873552.000000</td>
      <td>1.684259e+06</td>
      <td>896040.00000</td>
      <td>621000.000000</td>
      <td>426529.000000</td>
      <td>528666.000000</td>
    </tr>
  </tbody>
</table>
</div>



## 1. Check Data Unbalance


```python
value = df['default.payment.next.month'].value_counts()
amount = pd.DataFrame({'default.payment.next.month' : value.index, 'values' : value.values})
total = len(df)
plt.figure(figsize = (6,6))
plt.title('Check Data Unbalance [Default=1, Not Default=0 ]')
a = sns.barplot(data = amount, x = 'default.payment.next.month', y = 'values' )

for index,row in amount.iterrows():
    a.text(row['default.payment.next.month'],row['values'], '{:1.2f}'.format(row['values']/total), color='black', ha="center")

plt.show()
```


![png](output_26_0.png)


Since the target value unbalance is acceptable, we will go as it is. 

## 2. Data Exploration

### 1) LIMIT_BAL


```python
plt.figure(figsize = (12,6))
sns.set_context('paper', font_scale=1.5)

sns.distplot(a = df['LIMIT_BAL'])
plt.xlim([0,600000])
plt.title('Amount of Balance')
plt.show()
```


![png](output_30_0.png)



```python
x1 = list(df[df['default.payment.next.month'] == 1]['LIMIT_BAL'])
x2 = list(df[df['default.payment.next.month'] == 0]['LIMIT_BAL'])

plt.figure(figsize = (12,6))
sns.set_context('paper', font_scale=1.2)
plt.hist([x1, x2], bins = 40,  color=['steelblue', 'lightblue'])
plt.xlim([0,600000])
plt.legend(['Yes', 'No'], title = 'Default', loc='upper right', facecolor='white')
plt.title('Amount of Balance', size=15)

plt.show()
```


![png](output_31_0.png)


## 2) Client Information (Gender, Marriage, Education, Age)


```python
df.groupby(df['SEX']).size()
```




    SEX
    1    11888
    2    18112
    dtype: int64




```python
figure, ((ax1,ax2)) = plt.subplots(nrows=1, ncols=2)
figure.set_size_inches(14,8)

sns.countplot(data=df, x = 'SEX', ax = ax1)
sns.barplot(data=df, x = 'SEX', y = 'default.payment.next.month', ax = ax2)
ax1.set(title = 'Gender Size \n [Male=1, Female=2 ]')
ax2.set(title = 'Default Ratio by Gender \n [Male=1, Female=2 ]')

plt.show()
```


![png](output_34_0.png)



```python
figure, ((ax1,ax2)) = plt.subplots(nrows=1, ncols=2)
figure.set_size_inches(14,6)

sns.set_context('paper', font_scale=1.2)


sns.countplot(data=df, x = 'MARRIAGE', hue = 'default.payment.next.month', ax = ax1)

sns.countplot(data = df, x = 'EDUCATION', hue = 'default.payment.next.month', ax = ax2)
ax1.legend(['No', 'Yes'], title = 'Default', loc='upper right', facecolor='white')
plt.legend(['No', 'Yes'], title = 'Default', loc='upper right', facecolor='white')

figure, ((ax3)) = plt.subplots(nrows=1, ncols=1)
figure.set_size_inches(14,6)
sns.countplot(data = df, x = 'AGE', hue = 'default.payment.next.month', palette = 'pastel', ax = ax3)


ax1.set(title = 'Default Ratio by Marriage \n [Married=1, Single=2, Others=3]')
ax2.set(title = 'Default Ratio by Education \n [graduate school=1, university=2, high school=3, others=4, unknown=5, unknown=6]')
ax3.set(title = 'Age Distribution')
plt.legend(['No', 'Yes'], title = 'Default', loc='upper right', facecolor='white')
plt.show()


plt.show()
```


![png](output_35_0.png)



![png](output_35_1.png)



```python
a = pd.pivot_table(df, index="AGE", values="default.payment.next.month")
plt.figure(figsize = (12,6))
sns.barplot(data = a, x = a.index, y = 'default.payment.next.month', color = 'steelblue')
plt.title('Default Rate by Age')
plt.show()
```


![png](output_36_0.png)


#### In the education, 4,5,6 can be categorized as single value 4.

#### In the Marriage, there should be only 3 categories (1,2,3). So the value 0 can be categorized to 3. 


```python
df['EDUCATION']=np.where(df['EDUCATION'] == 5, 4, df['EDUCATION'])
df['EDUCATION']=np.where(df['EDUCATION'] == 6, 4, df['EDUCATION'])
df['EDUCATION']=np.where(df['EDUCATION'] == 0, 4, df['EDUCATION'])

df['MARRIAGE'] = np.where(df['MARRIAGE'] == 0, 3, df['MARRIAGE'])
```


```python
figure, ((ax1,ax2)) = plt.subplots(nrows=1, ncols=2)
figure.set_size_inches(14,6)
sns.set_context('paper', font_scale=1.2)

sns.countplot(data = df, x = 'MARRIAGE', hue = 'default.payment.next.month', ax = ax1)
sns.countplot(data = df, x = 'EDUCATION', hue = 'default.payment.next.month', ax = ax2)

ax1.set(title = 'Default Ratio by Marriage \n [Married=1, Single=2, Others=3]')
ax2.set(title = 'Default Ratio by Education \n [graduate school=1, university=2, high school=3, others=4]')

ax1.legend(['No', 'Yes'], title = 'Default', loc='upper right', facecolor='white')
ax2.legend(['No', 'Yes'], title = 'Default', loc='upper right', facecolor='white')

plt.show()
```


![png](output_39_0.png)


## 3) Repayment Status


```python
figure, ((ax1,ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3)
figure.set_size_inches(16,6)

sns.countplot(x="PAY_0", hue="default.payment.next.month", data=df, palette="pastel", ax=ax1)
sns.countplot(x="PAY_2", hue="default.payment.next.month", data=df, palette="pastel", ax=ax2)
sns.countplot(x="PAY_3", hue="default.payment.next.month", data=df, palette="pastel", ax=ax3)
sns.countplot(x="PAY_4", hue="default.payment.next.month", data=df, palette="pastel", ax=ax4)
sns.countplot(x="PAY_5", hue="default.payment.next.month", data=df, palette="pastel", ax=ax5)
sns.countplot(x="PAY_6", hue="default.payment.next.month", data=df, palette="pastel", ax=ax6)

ax1.legend(['No', 'Yes'], title = 'Default', loc='upper right', facecolor='white')
ax2.legend(['No', 'Yes'], title = 'Default', loc='upper right', facecolor='white')
ax3.legend(['No', 'Yes'], title = 'Default', loc='upper right', facecolor='white')
ax4.legend(['No', 'Yes'], title = 'Default', loc='upper right', facecolor='white')
ax5.legend(['No', 'Yes'], title = 'Default', loc='upper right', facecolor='white')
ax6.legend(['No', 'Yes'], title = 'Default', loc='upper right', facecolor='white')

ax1.set_frame_on(False)
ax2.set_frame_on(False)
ax3.set_frame_on(False)
ax4.set_frame_on(False)
ax5.set_frame_on(False)
ax6.set_frame_on(False)

plt.show()
```


![png](output_41_0.png)


The Pay data should only contain data from range -1 to 8, but -2 appears on the dataset.


```python
df['PAY_0'] = np.where(df['PAY_0'] == -2, -1, df['PAY_0'])
df['PAY_2'] = np.where(df['PAY_2'] == -2, -1, df['PAY_2'])
df['PAY_3'] = np.where(df['PAY_3'] == -2, -1, df['PAY_3'])
df['PAY_4'] = np.where(df['PAY_4'] == -2, -1, df['PAY_4'])
df['PAY_5'] = np.where(df['PAY_5'] == -2, -1, df['PAY_5'])
df['PAY_6'] = np.where(df['PAY_6'] == -2, -1, df['PAY_6'])
```


```python
figure, ((ax1,ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3)
figure.set_size_inches(16,6)

sns.countplot(x="PAY_0",  data=df, palette="pastel", ax=ax1)
sns.countplot(x="PAY_2",  data=df, palette="pastel", ax=ax2)
sns.countplot(x="PAY_3",  data=df, palette="pastel", ax=ax3)
sns.countplot(x="PAY_4",  data=df, palette="pastel", ax=ax4)
sns.countplot(x="PAY_5",  data=df, palette="pastel", ax=ax5)
sns.countplot(x="PAY_6",  data=df, palette="pastel", ax=ax6)

ax1.set_frame_on(False)
ax2.set_frame_on(False)
ax3.set_frame_on(False)
ax4.set_frame_on(False)
ax5.set_frame_on(False)
ax6.set_frame_on(False)

plt.show()
```


![png](output_44_0.png)



```python
figure, ((ax1,ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3)
figure.set_size_inches(16,6)

sns.countplot(x="PAY_0", hue="default.payment.next.month", data=df, palette="pastel", ax=ax1)
sns.countplot(x="PAY_2", hue="default.payment.next.month", data=df, palette="pastel", ax=ax2)
sns.countplot(x="PAY_3", hue="default.payment.next.month", data=df, palette="pastel", ax=ax3)
sns.countplot(x="PAY_4", hue="default.payment.next.month", data=df, palette="pastel", ax=ax4)
sns.countplot(x="PAY_5", hue="default.payment.next.month", data=df, palette="pastel", ax=ax5)
sns.countplot(x="PAY_6", hue="default.payment.next.month", data=df, palette="pastel", ax=ax6)

ax1.legend(['No', 'Yes'], title = 'Default', loc='upper right', facecolor='white')
ax2.legend(['No', 'Yes'], title = 'Default', loc='upper right', facecolor='white')
ax3.legend(['No', 'Yes'], title = 'Default', loc='upper right', facecolor='white')
ax4.legend(['No', 'Yes'], title = 'Default', loc='upper right', facecolor='white')
ax5.legend(['No', 'Yes'], title = 'Default', loc='upper right', facecolor='white')
ax6.legend(['No', 'Yes'], title = 'Default', loc='upper right', facecolor='white')

ax1.set_frame_on(False)
ax2.set_frame_on(False)
ax3.set_frame_on(False)
ax4.set_frame_on(False)
ax5.set_frame_on(False)
ax6.set_frame_on(False)

plt.show()
```


![png](output_45_0.png)


## 4) Amount of Bill Statement


```python
df.describe()
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
      <th>ID</th>
      <th>LIMIT_BAL</th>
      <th>SEX</th>
      <th>EDUCATION</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>PAY_0</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>...</th>
      <th>BILL_AMT4</th>
      <th>BILL_AMT5</th>
      <th>BILL_AMT6</th>
      <th>PAY_AMT1</th>
      <th>PAY_AMT2</th>
      <th>PAY_AMT3</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
      <th>default.payment.next.month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>...</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>3.000000e+04</td>
      <td>30000.00000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>15000.500000</td>
      <td>167484.322667</td>
      <td>1.603733</td>
      <td>1.842267</td>
      <td>1.557267</td>
      <td>35.485500</td>
      <td>0.075267</td>
      <td>-0.007700</td>
      <td>-0.030033</td>
      <td>-0.075733</td>
      <td>...</td>
      <td>43262.948967</td>
      <td>40311.400967</td>
      <td>38871.760400</td>
      <td>5663.580500</td>
      <td>5.921163e+03</td>
      <td>5225.68150</td>
      <td>4826.076867</td>
      <td>4799.387633</td>
      <td>5215.502567</td>
      <td>0.221200</td>
    </tr>
    <tr>
      <td>std</td>
      <td>8660.398374</td>
      <td>129747.661567</td>
      <td>0.489129</td>
      <td>0.744494</td>
      <td>0.521405</td>
      <td>9.217904</td>
      <td>0.990775</td>
      <td>1.035798</td>
      <td>1.025036</td>
      <td>0.987436</td>
      <td>...</td>
      <td>64332.856134</td>
      <td>60797.155770</td>
      <td>59554.107537</td>
      <td>16563.280354</td>
      <td>2.304087e+04</td>
      <td>17606.96147</td>
      <td>15666.159744</td>
      <td>15278.305679</td>
      <td>17777.465775</td>
      <td>0.415062</td>
    </tr>
    <tr>
      <td>min</td>
      <td>1.000000</td>
      <td>10000.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>21.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>...</td>
      <td>-170000.000000</td>
      <td>-81334.000000</td>
      <td>-339603.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>7500.750000</td>
      <td>50000.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>28.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>...</td>
      <td>2326.750000</td>
      <td>1763.000000</td>
      <td>1256.000000</td>
      <td>1000.000000</td>
      <td>8.330000e+02</td>
      <td>390.00000</td>
      <td>296.000000</td>
      <td>252.500000</td>
      <td>117.750000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>15000.500000</td>
      <td>140000.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>34.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>19052.000000</td>
      <td>18104.500000</td>
      <td>17071.000000</td>
      <td>2100.000000</td>
      <td>2.009000e+03</td>
      <td>1800.00000</td>
      <td>1500.000000</td>
      <td>1500.000000</td>
      <td>1500.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>22500.250000</td>
      <td>240000.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>41.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>54506.000000</td>
      <td>50190.500000</td>
      <td>49198.250000</td>
      <td>5006.000000</td>
      <td>5.000000e+03</td>
      <td>4505.00000</td>
      <td>4013.250000</td>
      <td>4031.500000</td>
      <td>4000.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>30000.000000</td>
      <td>1000000.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>79.000000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>...</td>
      <td>891586.000000</td>
      <td>927171.000000</td>
      <td>961664.000000</td>
      <td>873552.000000</td>
      <td>1.684259e+06</td>
      <td>896040.00000</td>
      <td>621000.000000</td>
      <td>426529.000000</td>
      <td>528666.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 25 columns</p>
</div>




```python
figure, ((ax1,ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3)
figure.set_size_inches(16,6)

sns.distplot(df["BILL_AMT1"], color = 'steelblue', ax=ax1)
sns.distplot(df["BILL_AMT2"], color = 'steelblue', ax=ax2)
sns.distplot(df["BILL_AMT3"], color = 'steelblue', ax=ax3)
sns.distplot(df["BILL_AMT4"], color = 'steelblue', ax=ax4)
sns.distplot(df["BILL_AMT5"], color = 'steelblue', ax=ax5)
sns.distplot(df["BILL_AMT6"], color = 'steelblue', ax=ax6)

ax1.set_xlim([-250000, 900000])
ax2.set_xlim([-250000, 900000])
ax3.set_xlim([-250000, 900000])
ax4.set_xlim([-250000, 900000])
ax5.set_xlim([-250000, 900000])
ax6.set_xlim([-250000, 900000])

ax1.set_frame_on(False)
ax2.set_frame_on(False)
ax3.set_frame_on(False)
ax4.set_frame_on(False)
ax5.set_frame_on(False)
ax6.set_frame_on(False)

plt.show()
```


![png](output_48_0.png)


Negative Number shouldn't appear on bill amount.   
Turn all negative number into 0.


```python
df['BILL_AMT1'] = np.where(df['BILL_AMT1'] < 0, 0, df['BILL_AMT1'])
df['BILL_AMT2'] = np.where(df['BILL_AMT2'] < 0, 0, df['BILL_AMT2'])
df['BILL_AMT3'] = np.where(df['BILL_AMT3'] < 0, 0, df['BILL_AMT3'])
df['BILL_AMT4'] = np.where(df['BILL_AMT4'] < 0, 0, df['BILL_AMT4'])
df['BILL_AMT5'] = np.where(df['BILL_AMT5'] < 0, 0, df['BILL_AMT5'])
df['BILL_AMT6'] = np.where(df['BILL_AMT6'] < 0, 0, df['BILL_AMT6'])
```


```python
sns.distplot(df["BILL_AMT6"], color = 'steelblue')

plt.xlim([-250000, 900000])

plt.show()
```


![png](output_51_0.png)


# 3. Feature Engineering

### New Variable : Prior

- With the theory that __those who have previous default records are more likely default__, I have created a new variable : __Prior__


```python
df['prior'] = np.where((df['PAY_0'] + df['PAY_2'] + df['PAY_3'] + df['PAY_4'] + df['PAY_5'] + df['PAY_6'])>0, 1, 0)
```


```python
df.groupby(df['prior']).size()
```




    prior
    0    22754
    1     7246
    dtype: int64




```python
sns.countplot(data = df, x = 'prior' )
```




    <AxesSubplot:xlabel='prior', ylabel='count'>




![png](output_56_1.png)


### New Variable : Gender, Marriage Combination


```python
df['Comb'] = 0
df.loc[(df['SEX'] == 1) & (df['MARRIAGE'] == 1), 'Comb'] = 1 # Married Male
df.loc[(df['SEX'] == 1) & (df['MARRIAGE'] == 2), 'Comb'] = 2 # Single Male
df.loc[(df['SEX'] == 1) & (df['MARRIAGE'] == 3), 'Comb'] = 3 # Divorced Male
df.loc[(df['SEX'] == 2) & (df['MARRIAGE'] == 1), 'Comb'] = 4 # Married Female
df.loc[(df['SEX'] == 2) & (df['MARRIAGE'] == 2), 'Comb'] = 5 # Single Female
df.loc[(df['SEX'] == 2) & (df['MARRIAGE'] == 3), 'Comb'] = 6 # Divorced Female
```


```python
df[['ID','Comb']].head()
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
      <th>ID</th>
      <th>Comb</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.countplot(data = df, x = 'Comb' )
```




    <AxesSubplot:xlabel='Comb', ylabel='count'>




![png](output_60_1.png)


### Renewed Variable : Age Distribution


```python
df['Age_dist'] = 0
df.loc[(df['AGE'] > 20) & (df['AGE'] < 31), 'Age_dist'] = 1
df.loc[(df['AGE'] >= 30) & (df['AGE'] < 41), 'Age_dist'] = 2
df.loc[(df['AGE'] >= 40) & (df['AGE'] < 51), 'Age_dist'] = 3
df.loc[(df['AGE'] >= 50) & (df['AGE'] < 61), 'Age_dist'] = 4
df.loc[(df['AGE'] >= 60) & (df['AGE'] < 71), 'Age_dist'] = 5
df.loc[(df['AGE'] >= 70) & (df['AGE'] < 81), 'Age_dist'] = 6
```


```python
sns.countplot(data = df, x = 'Age_dist')
```




    <AxesSubplot:xlabel='Age_dist', ylabel='count'>




![png](output_63_1.png)



```python
df.loc[df['Age_dist'] ==6, 'Age_dist'] = 5
```

# 5. Data Modeling


```python
df2 = df.copy()
```

### 5-1. One Hot Encoding


```python
onehotencoder = OneHotEncoder()
```


```python
encode = pd.DataFrame(onehotencoder.fit_transform(df2[['EDUCATION', 'MARRIAGE', 'Age_dist', 'Comb']]).toarray())
```


```python
df2.drop(['EDUCATION', 'MARRIAGE', 'Age_dist', 'Comb'], axis = 1, inplace = True)
df2 = df2.join(encode)
```


```python
df2.columns
```




    Index(['ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',
           'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
           'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
           'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
           'default.payment.next.month', 'prior', 'Comb', 'Age_dist'],
          dtype='object')




```python
df2.columns
```




    Index(['ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',
           'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
           'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
           'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
           'default.payment.next.month', 'prior', 'Comb', 'Age_dist'],
          dtype='object')




```python
features = [ 'ID','LIMIT_BAL','SEX','AGE','PAY_0','PAY_2','PAY_3','PAY_4',
'PAY_5','PAY_6','BILL_AMT1','BILL_AMT2', 'BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2',
'PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6','default.payment.next.month','prior',0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
```


```python
features = ['ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',
       'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
       'default.payment.next.month', 'prior', 'Comb', 'Age_dist']
```


```python
X = df2[features].copy()
y = df2['default.payment.next.month'].copy()
```

### 5-2. Standard Scaler


```python
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

### 5-3. Train_Test Split


```python
#train & test split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 16)
```


```python
X_train.shape, X_test.shape, y_train.shape, y_test.shape
```




    ((21000, 28), (9000, 28), (21000,), (9000,))



### 5-3. SMOTE - Oversampling Technique


```python
from imblearn.over_sampling import SMOTE
```


```python
sm = SMOTE(random_state=100)

X_smote, y_smote = sm.fit_sample(X_train, y_train)

print(len(y_smote))
print(y_smote.sum())
```

    32732
    16366
    

- Now we have 2 training sets :  
  __1. X_train, y_train__ : Unbalanced Data  
  __2. X_smote, y_smote__ : Balanced Data by Oversampling Technique

- Next, I would performn following procedure :  
  __1. K_fold Evaluation__ : In order to run different models and find which datasets (Balanced/Unbalanced) are more effective  
  __2. Hyperparameter Selection__ : To find best result from each algorithms  
  __3. Training__  
  __4. Testing__

### 5-4. Evaluation Method


```python
def metrics(y_test, pred) :
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_score = roc_auc_score(y_test, pred, average = 'macro')
    print('accuracy : {0:.2f}, precision : {1:.2f}, recall : {2:.2f}'.format(accuracy, precision, recall))
    print('f1-score : {0:.2f}, auc : {1:.2f}'.format(f1, roc_score))
```

### 5-5. Model Selection

Use combinations of different classifiers (ensembles) to create a more robuse model :

- Regression : Logistic Regression
- Bagging Ensembles : RandomForest
- Boosting Ensembles : Gradient Tree Boosting / XGB 


```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve, accuracy_score,precision_score,recall_score,roc_auc_score
from sklearn.model_selection import KFold
```


```python
def modeling(model, x_train, x_test, y_train, y_test) :
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    metrics(y_test, pred)
```

### Logistic Regression


```python
lr = LogisticRegression()
```


```python
modeling(lr, X_train, X_test, y_train, y_test)
```

    accuracy : 0.82, precision : 0.68, recall : 0.34
    f1-score : 0.46, auc : 0.65
    


```python
modeling(lr, X_smote, X_test, y_smote, y_test)
```

    accuracy : 0.75, precision : 0.45, recall : 0.61
    f1-score : 0.52, auc : 0.70
    


```python
clf_list = [LogisticRegression(),
            RandomForestClassifier(n_estimators = 100), 
            GradientBoostingClassifier(), 
            XGBClassifier()
           ]
```


```python
kf = KFold(n_splits = 5, random_state = 77, shuffle = True)


mdl = []
fold = []
scr = []


```


```python
for i,(train_index, test_index) in enumerate(kf.split(X)):
    training = X.iloc[train_index,:]
    valid = X.iloc[test_index,:]
    print(i)
    for clf in clf_list:
        model = clf.__class__.__name__
        print(model)
        feats = training[features] #defined above
        label = training['default.payment.next.month']
        valid_feats = valid[features]
        valid_label = valid['default.payment.next.month']
        clf.fit(feats,label) 
        pred = clf.predict(valid_feats)
        metrics(valid_label, pred)
        print('---------')

```

    0
    LogisticRegression
    accuracy : 0.78, precision : 1.00, recall : 0.00
    f1-score : 0.00, auc : 0.50
    ---------
    RandomForestClassifier
    accuracy : 1.00, precision : 1.00, recall : 1.00
    f1-score : 1.00, auc : 1.00
    ---------
    GradientBoostingClassifier
    


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-111-f861a35b1f2d> in <module>
         10         valid_feats = valid[features]
         11         valid_label = valid['default.payment.next.month']
    ---> 12         clf.fit(feats,label)
         13         pred = clf.predict(valid_feats)
         14         metrics(valid_label, pred)
    

    ~\Anaconda3\lib\site-packages\sklearn\ensemble\_gb.py in fit(self, X, y, sample_weight, monitor)
        498         n_stages = self._fit_stages(
        499             X, y, raw_predictions, sample_weight, self._rng, X_val, y_val,
    --> 500             sample_weight_val, begin_at_stage, monitor, X_idx_sorted)
        501 
        502         # change shape of arrays after fit (early-stopping or additional ests)
    

    ~\Anaconda3\lib\site-packages\sklearn\ensemble\_gb.py in _fit_stages(self, X, y, raw_predictions, sample_weight, random_state, X_val, y_val, sample_weight_val, begin_at_stage, monitor, X_idx_sorted)
        555             raw_predictions = self._fit_stage(
        556                 i, X, y, raw_predictions, sample_weight, sample_mask,
    --> 557                 random_state, X_idx_sorted, X_csc, X_csr)
        558 
        559             # track deviance (= loss)
    

    ~\Anaconda3\lib\site-packages\sklearn\ensemble\_gb.py in _fit_stage(self, i, X, y, raw_predictions, sample_weight, sample_mask, random_state, X_idx_sorted, X_csc, X_csr)
        210             X = X_csr if X_csr is not None else X
        211             tree.fit(X, residual, sample_weight=sample_weight,
    --> 212                      check_input=False, X_idx_sorted=X_idx_sorted)
        213 
        214             # update tree leaves
    

    ~\Anaconda3\lib\site-packages\sklearn\tree\_classes.py in fit(self, X, y, sample_weight, check_input, X_idx_sorted)
       1244             sample_weight=sample_weight,
       1245             check_input=check_input,
    -> 1246             X_idx_sorted=X_idx_sorted)
       1247         return self
       1248 
    

    ~\Anaconda3\lib\site-packages\sklearn\tree\_classes.py in fit(self, X, y, sample_weight, check_input, X_idx_sorted)
        373                                            min_impurity_split)
        374 
    --> 375         builder.build(self.tree_, X, y, sample_weight, X_idx_sorted)
        376 
        377         if self.n_outputs_ == 1 and is_classifier(self):
    

    KeyboardInterrupt: 



```python
n_iter = 0
for train_index,test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index],X.iloc[test_index]
    y_train, y_test = y.iloc[train_index],y.iloc[test_index]
    
    lr.fit(X_train,y_train)
    pred = lr.predict(X_test)
    n_iter+=1
    
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    
    print('Train Size:',train_size,'Test Size',test_size)
    metrics(y_test,pred)
    print('----------------------')

```

    Train Size: 24000 Test Size 6000
    accuracy : 0.78, precision : 1.00, recall : 0.00
    f1-score : 0.00, auc : 0.50
    ----------------------
    Train Size: 24000 Test Size 6000
    accuracy : 0.78, precision : 0.00, recall : 0.00
    f1-score : 0.00, auc : 0.50
    ----------------------
    Train Size: 24000 Test Size 6000
    accuracy : 0.78, precision : 0.00, recall : 0.00
    f1-score : 0.00, auc : 0.50
    ----------------------
    Train Size: 24000 Test Size 6000
    accuracy : 0.78, precision : 0.00, recall : 0.00
    f1-score : 0.00, auc : 0.50
    ----------------------
    Train Size: 24000 Test Size 6000
    accuracy : 0.77, precision : 0.00, recall : 0.00
    f1-score : 0.00, auc : 0.50
    ----------------------
    


```python
for train_index,test_index in kf.split(X_train):
    X_train_scaled, X_test_scaled  = X_train.iloc[train_index], 
    valid = df2.iloc[test_index, :]
    for clf in clf_list :
        model = clf.__class__.__name__
        feats = training
```

    [    1     2     3 ... 20997 20998 20999]
    [    0     1     2 ... 20997 20998 20999]
    [    0     3     5 ... 20996 20998 20999]
    [    0     1     2 ... 20995 20996 20997]
    [    0     1     2 ... 20997 20998 20999]
    


```python
for train_index,test_index in kfold.split(train_data):
    X_train, X_test = train_data.iloc[train_index],train_data.iloc[test_index]
    y_train, y_test = test_data.iloc[train_index],test_data.iloc[test_index]
    
    lr_clf.fit(X_train,y_train)
    pred = lr_clf.predict(X_test)
    n_iter+=1
    
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    
    print('Train Size:',train_size,'Test Size',test_size)
    get_clf_eval(y_test,pred)
```


```python
for i,(train_index, test_index) in enumerate(kf.split(df_train)):
    training = df.iloc[train_index,:]
    valid = df.iloc[test_index,:]
    print(i)
    for clf in clf_list:
        model = clf.__class__.__name__
        feats = training[features] #defined above
        label = training['def_pay']
        valid_feats = valid[features]
        valid_label = valid['def_pay']
        clf.fit(feats,label) 
        pred = clf.predict(valid_feats)
        score = f1_score(y_true = valid_label, y_pred = pred)
        fold.append(i+1)
        scr.append(score)
        mdl.append(model)
        print(model)
```

### Random Forest


```python
rf = RandomForestClassifier()
```


```python
modeling(rf, X_train, X_test, y_train, y_test)
```

    accuracy : 0.82, precision : 0.67, recall : 0.37
    f1-score : 0.47, auc : 0.66
    


```python
modeling(rf, X_smote, X_test, y_smote, y_test)
```

    accuracy : 0.80, precision : 0.56, recall : 0.47
    f1-score : 0.51, auc : 0.68
    

### Gradient Tree Boosting


```python
gbc = GradientBoostingClassifier()
```


```python
modeling(gbc, X_train, X_test, y_train, y_test)
```

    accuracy : 0.82, precision : 0.69, recall : 0.38
    f1-score : 0.49, auc : 0.66
    


```python
modeling(gbc, X_smote, X_test, y_smote, y_test)
```

    accuracy : 0.80, precision : 0.56, recall : 0.50
    f1-score : 0.53, auc : 0.69
    

### XGBoosting


```python
xgb = XGBClassifier()
```


```python
modeling(xgb, X_train, X_test, y_train, y_test)
```

    accuracy : 0.82, precision : 0.70, recall : 0.37
    f1-score : 0.48, auc : 0.66
    


```python
modeling(xgb, X_smote, X_test, y_smote, y_test)
```

    accuracy : 0.80, precision : 0.56, recall : 0.49
    f1-score : 0.53, auc : 0.69
    


```python

```
