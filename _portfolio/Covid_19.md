---
title: "Covid-19 Visualization"
date: 2020-07-26
classes: wide
toc: true
categories: DataAnalysis
---




```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.font_manager as fm
import plotly.io as pio
```


```python
#from IPython.core.display import display, HTML

#display(HTML("<style>.container { width:70% !important; }</style>"))
```


<style>.container { width:70% !important; }</style>



```python
%matplotlib inline
```


```python
plt.rcParams["font.family"] = 'NanumGothic'
plt.rcParams['font.size'] = 13
```

# 1. Load Data


```python
case = pd.read_csv('case.csv')
patientinfo = pd.read_csv('PatientInfo.csv')
policy = pd.read_csv('Policy.csv')
region = pd.read_csv('Region.csv')
searchtrend = pd.read_csv('SearchTrend.csv')
seoulfloating = pd.read_csv('SeoulFloating.csv')
time = pd.read_csv('Time.csv')
timeage = pd.read_csv('TimeAge.csv')
timegender = pd.read_csv('TimeGender.csv')
timeprovince = pd.read_csv('TimeProvince.csv')
weather = pd.read_csv('Weather.csv')
```


```python
print(time.shape)
time
```

    (163, 7)
    




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
      <th>date</th>
      <th>time</th>
      <th>test</th>
      <th>negative</th>
      <th>confirmed</th>
      <th>released</th>
      <th>deceased</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2020-01-20</td>
      <td>16</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2020-01-21</td>
      <td>16</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2020-01-22</td>
      <td>16</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2020-01-23</td>
      <td>16</td>
      <td>22</td>
      <td>21</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2020-01-24</td>
      <td>16</td>
      <td>27</td>
      <td>25</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
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
      <td>158</td>
      <td>2020-06-26</td>
      <td>0</td>
      <td>1232315</td>
      <td>1200885</td>
      <td>12602</td>
      <td>11172</td>
      <td>282</td>
    </tr>
    <tr>
      <td>159</td>
      <td>2020-06-27</td>
      <td>0</td>
      <td>1243780</td>
      <td>1211261</td>
      <td>12653</td>
      <td>11317</td>
      <td>282</td>
    </tr>
    <tr>
      <td>160</td>
      <td>2020-06-28</td>
      <td>0</td>
      <td>1251695</td>
      <td>1219975</td>
      <td>12715</td>
      <td>11364</td>
      <td>282</td>
    </tr>
    <tr>
      <td>161</td>
      <td>2020-06-29</td>
      <td>0</td>
      <td>1259954</td>
      <td>1228698</td>
      <td>12757</td>
      <td>11429</td>
      <td>282</td>
    </tr>
    <tr>
      <td>162</td>
      <td>2020-06-30</td>
      <td>0</td>
      <td>1273766</td>
      <td>1240157</td>
      <td>12800</td>
      <td>11537</td>
      <td>282</td>
    </tr>
  </tbody>
</table>
<p>163 rows × 7 columns</p>
</div>



## 1) Case Data
Case: Data of COVID-19 infection cases in South Korea

## 2) Patient Data
PatientInfo: Epidemiological data of COVID-19 patients in South Korea

## 3) Time Series Data
Time: Time series data of COVID-19 status in South Korea  

TimeAge: Time series data of COVID-19 status in terms of the age in South Korea

TimeGender: Time series data of COVID-19 status in terms of gender in South Korea

TimeProvince: Time series data of COVID-19 status in terms of the Province in South Korea

## 4) Additional Data
Region: Location and statistical data of the regions in South Korea

Weather: Data of the weather in the regions of South Korea

SearchTrend: Trend data of the keywords searched in NAVER which is one of the largest portals in South Korea

SeoulFloating: Data of floating population in Seoul, South Korea (from SK Telecom Big Data Hub)

Policy: Data of the government policy for COVID-19 in South Korea

# 2. Data Exploration


```python
print(patientinfo.shape, len(patientinfo.patient_id.unique()))
patientinfo.head()
```

    (5165, 14) 5164
    




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
      <th>patient_id</th>
      <th>sex</th>
      <th>age</th>
      <th>country</th>
      <th>province</th>
      <th>city</th>
      <th>infection_case</th>
      <th>infected_by</th>
      <th>contact_number</th>
      <th>symptom_onset_date</th>
      <th>confirmed_date</th>
      <th>released_date</th>
      <th>deceased_date</th>
      <th>state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1000000001</td>
      <td>male</td>
      <td>50s</td>
      <td>Korea</td>
      <td>Seoul</td>
      <td>Gangseo-gu</td>
      <td>overseas inflow</td>
      <td>NaN</td>
      <td>75</td>
      <td>2020-01-22</td>
      <td>2020-01-23</td>
      <td>2020-02-05</td>
      <td>NaN</td>
      <td>released</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1000000002</td>
      <td>male</td>
      <td>30s</td>
      <td>Korea</td>
      <td>Seoul</td>
      <td>Jungnang-gu</td>
      <td>overseas inflow</td>
      <td>NaN</td>
      <td>31</td>
      <td>NaN</td>
      <td>2020-01-30</td>
      <td>2020-03-02</td>
      <td>NaN</td>
      <td>released</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1000000003</td>
      <td>male</td>
      <td>50s</td>
      <td>Korea</td>
      <td>Seoul</td>
      <td>Jongno-gu</td>
      <td>contact with patient</td>
      <td>2002000001</td>
      <td>17</td>
      <td>NaN</td>
      <td>2020-01-30</td>
      <td>2020-02-19</td>
      <td>NaN</td>
      <td>released</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1000000004</td>
      <td>male</td>
      <td>20s</td>
      <td>Korea</td>
      <td>Seoul</td>
      <td>Mapo-gu</td>
      <td>overseas inflow</td>
      <td>NaN</td>
      <td>9</td>
      <td>2020-01-26</td>
      <td>2020-01-30</td>
      <td>2020-02-15</td>
      <td>NaN</td>
      <td>released</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1000000005</td>
      <td>female</td>
      <td>20s</td>
      <td>Korea</td>
      <td>Seoul</td>
      <td>Seongbuk-gu</td>
      <td>contact with patient</td>
      <td>1000000002</td>
      <td>2</td>
      <td>NaN</td>
      <td>2020-01-31</td>
      <td>2020-02-24</td>
      <td>NaN</td>
      <td>released</td>
    </tr>
  </tbody>
</table>
</div>




```python
# One of the patient_id is not unique
patientinfo['patient_id'].value_counts()[:10]
```




    1200012238    2
    2000001021    1
    6002000022    1
    1000000276    1
    5100000023    1
    1000000280    1
    4000000031    1
    1000000284    1
    1000000288    1
    2000000442    1
    Name: patient_id, dtype: int64




```python
patientinfo[patientinfo['patient_id'] == 1200012238]
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
      <th>patient_id</th>
      <th>sex</th>
      <th>age</th>
      <th>country</th>
      <th>province</th>
      <th>city</th>
      <th>infection_case</th>
      <th>infected_by</th>
      <th>contact_number</th>
      <th>symptom_onset_date</th>
      <th>confirmed_date</th>
      <th>released_date</th>
      <th>deceased_date</th>
      <th>state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1547</td>
      <td>1200012238</td>
      <td>female</td>
      <td>20s</td>
      <td>Korea</td>
      <td>Daegu</td>
      <td>Icheon-dong</td>
      <td>overseas inflow</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2020-06-17</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>isolated</td>
    </tr>
    <tr>
      <td>1555</td>
      <td>1200012238</td>
      <td>female</td>
      <td>20s</td>
      <td>Korea</td>
      <td>Daegu</td>
      <td>Nam-gu</td>
      <td>overseas inflow</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2020-06-17</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>isolated</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Since Icheon-dong belongs to Nam-gu, eliminate Icheon-dong 
patientinfo = patientinfo.drop(1547, 0)
```


```python
print(patientinfo.shape, len(patientinfo.patient_id.unique()))
```

    (5164, 14) 5164
    

## (1) Infection Route 감염 경로


```python
case.columns
```




    Index([' case_id', 'province', 'city', 'group', 'infection_case', 'confirmed',
           'latitude', 'longitude'],
          dtype='object')




```python
case.rename(columns = {' case_id':'case_id'}, inplace = True)
```


```python
case.columns
```




    Index(['case_id', 'province', 'city', 'group', 'infection_case', 'confirmed',
           'latitude', 'longitude'],
          dtype='object')




```python
print('types of infection case : ', len(patientinfo['infection_case'].unique()))

patientinfo['infection_case'].value_counts()

# 감영 경로가 다양해보임
```

    types of infection case :  52
    




    contact with patient                             1610
    overseas inflow                                   839
    etc                                               703
    Itaewon Clubs                                     162
    Richway                                           128
    Guro-gu Call Center                               112
    Shincheonji Church                                107
    Coupang Logistics Center                           80
    Yangcheon Table Tennis Club                        44
    Day Care Center                                    43
    SMR Newly Planted Churches Group                   36
    Onchun Church                                      33
    Bonghwa Pureun Nursing Home                        31
    gym facility in Cheonan                            30
    Ministry of Oceans and Fisheries                   28
    Wangsung Church                                    24
    Cheongdo Daenam Hospital                           21
    Dongan Church                                      17
    Eunpyeong St. Mary's Hospital                      16
    Gyeongsan Seorin Nursing Home                      15
    Seongdong-gu APT                                   13
    Dunsan Electronics Town                            13
    KB Life Insurance                                  13
    Gyeongsan Jeil Silver Town                         12
    Milal Shelter                                      11
    Gyeongsan Cham Joeun Community Center              10
    Korea Campus Crusade of Christ                      7
    Samsung Medical Center                              7
    Orange Town                                         7
    Gangnam Yeoksam-dong gathering                      6
    Geochang Church                                     6
    Geumcheon-gu rice milling machine manufacture       6
    Guri Collective Infection                           5
    Yeonana News Class                                  5
    Seocho Family                                       5
    Changnyeong Coin Karaoke                            4
    Yongin Brothers                                     4
    gym facility in Sejong                              4
    Samsung Fire & Marine Insurance                     4
    Yeongdeungpo Learning Institute                     3
    Seoul City Hall Station safety worker               3
    Biblical Language study meeting                     3
    Daezayeon Korea                                     3
    Suyeong-gu Kindergarten                             3
    Pilgrimage to Israel                                2
    Uiwang Logistics Center                             2
    Daejeon door-to-door sales                          1
    Gangnam Dongin Church                               1
    Orange Life                                         1
    River of Grace Community Church                     1
    Anyang Gunpo Pastors Group                          1
    Name: infection_case, dtype: int64




```python
type_infection = patientinfo.groupby(['infection_case'])['patient_id'].count()
type_infection = type_infection.reset_index()
type_infection.rename(columns = {'patient_id': 'count'}, inplace = True)


type_infection = type_infection.sort_values('count', ascending = False)
```


```python
type_infection[:3]
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
      <th>infection_case</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>46</td>
      <td>contact with patient</td>
      <td>1610</td>
    </tr>
    <tr>
      <td>50</td>
      <td>overseas inflow</td>
      <td>839</td>
    </tr>
    <tr>
      <td>47</td>
      <td>etc</td>
      <td>703</td>
    </tr>
  </tbody>
</table>
</div>




```python
type_infection.describe()
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
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>51.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>83.235294</td>
    </tr>
    <tr>
      <td>std</td>
      <td>265.399441</td>
    </tr>
    <tr>
      <td>min</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>10.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>30.500000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>1610.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize = (20,10))
plt.title('감염경로 (Infection Route)', position=(0.35, 2),fontsize = 20, pad = 25 )
sns.barplot(y = 'infection_case', x = 'count', data = type_infection)
plt.tick_params(labelsize=10)
display()


# 해외 유입, 기타, 환자 접촉, 신천지가 독보적으로 많은 영향을 끼쳤고, 나머지는 별로 많지 않다. 
# 별로 많지 않은 데이터 중 어디까지를 집단감염이라고 생각해야될까?
```


![png](output_22_0.png)


## (2) Patient Distribution 감염자 분포 확인


```python
print(patientinfo.shape)
patientinfo.head()
```

    (5164, 14)
    




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
      <th>patient_id</th>
      <th>sex</th>
      <th>age</th>
      <th>country</th>
      <th>province</th>
      <th>city</th>
      <th>infection_case</th>
      <th>infected_by</th>
      <th>contact_number</th>
      <th>symptom_onset_date</th>
      <th>confirmed_date</th>
      <th>released_date</th>
      <th>deceased_date</th>
      <th>state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1000000001</td>
      <td>male</td>
      <td>50s</td>
      <td>Korea</td>
      <td>Seoul</td>
      <td>Gangseo-gu</td>
      <td>overseas inflow</td>
      <td>NaN</td>
      <td>75</td>
      <td>2020-01-22</td>
      <td>2020-01-23</td>
      <td>2020-02-05</td>
      <td>NaN</td>
      <td>released</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1000000002</td>
      <td>male</td>
      <td>30s</td>
      <td>Korea</td>
      <td>Seoul</td>
      <td>Jungnang-gu</td>
      <td>overseas inflow</td>
      <td>NaN</td>
      <td>31</td>
      <td>NaN</td>
      <td>2020-01-30</td>
      <td>2020-03-02</td>
      <td>NaN</td>
      <td>released</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1000000003</td>
      <td>male</td>
      <td>50s</td>
      <td>Korea</td>
      <td>Seoul</td>
      <td>Jongno-gu</td>
      <td>contact with patient</td>
      <td>2002000001</td>
      <td>17</td>
      <td>NaN</td>
      <td>2020-01-30</td>
      <td>2020-02-19</td>
      <td>NaN</td>
      <td>released</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1000000004</td>
      <td>male</td>
      <td>20s</td>
      <td>Korea</td>
      <td>Seoul</td>
      <td>Mapo-gu</td>
      <td>overseas inflow</td>
      <td>NaN</td>
      <td>9</td>
      <td>2020-01-26</td>
      <td>2020-01-30</td>
      <td>2020-02-15</td>
      <td>NaN</td>
      <td>released</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1000000005</td>
      <td>female</td>
      <td>20s</td>
      <td>Korea</td>
      <td>Seoul</td>
      <td>Seongbuk-gu</td>
      <td>contact with patient</td>
      <td>1000000002</td>
      <td>2</td>
      <td>NaN</td>
      <td>2020-01-31</td>
      <td>2020-02-24</td>
      <td>NaN</td>
      <td>released</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.countplot(x = 'sex', data = patientinfo)
plt.title('남녀별 감염자 수 (Patient by Gender)', pad = 20)

plt.show()
```


![png](output_25_0.png)



```python
plt.figure(figsize = (12,6))

sns.countplot(x = 'age', data = patientinfo,order = patientinfo['age'].value_counts().index)
plt.title('연령별 감염자 수 (Patient by Age)', pad = 15)

plt.show()

#20대가 제일 많다 -> 이동을 제일 많이 하는 연령이라서 그런듯 + 조심성 부족...
```


![png](output_26_0.png)



```python
plt.figure(figsize = (22,6))

sns.countplot(x = 'province', data = patientinfo,order = patientinfo['province'].value_counts().index)
plt.title('지역별 감염자 수 (Patient by Region)',pad = 15, fontsize = 20,position=(0.5, 1.5) )
plt.tick_params(labelsize=12, rotation = -25)
plt.show()
```


![png](output_27_0.png)



```python
len(patientinfo['city'].unique())
```




    163




```python
print(time.shape)
time.head()
```

    (163, 7)
    




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
      <th>date</th>
      <th>time</th>
      <th>test</th>
      <th>negative</th>
      <th>confirmed</th>
      <th>released</th>
      <th>deceased</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2020-01-20</td>
      <td>16</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2020-01-21</td>
      <td>16</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2020-01-22</td>
      <td>16</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2020-01-23</td>
      <td>16</td>
      <td>22</td>
      <td>21</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2020-01-24</td>
      <td>16</td>
      <td>27</td>
      <td>25</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig = go.Figure()
fig.add_trace(go.Scatter(x=time['date'],y=time['released'],
             mode='lines+markers', name='released'))
fig.add_trace(go.Scatter(x=time['date'],y=time['confirmed'],
             mode='lines+markers', name='confirmed'))
fig.add_trace(go.Scatter(x=time['date'],y=time['deceased'],
             mode='lines+markers', name='deceased'))
fig.update_layout(height=600, width=1000, title_text="시간에 따른 확진자 추이", title_font_color="red", 
                  title = {"x" : 0.5, "xanchor" : "center"})
fig.update_xaxes(title_text="month")
fig.update_yaxes(title_text="Number")
pio.write_html(fig, file='시간에 따른 확진자 추이.html', auto_open=True)
fig.show()

# 확진자가 확 나오고, 그 다음에 완치자 크게 증가
# 확진자 수 증가세가 줄긴 하지만 여전히 상승...
# 시간 Interval 고려 (나중에)
```


<div>


            <div id="92cd94bd-074d-402a-a0fa-6d067d3b5c2c" class="plotly-graph-div" style="height:600px; width:1000px;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("92cd94bd-074d-402a-a0fa-6d067d3b5c2c")) {
                    Plotly.newPlot(
                        '92cd94bd-074d-402a-a0fa-6d067d3b5c2c',
                        [{"mode": "lines+markers", "name": "released", "type": "scatter", "x": ["2020-01-20", "2020-01-21", "2020-01-22", "2020-01-23", "2020-01-24", "2020-01-25", "2020-01-26", "2020-01-27", "2020-01-28", "2020-01-29", "2020-01-30", "2020-01-31", "2020-02-01", "2020-02-02", "2020-02-03", "2020-02-04", "2020-02-05", "2020-02-06", "2020-02-07", "2020-02-08", "2020-02-09", "2020-02-10", "2020-02-11", "2020-02-12", "2020-02-13", "2020-02-14", "2020-02-15", "2020-02-16", "2020-02-17", "2020-02-18", "2020-02-19", "2020-02-20", "2020-02-21", "2020-02-22", "2020-02-23", "2020-02-24", "2020-02-25", "2020-02-26", "2020-02-27", "2020-02-28", "2020-02-29", "2020-03-01", "2020-03-02", "2020-03-03", "2020-03-04", "2020-03-05", "2020-03-06", "2020-03-07", "2020-03-08", "2020-03-09", "2020-03-10", "2020-03-11", "2020-03-12", "2020-03-13", "2020-03-14", "2020-03-15", "2020-03-16", "2020-03-17", "2020-03-18", "2020-03-19", "2020-03-20", "2020-03-21", "2020-03-22", "2020-03-23", "2020-03-24", "2020-03-25", "2020-03-26", "2020-03-27", "2020-03-28", "2020-03-29", "2020-03-30", "2020-03-31", "2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04", "2020-04-05", "2020-04-06", "2020-04-07", "2020-04-08", "2020-04-09", "2020-04-10", "2020-04-11", "2020-04-12", "2020-04-13", "2020-04-14", "2020-04-15", "2020-04-16", "2020-04-17", "2020-04-18", "2020-04-19", "2020-04-20", "2020-04-21", "2020-04-22", "2020-04-23", "2020-04-24", "2020-04-25", "2020-04-26", "2020-04-27", "2020-04-28", "2020-04-29", "2020-04-30", "2020-05-01", "2020-05-02", "2020-05-03", "2020-05-04", "2020-05-05", "2020-05-06", "2020-05-07", "2020-05-08", "2020-05-09", "2020-05-10", "2020-05-11", "2020-05-12", "2020-05-13", "2020-05-14", "2020-05-15", "2020-05-16", "2020-05-17", "2020-05-18", "2020-05-19", "2020-05-20", "2020-05-21", "2020-05-22", "2020-05-23", "2020-05-24", "2020-05-25", "2020-05-26", "2020-05-27", "2020-05-28", "2020-05-29", "2020-05-30", "2020-05-31", "2020-06-01", "2020-06-02", "2020-06-03", "2020-06-04", "2020-06-05", "2020-06-06", "2020-06-07", "2020-06-08", "2020-06-09", "2020-06-10", "2020-06-11", "2020-06-12", "2020-06-13", "2020-06-14", "2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18", "2020-06-19", "2020-06-20", "2020-06-21", "2020-06-22", "2020-06-23", "2020-06-24", "2020-06-25", "2020-06-26", "2020-06-27", "2020-06-28", "2020-06-29", "2020-06-30"], "y": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 3, 4, 4, 7, 7, 7, 9, 9, 10, 12, 16, 16, 17, 18, 18, 24, 24, 24, 26, 27, 28, 30, 31, 34, 41, 88, 108, 118, 130, 166, 247, 288, 333, 510, 714, 834, 1137, 1401, 1540, 1947, 2233, 2612, 2909, 3166, 3507, 3730, 4144, 4528, 4811, 5033, 5228, 5408, 5567, 5828, 6021, 6325, 6463, 6598, 6694, 6776, 6973, 7117, 7243, 7368, 7447, 7534, 7616, 7757, 7829, 7937, 8042, 8114, 8213, 8277, 8411, 8501, 8635, 8717, 8764, 8854, 8922, 9059, 9072, 9123, 9183, 9217, 9283, 9333, 9419, 9484, 9568, 9610, 9632, 9670, 9695, 9762, 9821, 9851, 9888, 9904, 9938, 10066, 10135, 10162, 10194, 10213, 10226, 10275, 10295, 10340, 10363, 10398, 10405, 10422, 10446, 10467, 10499, 10506, 10531, 10552, 10563, 10589, 10611, 10654, 10669, 10691, 10718, 10730, 10760, 10774, 10800, 10835, 10856, 10868, 10881, 10908, 10930, 10974, 11172, 11317, 11364, 11429, 11537]}, {"mode": "lines+markers", "name": "confirmed", "type": "scatter", "x": ["2020-01-20", "2020-01-21", "2020-01-22", "2020-01-23", "2020-01-24", "2020-01-25", "2020-01-26", "2020-01-27", "2020-01-28", "2020-01-29", "2020-01-30", "2020-01-31", "2020-02-01", "2020-02-02", "2020-02-03", "2020-02-04", "2020-02-05", "2020-02-06", "2020-02-07", "2020-02-08", "2020-02-09", "2020-02-10", "2020-02-11", "2020-02-12", "2020-02-13", "2020-02-14", "2020-02-15", "2020-02-16", "2020-02-17", "2020-02-18", "2020-02-19", "2020-02-20", "2020-02-21", "2020-02-22", "2020-02-23", "2020-02-24", "2020-02-25", "2020-02-26", "2020-02-27", "2020-02-28", "2020-02-29", "2020-03-01", "2020-03-02", "2020-03-03", "2020-03-04", "2020-03-05", "2020-03-06", "2020-03-07", "2020-03-08", "2020-03-09", "2020-03-10", "2020-03-11", "2020-03-12", "2020-03-13", "2020-03-14", "2020-03-15", "2020-03-16", "2020-03-17", "2020-03-18", "2020-03-19", "2020-03-20", "2020-03-21", "2020-03-22", "2020-03-23", "2020-03-24", "2020-03-25", "2020-03-26", "2020-03-27", "2020-03-28", "2020-03-29", "2020-03-30", "2020-03-31", "2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04", "2020-04-05", "2020-04-06", "2020-04-07", "2020-04-08", "2020-04-09", "2020-04-10", "2020-04-11", "2020-04-12", "2020-04-13", "2020-04-14", "2020-04-15", "2020-04-16", "2020-04-17", "2020-04-18", "2020-04-19", "2020-04-20", "2020-04-21", "2020-04-22", "2020-04-23", "2020-04-24", "2020-04-25", "2020-04-26", "2020-04-27", "2020-04-28", "2020-04-29", "2020-04-30", "2020-05-01", "2020-05-02", "2020-05-03", "2020-05-04", "2020-05-05", "2020-05-06", "2020-05-07", "2020-05-08", "2020-05-09", "2020-05-10", "2020-05-11", "2020-05-12", "2020-05-13", "2020-05-14", "2020-05-15", "2020-05-16", "2020-05-17", "2020-05-18", "2020-05-19", "2020-05-20", "2020-05-21", "2020-05-22", "2020-05-23", "2020-05-24", "2020-05-25", "2020-05-26", "2020-05-27", "2020-05-28", "2020-05-29", "2020-05-30", "2020-05-31", "2020-06-01", "2020-06-02", "2020-06-03", "2020-06-04", "2020-06-05", "2020-06-06", "2020-06-07", "2020-06-08", "2020-06-09", "2020-06-10", "2020-06-11", "2020-06-12", "2020-06-13", "2020-06-14", "2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18", "2020-06-19", "2020-06-20", "2020-06-21", "2020-06-22", "2020-06-23", "2020-06-24", "2020-06-25", "2020-06-26", "2020-06-27", "2020-06-28", "2020-06-29", "2020-06-30"], "y": [1, 1, 1, 1, 2, 2, 3, 4, 4, 4, 6, 11, 12, 15, 15, 16, 18, 23, 24, 24, 27, 27, 28, 28, 28, 28, 28, 29, 30, 31, 51, 104, 204, 433, 602, 833, 977, 1261, 1766, 2337, 3150, 3736, 4212, 4812, 5328, 5766, 6284, 6767, 7134, 7382, 7513, 7755, 7869, 7979, 8086, 8126, 8236, 8320, 8413, 8565, 8652, 8799, 8897, 8961, 9037, 9137, 9241, 9332, 9478, 9583, 9661, 9786, 9887, 9976, 10062, 10156, 10237, 10284, 10331, 10384, 10423, 10450, 10480, 10512, 10537, 10564, 10591, 10613, 10635, 10653, 10661, 10674, 10683, 10694, 10702, 10708, 10718, 10728, 10738, 10752, 10761, 10765, 10774, 10780, 10793, 10801, 10804, 10806, 10810, 10822, 10840, 10874, 10909, 10936, 10962, 10991, 11018, 11037, 11050, 11065, 11078, 11110, 11122, 11142, 11165, 11190, 11206, 11225, 11265, 11344, 11402, 11441, 11468, 11503, 11541, 11590, 11629, 11668, 11719, 11776, 11814, 11852, 11902, 11947, 12003, 12051, 12085, 12121, 12155, 12198, 12257, 12306, 12373, 12421, 12438, 12484, 12535, 12563, 12602, 12653, 12715, 12757, 12800]}, {"mode": "lines+markers", "name": "deceased", "type": "scatter", "x": ["2020-01-20", "2020-01-21", "2020-01-22", "2020-01-23", "2020-01-24", "2020-01-25", "2020-01-26", "2020-01-27", "2020-01-28", "2020-01-29", "2020-01-30", "2020-01-31", "2020-02-01", "2020-02-02", "2020-02-03", "2020-02-04", "2020-02-05", "2020-02-06", "2020-02-07", "2020-02-08", "2020-02-09", "2020-02-10", "2020-02-11", "2020-02-12", "2020-02-13", "2020-02-14", "2020-02-15", "2020-02-16", "2020-02-17", "2020-02-18", "2020-02-19", "2020-02-20", "2020-02-21", "2020-02-22", "2020-02-23", "2020-02-24", "2020-02-25", "2020-02-26", "2020-02-27", "2020-02-28", "2020-02-29", "2020-03-01", "2020-03-02", "2020-03-03", "2020-03-04", "2020-03-05", "2020-03-06", "2020-03-07", "2020-03-08", "2020-03-09", "2020-03-10", "2020-03-11", "2020-03-12", "2020-03-13", "2020-03-14", "2020-03-15", "2020-03-16", "2020-03-17", "2020-03-18", "2020-03-19", "2020-03-20", "2020-03-21", "2020-03-22", "2020-03-23", "2020-03-24", "2020-03-25", "2020-03-26", "2020-03-27", "2020-03-28", "2020-03-29", "2020-03-30", "2020-03-31", "2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04", "2020-04-05", "2020-04-06", "2020-04-07", "2020-04-08", "2020-04-09", "2020-04-10", "2020-04-11", "2020-04-12", "2020-04-13", "2020-04-14", "2020-04-15", "2020-04-16", "2020-04-17", "2020-04-18", "2020-04-19", "2020-04-20", "2020-04-21", "2020-04-22", "2020-04-23", "2020-04-24", "2020-04-25", "2020-04-26", "2020-04-27", "2020-04-28", "2020-04-29", "2020-04-30", "2020-05-01", "2020-05-02", "2020-05-03", "2020-05-04", "2020-05-05", "2020-05-06", "2020-05-07", "2020-05-08", "2020-05-09", "2020-05-10", "2020-05-11", "2020-05-12", "2020-05-13", "2020-05-14", "2020-05-15", "2020-05-16", "2020-05-17", "2020-05-18", "2020-05-19", "2020-05-20", "2020-05-21", "2020-05-22", "2020-05-23", "2020-05-24", "2020-05-25", "2020-05-26", "2020-05-27", "2020-05-28", "2020-05-29", "2020-05-30", "2020-05-31", "2020-06-01", "2020-06-02", "2020-06-03", "2020-06-04", "2020-06-05", "2020-06-06", "2020-06-07", "2020-06-08", "2020-06-09", "2020-06-10", "2020-06-11", "2020-06-12", "2020-06-13", "2020-06-14", "2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18", "2020-06-19", "2020-06-20", "2020-06-21", "2020-06-22", "2020-06-23", "2020-06-24", "2020-06-25", "2020-06-26", "2020-06-27", "2020-06-28", "2020-06-29", "2020-06-30"], "y": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 6, 8, 12, 12, 13, 13, 17, 18, 22, 28, 32, 35, 42, 44, 50, 51, 54, 60, 66, 67, 72, 75, 75, 81, 84, 91, 94, 102, 104, 111, 120, 126, 131, 139, 144, 152, 158, 162, 165, 169, 174, 177, 183, 186, 192, 200, 204, 208, 211, 214, 217, 222, 225, 229, 230, 232, 234, 236, 237, 238, 240, 240, 240, 242, 243, 244, 246, 247, 248, 250, 250, 252, 254, 255, 256, 256, 256, 256, 256, 258, 259, 260, 260, 262, 262, 263, 263, 263, 264, 264, 266, 266, 267, 269, 269, 269, 269, 269, 270, 271, 272, 273, 273, 273, 273, 273, 273, 274, 276, 276, 277, 277, 277, 277, 278, 279, 280, 280, 280, 280, 280, 281, 281, 282, 282, 282, 282, 282, 282]}],
                        {"height": 600, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"font": {"color": "red"}, "text": "\uc2dc\uac04\uc5d0 \ub530\ub978 \ud655\uc9c4\uc790 \ucd94\uc774", "x": 0.5, "xanchor": "center"}, "width": 1000, "xaxis": {"title": {"text": "month"}}, "yaxis": {"title": {"text": "Number"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('92cd94bd-074d-402a-a0fa-6d067d3b5c2c');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>



```python

```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-29-e2147a726f6e> in <module>
    ----> 1 pio.write_html(fig, file='index.html', auto_open=True)
    

    NameError: name 'fig' is not defined



```python
fig = go.Figure()
fig.add_trace(go.Scatter(x=time['date'], y=time['deceased'], mode = 'lines+markers', name = 'deceased'))
fig.update_layout(height=600, width=1000, title_text="시간에 따른 사망자 추이 (Deceased)", title_font_color="red", 
                  title = {"x" : 0.5, "xanchor" : "center"})
fig.update_xaxes(title_text="month")
fig.update_yaxes(title_text="Number")
fig.show()
```


<div>


            <div id="3131e7d0-8413-49ec-b000-6a9cc54f0e31" class="plotly-graph-div" style="height:600px; width:1000px;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("3131e7d0-8413-49ec-b000-6a9cc54f0e31")) {
                    Plotly.newPlot(
                        '3131e7d0-8413-49ec-b000-6a9cc54f0e31',
                        [{"mode": "lines+markers", "name": "deceased", "type": "scatter", "x": ["2020-01-20", "2020-01-21", "2020-01-22", "2020-01-23", "2020-01-24", "2020-01-25", "2020-01-26", "2020-01-27", "2020-01-28", "2020-01-29", "2020-01-30", "2020-01-31", "2020-02-01", "2020-02-02", "2020-02-03", "2020-02-04", "2020-02-05", "2020-02-06", "2020-02-07", "2020-02-08", "2020-02-09", "2020-02-10", "2020-02-11", "2020-02-12", "2020-02-13", "2020-02-14", "2020-02-15", "2020-02-16", "2020-02-17", "2020-02-18", "2020-02-19", "2020-02-20", "2020-02-21", "2020-02-22", "2020-02-23", "2020-02-24", "2020-02-25", "2020-02-26", "2020-02-27", "2020-02-28", "2020-02-29", "2020-03-01", "2020-03-02", "2020-03-03", "2020-03-04", "2020-03-05", "2020-03-06", "2020-03-07", "2020-03-08", "2020-03-09", "2020-03-10", "2020-03-11", "2020-03-12", "2020-03-13", "2020-03-14", "2020-03-15", "2020-03-16", "2020-03-17", "2020-03-18", "2020-03-19", "2020-03-20", "2020-03-21", "2020-03-22", "2020-03-23", "2020-03-24", "2020-03-25", "2020-03-26", "2020-03-27", "2020-03-28", "2020-03-29", "2020-03-30", "2020-03-31", "2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04", "2020-04-05", "2020-04-06", "2020-04-07", "2020-04-08", "2020-04-09", "2020-04-10", "2020-04-11", "2020-04-12", "2020-04-13", "2020-04-14", "2020-04-15", "2020-04-16", "2020-04-17", "2020-04-18", "2020-04-19", "2020-04-20", "2020-04-21", "2020-04-22", "2020-04-23", "2020-04-24", "2020-04-25", "2020-04-26", "2020-04-27", "2020-04-28", "2020-04-29", "2020-04-30", "2020-05-01", "2020-05-02", "2020-05-03", "2020-05-04", "2020-05-05", "2020-05-06", "2020-05-07", "2020-05-08", "2020-05-09", "2020-05-10", "2020-05-11", "2020-05-12", "2020-05-13", "2020-05-14", "2020-05-15", "2020-05-16", "2020-05-17", "2020-05-18", "2020-05-19", "2020-05-20", "2020-05-21", "2020-05-22", "2020-05-23", "2020-05-24", "2020-05-25", "2020-05-26", "2020-05-27", "2020-05-28", "2020-05-29", "2020-05-30", "2020-05-31", "2020-06-01", "2020-06-02", "2020-06-03", "2020-06-04", "2020-06-05", "2020-06-06", "2020-06-07", "2020-06-08", "2020-06-09", "2020-06-10", "2020-06-11", "2020-06-12", "2020-06-13", "2020-06-14", "2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18", "2020-06-19", "2020-06-20", "2020-06-21", "2020-06-22", "2020-06-23", "2020-06-24", "2020-06-25", "2020-06-26", "2020-06-27", "2020-06-28", "2020-06-29", "2020-06-30"], "y": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 6, 8, 12, 12, 13, 13, 17, 18, 22, 28, 32, 35, 42, 44, 50, 51, 54, 60, 66, 67, 72, 75, 75, 81, 84, 91, 94, 102, 104, 111, 120, 126, 131, 139, 144, 152, 158, 162, 165, 169, 174, 177, 183, 186, 192, 200, 204, 208, 211, 214, 217, 222, 225, 229, 230, 232, 234, 236, 237, 238, 240, 240, 240, 242, 243, 244, 246, 247, 248, 250, 250, 252, 254, 255, 256, 256, 256, 256, 256, 258, 259, 260, 260, 262, 262, 263, 263, 263, 264, 264, 266, 266, 267, 269, 269, 269, 269, 269, 270, 271, 272, 273, 273, 273, 273, 273, 273, 274, 276, 276, 277, 277, 277, 277, 278, 279, 280, 280, 280, 280, 280, 281, 281, 282, 282, 282, 282, 282, 282]}],
                        {"height": 600, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"font": {"color": "red"}, "text": "\uc2dc\uac04\uc5d0 \ub530\ub978 \uc0ac\ub9dd\uc790 \ucd94\uc774 (Deceased)", "x": 0.5, "xanchor": "center"}, "width": 1000, "xaxis": {"title": {"text": "month"}}, "yaxis": {"title": {"text": "Number"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('3131e7d0-8413-49ec-b000-6a9cc54f0e31');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>



```python
time.head()
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
      <th>date</th>
      <th>time</th>
      <th>test</th>
      <th>negative</th>
      <th>confirmed</th>
      <th>released</th>
      <th>deceased</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2020-01-20</td>
      <td>16</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2020-01-21</td>
      <td>16</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2020-01-22</td>
      <td>16</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2020-01-23</td>
      <td>16</td>
      <td>22</td>
      <td>21</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2020-01-24</td>
      <td>16</td>
      <td>27</td>
      <td>25</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig = go.Figure()
fig.add_trace(go.Scatter(x=time['date'],y=time['test'],
             mode='lines+markers', name='test'))
fig.add_trace(go.Scatter(x=time['date'],y=time['negative'],
             mode='lines+markers', name='negative'))


fig.update_layout(height=600, width=1000, title_text="검사자와 음성판정 수", 
                  title = {'x' : 0.5, 'xanchor' : 'center', "y" : 0.88, "yanchor" : "bottom" } )
fig.update_xaxes(title_text="month")
fig.update_yaxes(title_text="Number")
fig

```


<div>


            <div id="58890cd2-0b14-48c9-bbed-c7170a016178" class="plotly-graph-div" style="height:600px; width:1000px;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("58890cd2-0b14-48c9-bbed-c7170a016178")) {
                    Plotly.newPlot(
                        '58890cd2-0b14-48c9-bbed-c7170a016178',
                        [{"mode": "lines+markers", "name": "test", "type": "scatter", "x": ["2020-01-20", "2020-01-21", "2020-01-22", "2020-01-23", "2020-01-24", "2020-01-25", "2020-01-26", "2020-01-27", "2020-01-28", "2020-01-29", "2020-01-30", "2020-01-31", "2020-02-01", "2020-02-02", "2020-02-03", "2020-02-04", "2020-02-05", "2020-02-06", "2020-02-07", "2020-02-08", "2020-02-09", "2020-02-10", "2020-02-11", "2020-02-12", "2020-02-13", "2020-02-14", "2020-02-15", "2020-02-16", "2020-02-17", "2020-02-18", "2020-02-19", "2020-02-20", "2020-02-21", "2020-02-22", "2020-02-23", "2020-02-24", "2020-02-25", "2020-02-26", "2020-02-27", "2020-02-28", "2020-02-29", "2020-03-01", "2020-03-02", "2020-03-03", "2020-03-04", "2020-03-05", "2020-03-06", "2020-03-07", "2020-03-08", "2020-03-09", "2020-03-10", "2020-03-11", "2020-03-12", "2020-03-13", "2020-03-14", "2020-03-15", "2020-03-16", "2020-03-17", "2020-03-18", "2020-03-19", "2020-03-20", "2020-03-21", "2020-03-22", "2020-03-23", "2020-03-24", "2020-03-25", "2020-03-26", "2020-03-27", "2020-03-28", "2020-03-29", "2020-03-30", "2020-03-31", "2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04", "2020-04-05", "2020-04-06", "2020-04-07", "2020-04-08", "2020-04-09", "2020-04-10", "2020-04-11", "2020-04-12", "2020-04-13", "2020-04-14", "2020-04-15", "2020-04-16", "2020-04-17", "2020-04-18", "2020-04-19", "2020-04-20", "2020-04-21", "2020-04-22", "2020-04-23", "2020-04-24", "2020-04-25", "2020-04-26", "2020-04-27", "2020-04-28", "2020-04-29", "2020-04-30", "2020-05-01", "2020-05-02", "2020-05-03", "2020-05-04", "2020-05-05", "2020-05-06", "2020-05-07", "2020-05-08", "2020-05-09", "2020-05-10", "2020-05-11", "2020-05-12", "2020-05-13", "2020-05-14", "2020-05-15", "2020-05-16", "2020-05-17", "2020-05-18", "2020-05-19", "2020-05-20", "2020-05-21", "2020-05-22", "2020-05-23", "2020-05-24", "2020-05-25", "2020-05-26", "2020-05-27", "2020-05-28", "2020-05-29", "2020-05-30", "2020-05-31", "2020-06-01", "2020-06-02", "2020-06-03", "2020-06-04", "2020-06-05", "2020-06-06", "2020-06-07", "2020-06-08", "2020-06-09", "2020-06-10", "2020-06-11", "2020-06-12", "2020-06-13", "2020-06-14", "2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18", "2020-06-19", "2020-06-20", "2020-06-21", "2020-06-22", "2020-06-23", "2020-06-24", "2020-06-25", "2020-06-26", "2020-06-27", "2020-06-28", "2020-06-29", "2020-06-30"], "y": [1, 1, 4, 22, 27, 27, 51, 61, 116, 187, 246, 312, 371, 429, 490, 607, 714, 885, 1352, 2097, 2598, 3110, 4325, 5624, 6511, 7242, 7734, 8161, 8718, 9772, 11173, 13202, 16400, 21586, 26179, 32756, 40304, 53553, 66652, 81167, 94055, 98921, 109591, 125851, 136707, 146541, 164740, 178189, 188518, 196618, 210144, 222395, 234998, 248647, 261335, 268212, 274504, 286716, 295647, 307024, 316664, 327509, 331780, 338036, 348582, 357896, 364942, 376961, 387925, 394141, 395194, 410564, 421547, 431743, 443273, 455032, 461233, 466804, 477304, 486003, 494711, 503051, 510479, 514621, 518743, 527438, 534552, 538775, 546463, 554834, 559109, 563035, 571014, 577959, 583971, 589520, 595161, 598285, 601660, 608514, 614197, 619881, 623069, 627562, 630973, 633921, 640237, 643095, 649388, 654863, 660030, 663886, 668492, 680890, 695920, 711484, 726747, 740645, 747653, 753211, 765574, 776433, 788684, 802418, 814420, 820289, 826437, 839475, 852876, 868666, 885120, 902901, 910822, 921391, 939851, 956852, 973858, 990960, 1005305, 1012769, 1018214, 1035997, 1051972, 1066888, 1081487, 1094704, 1100328, 1105719, 1119767, 1132823, 1145712, 1158063, 1170901, 1176463, 1182066, 1196012, 1208597, 1220478, 1232315, 1243780, 1251695, 1259954, 1273766]}, {"mode": "lines+markers", "name": "negative", "type": "scatter", "x": ["2020-01-20", "2020-01-21", "2020-01-22", "2020-01-23", "2020-01-24", "2020-01-25", "2020-01-26", "2020-01-27", "2020-01-28", "2020-01-29", "2020-01-30", "2020-01-31", "2020-02-01", "2020-02-02", "2020-02-03", "2020-02-04", "2020-02-05", "2020-02-06", "2020-02-07", "2020-02-08", "2020-02-09", "2020-02-10", "2020-02-11", "2020-02-12", "2020-02-13", "2020-02-14", "2020-02-15", "2020-02-16", "2020-02-17", "2020-02-18", "2020-02-19", "2020-02-20", "2020-02-21", "2020-02-22", "2020-02-23", "2020-02-24", "2020-02-25", "2020-02-26", "2020-02-27", "2020-02-28", "2020-02-29", "2020-03-01", "2020-03-02", "2020-03-03", "2020-03-04", "2020-03-05", "2020-03-06", "2020-03-07", "2020-03-08", "2020-03-09", "2020-03-10", "2020-03-11", "2020-03-12", "2020-03-13", "2020-03-14", "2020-03-15", "2020-03-16", "2020-03-17", "2020-03-18", "2020-03-19", "2020-03-20", "2020-03-21", "2020-03-22", "2020-03-23", "2020-03-24", "2020-03-25", "2020-03-26", "2020-03-27", "2020-03-28", "2020-03-29", "2020-03-30", "2020-03-31", "2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04", "2020-04-05", "2020-04-06", "2020-04-07", "2020-04-08", "2020-04-09", "2020-04-10", "2020-04-11", "2020-04-12", "2020-04-13", "2020-04-14", "2020-04-15", "2020-04-16", "2020-04-17", "2020-04-18", "2020-04-19", "2020-04-20", "2020-04-21", "2020-04-22", "2020-04-23", "2020-04-24", "2020-04-25", "2020-04-26", "2020-04-27", "2020-04-28", "2020-04-29", "2020-04-30", "2020-05-01", "2020-05-02", "2020-05-03", "2020-05-04", "2020-05-05", "2020-05-06", "2020-05-07", "2020-05-08", "2020-05-09", "2020-05-10", "2020-05-11", "2020-05-12", "2020-05-13", "2020-05-14", "2020-05-15", "2020-05-16", "2020-05-17", "2020-05-18", "2020-05-19", "2020-05-20", "2020-05-21", "2020-05-22", "2020-05-23", "2020-05-24", "2020-05-25", "2020-05-26", "2020-05-27", "2020-05-28", "2020-05-29", "2020-05-30", "2020-05-31", "2020-06-01", "2020-06-02", "2020-06-03", "2020-06-04", "2020-06-05", "2020-06-06", "2020-06-07", "2020-06-08", "2020-06-09", "2020-06-10", "2020-06-11", "2020-06-12", "2020-06-13", "2020-06-14", "2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18", "2020-06-19", "2020-06-20", "2020-06-21", "2020-06-22", "2020-06-23", "2020-06-24", "2020-06-25", "2020-06-26", "2020-06-27", "2020-06-28", "2020-06-29", "2020-06-30"], "y": [0, 0, 3, 21, 25, 25, 47, 56, 97, 155, 199, 245, 289, 327, 414, 462, 522, 693, 1001, 1134, 1683, 2552, 3535, 4811, 5921, 6679, 7148, 7647, 7980, 8923, 9973, 11238, 13016, 15116, 17520, 20292, 25447, 31576, 39318, 48593, 55723, 61825, 71580, 85484, 102965, 118965, 136624, 151802, 162008, 171778, 184179, 196100, 209402, 222728, 235615, 243778, 251297, 261105, 270888, 282555, 292487, 303006, 308343, 315447, 324105, 334481, 341332, 352410, 361883, 369530, 372002, 383886, 395075, 403882, 414303, 424732, 431425, 437225, 446323, 457761, 468779, 477303, 485929, 490321, 494815, 502223, 508935, 513894, 521642, 530631, 536205, 540380, 547610, 555144, 563130, 569212, 575184, 578558, 582027, 588559, 595129, 600482, 603610, 608286, 611592, 614944, 620575, 624280, 630149, 635174, 640037, 642884, 646661, 653624, 665379, 679771, 695854, 711265, 718943, 726053, 737571, 748972, 759473, 770990, 781686, 788766, 796142, 806206, 820550, 834952, 849161, 865162, 876060, 885830, 899388, 917397, 934030, 950526, 965632, 974512, 982026, 996686, 1013847, 1029447, 1045240, 1059301, 1066887, 1072805, 1084980, 1099136, 1111741, 1124567, 1137058, 1143971, 1150225, 1161250, 1175817, 1189015, 1200885, 1211261, 1219975, 1228698, 1240157]}],
                        {"height": 600, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "\uac80\uc0ac\uc790\uc640 \uc74c\uc131\ud310\uc815 \uc218", "x": 0.5, "xanchor": "center", "y": 0.88, "yanchor": "bottom"}, "width": 1000, "xaxis": {"title": {"text": "month"}}, "yaxis": {"title": {"text": "Number"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('58890cd2-0b14-48c9-bbed-c7170a016178');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>



```python
## https://minus31.github.io/2018/07/28/python-date/ 
## 날짜 datetime으로 처리 

time['date'] = pd.to_datetime(time['date'])
time['date'].describe()
```




    count                     163
    unique                    163
    top       2020-03-13 00:00:00
    freq                        1
    first     2020-01-20 00:00:00
    last      2020-06-30 00:00:00
    Name: date, dtype: object




```python
fig = go.Figure()

fig.add_trace(go.Bar(x=time['date'],y=time['confirmed'].diff(), name='confirmed'))
fig.update_layout(title_text="일별 확진자 수 (Number of Patients)", 
                  title = {'x' : 0.5, 'xanchor' : 'center', "y" : 0.85, "yanchor" : "bottom" } )


fig.show()

#3월에 확 터지고 (아마 대구 신천지), 5월에 K방역 성공하는듯 했으나 이태원으로 다시 터짐
```


<div>


            <div id="8dc6eab1-6c36-4fa5-a1ed-3adca352d829" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("8dc6eab1-6c36-4fa5-a1ed-3adca352d829")) {
                    Plotly.newPlot(
                        '8dc6eab1-6c36-4fa5-a1ed-3adca352d829',
                        [{"name": "confirmed", "type": "bar", "x": ["2020-01-20T00:00:00", "2020-01-21T00:00:00", "2020-01-22T00:00:00", "2020-01-23T00:00:00", "2020-01-24T00:00:00", "2020-01-25T00:00:00", "2020-01-26T00:00:00", "2020-01-27T00:00:00", "2020-01-28T00:00:00", "2020-01-29T00:00:00", "2020-01-30T00:00:00", "2020-01-31T00:00:00", "2020-02-01T00:00:00", "2020-02-02T00:00:00", "2020-02-03T00:00:00", "2020-02-04T00:00:00", "2020-02-05T00:00:00", "2020-02-06T00:00:00", "2020-02-07T00:00:00", "2020-02-08T00:00:00", "2020-02-09T00:00:00", "2020-02-10T00:00:00", "2020-02-11T00:00:00", "2020-02-12T00:00:00", "2020-02-13T00:00:00", "2020-02-14T00:00:00", "2020-02-15T00:00:00", "2020-02-16T00:00:00", "2020-02-17T00:00:00", "2020-02-18T00:00:00", "2020-02-19T00:00:00", "2020-02-20T00:00:00", "2020-02-21T00:00:00", "2020-02-22T00:00:00", "2020-02-23T00:00:00", "2020-02-24T00:00:00", "2020-02-25T00:00:00", "2020-02-26T00:00:00", "2020-02-27T00:00:00", "2020-02-28T00:00:00", "2020-02-29T00:00:00", "2020-03-01T00:00:00", "2020-03-02T00:00:00", "2020-03-03T00:00:00", "2020-03-04T00:00:00", "2020-03-05T00:00:00", "2020-03-06T00:00:00", "2020-03-07T00:00:00", "2020-03-08T00:00:00", "2020-03-09T00:00:00", "2020-03-10T00:00:00", "2020-03-11T00:00:00", "2020-03-12T00:00:00", "2020-03-13T00:00:00", "2020-03-14T00:00:00", "2020-03-15T00:00:00", "2020-03-16T00:00:00", "2020-03-17T00:00:00", "2020-03-18T00:00:00", "2020-03-19T00:00:00", "2020-03-20T00:00:00", "2020-03-21T00:00:00", "2020-03-22T00:00:00", "2020-03-23T00:00:00", "2020-03-24T00:00:00", "2020-03-25T00:00:00", "2020-03-26T00:00:00", "2020-03-27T00:00:00", "2020-03-28T00:00:00", "2020-03-29T00:00:00", "2020-03-30T00:00:00", "2020-03-31T00:00:00", "2020-04-01T00:00:00", "2020-04-02T00:00:00", "2020-04-03T00:00:00", "2020-04-04T00:00:00", "2020-04-05T00:00:00", "2020-04-06T00:00:00", "2020-04-07T00:00:00", "2020-04-08T00:00:00", "2020-04-09T00:00:00", "2020-04-10T00:00:00", "2020-04-11T00:00:00", "2020-04-12T00:00:00", "2020-04-13T00:00:00", "2020-04-14T00:00:00", "2020-04-15T00:00:00", "2020-04-16T00:00:00", "2020-04-17T00:00:00", "2020-04-18T00:00:00", "2020-04-19T00:00:00", "2020-04-20T00:00:00", "2020-04-21T00:00:00", "2020-04-22T00:00:00", "2020-04-23T00:00:00", "2020-04-24T00:00:00", "2020-04-25T00:00:00", "2020-04-26T00:00:00", "2020-04-27T00:00:00", "2020-04-28T00:00:00", "2020-04-29T00:00:00", "2020-04-30T00:00:00", "2020-05-01T00:00:00", "2020-05-02T00:00:00", "2020-05-03T00:00:00", "2020-05-04T00:00:00", "2020-05-05T00:00:00", "2020-05-06T00:00:00", "2020-05-07T00:00:00", "2020-05-08T00:00:00", "2020-05-09T00:00:00", "2020-05-10T00:00:00", "2020-05-11T00:00:00", "2020-05-12T00:00:00", "2020-05-13T00:00:00", "2020-05-14T00:00:00", "2020-05-15T00:00:00", "2020-05-16T00:00:00", "2020-05-17T00:00:00", "2020-05-18T00:00:00", "2020-05-19T00:00:00", "2020-05-20T00:00:00", "2020-05-21T00:00:00", "2020-05-22T00:00:00", "2020-05-23T00:00:00", "2020-05-24T00:00:00", "2020-05-25T00:00:00", "2020-05-26T00:00:00", "2020-05-27T00:00:00", "2020-05-28T00:00:00", "2020-05-29T00:00:00", "2020-05-30T00:00:00", "2020-05-31T00:00:00", "2020-06-01T00:00:00", "2020-06-02T00:00:00", "2020-06-03T00:00:00", "2020-06-04T00:00:00", "2020-06-05T00:00:00", "2020-06-06T00:00:00", "2020-06-07T00:00:00", "2020-06-08T00:00:00", "2020-06-09T00:00:00", "2020-06-10T00:00:00", "2020-06-11T00:00:00", "2020-06-12T00:00:00", "2020-06-13T00:00:00", "2020-06-14T00:00:00", "2020-06-15T00:00:00", "2020-06-16T00:00:00", "2020-06-17T00:00:00", "2020-06-18T00:00:00", "2020-06-19T00:00:00", "2020-06-20T00:00:00", "2020-06-21T00:00:00", "2020-06-22T00:00:00", "2020-06-23T00:00:00", "2020-06-24T00:00:00", "2020-06-25T00:00:00", "2020-06-26T00:00:00", "2020-06-27T00:00:00", "2020-06-28T00:00:00", "2020-06-29T00:00:00", "2020-06-30T00:00:00"], "y": [null, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 5.0, 1.0, 3.0, 0.0, 1.0, 2.0, 5.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 20.0, 53.0, 100.0, 229.0, 169.0, 231.0, 144.0, 284.0, 505.0, 571.0, 813.0, 586.0, 476.0, 600.0, 516.0, 438.0, 518.0, 483.0, 367.0, 248.0, 131.0, 242.0, 114.0, 110.0, 107.0, 40.0, 110.0, 84.0, 93.0, 152.0, 87.0, 147.0, 98.0, 64.0, 76.0, 100.0, 104.0, 91.0, 146.0, 105.0, 78.0, 125.0, 101.0, 89.0, 86.0, 94.0, 81.0, 47.0, 47.0, 53.0, 39.0, 27.0, 30.0, 32.0, 25.0, 27.0, 27.0, 22.0, 22.0, 18.0, 8.0, 13.0, 9.0, 11.0, 8.0, 6.0, 10.0, 10.0, 10.0, 14.0, 9.0, 4.0, 9.0, 6.0, 13.0, 8.0, 3.0, 2.0, 4.0, 12.0, 18.0, 34.0, 35.0, 27.0, 26.0, 29.0, 27.0, 19.0, 13.0, 15.0, 13.0, 32.0, 12.0, 20.0, 23.0, 25.0, 16.0, 19.0, 40.0, 79.0, 58.0, 39.0, 27.0, 35.0, 38.0, 49.0, 39.0, 39.0, 51.0, 57.0, 38.0, 38.0, 50.0, 45.0, 56.0, 48.0, 34.0, 36.0, 34.0, 43.0, 59.0, 49.0, 67.0, 48.0, 17.0, 46.0, 51.0, 28.0, 39.0, 51.0, 62.0, 42.0, 43.0]}],
                        {"template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "\uc77c\ubcc4 \ud655\uc9c4\uc790 \uc218 (Number of Patients)", "x": 0.5, "xanchor": "center", "y": 0.85, "yanchor": "bottom"}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('8dc6eab1-6c36-4fa5-a1ed-3adca352d829');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>



```python
print(timeage.shape)
timeage.head()
```

    (1089, 5)
    




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
      <th>date</th>
      <th>time</th>
      <th>age</th>
      <th>confirmed</th>
      <th>deceased</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2020-03-02</td>
      <td>0</td>
      <td>0s</td>
      <td>32</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2020-03-02</td>
      <td>0</td>
      <td>10s</td>
      <td>169</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2020-03-02</td>
      <td>0</td>
      <td>20s</td>
      <td>1235</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2020-03-02</td>
      <td>0</td>
      <td>30s</td>
      <td>506</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2020-03-02</td>
      <td>0</td>
      <td>40s</td>
      <td>633</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig = px.bar(timeage, x='date', y='confirmed', hover_data=['age'], color='age')

fig.update_traces(marker_line_width= 0.5)


fig.update_layout(height=600, width=1000, title_text="연령별 확진자 수 추이", 
                 title = {'x' : 0.5, 'xanchor' : 'center', "y" : 0.93, "yanchor" : "bottom" })

fig


#20대가 제일 많고, 그리고 50대
```


<div>


            <div id="36db25e0-860e-48f1-b6f4-08681b132f96" class="plotly-graph-div" style="height:600px; width:1000px;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("36db25e0-860e-48f1-b6f4-08681b132f96")) {
                    Plotly.newPlot(
                        '36db25e0-860e-48f1-b6f4-08681b132f96',
                        [{"alignmentgroup": "True", "customdata": [["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"], ["0s"]], "hovertemplate": "age=%{customdata[0]}<br>date=%{x}<br>confirmed=%{y}<extra></extra>", "legendgroup": "0s", "marker": {"color": "#636efa", "line": {"width": 0.5}}, "name": "0s", "offsetgroup": "0s", "orientation": "v", "showlegend": true, "textposition": "auto", "type": "bar", "x": ["2020-03-02", "2020-03-03", "2020-03-04", "2020-03-05", "2020-03-06", "2020-03-07", "2020-03-08", "2020-03-09", "2020-03-10", "2020-03-11", "2020-03-12", "2020-03-13", "2020-03-14", "2020-03-15", "2020-03-16", "2020-03-17", "2020-03-18", "2020-03-19", "2020-03-20", "2020-03-21", "2020-03-22", "2020-03-23", "2020-03-24", "2020-03-25", "2020-03-26", "2020-03-27", "2020-03-28", "2020-03-29", "2020-03-30", "2020-03-31", "2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04", "2020-04-05", "2020-04-06", "2020-04-07", "2020-04-08", "2020-04-09", "2020-04-10", "2020-04-11", "2020-04-12", "2020-04-13", "2020-04-14", "2020-04-15", "2020-04-16", "2020-04-17", "2020-04-18", "2020-04-19", "2020-04-20", "2020-04-21", "2020-04-22", "2020-04-23", "2020-04-24", "2020-04-25", "2020-04-26", "2020-04-27", "2020-04-28", "2020-04-29", "2020-04-30", "2020-05-01", "2020-05-02", "2020-05-03", "2020-05-04", "2020-05-05", "2020-05-06", "2020-05-07", "2020-05-08", "2020-05-09", "2020-05-10", "2020-05-11", "2020-05-12", "2020-05-13", "2020-05-14", "2020-05-15", "2020-05-16", "2020-05-17", "2020-05-18", "2020-05-19", "2020-05-20", "2020-05-21", "2020-05-22", "2020-05-23", "2020-05-24", "2020-05-25", "2020-05-26", "2020-05-27", "2020-05-28", "2020-05-29", "2020-05-30", "2020-05-31", "2020-06-01", "2020-06-02", "2020-06-03", "2020-06-04", "2020-06-05", "2020-06-06", "2020-06-07", "2020-06-08", "2020-06-09", "2020-06-10", "2020-06-11", "2020-06-12", "2020-06-13", "2020-06-14", "2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18", "2020-06-19", "2020-06-20", "2020-06-21", "2020-06-22", "2020-06-23", "2020-06-24", "2020-06-25", "2020-06-26", "2020-06-27", "2020-06-28", "2020-06-29", "2020-06-30"], "xaxis": "x", "y": [32, 34, 34, 38, 45, 52, 58, 66, 67, 75, 76, 77, 81, 83, 85, 86, 87, 91, 97, 99, 101, 103, 105, 105, 106, 108, 109, 111, 112, 112, 116, 119, 121, 122, 124, 126, 126, 126, 128, 129, 130, 130, 132, 132, 132, 135, 136, 138, 138, 138, 138, 138, 138, 138, 139, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 141, 141, 141, 141, 142, 143, 143, 145, 147, 147, 147, 148, 148, 148, 149, 149, 149, 149, 150, 153, 155, 156, 157, 157, 158, 158, 159, 160, 161, 164, 165, 166, 166, 167, 168, 168, 168, 168, 170, 172, 174, 175, 175, 177, 177, 177, 180, 182, 183, 184, 185, 187, 190, 193], "yaxis": "y"}, {"alignmentgroup": "True", "customdata": [["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"], ["10s"]], "hovertemplate": "age=%{customdata[0]}<br>date=%{x}<br>confirmed=%{y}<extra></extra>", "legendgroup": "10s", "marker": {"color": "#EF553B", "line": {"width": 0.5}}, "name": "10s", "offsetgroup": "10s", "orientation": "v", "showlegend": true, "textposition": "auto", "type": "bar", "x": ["2020-03-02", "2020-03-03", "2020-03-04", "2020-03-05", "2020-03-06", "2020-03-07", "2020-03-08", "2020-03-09", "2020-03-10", "2020-03-11", "2020-03-12", "2020-03-13", "2020-03-14", "2020-03-15", "2020-03-16", "2020-03-17", "2020-03-18", "2020-03-19", "2020-03-20", "2020-03-21", "2020-03-22", "2020-03-23", "2020-03-24", "2020-03-25", "2020-03-26", "2020-03-27", "2020-03-28", "2020-03-29", "2020-03-30", "2020-03-31", "2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04", "2020-04-05", "2020-04-06", "2020-04-07", "2020-04-08", "2020-04-09", "2020-04-10", "2020-04-11", "2020-04-12", "2020-04-13", "2020-04-14", "2020-04-15", "2020-04-16", "2020-04-17", "2020-04-18", "2020-04-19", "2020-04-20", "2020-04-21", "2020-04-22", "2020-04-23", "2020-04-24", "2020-04-25", "2020-04-26", "2020-04-27", "2020-04-28", "2020-04-29", "2020-04-30", "2020-05-01", "2020-05-02", "2020-05-03", "2020-05-04", "2020-05-05", "2020-05-06", "2020-05-07", "2020-05-08", "2020-05-09", "2020-05-10", "2020-05-11", "2020-05-12", "2020-05-13", "2020-05-14", "2020-05-15", "2020-05-16", "2020-05-17", "2020-05-18", "2020-05-19", "2020-05-20", "2020-05-21", "2020-05-22", "2020-05-23", "2020-05-24", "2020-05-25", "2020-05-26", "2020-05-27", "2020-05-28", "2020-05-29", "2020-05-30", "2020-05-31", "2020-06-01", "2020-06-02", "2020-06-03", "2020-06-04", "2020-06-05", "2020-06-06", "2020-06-07", "2020-06-08", "2020-06-09", "2020-06-10", "2020-06-11", "2020-06-12", "2020-06-13", "2020-06-14", "2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18", "2020-06-19", "2020-06-20", "2020-06-21", "2020-06-22", "2020-06-23", "2020-06-24", "2020-06-25", "2020-06-26", "2020-06-27", "2020-06-28", "2020-06-29", "2020-06-30"], "xaxis": "x", "y": [169, 204, 233, 257, 292, 327, 360, 381, 393, 405, 412, 421, 424, 427, 432, 436, 438, 444, 452, 457, 460, 460, 468, 475, 488, 496, 501, 508, 513, 515, 519, 522, 528, 530, 535, 542, 544, 548, 552, 553, 558, 561, 562, 565, 569, 572, 576, 576, 578, 581, 582, 583, 584, 585, 586, 586, 588, 590, 590, 590, 590, 590, 591, 591, 591, 591, 592, 592, 594, 597, 598, 602, 603, 614, 617, 619, 620, 621, 621, 627, 631, 633, 634, 636, 637, 638, 640, 644, 650, 655, 655, 657, 659, 661, 661, 663, 664, 670, 672, 672, 674, 677, 680, 681, 681, 681, 682, 683, 684, 686, 690, 691, 692, 692, 693, 696, 698, 700, 703, 704, 708], "yaxis": "y"}, {"alignmentgroup": "True", "customdata": [["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"], ["20s"]], "hovertemplate": "age=%{customdata[0]}<br>date=%{x}<br>confirmed=%{y}<extra></extra>", "legendgroup": "20s", "marker": {"color": "#00cc96", "line": {"width": 0.5}}, "name": "20s", "offsetgroup": "20s", "orientation": "v", "showlegend": true, "textposition": "auto", "type": "bar", "x": ["2020-03-02", "2020-03-03", "2020-03-04", "2020-03-05", "2020-03-06", "2020-03-07", "2020-03-08", "2020-03-09", "2020-03-10", "2020-03-11", "2020-03-12", "2020-03-13", "2020-03-14", "2020-03-15", "2020-03-16", "2020-03-17", "2020-03-18", "2020-03-19", "2020-03-20", "2020-03-21", "2020-03-22", "2020-03-23", "2020-03-24", "2020-03-25", "2020-03-26", "2020-03-27", "2020-03-28", "2020-03-29", "2020-03-30", "2020-03-31", "2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04", "2020-04-05", "2020-04-06", "2020-04-07", "2020-04-08", "2020-04-09", "2020-04-10", "2020-04-11", "2020-04-12", "2020-04-13", "2020-04-14", "2020-04-15", "2020-04-16", "2020-04-17", "2020-04-18", "2020-04-19", "2020-04-20", "2020-04-21", "2020-04-22", "2020-04-23", "2020-04-24", "2020-04-25", "2020-04-26", "2020-04-27", "2020-04-28", "2020-04-29", "2020-04-30", "2020-05-01", "2020-05-02", "2020-05-03", "2020-05-04", "2020-05-05", "2020-05-06", "2020-05-07", "2020-05-08", "2020-05-09", "2020-05-10", "2020-05-11", "2020-05-12", "2020-05-13", "2020-05-14", "2020-05-15", "2020-05-16", "2020-05-17", "2020-05-18", "2020-05-19", "2020-05-20", "2020-05-21", "2020-05-22", "2020-05-23", "2020-05-24", "2020-05-25", "2020-05-26", "2020-05-27", "2020-05-28", "2020-05-29", "2020-05-30", "2020-05-31", "2020-06-01", "2020-06-02", "2020-06-03", "2020-06-04", "2020-06-05", "2020-06-06", "2020-06-07", "2020-06-08", "2020-06-09", "2020-06-10", "2020-06-11", "2020-06-12", "2020-06-13", "2020-06-14", "2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18", "2020-06-19", "2020-06-20", "2020-06-21", "2020-06-22", "2020-06-23", "2020-06-24", "2020-06-25", "2020-06-26", "2020-06-27", "2020-06-28", "2020-06-29", "2020-06-30"], "xaxis": "x", "y": [1235, 1417, 1575, 1727, 1877, 2038, 2133, 2190, 2213, 2238, 2261, 2274, 2287, 2301, 2313, 2330, 2342, 2358, 2365, 2380, 2396, 2417, 2438, 2473, 2508, 2532, 2567, 2602, 2630, 2656, 2682, 2704, 2734, 2761, 2789, 2804, 2819, 2832, 2844, 2851, 2856, 2869, 2879, 2886, 2895, 2900, 2909, 2918, 2921, 2926, 2928, 2931, 2933, 2934, 2937, 2940, 2943, 2948, 2951, 2952, 2957, 2960, 2962, 2964, 2964, 2964, 2966, 2967, 2979, 2998, 3019, 3029, 3042, 3056, 3066, 3074, 3079, 3082, 3087, 3100, 3103, 3111, 3113, 3117, 3120, 3123, 3131, 3146, 3158, 3167, 3176, 3178, 3183, 3188, 3193, 3198, 3200, 3203, 3208, 3211, 3217, 3223, 3234, 3243, 3251, 3256, 3259, 3267, 3277, 3285, 3294, 3299, 3302, 3306, 3309, 3309, 3317, 3331, 3343, 3352, 3362], "yaxis": "y"}, {"alignmentgroup": "True", "customdata": [["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"], ["30s"]], "hovertemplate": "age=%{customdata[0]}<br>date=%{x}<br>confirmed=%{y}<extra></extra>", "legendgroup": "30s", "marker": {"color": "#ab63fa", "line": {"width": 0.5}}, "name": "30s", "offsetgroup": "30s", "orientation": "v", "showlegend": true, "textposition": "auto", "type": "bar", "x": ["2020-03-02", "2020-03-03", "2020-03-04", "2020-03-05", "2020-03-06", "2020-03-07", "2020-03-08", "2020-03-09", "2020-03-10", "2020-03-11", "2020-03-12", "2020-03-13", "2020-03-14", "2020-03-15", "2020-03-16", "2020-03-17", "2020-03-18", "2020-03-19", "2020-03-20", "2020-03-21", "2020-03-22", "2020-03-23", "2020-03-24", "2020-03-25", "2020-03-26", "2020-03-27", "2020-03-28", "2020-03-29", "2020-03-30", "2020-03-31", "2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04", "2020-04-05", "2020-04-06", "2020-04-07", "2020-04-08", "2020-04-09", "2020-04-10", "2020-04-11", "2020-04-12", "2020-04-13", "2020-04-14", "2020-04-15", "2020-04-16", "2020-04-17", "2020-04-18", "2020-04-19", "2020-04-20", "2020-04-21", "2020-04-22", "2020-04-23", "2020-04-24", "2020-04-25", "2020-04-26", "2020-04-27", "2020-04-28", "2020-04-29", "2020-04-30", "2020-05-01", "2020-05-02", "2020-05-03", "2020-05-04", "2020-05-05", "2020-05-06", "2020-05-07", "2020-05-08", "2020-05-09", "2020-05-10", "2020-05-11", "2020-05-12", "2020-05-13", "2020-05-14", "2020-05-15", "2020-05-16", "2020-05-17", "2020-05-18", "2020-05-19", "2020-05-20", "2020-05-21", "2020-05-22", "2020-05-23", "2020-05-24", "2020-05-25", "2020-05-26", "2020-05-27", "2020-05-28", "2020-05-29", "2020-05-30", "2020-05-31", "2020-06-01", "2020-06-02", "2020-06-03", "2020-06-04", "2020-06-05", "2020-06-06", "2020-06-07", "2020-06-08", "2020-06-09", "2020-06-10", "2020-06-11", "2020-06-12", "2020-06-13", "2020-06-14", "2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18", "2020-06-19", "2020-06-20", "2020-06-21", "2020-06-22", "2020-06-23", "2020-06-24", "2020-06-25", "2020-06-26", "2020-06-27", "2020-06-28", "2020-06-29", "2020-06-30"], "xaxis": "x", "y": [506, 578, 631, 659, 693, 727, 760, 779, 789, 804, 812, 823, 833, 842, 849, 856, 873, 886, 893, 900, 909, 917, 921, 943, 955, 960, 978, 993, 1002, 1012, 1027, 1043, 1052, 1066, 1083, 1086, 1092, 1102, 1109, 1113, 1115, 1120, 1122, 1126, 1131, 1132, 1135, 1136, 1137, 1139, 1139, 1140, 1141, 1142, 1143, 1147, 1152, 1154, 1156, 1158, 1159, 1160, 1163, 1164, 1165, 1166, 1167, 1174, 1177, 1180, 1188, 1194, 1199, 1202, 1207, 1209, 1211, 1215, 1215, 1219, 1221, 1225, 1231, 1235, 1238, 1242, 1248, 1274, 1283, 1285, 1292, 1296, 1299, 1305, 1308, 1310, 1316, 1322, 1324, 1326, 1335, 1345, 1353, 1354, 1357, 1362, 1366, 1380, 1384, 1393, 1406, 1416, 1420, 1429, 1444, 1453, 1463, 1473, 1485, 1490, 1496], "yaxis": "y"}, {"alignmentgroup": "True", "customdata": [["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"], ["40s"]], "hovertemplate": "age=%{customdata[0]}<br>date=%{x}<br>confirmed=%{y}<extra></extra>", "legendgroup": "40s", "marker": {"color": "#FFA15A", "line": {"width": 0.5}}, "name": "40s", "offsetgroup": "40s", "orientation": "v", "showlegend": true, "textposition": "auto", "type": "bar", "x": ["2020-03-02", "2020-03-03", "2020-03-04", "2020-03-05", "2020-03-06", "2020-03-07", "2020-03-08", "2020-03-09", "2020-03-10", "2020-03-11", "2020-03-12", "2020-03-13", "2020-03-14", "2020-03-15", "2020-03-16", "2020-03-17", "2020-03-18", "2020-03-19", "2020-03-20", "2020-03-21", "2020-03-22", "2020-03-23", "2020-03-24", "2020-03-25", "2020-03-26", "2020-03-27", "2020-03-28", "2020-03-29", "2020-03-30", "2020-03-31", "2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04", "2020-04-05", "2020-04-06", "2020-04-07", "2020-04-08", "2020-04-09", "2020-04-10", "2020-04-11", "2020-04-12", "2020-04-13", "2020-04-14", "2020-04-15", "2020-04-16", "2020-04-17", "2020-04-18", "2020-04-19", "2020-04-20", "2020-04-21", "2020-04-22", "2020-04-23", "2020-04-24", "2020-04-25", "2020-04-26", "2020-04-27", "2020-04-28", "2020-04-29", "2020-04-30", "2020-05-01", "2020-05-02", "2020-05-03", "2020-05-04", "2020-05-05", "2020-05-06", "2020-05-07", "2020-05-08", "2020-05-09", "2020-05-10", "2020-05-11", "2020-05-12", "2020-05-13", "2020-05-14", "2020-05-15", "2020-05-16", "2020-05-17", "2020-05-18", "2020-05-19", "2020-05-20", "2020-05-21", "2020-05-22", "2020-05-23", "2020-05-24", "2020-05-25", "2020-05-26", "2020-05-27", "2020-05-28", "2020-05-29", "2020-05-30", "2020-05-31", "2020-06-01", "2020-06-02", "2020-06-03", "2020-06-04", "2020-06-05", "2020-06-06", "2020-06-07", "2020-06-08", "2020-06-09", "2020-06-10", "2020-06-11", "2020-06-12", "2020-06-13", "2020-06-14", "2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18", "2020-06-19", "2020-06-20", "2020-06-21", "2020-06-22", "2020-06-23", "2020-06-24", "2020-06-25", "2020-06-26", "2020-06-27", "2020-06-28", "2020-06-29", "2020-06-30"], "xaxis": "x", "y": [633, 713, 790, 847, 889, 941, 975, 1005, 1030, 1082, 1101, 1117, 1133, 1141, 1147, 1164, 1171, 1181, 1193, 1203, 1221, 1228, 1234, 1246, 1252, 1256, 1278, 1292, 1297, 1312, 1323, 1336, 1350, 1358, 1370, 1375, 1382, 1387, 1394, 1396, 1399, 1400, 1401, 1405, 1408, 1410, 1411, 1412, 1412, 1412, 1413, 1417, 1418, 1418, 1420, 1421, 1422, 1423, 1426, 1427, 1429, 1430, 1432, 1435, 1436, 1436, 1436, 1438, 1438, 1442, 1446, 1448, 1450, 1451, 1453, 1454, 1457, 1462, 1462, 1468, 1471, 1473, 1474, 1481, 1483, 1486, 1489, 1503, 1513, 1517, 1521, 1527, 1529, 1534, 1537, 1540, 1551, 1559, 1564, 1568, 1569, 1574, 1575, 1578, 1584, 1595, 1599, 1601, 1607, 1610, 1618, 1623, 1624, 1633, 1644, 1646, 1651, 1657, 1667, 1673, 1681], "yaxis": "y"}, {"alignmentgroup": "True", "customdata": [["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"], ["50s"]], "hovertemplate": "age=%{customdata[0]}<br>date=%{x}<br>confirmed=%{y}<extra></extra>", "legendgroup": "50s", "marker": {"color": "#19d3f3", "line": {"width": 0.5}}, "name": "50s", "offsetgroup": "50s", "orientation": "v", "showlegend": true, "textposition": "auto", "type": "bar", "x": ["2020-03-02", "2020-03-03", "2020-03-04", "2020-03-05", "2020-03-06", "2020-03-07", "2020-03-08", "2020-03-09", "2020-03-10", "2020-03-11", "2020-03-12", "2020-03-13", "2020-03-14", "2020-03-15", "2020-03-16", "2020-03-17", "2020-03-18", "2020-03-19", "2020-03-20", "2020-03-21", "2020-03-22", "2020-03-23", "2020-03-24", "2020-03-25", "2020-03-26", "2020-03-27", "2020-03-28", "2020-03-29", "2020-03-30", "2020-03-31", "2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04", "2020-04-05", "2020-04-06", "2020-04-07", "2020-04-08", "2020-04-09", "2020-04-10", "2020-04-11", "2020-04-12", "2020-04-13", "2020-04-14", "2020-04-15", "2020-04-16", "2020-04-17", "2020-04-18", "2020-04-19", "2020-04-20", "2020-04-21", "2020-04-22", "2020-04-23", "2020-04-24", "2020-04-25", "2020-04-26", "2020-04-27", "2020-04-28", "2020-04-29", "2020-04-30", "2020-05-01", "2020-05-02", "2020-05-03", "2020-05-04", "2020-05-05", "2020-05-06", "2020-05-07", "2020-05-08", "2020-05-09", "2020-05-10", "2020-05-11", "2020-05-12", "2020-05-13", "2020-05-14", "2020-05-15", "2020-05-16", "2020-05-17", "2020-05-18", "2020-05-19", "2020-05-20", "2020-05-21", "2020-05-22", "2020-05-23", "2020-05-24", "2020-05-25", "2020-05-26", "2020-05-27", "2020-05-28", "2020-05-29", "2020-05-30", "2020-05-31", "2020-06-01", "2020-06-02", "2020-06-03", "2020-06-04", "2020-06-05", "2020-06-06", "2020-06-07", "2020-06-08", "2020-06-09", "2020-06-10", "2020-06-11", "2020-06-12", "2020-06-13", "2020-06-14", "2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18", "2020-06-19", "2020-06-20", "2020-06-21", "2020-06-22", "2020-06-23", "2020-06-24", "2020-06-25", "2020-06-26", "2020-06-27", "2020-06-28", "2020-06-29", "2020-06-30"], "xaxis": "x", "y": [834, 952, 1051, 1127, 1217, 1287, 1349, 1391, 1416, 1472, 1495, 1523, 1551, 1568, 1585, 1602, 1615, 1642, 1656, 1672, 1691, 1702, 1716, 1724, 1738, 1752, 1780, 1798, 1812, 1851, 1865, 1878, 1887, 1898, 1904, 1906, 1909, 1915, 1917, 1920, 1926, 1930, 1932, 1935, 1937, 1940, 1942, 1944, 1945, 1948, 1951, 1951, 1952, 1953, 1953, 1953, 1953, 1955, 1956, 1956, 1956, 1956, 1956, 1956, 1957, 1957, 1957, 1957, 1958, 1960, 1960, 1963, 1964, 1964, 1965, 1966, 1967, 1968, 1971, 1972, 1972, 1974, 1983, 1987, 1992, 1996, 2002, 2014, 2023, 2033, 2039, 2052, 2061, 2070, 2079, 2086, 2094, 2105, 2111, 2121, 2128, 2137, 2154, 2163, 2170, 2176, 2182, 2189, 2198, 2210, 2222, 2230, 2235, 2244, 2254, 2260, 2264, 2269, 2275, 2280, 2286], "yaxis": "y"}, {"alignmentgroup": "True", "customdata": [["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"], ["60s"]], "hovertemplate": "age=%{customdata[0]}<br>date=%{x}<br>confirmed=%{y}<extra></extra>", "legendgroup": "60s", "marker": {"color": "#FF6692", "line": {"width": 0.5}}, "name": "60s", "offsetgroup": "60s", "orientation": "v", "showlegend": true, "textposition": "auto", "type": "bar", "x": ["2020-03-02", "2020-03-03", "2020-03-04", "2020-03-05", "2020-03-06", "2020-03-07", "2020-03-08", "2020-03-09", "2020-03-10", "2020-03-11", "2020-03-12", "2020-03-13", "2020-03-14", "2020-03-15", "2020-03-16", "2020-03-17", "2020-03-18", "2020-03-19", "2020-03-20", "2020-03-21", "2020-03-22", "2020-03-23", "2020-03-24", "2020-03-25", "2020-03-26", "2020-03-27", "2020-03-28", "2020-03-29", "2020-03-30", "2020-03-31", "2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04", "2020-04-05", "2020-04-06", "2020-04-07", "2020-04-08", "2020-04-09", "2020-04-10", "2020-04-11", "2020-04-12", "2020-04-13", "2020-04-14", "2020-04-15", "2020-04-16", "2020-04-17", "2020-04-18", "2020-04-19", "2020-04-20", "2020-04-21", "2020-04-22", "2020-04-23", "2020-04-24", "2020-04-25", "2020-04-26", "2020-04-27", "2020-04-28", "2020-04-29", "2020-04-30", "2020-05-01", "2020-05-02", "2020-05-03", "2020-05-04", "2020-05-05", "2020-05-06", "2020-05-07", "2020-05-08", "2020-05-09", "2020-05-10", "2020-05-11", "2020-05-12", "2020-05-13", "2020-05-14", "2020-05-15", "2020-05-16", "2020-05-17", "2020-05-18", "2020-05-19", "2020-05-20", "2020-05-21", "2020-05-22", "2020-05-23", "2020-05-24", "2020-05-25", "2020-05-26", "2020-05-27", "2020-05-28", "2020-05-29", "2020-05-30", "2020-05-31", "2020-06-01", "2020-06-02", "2020-06-03", "2020-06-04", "2020-06-05", "2020-06-06", "2020-06-07", "2020-06-08", "2020-06-09", "2020-06-10", "2020-06-11", "2020-06-12", "2020-06-13", "2020-06-14", "2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18", "2020-06-19", "2020-06-20", "2020-06-21", "2020-06-22", "2020-06-23", "2020-06-24", "2020-06-25", "2020-06-26", "2020-06-27", "2020-06-28", "2020-06-29", "2020-06-30"], "xaxis": "x", "y": [530, 597, 646, 699, 763, 830, 878, 916, 929, 960, 972, 985, 999, 1012, 1024, 1033, 1059, 1080, 1099, 1118, 1132, 1139, 1146, 1154, 1162, 1170, 1201, 1210, 1218, 1235, 1245, 1258, 1266, 1282, 1289, 1294, 1304, 1312, 1314, 1320, 1327, 1330, 1335, 1338, 1339, 1341, 1342, 1343, 1343, 1343, 1343, 1344, 1344, 1345, 1347, 1347, 1347, 1348, 1348, 1348, 1348, 1349, 1351, 1353, 1353, 1354, 1354, 1355, 1355, 1357, 1358, 1358, 1359, 1359, 1361, 1364, 1364, 1365, 1368, 1369, 1369, 1369, 1372, 1375, 1377, 1378, 1386, 1392, 1400, 1404, 1405, 1410, 1421, 1436, 1445, 1455, 1464, 1476, 1487, 1496, 1512, 1517, 1530, 1538, 1546, 1550, 1558, 1561, 1574, 1585, 1598, 1608, 1610, 1617, 1622, 1627, 1633, 1640, 1653, 1665, 1668], "yaxis": "y"}, {"alignmentgroup": "True", "customdata": [["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"], ["70s"]], "hovertemplate": "age=%{customdata[0]}<br>date=%{x}<br>confirmed=%{y}<extra></extra>", "legendgroup": "70s", "marker": {"color": "#B6E880", "line": {"width": 0.5}}, "name": "70s", "offsetgroup": "70s", "orientation": "v", "showlegend": true, "textposition": "auto", "type": "bar", "x": ["2020-03-02", "2020-03-03", "2020-03-04", "2020-03-05", "2020-03-06", "2020-03-07", "2020-03-08", "2020-03-09", "2020-03-10", "2020-03-11", "2020-03-12", "2020-03-13", "2020-03-14", "2020-03-15", "2020-03-16", "2020-03-17", "2020-03-18", "2020-03-19", "2020-03-20", "2020-03-21", "2020-03-22", "2020-03-23", "2020-03-24", "2020-03-25", "2020-03-26", "2020-03-27", "2020-03-28", "2020-03-29", "2020-03-30", "2020-03-31", "2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04", "2020-04-05", "2020-04-06", "2020-04-07", "2020-04-08", "2020-04-09", "2020-04-10", "2020-04-11", "2020-04-12", "2020-04-13", "2020-04-14", "2020-04-15", "2020-04-16", "2020-04-17", "2020-04-18", "2020-04-19", "2020-04-20", "2020-04-21", "2020-04-22", "2020-04-23", "2020-04-24", "2020-04-25", "2020-04-26", "2020-04-27", "2020-04-28", "2020-04-29", "2020-04-30", "2020-05-01", "2020-05-02", "2020-05-03", "2020-05-04", "2020-05-05", "2020-05-06", "2020-05-07", "2020-05-08", "2020-05-09", "2020-05-10", "2020-05-11", "2020-05-12", "2020-05-13", "2020-05-14", "2020-05-15", "2020-05-16", "2020-05-17", "2020-05-18", "2020-05-19", "2020-05-20", "2020-05-21", "2020-05-22", "2020-05-23", "2020-05-24", "2020-05-25", "2020-05-26", "2020-05-27", "2020-05-28", "2020-05-29", "2020-05-30", "2020-05-31", "2020-06-01", "2020-06-02", "2020-06-03", "2020-06-04", "2020-06-05", "2020-06-06", "2020-06-07", "2020-06-08", "2020-06-09", "2020-06-10", "2020-06-11", "2020-06-12", "2020-06-13", "2020-06-14", "2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18", "2020-06-19", "2020-06-20", "2020-06-21", "2020-06-22", "2020-06-23", "2020-06-24", "2020-06-25", "2020-06-26", "2020-06-27", "2020-06-28", "2020-06-29", "2020-06-30"], "xaxis": "x", "y": [192, 224, 260, 288, 340, 384, 409, 438, 454, 483, 497, 506, 515, 525, 531, 539, 542, 562, 568, 589, 595, 599, 608, 611, 616, 630, 632, 635, 640, 651, 658, 663, 668, 678, 681, 686, 689, 692, 692, 694, 694, 697, 698, 700, 702, 703, 704, 705, 705, 705, 706, 707, 708, 708, 708, 708, 708, 709, 709, 709, 709, 709, 710, 710, 710, 710, 710, 710, 710, 711, 711, 711, 712, 712, 714, 714, 715, 715, 716, 717, 717, 718, 718, 719, 719, 721, 724, 724, 724, 725, 725, 727, 732, 738, 744, 751, 759, 767, 770, 775, 782, 787, 789, 796, 797, 799, 805, 810, 817, 820, 823, 829, 830, 834, 836, 838, 839, 843, 846, 847, 850], "yaxis": "y"}, {"alignmentgroup": "True", "customdata": [["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"], ["80s"]], "hovertemplate": "age=%{customdata[0]}<br>date=%{x}<br>confirmed=%{y}<extra></extra>", "legendgroup": "80s", "marker": {"color": "#FF97FF", "line": {"width": 0.5}}, "name": "80s", "offsetgroup": "80s", "orientation": "v", "showlegend": true, "textposition": "auto", "type": "bar", "x": ["2020-03-02", "2020-03-03", "2020-03-04", "2020-03-05", "2020-03-06", "2020-03-07", "2020-03-08", "2020-03-09", "2020-03-10", "2020-03-11", "2020-03-12", "2020-03-13", "2020-03-14", "2020-03-15", "2020-03-16", "2020-03-17", "2020-03-18", "2020-03-19", "2020-03-20", "2020-03-21", "2020-03-22", "2020-03-23", "2020-03-24", "2020-03-25", "2020-03-26", "2020-03-27", "2020-03-28", "2020-03-29", "2020-03-30", "2020-03-31", "2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04", "2020-04-05", "2020-04-06", "2020-04-07", "2020-04-08", "2020-04-09", "2020-04-10", "2020-04-11", "2020-04-12", "2020-04-13", "2020-04-14", "2020-04-15", "2020-04-16", "2020-04-17", "2020-04-18", "2020-04-19", "2020-04-20", "2020-04-21", "2020-04-22", "2020-04-23", "2020-04-24", "2020-04-25", "2020-04-26", "2020-04-27", "2020-04-28", "2020-04-29", "2020-04-30", "2020-05-01", "2020-05-02", "2020-05-03", "2020-05-04", "2020-05-05", "2020-05-06", "2020-05-07", "2020-05-08", "2020-05-09", "2020-05-10", "2020-05-11", "2020-05-12", "2020-05-13", "2020-05-14", "2020-05-15", "2020-05-16", "2020-05-17", "2020-05-18", "2020-05-19", "2020-05-20", "2020-05-21", "2020-05-22", "2020-05-23", "2020-05-24", "2020-05-25", "2020-05-26", "2020-05-27", "2020-05-28", "2020-05-29", "2020-05-30", "2020-05-31", "2020-06-01", "2020-06-02", "2020-06-03", "2020-06-04", "2020-06-05", "2020-06-06", "2020-06-07", "2020-06-08", "2020-06-09", "2020-06-10", "2020-06-11", "2020-06-12", "2020-06-13", "2020-06-14", "2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18", "2020-06-19", "2020-06-20", "2020-06-21", "2020-06-22", "2020-06-23", "2020-06-24", "2020-06-25", "2020-06-26", "2020-06-27", "2020-06-28", "2020-06-29", "2020-06-30"], "xaxis": "x", "y": [81, 93, 108, 124, 168, 191, 212, 216, 222, 236, 243, 253, 263, 263, 270, 274, 286, 321, 329, 381, 392, 396, 401, 406, 416, 428, 432, 434, 437, 442, 452, 453, 456, 461, 462, 465, 466, 470, 473, 474, 475, 475, 476, 477, 478, 480, 480, 481, 482, 482, 483, 483, 484, 485, 485, 485, 485, 485, 485, 485, 486, 486, 488, 488, 488, 488, 488, 488, 488, 488, 488, 489, 490, 490, 490, 490, 490, 490, 490, 490, 490, 490, 491, 491, 491, 491, 492, 492, 495, 498, 498, 498, 499, 499, 502, 504, 507, 509, 512, 517, 518, 519, 520, 530, 531, 532, 532, 533, 541, 542, 545, 548, 548, 549, 551, 551, 553, 555, 556, 556, 556], "yaxis": "y"}],
                        {"barmode": "relative", "height": 600, "legend": {"title": {"text": "age"}, "tracegroupgap": 0}, "margin": {"t": 60}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "\uc5f0\ub839\ubcc4 \ud655\uc9c4\uc790 \uc218 \ucd94\uc774", "x": 0.5, "xanchor": "center", "y": 0.93, "yanchor": "bottom"}, "width": 1000, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "date"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "confirmed"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('36db25e0-860e-48f1-b6f4-08681b132f96');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>



```python
timeage_ = timeage.pivot_table(index = ['date'],columns=['age'], aggfunc=sum)

timeage_['deceased']
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
      <th>age</th>
      <th>0s</th>
      <th>10s</th>
      <th>20s</th>
      <th>30s</th>
      <th>40s</th>
      <th>50s</th>
      <th>60s</th>
      <th>70s</th>
      <th>80s</th>
    </tr>
    <tr>
      <th>date</th>
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
      <td>2020-03-02</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>6</td>
      <td>6</td>
      <td>3</td>
    </tr>
    <tr>
      <td>2020-03-03</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>7</td>
      <td>9</td>
      <td>5</td>
    </tr>
    <tr>
      <td>2020-03-04</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>7</td>
      <td>12</td>
      <td>6</td>
    </tr>
    <tr>
      <td>2020-03-05</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>8</td>
      <td>13</td>
      <td>7</td>
    </tr>
    <tr>
      <td>2020-03-06</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>11</td>
      <td>14</td>
      <td>10</td>
    </tr>
    <tr>
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
      <td>2020-06-26</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>15</td>
      <td>41</td>
      <td>82</td>
      <td>139</td>
    </tr>
    <tr>
      <td>2020-06-27</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>15</td>
      <td>41</td>
      <td>82</td>
      <td>139</td>
    </tr>
    <tr>
      <td>2020-06-28</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>15</td>
      <td>41</td>
      <td>82</td>
      <td>139</td>
    </tr>
    <tr>
      <td>2020-06-29</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>15</td>
      <td>41</td>
      <td>82</td>
      <td>139</td>
    </tr>
    <tr>
      <td>2020-06-30</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>15</td>
      <td>41</td>
      <td>82</td>
      <td>139</td>
    </tr>
  </tbody>
</table>
<p>121 rows × 9 columns</p>
</div>




```python
fig = go.Figure()
for col in timeage_['deceased'].columns:
    fig.add_trace(go.Scatter(x=timeage_.index,y= timeage_['deceased'][col], mode='lines+markers', name=col))
    
fig.update_layout(height=600, width=1000, title_text="사망자 추이 (Deceased by Age)", 
                  title = {'x' : 0.5, 'xanchor' : 'center', "y" : 0.88, "yanchor" : "bottom" })
fig

#연령과 사망률이 매우 밀접한 관계임을 다시 한번 확인할 수 있다.
```


<div>


            <div id="35d79fbb-9c95-4b9f-9de9-c23f4cedb662" class="plotly-graph-div" style="height:600px; width:1000px;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("35d79fbb-9c95-4b9f-9de9-c23f4cedb662")) {
                    Plotly.newPlot(
                        '35d79fbb-9c95-4b9f-9de9-c23f4cedb662',
                        [{"mode": "lines+markers", "name": "0s", "type": "scatter", "x": ["2020-03-02", "2020-03-03", "2020-03-04", "2020-03-05", "2020-03-06", "2020-03-07", "2020-03-08", "2020-03-09", "2020-03-10", "2020-03-11", "2020-03-12", "2020-03-13", "2020-03-14", "2020-03-15", "2020-03-16", "2020-03-17", "2020-03-18", "2020-03-19", "2020-03-20", "2020-03-21", "2020-03-22", "2020-03-23", "2020-03-24", "2020-03-25", "2020-03-26", "2020-03-27", "2020-03-28", "2020-03-29", "2020-03-30", "2020-03-31", "2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04", "2020-04-05", "2020-04-06", "2020-04-07", "2020-04-08", "2020-04-09", "2020-04-10", "2020-04-11", "2020-04-12", "2020-04-13", "2020-04-14", "2020-04-15", "2020-04-16", "2020-04-17", "2020-04-18", "2020-04-19", "2020-04-20", "2020-04-21", "2020-04-22", "2020-04-23", "2020-04-24", "2020-04-25", "2020-04-26", "2020-04-27", "2020-04-28", "2020-04-29", "2020-04-30", "2020-05-01", "2020-05-02", "2020-05-03", "2020-05-04", "2020-05-05", "2020-05-06", "2020-05-07", "2020-05-08", "2020-05-09", "2020-05-10", "2020-05-11", "2020-05-12", "2020-05-13", "2020-05-14", "2020-05-15", "2020-05-16", "2020-05-17", "2020-05-18", "2020-05-19", "2020-05-20", "2020-05-21", "2020-05-22", "2020-05-23", "2020-05-24", "2020-05-25", "2020-05-26", "2020-05-27", "2020-05-28", "2020-05-29", "2020-05-30", "2020-05-31", "2020-06-01", "2020-06-02", "2020-06-03", "2020-06-04", "2020-06-05", "2020-06-06", "2020-06-07", "2020-06-08", "2020-06-09", "2020-06-10", "2020-06-11", "2020-06-12", "2020-06-13", "2020-06-14", "2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18", "2020-06-19", "2020-06-20", "2020-06-21", "2020-06-22", "2020-06-23", "2020-06-24", "2020-06-25", "2020-06-26", "2020-06-27", "2020-06-28", "2020-06-29", "2020-06-30"], "y": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}, {"mode": "lines+markers", "name": "10s", "type": "scatter", "x": ["2020-03-02", "2020-03-03", "2020-03-04", "2020-03-05", "2020-03-06", "2020-03-07", "2020-03-08", "2020-03-09", "2020-03-10", "2020-03-11", "2020-03-12", "2020-03-13", "2020-03-14", "2020-03-15", "2020-03-16", "2020-03-17", "2020-03-18", "2020-03-19", "2020-03-20", "2020-03-21", "2020-03-22", "2020-03-23", "2020-03-24", "2020-03-25", "2020-03-26", "2020-03-27", "2020-03-28", "2020-03-29", "2020-03-30", "2020-03-31", "2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04", "2020-04-05", "2020-04-06", "2020-04-07", "2020-04-08", "2020-04-09", "2020-04-10", "2020-04-11", "2020-04-12", "2020-04-13", "2020-04-14", "2020-04-15", "2020-04-16", "2020-04-17", "2020-04-18", "2020-04-19", "2020-04-20", "2020-04-21", "2020-04-22", "2020-04-23", "2020-04-24", "2020-04-25", "2020-04-26", "2020-04-27", "2020-04-28", "2020-04-29", "2020-04-30", "2020-05-01", "2020-05-02", "2020-05-03", "2020-05-04", "2020-05-05", "2020-05-06", "2020-05-07", "2020-05-08", "2020-05-09", "2020-05-10", "2020-05-11", "2020-05-12", "2020-05-13", "2020-05-14", "2020-05-15", "2020-05-16", "2020-05-17", "2020-05-18", "2020-05-19", "2020-05-20", "2020-05-21", "2020-05-22", "2020-05-23", "2020-05-24", "2020-05-25", "2020-05-26", "2020-05-27", "2020-05-28", "2020-05-29", "2020-05-30", "2020-05-31", "2020-06-01", "2020-06-02", "2020-06-03", "2020-06-04", "2020-06-05", "2020-06-06", "2020-06-07", "2020-06-08", "2020-06-09", "2020-06-10", "2020-06-11", "2020-06-12", "2020-06-13", "2020-06-14", "2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18", "2020-06-19", "2020-06-20", "2020-06-21", "2020-06-22", "2020-06-23", "2020-06-24", "2020-06-25", "2020-06-26", "2020-06-27", "2020-06-28", "2020-06-29", "2020-06-30"], "y": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}, {"mode": "lines+markers", "name": "20s", "type": "scatter", "x": ["2020-03-02", "2020-03-03", "2020-03-04", "2020-03-05", "2020-03-06", "2020-03-07", "2020-03-08", "2020-03-09", "2020-03-10", "2020-03-11", "2020-03-12", "2020-03-13", "2020-03-14", "2020-03-15", "2020-03-16", "2020-03-17", "2020-03-18", "2020-03-19", "2020-03-20", "2020-03-21", "2020-03-22", "2020-03-23", "2020-03-24", "2020-03-25", "2020-03-26", "2020-03-27", "2020-03-28", "2020-03-29", "2020-03-30", "2020-03-31", "2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04", "2020-04-05", "2020-04-06", "2020-04-07", "2020-04-08", "2020-04-09", "2020-04-10", "2020-04-11", "2020-04-12", "2020-04-13", "2020-04-14", "2020-04-15", "2020-04-16", "2020-04-17", "2020-04-18", "2020-04-19", "2020-04-20", "2020-04-21", "2020-04-22", "2020-04-23", "2020-04-24", "2020-04-25", "2020-04-26", "2020-04-27", "2020-04-28", "2020-04-29", "2020-04-30", "2020-05-01", "2020-05-02", "2020-05-03", "2020-05-04", "2020-05-05", "2020-05-06", "2020-05-07", "2020-05-08", "2020-05-09", "2020-05-10", "2020-05-11", "2020-05-12", "2020-05-13", "2020-05-14", "2020-05-15", "2020-05-16", "2020-05-17", "2020-05-18", "2020-05-19", "2020-05-20", "2020-05-21", "2020-05-22", "2020-05-23", "2020-05-24", "2020-05-25", "2020-05-26", "2020-05-27", "2020-05-28", "2020-05-29", "2020-05-30", "2020-05-31", "2020-06-01", "2020-06-02", "2020-06-03", "2020-06-04", "2020-06-05", "2020-06-06", "2020-06-07", "2020-06-08", "2020-06-09", "2020-06-10", "2020-06-11", "2020-06-12", "2020-06-13", "2020-06-14", "2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18", "2020-06-19", "2020-06-20", "2020-06-21", "2020-06-22", "2020-06-23", "2020-06-24", "2020-06-25", "2020-06-26", "2020-06-27", "2020-06-28", "2020-06-29", "2020-06-30"], "y": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}, {"mode": "lines+markers", "name": "30s", "type": "scatter", "x": ["2020-03-02", "2020-03-03", "2020-03-04", "2020-03-05", "2020-03-06", "2020-03-07", "2020-03-08", "2020-03-09", "2020-03-10", "2020-03-11", "2020-03-12", "2020-03-13", "2020-03-14", "2020-03-15", "2020-03-16", "2020-03-17", "2020-03-18", "2020-03-19", "2020-03-20", "2020-03-21", "2020-03-22", "2020-03-23", "2020-03-24", "2020-03-25", "2020-03-26", "2020-03-27", "2020-03-28", "2020-03-29", "2020-03-30", "2020-03-31", "2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04", "2020-04-05", "2020-04-06", "2020-04-07", "2020-04-08", "2020-04-09", "2020-04-10", "2020-04-11", "2020-04-12", "2020-04-13", "2020-04-14", "2020-04-15", "2020-04-16", "2020-04-17", "2020-04-18", "2020-04-19", "2020-04-20", "2020-04-21", "2020-04-22", "2020-04-23", "2020-04-24", "2020-04-25", "2020-04-26", "2020-04-27", "2020-04-28", "2020-04-29", "2020-04-30", "2020-05-01", "2020-05-02", "2020-05-03", "2020-05-04", "2020-05-05", "2020-05-06", "2020-05-07", "2020-05-08", "2020-05-09", "2020-05-10", "2020-05-11", "2020-05-12", "2020-05-13", "2020-05-14", "2020-05-15", "2020-05-16", "2020-05-17", "2020-05-18", "2020-05-19", "2020-05-20", "2020-05-21", "2020-05-22", "2020-05-23", "2020-05-24", "2020-05-25", "2020-05-26", "2020-05-27", "2020-05-28", "2020-05-29", "2020-05-30", "2020-05-31", "2020-06-01", "2020-06-02", "2020-06-03", "2020-06-04", "2020-06-05", "2020-06-06", "2020-06-07", "2020-06-08", "2020-06-09", "2020-06-10", "2020-06-11", "2020-06-12", "2020-06-13", "2020-06-14", "2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18", "2020-06-19", "2020-06-20", "2020-06-21", "2020-06-22", "2020-06-23", "2020-06-24", "2020-06-25", "2020-06-26", "2020-06-27", "2020-06-28", "2020-06-29", "2020-06-30"], "y": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]}, {"mode": "lines+markers", "name": "40s", "type": "scatter", "x": ["2020-03-02", "2020-03-03", "2020-03-04", "2020-03-05", "2020-03-06", "2020-03-07", "2020-03-08", "2020-03-09", "2020-03-10", "2020-03-11", "2020-03-12", "2020-03-13", "2020-03-14", "2020-03-15", "2020-03-16", "2020-03-17", "2020-03-18", "2020-03-19", "2020-03-20", "2020-03-21", "2020-03-22", "2020-03-23", "2020-03-24", "2020-03-25", "2020-03-26", "2020-03-27", "2020-03-28", "2020-03-29", "2020-03-30", "2020-03-31", "2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04", "2020-04-05", "2020-04-06", "2020-04-07", "2020-04-08", "2020-04-09", "2020-04-10", "2020-04-11", "2020-04-12", "2020-04-13", "2020-04-14", "2020-04-15", "2020-04-16", "2020-04-17", "2020-04-18", "2020-04-19", "2020-04-20", "2020-04-21", "2020-04-22", "2020-04-23", "2020-04-24", "2020-04-25", "2020-04-26", "2020-04-27", "2020-04-28", "2020-04-29", "2020-04-30", "2020-05-01", "2020-05-02", "2020-05-03", "2020-05-04", "2020-05-05", "2020-05-06", "2020-05-07", "2020-05-08", "2020-05-09", "2020-05-10", "2020-05-11", "2020-05-12", "2020-05-13", "2020-05-14", "2020-05-15", "2020-05-16", "2020-05-17", "2020-05-18", "2020-05-19", "2020-05-20", "2020-05-21", "2020-05-22", "2020-05-23", "2020-05-24", "2020-05-25", "2020-05-26", "2020-05-27", "2020-05-28", "2020-05-29", "2020-05-30", "2020-05-31", "2020-06-01", "2020-06-02", "2020-06-03", "2020-06-04", "2020-06-05", "2020-06-06", "2020-06-07", "2020-06-08", "2020-06-09", "2020-06-10", "2020-06-11", "2020-06-12", "2020-06-13", "2020-06-14", "2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18", "2020-06-19", "2020-06-20", "2020-06-21", "2020-06-22", "2020-06-23", "2020-06-24", "2020-06-25", "2020-06-26", "2020-06-27", "2020-06-28", "2020-06-29", "2020-06-30"], "y": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]}, {"mode": "lines+markers", "name": "50s", "type": "scatter", "x": ["2020-03-02", "2020-03-03", "2020-03-04", "2020-03-05", "2020-03-06", "2020-03-07", "2020-03-08", "2020-03-09", "2020-03-10", "2020-03-11", "2020-03-12", "2020-03-13", "2020-03-14", "2020-03-15", "2020-03-16", "2020-03-17", "2020-03-18", "2020-03-19", "2020-03-20", "2020-03-21", "2020-03-22", "2020-03-23", "2020-03-24", "2020-03-25", "2020-03-26", "2020-03-27", "2020-03-28", "2020-03-29", "2020-03-30", "2020-03-31", "2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04", "2020-04-05", "2020-04-06", "2020-04-07", "2020-04-08", "2020-04-09", "2020-04-10", "2020-04-11", "2020-04-12", "2020-04-13", "2020-04-14", "2020-04-15", "2020-04-16", "2020-04-17", "2020-04-18", "2020-04-19", "2020-04-20", "2020-04-21", "2020-04-22", "2020-04-23", "2020-04-24", "2020-04-25", "2020-04-26", "2020-04-27", "2020-04-28", "2020-04-29", "2020-04-30", "2020-05-01", "2020-05-02", "2020-05-03", "2020-05-04", "2020-05-05", "2020-05-06", "2020-05-07", "2020-05-08", "2020-05-09", "2020-05-10", "2020-05-11", "2020-05-12", "2020-05-13", "2020-05-14", "2020-05-15", "2020-05-16", "2020-05-17", "2020-05-18", "2020-05-19", "2020-05-20", "2020-05-21", "2020-05-22", "2020-05-23", "2020-05-24", "2020-05-25", "2020-05-26", "2020-05-27", "2020-05-28", "2020-05-29", "2020-05-30", "2020-05-31", "2020-06-01", "2020-06-02", "2020-06-03", "2020-06-04", "2020-06-05", "2020-06-06", "2020-06-07", "2020-06-08", "2020-06-09", "2020-06-10", "2020-06-11", "2020-06-12", "2020-06-13", "2020-06-14", "2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18", "2020-06-19", "2020-06-20", "2020-06-21", "2020-06-22", "2020-06-23", "2020-06-24", "2020-06-25", "2020-06-26", "2020-06-27", "2020-06-28", "2020-06-29", "2020-06-30"], "y": [5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 10, 10, 10, 10, 10, 10, 10, 10, 11, 12, 12, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]}, {"mode": "lines+markers", "name": "60s", "type": "scatter", "x": ["2020-03-02", "2020-03-03", "2020-03-04", "2020-03-05", "2020-03-06", "2020-03-07", "2020-03-08", "2020-03-09", "2020-03-10", "2020-03-11", "2020-03-12", "2020-03-13", "2020-03-14", "2020-03-15", "2020-03-16", "2020-03-17", "2020-03-18", "2020-03-19", "2020-03-20", "2020-03-21", "2020-03-22", "2020-03-23", "2020-03-24", "2020-03-25", "2020-03-26", "2020-03-27", "2020-03-28", "2020-03-29", "2020-03-30", "2020-03-31", "2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04", "2020-04-05", "2020-04-06", "2020-04-07", "2020-04-08", "2020-04-09", "2020-04-10", "2020-04-11", "2020-04-12", "2020-04-13", "2020-04-14", "2020-04-15", "2020-04-16", "2020-04-17", "2020-04-18", "2020-04-19", "2020-04-20", "2020-04-21", "2020-04-22", "2020-04-23", "2020-04-24", "2020-04-25", "2020-04-26", "2020-04-27", "2020-04-28", "2020-04-29", "2020-04-30", "2020-05-01", "2020-05-02", "2020-05-03", "2020-05-04", "2020-05-05", "2020-05-06", "2020-05-07", "2020-05-08", "2020-05-09", "2020-05-10", "2020-05-11", "2020-05-12", "2020-05-13", "2020-05-14", "2020-05-15", "2020-05-16", "2020-05-17", "2020-05-18", "2020-05-19", "2020-05-20", "2020-05-21", "2020-05-22", "2020-05-23", "2020-05-24", "2020-05-25", "2020-05-26", "2020-05-27", "2020-05-28", "2020-05-29", "2020-05-30", "2020-05-31", "2020-06-01", "2020-06-02", "2020-06-03", "2020-06-04", "2020-06-05", "2020-06-06", "2020-06-07", "2020-06-08", "2020-06-09", "2020-06-10", "2020-06-11", "2020-06-12", "2020-06-13", "2020-06-14", "2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18", "2020-06-19", "2020-06-20", "2020-06-21", "2020-06-22", "2020-06-23", "2020-06-24", "2020-06-25", "2020-06-26", "2020-06-27", "2020-06-28", "2020-06-29", "2020-06-30"], "y": [6, 7, 7, 8, 11, 11, 11, 12, 13, 14, 14, 14, 14, 14, 14, 16, 16, 17, 17, 17, 17, 18, 20, 20, 20, 21, 21, 21, 21, 22, 23, 23, 24, 25, 25, 26, 26, 27, 27, 28, 29, 31, 32, 33, 33, 33, 33, 33, 34, 34, 34, 34, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 36, 36, 36, 37, 37, 37, 37, 37, 37, 37, 38, 38, 38, 38, 38, 38, 38, 38, 38, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 41, 41, 41, 41, 41, 41, 41, 41]}, {"mode": "lines+markers", "name": "70s", "type": "scatter", "x": ["2020-03-02", "2020-03-03", "2020-03-04", "2020-03-05", "2020-03-06", "2020-03-07", "2020-03-08", "2020-03-09", "2020-03-10", "2020-03-11", "2020-03-12", "2020-03-13", "2020-03-14", "2020-03-15", "2020-03-16", "2020-03-17", "2020-03-18", "2020-03-19", "2020-03-20", "2020-03-21", "2020-03-22", "2020-03-23", "2020-03-24", "2020-03-25", "2020-03-26", "2020-03-27", "2020-03-28", "2020-03-29", "2020-03-30", "2020-03-31", "2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04", "2020-04-05", "2020-04-06", "2020-04-07", "2020-04-08", "2020-04-09", "2020-04-10", "2020-04-11", "2020-04-12", "2020-04-13", "2020-04-14", "2020-04-15", "2020-04-16", "2020-04-17", "2020-04-18", "2020-04-19", "2020-04-20", "2020-04-21", "2020-04-22", "2020-04-23", "2020-04-24", "2020-04-25", "2020-04-26", "2020-04-27", "2020-04-28", "2020-04-29", "2020-04-30", "2020-05-01", "2020-05-02", "2020-05-03", "2020-05-04", "2020-05-05", "2020-05-06", "2020-05-07", "2020-05-08", "2020-05-09", "2020-05-10", "2020-05-11", "2020-05-12", "2020-05-13", "2020-05-14", "2020-05-15", "2020-05-16", "2020-05-17", "2020-05-18", "2020-05-19", "2020-05-20", "2020-05-21", "2020-05-22", "2020-05-23", "2020-05-24", "2020-05-25", "2020-05-26", "2020-05-27", "2020-05-28", "2020-05-29", "2020-05-30", "2020-05-31", "2020-06-01", "2020-06-02", "2020-06-03", "2020-06-04", "2020-06-05", "2020-06-06", "2020-06-07", "2020-06-08", "2020-06-09", "2020-06-10", "2020-06-11", "2020-06-12", "2020-06-13", "2020-06-14", "2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18", "2020-06-19", "2020-06-20", "2020-06-21", "2020-06-22", "2020-06-23", "2020-06-24", "2020-06-25", "2020-06-26", "2020-06-27", "2020-06-28", "2020-06-29", "2020-06-30"], "y": [6, 9, 12, 13, 14, 14, 18, 18, 19, 21, 24, 24, 27, 28, 28, 29, 29, 34, 35, 37, 37, 38, 38, 39, 41, 41, 41, 43, 45, 46, 46, 47, 49, 50, 51, 52, 57, 60, 60, 62, 63, 63, 64, 65, 68, 68, 68, 68, 68, 69, 70, 71, 71, 71, 71, 72, 72, 73, 73, 74, 75, 75, 75, 76, 76, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 78, 78, 78, 78, 78, 78, 78, 79, 79, 79, 79, 79, 79, 80, 80, 80, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 82, 82, 82, 82, 82, 82]}, {"mode": "lines+markers", "name": "80s", "type": "scatter", "x": ["2020-03-02", "2020-03-03", "2020-03-04", "2020-03-05", "2020-03-06", "2020-03-07", "2020-03-08", "2020-03-09", "2020-03-10", "2020-03-11", "2020-03-12", "2020-03-13", "2020-03-14", "2020-03-15", "2020-03-16", "2020-03-17", "2020-03-18", "2020-03-19", "2020-03-20", "2020-03-21", "2020-03-22", "2020-03-23", "2020-03-24", "2020-03-25", "2020-03-26", "2020-03-27", "2020-03-28", "2020-03-29", "2020-03-30", "2020-03-31", "2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04", "2020-04-05", "2020-04-06", "2020-04-07", "2020-04-08", "2020-04-09", "2020-04-10", "2020-04-11", "2020-04-12", "2020-04-13", "2020-04-14", "2020-04-15", "2020-04-16", "2020-04-17", "2020-04-18", "2020-04-19", "2020-04-20", "2020-04-21", "2020-04-22", "2020-04-23", "2020-04-24", "2020-04-25", "2020-04-26", "2020-04-27", "2020-04-28", "2020-04-29", "2020-04-30", "2020-05-01", "2020-05-02", "2020-05-03", "2020-05-04", "2020-05-05", "2020-05-06", "2020-05-07", "2020-05-08", "2020-05-09", "2020-05-10", "2020-05-11", "2020-05-12", "2020-05-13", "2020-05-14", "2020-05-15", "2020-05-16", "2020-05-17", "2020-05-18", "2020-05-19", "2020-05-20", "2020-05-21", "2020-05-22", "2020-05-23", "2020-05-24", "2020-05-25", "2020-05-26", "2020-05-27", "2020-05-28", "2020-05-29", "2020-05-30", "2020-05-31", "2020-06-01", "2020-06-02", "2020-06-03", "2020-06-04", "2020-06-05", "2020-06-06", "2020-06-07", "2020-06-08", "2020-06-09", "2020-06-10", "2020-06-11", "2020-06-12", "2020-06-13", "2020-06-14", "2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18", "2020-06-19", "2020-06-20", "2020-06-21", "2020-06-22", "2020-06-23", "2020-06-24", "2020-06-25", "2020-06-26", "2020-06-27", "2020-06-28", "2020-06-29", "2020-06-30"], "y": [3, 5, 6, 7, 10, 12, 14, 14, 15, 17, 20, 21, 23, 25, 25, 28, 31, 31, 33, 39, 41, 46, 52, 55, 58, 65, 70, 76, 80, 82, 84, 85, 86, 87, 91, 92, 93, 96, 100, 101, 101, 102, 103, 106, 106, 110, 111, 112, 112, 113, 113, 113, 114, 114, 114, 115, 116, 116, 118, 118, 118, 120, 120, 120, 122, 122, 122, 122, 122, 122, 122, 124, 125, 125, 125, 127, 127, 127, 127, 127, 128, 128, 129, 129, 129, 131, 131, 131, 131, 131, 131, 132, 133, 133, 133, 133, 133, 133, 133, 134, 136, 136, 136, 136, 136, 136, 137, 138, 139, 139, 139, 139, 139, 139, 139, 139, 139, 139, 139, 139, 139]}],
                        {"height": 600, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "\uc0ac\ub9dd\uc790 \ucd94\uc774 (Deceased by Age)", "x": 0.5, "xanchor": "center", "y": 0.88, "yanchor": "bottom"}, "width": 1000},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('35d79fbb-9c95-4b9f-9de9-c23f4cedb662');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>



```python
print(timegender.shape)
timegender.head()
```

    (242, 5)
    




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
      <th>date</th>
      <th>time</th>
      <th>sex</th>
      <th>confirmed</th>
      <th>deceased</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2020-03-02</td>
      <td>0</td>
      <td>male</td>
      <td>1591</td>
      <td>13</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2020-03-02</td>
      <td>0</td>
      <td>female</td>
      <td>2621</td>
      <td>9</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2020-03-03</td>
      <td>0</td>
      <td>male</td>
      <td>1810</td>
      <td>16</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2020-03-03</td>
      <td>0</td>
      <td>female</td>
      <td>3002</td>
      <td>12</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2020-03-04</td>
      <td>0</td>
      <td>male</td>
      <td>1996</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
</div>




```python
timegender.columns
```




    Index(['date', 'time', 'sex', 'confirmed', 'deceased'], dtype='object')




```python
gender_ = timegender.pivot_table(index = ['date'], columns = ['sex'], aggfunc = sum)

gender_conf = gender_['confirmed']
gender_dec = gender_['deceased']

gender_dec
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
      <th>sex</th>
      <th>female</th>
      <th>male</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2020-03-02</td>
      <td>9</td>
      <td>13</td>
    </tr>
    <tr>
      <td>2020-03-03</td>
      <td>12</td>
      <td>16</td>
    </tr>
    <tr>
      <td>2020-03-04</td>
      <td>12</td>
      <td>20</td>
    </tr>
    <tr>
      <td>2020-03-05</td>
      <td>14</td>
      <td>21</td>
    </tr>
    <tr>
      <td>2020-03-06</td>
      <td>17</td>
      <td>25</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>2020-06-26</td>
      <td>131</td>
      <td>151</td>
    </tr>
    <tr>
      <td>2020-06-27</td>
      <td>131</td>
      <td>151</td>
    </tr>
    <tr>
      <td>2020-06-28</td>
      <td>131</td>
      <td>151</td>
    </tr>
    <tr>
      <td>2020-06-29</td>
      <td>131</td>
      <td>151</td>
    </tr>
    <tr>
      <td>2020-06-30</td>
      <td>131</td>
      <td>151</td>
    </tr>
  </tbody>
</table>
<p>121 rows × 2 columns</p>
</div>




```python
gender_conf
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
      <th>sex</th>
      <th>female</th>
      <th>male</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2020-03-02</td>
      <td>2621</td>
      <td>1591</td>
    </tr>
    <tr>
      <td>2020-03-03</td>
      <td>3002</td>
      <td>1810</td>
    </tr>
    <tr>
      <td>2020-03-04</td>
      <td>3332</td>
      <td>1996</td>
    </tr>
    <tr>
      <td>2020-03-05</td>
      <td>3617</td>
      <td>2149</td>
    </tr>
    <tr>
      <td>2020-03-06</td>
      <td>3939</td>
      <td>2345</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>2020-06-26</td>
      <td>7218</td>
      <td>5384</td>
    </tr>
    <tr>
      <td>2020-06-27</td>
      <td>7241</td>
      <td>5412</td>
    </tr>
    <tr>
      <td>2020-06-28</td>
      <td>7265</td>
      <td>5450</td>
    </tr>
    <tr>
      <td>2020-06-29</td>
      <td>7287</td>
      <td>5470</td>
    </tr>
    <tr>
      <td>2020-06-30</td>
      <td>7305</td>
      <td>5495</td>
    </tr>
  </tbody>
</table>
<p>121 rows × 2 columns</p>
</div>




```python
fig = make_subplots(rows = 1, cols = 2, subplot_titles = ('성별 확진자 추이 (Confirmed by Age)', '성별 사망자 추이 (Deceased by Age)'))

fig.add_trace(go.Scatter(x = gender_conf.index, y=gender_conf['male'], line=dict(color='#334eff'), mode = 'lines', name = 'male'), row = 1, col = 1)
fig.add_trace(go.Scatter(x = gender_conf.index, y=gender_conf['female'],line=dict(color= '#eb2a2d'), mode = 'lines', name = 'female'), row = 1, col = 1)


fig.add_trace(go.Scatter(x = gender_dec.index, y=gender_dec['male'], line=dict(color='#334eff'), showlegend=False, mode = 'lines', name = 'male'), row = 1, col = 2)
fig.add_trace(go.Scatter(x = gender_dec.index, y=gender_dec['female'], line=dict(color= '#eb2a2d'), showlegend=False, mode = 'lines', name = 'female'), row = 1, col = 2)

fig.update_xaxes(title_text="Month", row=1, col=1)
fig.update_xaxes(title_text="Month", row=1, col=2)

fig.update_yaxes(title_text="Number", row=1, col=1)
fig.update_yaxes(title_text="Number", row=1, col=2)

fig.update_layout(title_text="성별 확진자/사망자 추이 (Confirmed/Deceased by Gender)",
                 title = {'x' : 0.5, 'xanchor' : 'center', "y" : 0.92, "yanchor" : "bottom" })

fig
```


<div>


            <div id="7178e4db-4e81-453b-9353-bba56f18728c" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("7178e4db-4e81-453b-9353-bba56f18728c")) {
                    Plotly.newPlot(
                        '7178e4db-4e81-453b-9353-bba56f18728c',
                        [{"line": {"color": "#334eff"}, "mode": "lines", "name": "male", "type": "scatter", "x": ["2020-03-02", "2020-03-03", "2020-03-04", "2020-03-05", "2020-03-06", "2020-03-07", "2020-03-08", "2020-03-09", "2020-03-10", "2020-03-11", "2020-03-12", "2020-03-13", "2020-03-14", "2020-03-15", "2020-03-16", "2020-03-17", "2020-03-18", "2020-03-19", "2020-03-20", "2020-03-21", "2020-03-22", "2020-03-23", "2020-03-24", "2020-03-25", "2020-03-26", "2020-03-27", "2020-03-28", "2020-03-29", "2020-03-30", "2020-03-31", "2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04", "2020-04-05", "2020-04-06", "2020-04-07", "2020-04-08", "2020-04-09", "2020-04-10", "2020-04-11", "2020-04-12", "2020-04-13", "2020-04-14", "2020-04-15", "2020-04-16", "2020-04-17", "2020-04-18", "2020-04-19", "2020-04-20", "2020-04-21", "2020-04-22", "2020-04-23", "2020-04-24", "2020-04-25", "2020-04-26", "2020-04-27", "2020-04-28", "2020-04-29", "2020-04-30", "2020-05-01", "2020-05-02", "2020-05-03", "2020-05-04", "2020-05-05", "2020-05-06", "2020-05-07", "2020-05-08", "2020-05-09", "2020-05-10", "2020-05-11", "2020-05-12", "2020-05-13", "2020-05-14", "2020-05-15", "2020-05-16", "2020-05-17", "2020-05-18", "2020-05-19", "2020-05-20", "2020-05-21", "2020-05-22", "2020-05-23", "2020-05-24", "2020-05-25", "2020-05-26", "2020-05-27", "2020-05-28", "2020-05-29", "2020-05-30", "2020-05-31", "2020-06-01", "2020-06-02", "2020-06-03", "2020-06-04", "2020-06-05", "2020-06-06", "2020-06-07", "2020-06-08", "2020-06-09", "2020-06-10", "2020-06-11", "2020-06-12", "2020-06-13", "2020-06-14", "2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18", "2020-06-19", "2020-06-20", "2020-06-21", "2020-06-22", "2020-06-23", "2020-06-24", "2020-06-25", "2020-06-26", "2020-06-27", "2020-06-28", "2020-06-29", "2020-06-30"], "xaxis": "x", "y": [1591, 1810, 1996, 2149, 2345, 2522, 2694, 2799, 2852, 2947, 2994, 3043, 3100, 3136, 3169, 3200, 3240, 3296, 3330, 3387, 3430, 3457, 3497, 3550, 3598, 3638, 3736, 3799, 3834, 3905, 3946, 3979, 4013, 4052, 4098, 4118, 4138, 4163, 4185, 4200, 4215, 4229, 4243, 4256, 4266, 4272, 4286, 4293, 4297, 4302, 4308, 4314, 4318, 4323, 4327, 4333, 4337, 4344, 4348, 4352, 4356, 4362, 4369, 4375, 4377, 4379, 4382, 4389, 4406, 4430, 4461, 4481, 4499, 4516, 4532, 4544, 4549, 4560, 4569, 4592, 4601, 4616, 4628, 4643, 4651, 4664, 4680, 4727, 4759, 4780, 4795, 4806, 4829, 4852, 4872, 4893, 4920, 4950, 4962, 4980, 5002, 5030, 5061, 5081, 5097, 5116, 5138, 5156, 5181, 5206, 5245, 5270, 5279, 5314, 5345, 5360, 5384, 5412, 5450, 5470, 5495], "yaxis": "y"}, {"line": {"color": "#eb2a2d"}, "mode": "lines", "name": "female", "type": "scatter", "x": ["2020-03-02", "2020-03-03", "2020-03-04", "2020-03-05", "2020-03-06", "2020-03-07", "2020-03-08", "2020-03-09", "2020-03-10", "2020-03-11", "2020-03-12", "2020-03-13", "2020-03-14", "2020-03-15", "2020-03-16", "2020-03-17", "2020-03-18", "2020-03-19", "2020-03-20", "2020-03-21", "2020-03-22", "2020-03-23", "2020-03-24", "2020-03-25", "2020-03-26", "2020-03-27", "2020-03-28", "2020-03-29", "2020-03-30", "2020-03-31", "2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04", "2020-04-05", "2020-04-06", "2020-04-07", "2020-04-08", "2020-04-09", "2020-04-10", "2020-04-11", "2020-04-12", "2020-04-13", "2020-04-14", "2020-04-15", "2020-04-16", "2020-04-17", "2020-04-18", "2020-04-19", "2020-04-20", "2020-04-21", "2020-04-22", "2020-04-23", "2020-04-24", "2020-04-25", "2020-04-26", "2020-04-27", "2020-04-28", "2020-04-29", "2020-04-30", "2020-05-01", "2020-05-02", "2020-05-03", "2020-05-04", "2020-05-05", "2020-05-06", "2020-05-07", "2020-05-08", "2020-05-09", "2020-05-10", "2020-05-11", "2020-05-12", "2020-05-13", "2020-05-14", "2020-05-15", "2020-05-16", "2020-05-17", "2020-05-18", "2020-05-19", "2020-05-20", "2020-05-21", "2020-05-22", "2020-05-23", "2020-05-24", "2020-05-25", "2020-05-26", "2020-05-27", "2020-05-28", "2020-05-29", "2020-05-30", "2020-05-31", "2020-06-01", "2020-06-02", "2020-06-03", "2020-06-04", "2020-06-05", "2020-06-06", "2020-06-07", "2020-06-08", "2020-06-09", "2020-06-10", "2020-06-11", "2020-06-12", "2020-06-13", "2020-06-14", "2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18", "2020-06-19", "2020-06-20", "2020-06-21", "2020-06-22", "2020-06-23", "2020-06-24", "2020-06-25", "2020-06-26", "2020-06-27", "2020-06-28", "2020-06-29", "2020-06-30"], "xaxis": "x", "y": [2621, 3002, 3332, 3617, 3939, 4245, 4440, 4583, 4661, 4808, 4875, 4936, 4986, 5026, 5067, 5120, 5173, 5269, 5322, 5412, 5467, 5504, 5540, 5587, 5643, 5694, 5742, 5784, 5827, 5881, 5941, 5997, 6049, 6104, 6139, 6166, 6193, 6221, 6238, 6250, 6265, 6283, 6294, 6308, 6325, 6341, 6349, 6360, 6364, 6372, 6375, 6380, 6384, 6385, 6391, 6395, 6401, 6408, 6413, 6413, 6418, 6418, 6424, 6426, 6427, 6427, 6428, 6433, 6434, 6444, 6448, 6455, 6463, 6475, 6486, 6493, 6501, 6505, 6509, 6518, 6521, 6526, 6537, 6547, 6555, 6561, 6585, 6617, 6643, 6661, 6673, 6697, 6712, 6738, 6757, 6775, 6799, 6826, 6852, 6872, 6900, 6917, 6942, 6970, 6988, 7005, 7017, 7042, 7076, 7100, 7128, 7151, 7159, 7170, 7190, 7203, 7218, 7241, 7265, 7287, 7305], "yaxis": "y"}, {"line": {"color": "#334eff"}, "mode": "lines", "name": "male", "showlegend": false, "type": "scatter", "x": ["2020-03-02", "2020-03-03", "2020-03-04", "2020-03-05", "2020-03-06", "2020-03-07", "2020-03-08", "2020-03-09", "2020-03-10", "2020-03-11", "2020-03-12", "2020-03-13", "2020-03-14", "2020-03-15", "2020-03-16", "2020-03-17", "2020-03-18", "2020-03-19", "2020-03-20", "2020-03-21", "2020-03-22", "2020-03-23", "2020-03-24", "2020-03-25", "2020-03-26", "2020-03-27", "2020-03-28", "2020-03-29", "2020-03-30", "2020-03-31", "2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04", "2020-04-05", "2020-04-06", "2020-04-07", "2020-04-08", "2020-04-09", "2020-04-10", "2020-04-11", "2020-04-12", "2020-04-13", "2020-04-14", "2020-04-15", "2020-04-16", "2020-04-17", "2020-04-18", "2020-04-19", "2020-04-20", "2020-04-21", "2020-04-22", "2020-04-23", "2020-04-24", "2020-04-25", "2020-04-26", "2020-04-27", "2020-04-28", "2020-04-29", "2020-04-30", "2020-05-01", "2020-05-02", "2020-05-03", "2020-05-04", "2020-05-05", "2020-05-06", "2020-05-07", "2020-05-08", "2020-05-09", "2020-05-10", "2020-05-11", "2020-05-12", "2020-05-13", "2020-05-14", "2020-05-15", "2020-05-16", "2020-05-17", "2020-05-18", "2020-05-19", "2020-05-20", "2020-05-21", "2020-05-22", "2020-05-23", "2020-05-24", "2020-05-25", "2020-05-26", "2020-05-27", "2020-05-28", "2020-05-29", "2020-05-30", "2020-05-31", "2020-06-01", "2020-06-02", "2020-06-03", "2020-06-04", "2020-06-05", "2020-06-06", "2020-06-07", "2020-06-08", "2020-06-09", "2020-06-10", "2020-06-11", "2020-06-12", "2020-06-13", "2020-06-14", "2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18", "2020-06-19", "2020-06-20", "2020-06-21", "2020-06-22", "2020-06-23", "2020-06-24", "2020-06-25", "2020-06-26", "2020-06-27", "2020-06-28", "2020-06-29", "2020-06-30"], "xaxis": "x2", "y": [13, 16, 20, 21, 25, 26, 29, 31, 33, 35, 38, 39, 41, 41, 41, 44, 45, 49, 51, 54, 55, 58, 60, 64, 68, 75, 77, 78, 80, 82, 84, 87, 92, 94, 97, 98, 101, 106, 107, 109, 111, 113, 115, 116, 118, 122, 122, 124, 124, 125, 125, 126, 127, 127, 127, 127, 127, 128, 129, 130, 130, 130, 130, 131, 131, 132, 133, 133, 133, 133, 133, 133, 134, 135, 135, 136, 136, 137, 137, 137, 138, 138, 140, 140, 141, 142, 142, 142, 142, 142, 143, 143, 144, 145, 145, 145, 145, 145, 145, 146, 147, 147, 147, 147, 147, 147, 148, 149, 150, 150, 150, 150, 150, 150, 150, 151, 151, 151, 151, 151, 151], "yaxis": "y2"}, {"line": {"color": "#eb2a2d"}, "mode": "lines", "name": "female", "showlegend": false, "type": "scatter", "x": ["2020-03-02", "2020-03-03", "2020-03-04", "2020-03-05", "2020-03-06", "2020-03-07", "2020-03-08", "2020-03-09", "2020-03-10", "2020-03-11", "2020-03-12", "2020-03-13", "2020-03-14", "2020-03-15", "2020-03-16", "2020-03-17", "2020-03-18", "2020-03-19", "2020-03-20", "2020-03-21", "2020-03-22", "2020-03-23", "2020-03-24", "2020-03-25", "2020-03-26", "2020-03-27", "2020-03-28", "2020-03-29", "2020-03-30", "2020-03-31", "2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04", "2020-04-05", "2020-04-06", "2020-04-07", "2020-04-08", "2020-04-09", "2020-04-10", "2020-04-11", "2020-04-12", "2020-04-13", "2020-04-14", "2020-04-15", "2020-04-16", "2020-04-17", "2020-04-18", "2020-04-19", "2020-04-20", "2020-04-21", "2020-04-22", "2020-04-23", "2020-04-24", "2020-04-25", "2020-04-26", "2020-04-27", "2020-04-28", "2020-04-29", "2020-04-30", "2020-05-01", "2020-05-02", "2020-05-03", "2020-05-04", "2020-05-05", "2020-05-06", "2020-05-07", "2020-05-08", "2020-05-09", "2020-05-10", "2020-05-11", "2020-05-12", "2020-05-13", "2020-05-14", "2020-05-15", "2020-05-16", "2020-05-17", "2020-05-18", "2020-05-19", "2020-05-20", "2020-05-21", "2020-05-22", "2020-05-23", "2020-05-24", "2020-05-25", "2020-05-26", "2020-05-27", "2020-05-28", "2020-05-29", "2020-05-30", "2020-05-31", "2020-06-01", "2020-06-02", "2020-06-03", "2020-06-04", "2020-06-05", "2020-06-06", "2020-06-07", "2020-06-08", "2020-06-09", "2020-06-10", "2020-06-11", "2020-06-12", "2020-06-13", "2020-06-14", "2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18", "2020-06-19", "2020-06-20", "2020-06-21", "2020-06-22", "2020-06-23", "2020-06-24", "2020-06-25", "2020-06-26", "2020-06-27", "2020-06-28", "2020-06-29", "2020-06-30"], "xaxis": "x2", "y": [9, 12, 12, 14, 17, 18, 20, 20, 21, 25, 28, 28, 31, 34, 34, 37, 39, 42, 43, 48, 49, 53, 60, 62, 63, 64, 67, 74, 78, 80, 81, 82, 82, 83, 86, 88, 91, 94, 97, 99, 100, 101, 102, 106, 107, 107, 108, 108, 110, 111, 112, 112, 113, 113, 113, 115, 116, 116, 117, 117, 118, 120, 120, 121, 123, 123, 123, 123, 123, 123, 123, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 128, 128, 128, 128, 128, 128, 128, 128, 128, 129, 129, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 131, 131, 131, 131, 131, 131, 131, 131], "yaxis": "y2"}],
                        {"annotations": [{"font": {"size": 16}, "showarrow": false, "text": "\uc131\ubcc4 \ud655\uc9c4\uc790 \ucd94\uc774 (Confirmed by Age)", "x": 0.225, "xanchor": "center", "xref": "paper", "y": 1.0, "yanchor": "bottom", "yref": "paper"}, {"font": {"size": 16}, "showarrow": false, "text": "\uc131\ubcc4 \uc0ac\ub9dd\uc790 \ucd94\uc774 (Deceased by Age)", "x": 0.775, "xanchor": "center", "xref": "paper", "y": 1.0, "yanchor": "bottom", "yref": "paper"}], "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "\uc131\ubcc4 \ud655\uc9c4\uc790/\uc0ac\ub9dd\uc790 \ucd94\uc774 (Confirmed/Deceased by Gender)", "x": 0.5, "xanchor": "center", "y": 0.92, "yanchor": "bottom"}, "xaxis": {"anchor": "y", "domain": [0.0, 0.45], "title": {"text": "Month"}}, "xaxis2": {"anchor": "y2", "domain": [0.55, 1.0], "title": {"text": "Month"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "Number"}}, "yaxis2": {"anchor": "x2", "domain": [0.0, 1.0], "title": {"text": "Number"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('7178e4db-4e81-453b-9353-bba56f18728c');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>



```python
print(case.shape)
case.head()
```

    (174, 8)
    




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
      <th>case_id</th>
      <th>province</th>
      <th>city</th>
      <th>group</th>
      <th>infection_case</th>
      <th>confirmed</th>
      <th>latitude</th>
      <th>longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1000001</td>
      <td>Seoul</td>
      <td>Yongsan-gu</td>
      <td>True</td>
      <td>Itaewon Clubs</td>
      <td>139</td>
      <td>37.538621</td>
      <td>126.992652</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1000002</td>
      <td>Seoul</td>
      <td>Gwanak-gu</td>
      <td>True</td>
      <td>Richway</td>
      <td>119</td>
      <td>37.48208</td>
      <td>126.901384</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1000003</td>
      <td>Seoul</td>
      <td>Guro-gu</td>
      <td>True</td>
      <td>Guro-gu Call Center</td>
      <td>95</td>
      <td>37.508163</td>
      <td>126.884387</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1000004</td>
      <td>Seoul</td>
      <td>Yangcheon-gu</td>
      <td>True</td>
      <td>Yangcheon Table Tennis Club</td>
      <td>43</td>
      <td>37.546061</td>
      <td>126.874209</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1000005</td>
      <td>Seoul</td>
      <td>Dobong-gu</td>
      <td>True</td>
      <td>Day Care Center</td>
      <td>43</td>
      <td>37.679422</td>
      <td>127.044374</td>
    </tr>
  </tbody>
</table>
</div>




```python
case_ = case.copy()
```


```python
print(case_.shape)
case_
```

    (174, 8)
    




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
      <th>case_id</th>
      <th>province</th>
      <th>city</th>
      <th>group</th>
      <th>infection_case</th>
      <th>confirmed</th>
      <th>latitude</th>
      <th>longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1000001</td>
      <td>Seoul</td>
      <td>Yongsan-gu</td>
      <td>True</td>
      <td>Itaewon Clubs</td>
      <td>139</td>
      <td>37.538621</td>
      <td>126.992652</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1000002</td>
      <td>Seoul</td>
      <td>Gwanak-gu</td>
      <td>True</td>
      <td>Richway</td>
      <td>119</td>
      <td>37.48208</td>
      <td>126.901384</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1000003</td>
      <td>Seoul</td>
      <td>Guro-gu</td>
      <td>True</td>
      <td>Guro-gu Call Center</td>
      <td>95</td>
      <td>37.508163</td>
      <td>126.884387</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1000004</td>
      <td>Seoul</td>
      <td>Yangcheon-gu</td>
      <td>True</td>
      <td>Yangcheon Table Tennis Club</td>
      <td>43</td>
      <td>37.546061</td>
      <td>126.874209</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1000005</td>
      <td>Seoul</td>
      <td>Dobong-gu</td>
      <td>True</td>
      <td>Day Care Center</td>
      <td>43</td>
      <td>37.679422</td>
      <td>127.044374</td>
    </tr>
    <tr>
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
      <td>169</td>
      <td>6100012</td>
      <td>Gyeongsangnam-do</td>
      <td>-</td>
      <td>False</td>
      <td>etc</td>
      <td>20</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>170</td>
      <td>7000001</td>
      <td>Jeju-do</td>
      <td>-</td>
      <td>False</td>
      <td>overseas inflow</td>
      <td>14</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>171</td>
      <td>7000002</td>
      <td>Jeju-do</td>
      <td>-</td>
      <td>False</td>
      <td>contact with patient</td>
      <td>0</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>172</td>
      <td>7000003</td>
      <td>Jeju-do</td>
      <td>-</td>
      <td>False</td>
      <td>etc</td>
      <td>4</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>173</td>
      <td>7000004</td>
      <td>Jeju-do</td>
      <td>from other city</td>
      <td>True</td>
      <td>Itaewon Clubs</td>
      <td>1</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
</table>
<p>174 rows × 8 columns</p>
</div>




```python
case_ = case_[case_['latitude'] != '-']
```


```python
print(case_.shape)
case_
```

    (65, 8)
    




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
      <th>case_id</th>
      <th>province</th>
      <th>city</th>
      <th>group</th>
      <th>infection_case</th>
      <th>confirmed</th>
      <th>latitude</th>
      <th>longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1000001</td>
      <td>Seoul</td>
      <td>Yongsan-gu</td>
      <td>True</td>
      <td>Itaewon Clubs</td>
      <td>139</td>
      <td>37.538621</td>
      <td>126.992652</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1000002</td>
      <td>Seoul</td>
      <td>Gwanak-gu</td>
      <td>True</td>
      <td>Richway</td>
      <td>119</td>
      <td>37.48208</td>
      <td>126.901384</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1000003</td>
      <td>Seoul</td>
      <td>Guro-gu</td>
      <td>True</td>
      <td>Guro-gu Call Center</td>
      <td>95</td>
      <td>37.508163</td>
      <td>126.884387</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1000004</td>
      <td>Seoul</td>
      <td>Yangcheon-gu</td>
      <td>True</td>
      <td>Yangcheon Table Tennis Club</td>
      <td>43</td>
      <td>37.546061</td>
      <td>126.874209</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1000005</td>
      <td>Seoul</td>
      <td>Dobong-gu</td>
      <td>True</td>
      <td>Day Care Center</td>
      <td>43</td>
      <td>37.679422</td>
      <td>127.044374</td>
    </tr>
    <tr>
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
      <td>160</td>
      <td>6100003</td>
      <td>Gyeongsangnam-do</td>
      <td>Jinju-si</td>
      <td>True</td>
      <td>Wings Tower</td>
      <td>9</td>
      <td>35.164845</td>
      <td>128.126969</td>
    </tr>
    <tr>
      <td>161</td>
      <td>6100004</td>
      <td>Gyeongsangnam-do</td>
      <td>Geochang-gun</td>
      <td>True</td>
      <td>Geochang-gun Woongyang-myeon</td>
      <td>8</td>
      <td>35.805681</td>
      <td>127.917805</td>
    </tr>
    <tr>
      <td>162</td>
      <td>6100005</td>
      <td>Gyeongsangnam-do</td>
      <td>Changwon-si</td>
      <td>True</td>
      <td>Hanmaeum Changwon Hospital</td>
      <td>7</td>
      <td>35.22115</td>
      <td>128.6866</td>
    </tr>
    <tr>
      <td>163</td>
      <td>6100006</td>
      <td>Gyeongsangnam-do</td>
      <td>Changnyeong-gun</td>
      <td>True</td>
      <td>Changnyeong Coin Karaoke</td>
      <td>7</td>
      <td>35.54127</td>
      <td>128.5008</td>
    </tr>
    <tr>
      <td>164</td>
      <td>6100007</td>
      <td>Gyeongsangnam-do</td>
      <td>Yangsan-si</td>
      <td>True</td>
      <td>Soso Seowon</td>
      <td>3</td>
      <td>35.338811</td>
      <td>129.017508</td>
    </tr>
  </tbody>
</table>
<p>65 rows × 8 columns</p>
</div>




```python
case_["latitude"] = case_["latitude"].apply(pd.to_numeric)
case_["longitude"] = case_["longitude"].apply(pd.to_numeric)
```


```python
case_.rename(columns = {' case_id':'case_id'}, inplace = True)
```


```python
token = open('token.txt', 'r')
line = token.readline()
```


```python
 px.set_mapbox_access_token(line)
```


```python
fig_map = px.scatter_mapbox(case_, lat="latitude", lon="longitude", color='case_id', 
                            color_continuous_scale="Rainbow", size='confirmed', size_max=50, zoom=7)


fig_map.update_layout(mapbox_style="open-street-map",width=900, height=700)
fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig_map.show('svg')
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-164-842cd0295f0f> in <module>
          5 fig_map.update_layout(mapbox_style="open-street-map",width=900, height=700)
          6 fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    ----> 7 fig_map.show('svg')
    

    ~\Anaconda3\lib\site-packages\plotly\basedatatypes.py in show(self, *args, **kwargs)
       2869         import plotly.io as pio
       2870 
    -> 2871         return pio.show(self, *args, **kwargs)
       2872 
       2873     def to_json(self, *args, **kwargs):
    

    ~\Anaconda3\lib\site-packages\plotly\io\_renderers.py in show(fig, renderer, validate, **kwargs)
        375 
        376     # Mimetype renderers
    --> 377     bundle = renderers._build_mime_bundle(fig_dict, renderers_string=renderer, **kwargs)
        378     if bundle:
        379         if not ipython_display:
    

    ~\Anaconda3\lib\site-packages\plotly\io\_renderers.py in _build_mime_bundle(self, fig_dict, renderers_string, **kwargs)
        281             for renderer in renderers_list:
        282                 if isinstance(renderer, MimetypeRenderer):
    --> 283                     renderer.activate()
        284         else:
        285             # Activate any pending default renderers
    

    ~\Anaconda3\lib\site-packages\plotly\io\_base_renderers.py in activate(self)
        124     def activate(self):
        125         # Start up orca server to reduce the delay on first render
    --> 126         ensure_server()
        127 
        128     def to_mimebundle(self, fig_dict):
    

    ~\Anaconda3\lib\site-packages\plotly\io\_orca.py in ensure_server()
       1388         # Validate orca executable only if server_url is not provided
       1389         if status.state == "unvalidated":
    -> 1390             validate_executable()
       1391         # Acquire lock to make sure that we keep the properties of orca_state
       1392         # consistent across threads
    

    ~\Anaconda3\lib\site-packages\plotly\io\_orca.py in validate_executable()
       1085                 executable=config.executable,
       1086                 formatted_path=formatted_path,
    -> 1087                 instructions=install_location_instructions,
       1088             )
       1089         )
    

    ValueError: 
    The orca executable is required to export figures as static images,
    but it could not be found on the system path.
    
    Searched for executable 'orca' on the following path:
        C:\Users\AndrewKim\Anaconda3
        C:\Users\AndrewKim\Anaconda3\Library\mingw-w64\bin
        C:\Users\AndrewKim\Anaconda3\Library\usr\bin
        C:\Users\AndrewKim\Anaconda3\Library\bin
        C:\Users\AndrewKim\Anaconda3\Scripts
        C:\Program Files (x86)\Common Files\Oracle\Java\javapath
        C:\app\AndrewKim\product\11.2.0\dbhome_1\bin
        C:\Program Files (x86)\Intel\Intel(R) Management Engine Components\iCLS\
        C:\Program Files\Intel\Intel(R) Management Engine Components\iCLS\
        C:\Windows\system32
        C:\Windows
        C:\Windows\System32\Wbem
        C:\Windows\System32\WindowsPowerShell\v1.0\
        C:\Windows\System32\OpenSSH\
        C:\Program Files (x86)\Intel\Intel(R) Management Engine Components\DAL
        C:\Program Files\Intel\Intel(R) Management Engine Components\DAL
        C:\Program Files\Git\cmd
        C:\Program Files (x86)\Windows Kits\8.1\Windows Performance Toolkit\
        C:\Program Files\MySQL\MySQL Shell 8.0\bin\
        C:\Users\AndrewKim\AppData\Local\Programs\Python\Python38-32\Scripts\
        C:\Users\AndrewKim\AppData\Local\Programs\Python\Python38-32\
        C:\Users\AndrewKim\AppData\Local\Microsoft\WindowsApps
        C:\Users\AndrewKim\Anaconda3
        C:\Users\AndrewKim\Anaconda3\Library\bin
        C:\Users\AndrewKim\Anaconda3\Scripts
        
    
    If you haven't installed orca yet, you can do so using conda as follows:
    
        $ conda install -c plotly plotly-orca
    
    Alternatively, see other installation methods in the orca project README at
    https://github.com/plotly/orca
    
    After installation is complete, no further configuration should be needed.
    
    If you have installed orca, then for some reason plotly.py was unable to
    locate it. In this case, set the `plotly.io.orca.config.executable`
    property to the full path of your orca executable. For example:
    
        >>> plotly.io.orca.config.executable = '/path/to/orca'
    
    After updating this executable property, try the export operation again.
    If it is successful then you may want to save this configuration so that it
    will be applied automatically in future sessions. You can do this as follows:
    
        >>> plotly.io.orca.config.save()
    
    If you're still having trouble, feel free to ask for help on the forums at
    https://community.plot.ly/c/api/python
    



```python
fig_map = px.scatter_mapbox(case_, lat="latitude", lon="longitude", color='case_id', color_continuous_scale="Rainbow", size='confirmed', size_max=50, zoom=7)


fig_map.update_layout(mapbox_style="carto-positron",width=900, height=700)
fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig_map.show()
```


<div>


            <div id="66e2c694-f245-4d78-acd5-a14aac1ed67a" class="plotly-graph-div" style="height:700px; width:900px;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("66e2c694-f245-4d78-acd5-a14aac1ed67a")) {
                    Plotly.newPlot(
                        '66e2c694-f245-4d78-acd5-a14aac1ed67a',
                        [{"hovertemplate": "confirmed=%{marker.size}<br>latitude=%{lat}<br>longitude=%{lon}<br>case_id=%{marker.color}<extra></extra>", "lat": [37.538621, 37.482079999999996, 37.508163, 37.546061, 37.679421999999995, 37.481059, 37.592888, 37.481735, 37.63369, 37.55713, 37.576809999999995, 37.48825, 37.562405, 37.558147, 37.594782, 37.560899, 37.520846, 37.522331, 37.524623, 37.498279, 37.559649, 37.565699, 37.486837, 35.21628, 35.16708, 35.20599, 35.17371, 35.84008, 35.857375, 35.885591999999995, 35.857393, 35.88395, 35.136035, 36.3400973, 36.3398739, 36.346869, 36.358123, 36.504713, 36.48025, 37.455687, 37.530578999999996, 37.758635, 37.388329999999996, 37.381784, 37.287356, 37.403721999999995, 37.2376, 37.342762, 36.824220000000004, 36.81503, 37.000353999999994, 35.078825, 35.64887, 36.92757, 35.782149, 36.646845, 36.0581, 35.84819, 35.82558, 35.685559999999995, 35.164845, 35.805681, 35.22115, 35.541270000000004, 35.338811], "legendgroup": "", "lon": [126.992652, 126.90138400000001, 126.88438700000002, 126.87420900000001, 127.044374, 126.89434299999999, 127.056766, 126.930121, 126.9165, 127.0403, 127.006, 127.08559, 126.984377, 126.943799, 126.96802199999999, 126.966998, 126.931278, 127.05738799999999, 126.84311799999999, 127.03013899999999, 126.835102, 126.977079, 126.89316299999999, 129.0771, 129.1124, 129.1256, 129.0633, 128.5667, 128.466651, 128.556649, 128.466653, 128.62405900000002, 126.956405, 127.39270990000001, 127.38197439999999, 127.36859399999999, 127.388856, 127.26517199999999, 127.289, 127.161627, 126.775254, 127.077716, 127.1218, 126.93615, 127.013827, 126.954939, 127.0517, 127.98381499999999, 127.9552, 127.1139, 126.35444299999999, 126.316746, 128.7368, 128.9099, 128.801498, 128.43741599999998, 128.4941, 128.7621, 128.7373, 127.9127, 128.126969, 127.917805, 128.6866, 128.5008, 129.01750800000002], "marker": {"color": [1000001, 1000002, 1000003, 1000004, 1000005, 1000006, 1000008, 1000010, 1000011, 1000012, 1000013, 1000014, 1000015, 1000016, 1000017, 1000023, 1000024, 1000025, 1000026, 1000029, 1000030, 1000032, 1000035, 1100001, 1100003, 1100004, 1100005, 1200001, 1200002, 1200003, 1200004, 1200005, 1300001, 1500002, 1500003, 1500004, 1500005, 1700001, 1700002, 2000001, 2000002, 2000005, 2000010, 2000011, 2000012, 2000013, 2000014, 3000003, 4000001, 4100001, 4100003, 5100001, 6000002, 6000003, 6000004, 6000006, 6000007, 6000008, 6000009, 6100002, 6100003, 6100004, 6100005, 6100006, 6100007], "coloraxis": "coloraxis", "size": [139, 119, 95, 43, 43, 41, 17, 30, 14, 13, 10, 7, 7, 5, 7, 13, 3, 1, 3, 4, 0, 3, 3, 39, 5, 6, 4, 4511, 196, 124, 101, 39, 5, 13, 7, 4, 3, 31, 8, 67, 67, 50, 22, 22, 15, 17, 10, 4, 11, 103, 9, 2, 119, 68, 66, 40, 36, 17, 16, 10, 9, 8, 7, 7, 3], "sizemode": "area", "sizeref": 1.8044}, "mode": "markers", "name": "", "showlegend": false, "subplot": "mapbox", "type": "scattermapbox"}],
                        {"coloraxis": {"colorbar": {"title": {"text": "case_id"}}, "colorscale": [[0.0, "rgb(150,0,90)"], [0.125, "rgb(0,0,200)"], [0.25, "rgb(0,25,255)"], [0.375, "rgb(0,152,255)"], [0.5, "rgb(44,255,150)"], [0.625, "rgb(151,255,0)"], [0.75, "rgb(255,234,0)"], [0.875, "rgb(255,111,0)"], [1.0, "rgb(255,0,0)"]]}, "height": 700, "legend": {"itemsizing": "constant", "tracegroupgap": 0}, "mapbox": {"accesstoken": "pk.eyJ1IjoicnVkZmJmMyIsImEiOiJja2NwMHNqbmEwY2hsMnJsZWJ1Z2RidW9uIn0.ykG4EnjeO8toraCGglt-Lg", "center": {"lat": 36.69405111076924, "lon": 127.58488500461539}, "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]}, "style": "carto-positron", "zoom": 7}, "margin": {"b": 0, "l": 0, "r": 0, "t": 0}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "width": 900},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('66e2c694-f245-4d78-acd5-a14aac1ed67a');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>



```python

```


```python

```
