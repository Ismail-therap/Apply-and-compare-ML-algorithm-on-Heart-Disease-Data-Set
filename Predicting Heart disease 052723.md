# Original data source:

https://archive.ics.uci.edu/ml/datasets/heart+Disease

# Loading and Preparing data


```python
# Required packages
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
```


```python
# Setting data directory
os.chdir('C:/Users/stati/OneDrive/Desktop/Research_with_Nobel_vai/Results')
```


```python
# Define the data types for each column



my_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 
            'fbs', 'restecg', 'thalach', 'exang', 
            'oldpeak', 'slope', 'ca', 'thal', 'num']
```


```python
clev = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data',header=None, names=my_columns)
hung = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data',header=None, names=my_columns)
swit = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data',header=None, names=my_columns)
va = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data',header=None, names=my_columns)
```


```python
# Concatenate the data frames
df = pd.concat([clev, hung, swit,va])
```


```python
# Print value counts for each column to explore any unusual data points
for column in df.columns:
    print(column, ":\n", df[column].value_counts(), "\n")
```

    age :
     54.0    51
    58.0    43
    55.0    41
    56.0    38
    57.0    38
    52.0    36
    62.0    35
    51.0    35
    59.0    35
    53.0    33
    60.0    32
    61.0    31
    48.0    31
    63.0    30
    50.0    25
    41.0    24
    46.0    24
    43.0    24
    64.0    22
    49.0    22
    65.0    21
    44.0    19
    47.0    19
    45.0    18
    42.0    18
    38.0    16
    67.0    15
    39.0    15
    69.0    13
    40.0    13
    66.0    13
    35.0    11
    37.0    11
    68.0    10
    34.0     7
    70.0     7
    74.0     7
    36.0     6
    32.0     5
    71.0     5
    72.0     4
    29.0     3
    75.0     3
    31.0     2
    33.0     2
    76.0     2
    77.0     2
    30.0     1
    28.0     1
    73.0     1
    Name: age, dtype: int64 
    
    sex :
     1.0    726
    0.0    194
    Name: sex, dtype: int64 
    
    cp :
     4.0    496
    3.0    204
    2.0    174
    1.0     46
    Name: cp, dtype: int64 
    
    trestbps :
     120      94
    130      79
    140      70
    ?        59
    110      40
             ..
    108       1
    129.0     1
    113       1
    92        1
    127       1
    Name: trestbps, Length: 103, dtype: int64 
    
    chol :
     0        123
    0         49
    ?         30
    220        7
    216        7
            ... 
    187        1
    141.0      1
    328        1
    168        1
    385        1
    Name: chol, Length: 336, dtype: int64 
    
    fbs :
     0      434
    0.0    258
    1       93
    ?       90
    1.0     45
    Name: fbs, dtype: int64 
    
    restecg :
     0      320
    0.0    231
    2.0    175
    1.0     97
    1       82
    2       13
    ?        2
    Name: restecg, dtype: int64 
    
    thalach :
     ?        55
    150      36
    140      35
    120      32
    130      26
             ..
    95.0      1
    179       1
    192.0     1
    127.0     1
    151       1
    Name: thalach, Length: 198, dtype: int64 
    
    exang :
     0      324
    1      238
    0.0    204
    1.0     99
    ?       55
    Name: exang, dtype: int64 
    
    oldpeak :
     0.0     288
    0        82
    ?        62
    1.0      55
    2.0      40
           ... 
    -.1       1
    4.4       1
    5.0       1
    -1.5      1
    0.8       1
    Name: oldpeak, Length: 82, dtype: int64 
    
    slope :
     ?      309
    2      205
    1.0    142
    2.0    140
    1       61
    3       42
    3.0     21
    Name: slope, dtype: int64 
    
    ca :
     ?      611
    0.0    176
    1.0     65
    2.0     38
    3.0     20
    0        5
    2        3
    1        2
    Name: ca, dtype: int64 
    
    thal :
     ?      486
    3.0    166
    7.0    117
    7       75
    3       30
    6       28
    6.0     18
    Name: thal, dtype: int64 
    
    num :
     0    411
    1    265
    2    109
    3    107
    4     28
    Name: num, dtype: int64 
    
    


```python
df.replace('?', float('NaN'), inplace=True)
```


```python
# Re-declare data types for appropriate columns
df['age'] = df['age'].astype('int')
df['sex'] = df['sex'].astype('category')
df['cp'] = df['cp'].astype('category')
df['trestbps'] = df['trestbps'].astype('float')
df['chol'] = df['chol'].astype('float')
df['fbs'] = df['fbs'].astype('category')
df['restecg'] = df['restecg'].astype('category')
df['thalach'] = df['thalach'].astype('float')
df['exang'] = df['exang'].astype('category')
df['oldpeak'] = df['oldpeak'].astype('float')
df['slope'] = df['slope'].astype('category')
df['ca'] = df['ca'].astype('float')
df['thal'] = df['thal'].astype('float')
df['num'] = df['num'].astype('category')

```

# Exploratory analysis:


```python
# Check the first few rows of the DataFrame
print(df.head())
```

       age  sex   cp  trestbps   chol  fbs restecg  thalach exang  oldpeak slope  \
    0   63  1.0  1.0     145.0  233.0  1.0     2.0    150.0   0.0      2.3   3.0   
    1   67  1.0  4.0     160.0  286.0  0.0     2.0    108.0   1.0      1.5   2.0   
    2   67  1.0  4.0     120.0  229.0  0.0     2.0    129.0   1.0      2.6   2.0   
    3   37  1.0  3.0     130.0  250.0  0.0     0.0    187.0   0.0      3.5   3.0   
    4   41  0.0  2.0     130.0  204.0  0.0     2.0    172.0   0.0      1.4   1.0   
    
        ca  thal num  
    0  0.0   6.0   0  
    1  3.0   3.0   2  
    2  2.0   7.0   1  
    3  0.0   3.0   0  
    4  0.0   3.0   0  
    


```python
# Check the shape of the DataFrame
print(df.shape)
```

    (920, 14)
    


```python
# Check the data types of the columns
print(df.dtypes)
```

    age            int32
    sex         category
    cp          category
    trestbps     float64
    chol         float64
    fbs         category
    restecg     category
    thalach      float64
    exang       category
    oldpeak      float64
    slope       category
    ca           float64
    thal         float64
    num         category
    dtype: object
    


```python
# Check the summary statistics of the numerical columns
print(df.describe())
```

                  age    trestbps        chol     thalach     oldpeak          ca  \
    count  920.000000  861.000000  890.000000  865.000000  858.000000  309.000000   
    mean    53.510870  132.132404  199.130337  137.545665    0.878788    0.676375   
    std      9.424685   19.066070  110.780810   25.926276    1.091226    0.935653   
    min     28.000000    0.000000    0.000000   60.000000   -2.600000    0.000000   
    25%     47.000000  120.000000  175.000000  120.000000    0.000000    0.000000   
    50%     54.000000  130.000000  223.000000  140.000000    0.500000    0.000000   
    75%     60.000000  140.000000  268.000000  157.000000    1.500000    1.000000   
    max     77.000000  200.000000  603.000000  202.000000    6.200000    3.000000   
    
                 thal  
    count  434.000000  
    mean     5.087558  
    std      1.919075  
    min      3.000000  
    25%      3.000000  
    50%      6.000000  
    75%      7.000000  
    max      7.000000  
    


```python
# Check the distribution of the target variable
sns.countplot(x='num', data=df)
plt.show()

```


    
![png](output_15_0.png)
    



```python
# Number of missing value per columns
```


```python
# Count the number of missing values per column
missing_values_count = df.isna().sum()

# Print the result
print(missing_values_count)
```

    age           0
    sex           0
    cp            0
    trestbps     59
    chol         30
    fbs          90
    restecg       2
    thalach      55
    exang        55
    oldpeak      62
    slope       309
    ca          611
    thal        486
    num           0
    dtype: int64
    


```python
# Remove missing entries from DataFrame
df = df.dropna()
```


```python
print(df.shape)
```

    (299, 14)
    


```python
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
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalach</th>
      <th>exang</th>
      <th>oldpeak</th>
      <th>slope</th>
      <th>ca</th>
      <th>thal</th>
      <th>num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>145.0</td>
      <td>233.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>150.0</td>
      <td>0.0</td>
      <td>2.3</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>67</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>160.0</td>
      <td>286.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>108.0</td>
      <td>1.0</td>
      <td>1.5</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>67</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>120.0</td>
      <td>229.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>129.0</td>
      <td>1.0</td>
      <td>2.6</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>37</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>130.0</td>
      <td>250.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>187.0</td>
      <td>0.0</td>
      <td>3.5</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>41</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>130.0</td>
      <td>204.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>172.0</td>
      <td>0.0</td>
      <td>1.4</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
## Defining the dependent variable:

# thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

df['heart_patient'] = np.where(df['thal'] == 3, 0, 1)

```


```python
df = df.drop('thal',axis = 1)
```


```python
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
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalach</th>
      <th>exang</th>
      <th>oldpeak</th>
      <th>slope</th>
      <th>ca</th>
      <th>num</th>
      <th>heart_patient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>145.0</td>
      <td>233.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>150.0</td>
      <td>0.0</td>
      <td>2.3</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>67</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>160.0</td>
      <td>286.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>108.0</td>
      <td>1.0</td>
      <td>1.5</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>67</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>120.0</td>
      <td>229.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>129.0</td>
      <td>1.0</td>
      <td>2.6</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>37</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>130.0</td>
      <td>250.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>187.0</td>
      <td>0.0</td>
      <td>3.5</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>41</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>130.0</td>
      <td>204.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>172.0</td>
      <td>0.0</td>
      <td>1.4</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



So, we have dependent variable 'heart_patient', 0 indicating that the patient do not have have heart condition, 1 means a heart patient. 

We have seen common ML models like Logistic regression, Random forest, XGBoost, CNN already used for this data to classify the patients. Now, we will try to use 'Language models' like BERT, LSTM, Fasttext to explore their performance in classifying the heart patients. 

# Prepare the data to feed in the model:


```python
df.columns
```




    Index(['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
           'exang', 'oldpeak', 'slope', 'ca', 'num', 'heart_patient'],
          dtype='object')




```python

```

# 1. LSTM:




```python
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
```


```python
# Prepare the input features (independent variables) and target variable (heart_patient)
X = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'num']].values
y = df['heart_patient'].values

X = X.astype(np.float32)
y = y.astype(np.float32)
# Split the data into train and test sets with an 80-20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the input features for LSTM model
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Build the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(1, X_train.shape[2])))
model.add(Dense(1, activation='sigmoid'))
# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

```


```python
# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)


```

    Epoch 1/10
    8/8 [==============================] - 2s 4ms/step - loss: 0.7126 - accuracy: 0.4435
    Epoch 2/10
    8/8 [==============================] - 0s 3ms/step - loss: 0.6871 - accuracy: 0.4979
    Epoch 3/10
    8/8 [==============================] - 0s 2ms/step - loss: 0.6756 - accuracy: 0.6360
    Epoch 4/10
    8/8 [==============================] - 0s 3ms/step - loss: 0.6652 - accuracy: 0.6653
    Epoch 5/10
    8/8 [==============================] - 0s 2ms/step - loss: 0.6584 - accuracy: 0.6569
    Epoch 6/10
    8/8 [==============================] - 0s 3ms/step - loss: 0.6496 - accuracy: 0.6611
    Epoch 7/10
    8/8 [==============================] - 0s 3ms/step - loss: 0.6438 - accuracy: 0.6653
    Epoch 8/10
    8/8 [==============================] - 0s 2ms/step - loss: 0.6426 - accuracy: 0.6695
    Epoch 9/10
    8/8 [==============================] - 0s 4ms/step - loss: 0.6390 - accuracy: 0.6736
    Epoch 10/10
    8/8 [==============================] - 0s 2ms/step - loss: 0.6368 - accuracy: 0.6695
    




    <keras.callbacks.History at 0x221d4fcddf0>




```python
# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)
```

    2/2 [==============================] - 0s 8ms/step - loss: 0.6847 - accuracy: 0.5333
    Test Loss: 0.6847085356712341
    Test Accuracy: 0.5333333611488342
    

# 2. Fasttext:

# 3. BERT


```python

```
