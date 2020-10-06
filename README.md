# `pd.get_dummies` vs. `OneHotEncoder` for Machine Learning

This notebook demonstrates why `OneHotEncoder` is better than `pd.get_dummies` for creating dummy categorical variables in a machine learning context


```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
```

Let's use a made-up dataset for the purpose of this example.  Let's say we have total purchase amounts from customers in different states.


```python
np.random.seed(2020)
```


```python
amounts = np.random.choice(1000, 10)
ages = np.random.choice(100, 10)
states = np.random.choice(["Washington", "California", "Illinois"], 10)
```


```python
df = pd.DataFrame([amounts, ages, states]).T
df.columns = ["Amount", "Age", "State"]
```


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
      <th>Amount</th>
      <th>Age</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>864</td>
      <td>29</td>
      <td>Illinois</td>
    </tr>
    <tr>
      <th>1</th>
      <td>392</td>
      <td>48</td>
      <td>California</td>
    </tr>
    <tr>
      <th>2</th>
      <td>323</td>
      <td>32</td>
      <td>Washington</td>
    </tr>
    <tr>
      <th>3</th>
      <td>630</td>
      <td>24</td>
      <td>Washington</td>
    </tr>
    <tr>
      <th>4</th>
      <td>707</td>
      <td>74</td>
      <td>Washington</td>
    </tr>
    <tr>
      <th>5</th>
      <td>91</td>
      <td>9</td>
      <td>Washington</td>
    </tr>
    <tr>
      <th>6</th>
      <td>637</td>
      <td>51</td>
      <td>Illinois</td>
    </tr>
    <tr>
      <th>7</th>
      <td>643</td>
      <td>11</td>
      <td>Washington</td>
    </tr>
    <tr>
      <th>8</th>
      <td>583</td>
      <td>55</td>
      <td>California</td>
    </tr>
    <tr>
      <th>9</th>
      <td>952</td>
      <td>62</td>
      <td>California</td>
    </tr>
  </tbody>
</table>
</div>



Ok, let's say this is our training dataset.  We want a linear regression model to predict the amount based on the age and state of the customer

## Preprocessing with `pd.get_dummies`
To use this data in a linear regression model, we need to convert the categorical data to dummied-out numbers.  First, let's try doing that with `pd.get_dummies`


```python
dummies_df = pd.get_dummies(df, columns=["State"])
dummies_df
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
      <th>Amount</th>
      <th>Age</th>
      <th>State_California</th>
      <th>State_Illinois</th>
      <th>State_Washington</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>864</td>
      <td>29</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>392</td>
      <td>48</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>323</td>
      <td>32</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>630</td>
      <td>24</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>707</td>
      <td>74</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>91</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>637</td>
      <td>51</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>643</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>583</td>
      <td>55</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>952</td>
      <td>62</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Fitting a Model to Training Data
That was very easy, let's fit a linear regression model


```python
dummies_model = LinearRegression()
dummies_model.fit(dummies_df.drop("Amount", axis=1), dummies_df["Amount"])
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python
dummies_coef = dummies_model.coef_
dummies_coef
```




    array([  4.89704939, -46.83843632, 134.78397121, -87.94553489])




```python
dummies_intercept = dummies_model.intercept_
dummies_intercept
```




    419.83405316798473




```python
dummies_model.score(dummies_df.drop("Amount", axis=1), dummies_df["Amount"])
```




    0.3343722589232698



### Testing on Unseen Data

So, we have an r-squared of 0.33 for our training data. Let's make up a few more records for testing on unseen data


```python
np.random.seed(1)
```


```python
test_amounts = np.random.choice(1000, 5)
test_ages = np.random.choice(100, 5)
test_states = np.random.choice(["Washington", "California", "Illinois"], 5)
```


```python
test_df = pd.DataFrame([test_amounts, test_ages, test_states]).T
test_df.columns = ["Amount", "Age", "State"]
```


```python
test_df
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
      <th>Amount</th>
      <th>Age</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>37</td>
      <td>9</td>
      <td>Washington</td>
    </tr>
    <tr>
      <th>1</th>
      <td>235</td>
      <td>75</td>
      <td>California</td>
    </tr>
    <tr>
      <th>2</th>
      <td>908</td>
      <td>5</td>
      <td>Washington</td>
    </tr>
    <tr>
      <th>3</th>
      <td>72</td>
      <td>79</td>
      <td>California</td>
    </tr>
    <tr>
      <th>4</th>
      <td>767</td>
      <td>64</td>
      <td>Washington</td>
    </tr>
  </tbody>
</table>
</div>



The only states we have here are Washington and California.  Let's dummy those out:


```python
test_dummies_df = pd.get_dummies(test_df, columns=["State"])
test_dummies_df
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
      <th>Amount</th>
      <th>Age</th>
      <th>State_California</th>
      <th>State_Washington</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>37</td>
      <td>9</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>235</td>
      <td>75</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>908</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>72</td>
      <td>79</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>767</td>
      <td>64</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Now let's try to score our model on these:


```python
dummies_model.score(test_dummies_df.drop("Amount", axis=1), test_dummies_df["Amount"])
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-16-1561818d2ab8> in <module>
    ----> 1 dummies_model.score(test_dummies_df.drop("Amount", axis=1), test_dummies_df["Amount"])
    

    ~/.conda/envs/prework-labs/lib/python3.7/site-packages/sklearn/base.py in score(self, X, y, sample_weight)
        420         from .metrics import r2_score
        421         from .metrics._regression import _check_reg_targets
    --> 422         y_pred = self.predict(X)
        423         # XXX: Remove the check in 0.23
        424         y_type, _, _, _ = _check_reg_targets(y, y_pred, None)


    ~/.conda/envs/prework-labs/lib/python3.7/site-packages/sklearn/linear_model/_base.py in predict(self, X)
        223             Returns predicted values.
        224         """
    --> 225         return self._decision_function(X)
        226 
        227     _preprocess_data = staticmethod(_preprocess_data)


    ~/.conda/envs/prework-labs/lib/python3.7/site-packages/sklearn/linear_model/_base.py in _decision_function(self, X)
        207         X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])
        208         return safe_sparse_dot(X, self.coef_.T,
    --> 209                                dense_output=True) + self.intercept_
        210 
        211     def predict(self, X):


    ~/.conda/envs/prework-labs/lib/python3.7/site-packages/sklearn/utils/extmath.py in safe_sparse_dot(a, b, dense_output)
        149             ret = np.dot(a, b)
        150     else:
    --> 151         ret = a @ b
        152 
        153     if (sparse.issparse(a) and sparse.issparse(b)


    ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 4 is different from 3)


#### Error!

We get an error, since the model was trained on a dataset with 4 features, but now we are trying to pass in only 3 features

## Preprocessing with `OneHotEncoder`

This process will be a bit more annoying, but it won't break with the new data


```python
# sparse=False makes it more readable but less efficient
ohe = OneHotEncoder(categories="auto", handle_unknown="ignore", sparse=False)
```


```python
ohe_states_array = ohe.fit_transform(df[["State"]])
```


```python
ohe_states_array
```




    array([[0., 1., 0.],
           [1., 0., 0.],
           [0., 0., 1.],
           [0., 0., 1.],
           [0., 0., 1.],
           [0., 0., 1.],
           [0., 1., 0.],
           [0., 0., 1.],
           [1., 0., 0.],
           [1., 0., 0.]])




```python
ohe_states_df = pd.DataFrame(ohe_states_array, index=df.index, columns=ohe.categories_[0])
```


```python
ohe_states_df
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
      <th>California</th>
      <th>Illinois</th>
      <th>Washington</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
ohe_df = pd.concat([df.drop("State", axis=1), ohe_states_df], axis=1)
ohe_df
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
      <th>Amount</th>
      <th>Age</th>
      <th>California</th>
      <th>Illinois</th>
      <th>Washington</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>864</td>
      <td>29</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>392</td>
      <td>48</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>323</td>
      <td>32</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>630</td>
      <td>24</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>707</td>
      <td>74</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>91</td>
      <td>9</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>637</td>
      <td>51</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>643</td>
      <td>11</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>583</td>
      <td>55</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>952</td>
      <td>62</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



### Fitting a Model to Training Data

This will look the same as the `pd.get_dummies` version


```python
ohe_model = LinearRegression()
ohe_model.fit(ohe_df.drop("Amount", axis=1), ohe_df["Amount"])
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python
print("Dummies Model:", dummies_coef)
print("OHE Model:", ohe_model.coef_)
```

    Dummies Model: [  4.89704939 -46.83843632 134.78397121 -87.94553489]
    OHE Model: [  4.89704939 -46.83843632 134.78397121 -87.94553489]



```python
print("Dummies Model:", dummies_intercept)
print("OHE Model:", ohe_model.intercept_)
```

    Dummies Model: 419.83405316798473
    OHE Model: 419.83405316798473


### Testing on Unseen Data

This is where the encoder makes a difference!


```python
# Reminder that this is our test data
test_df
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
      <th>Amount</th>
      <th>Age</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>37</td>
      <td>9</td>
      <td>Washington</td>
    </tr>
    <tr>
      <th>1</th>
      <td>235</td>
      <td>75</td>
      <td>California</td>
    </tr>
    <tr>
      <th>2</th>
      <td>908</td>
      <td>5</td>
      <td>Washington</td>
    </tr>
    <tr>
      <th>3</th>
      <td>72</td>
      <td>79</td>
      <td>California</td>
    </tr>
    <tr>
      <th>4</th>
      <td>767</td>
      <td>64</td>
      <td>Washington</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_ohe_states_array = ohe.transform(test_df[["State"]])
test_ohe_states_df = pd.DataFrame(test_ohe_states_array, index=test_df.index, columns=ohe.categories_[0])
test_ohe_states_df
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
      <th>California</th>
      <th>Illinois</th>
      <th>Washington</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



Notice that we now have the same columns as the training data, even though there were no "Illinois" values in the testing data


```python
test_ohe_df = pd.concat([test_df.drop("State", axis=1), test_ohe_states_df], axis=1)
test_ohe_df
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
      <th>Amount</th>
      <th>Age</th>
      <th>California</th>
      <th>Illinois</th>
      <th>Washington</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>37</td>
      <td>9</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>235</td>
      <td>75</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>908</td>
      <td>5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>72</td>
      <td>79</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>767</td>
      <td>64</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
ohe_model.score(test_ohe_df.drop("Amount", axis=1), test_ohe_df["Amount"])
```




    -0.7632751620783784



#### No Error!

That is a very bad r-squared score, but that is to be expected for truly random data like this.  The point is that we were able to make predictions on the new data, even though the categories present were not the exact same as the categories in the training data!


```python

```
