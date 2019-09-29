# Feature Engineering Techniques

### Label Encoding 
- Always encode train and test together
- Once encode columns can be downcast to reduce space

``

```python
df[col],_ = df[col].factorize()

if df[col].max()<128: df[col] = df[col].astype('int8')
elif df[col].max()<32768: df[col] = df[col].astype('int16')
else: df[col].astype('int32')
```



### Splitting

- String features can be split to create group of features

- Numeric features can be split e,g, into integer and decimal points

  

### Combining / Transforming / Interaction

Two or more string or numeric columns can be combined to create a correlation with the target variable.



## Frequency Encoding

Shows how common or rare a value is. This can used to help look for outliers

``

```python
temp = df['col'].value_counts().to_dict()
df['col_counts'] = df['col'].map(temp)
```



### Aggregations and Group Statistics

Can help determine if values are common or rare for a group:

``

```python
temp = df.groupby('col')['TransactionAmt'].agg(['mean'])   
    .rename({'mean':'TransactionAmt_col_mean'},axis=1)
df = pd.merge(df,temp,on='col',how='left')
```







##### References

https://www.kaggle.com/c/ieee-fraud-detection/discussion/108575
