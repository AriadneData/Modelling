# LightGBM 

**Light GBM grows tree vertically** while other algorithm grows trees horizontally meaning that Light GBM grows tree **leaf-wise** while other algorithm grows level-wise. Compared with depth-wise growth, the leaf-wise algorithm can converge much faster. However, the leaf-wise growth may be over-fitting if not used with the appropriate parameters. Can also use GPU

#### Disadvantage:

Light GBM is **sensitive to overfitting** and can easily overfit small data. Their is no threshold on the number of rows but my experience suggests me to use it only for data with 10,000+ rows.



#### Main Parameters

1. **num_leaves** - the main parameter effecting the complexity. This should be less than 2^(max_depth).  	
2. **min_data_in_leaf** - important to prevent overfitting.  Setting to 100s to 1000s is enough for a large dataset.
3. **max_depth** - sets the depth of the tree 	

#### For Faster Speed

- Use bagging by setting `bagging_fraction` and `bagging_freq`
- Use feature sub-sampling by setting `feature_fraction`
- Use small `max_bin`

#### For Better Accuracy

- Use large `max_bin` (may be slower)
- Use small `learning_rate` with large `num_iterations`
- Use large `num_leaves` (may cause over-fitting)
- Use bigger training data

#### Over-fitting

- Use small `max_bin`
- Use small `num_leaves`
- Use `min_data_in_leaf` and `min_sum_hessian_in_leaf`
- Use bagging by set `bagging_fraction` and `bagging_freq`
- Use feature sub-sampling by set `feature_fraction`
- Use bigger training data
- Try `lambda_l1`, `lambda_l2` and `min_gain_to_split` for regularization
- Try `max_depth` to avoid growing deep tree

#### I/O Parameters

**categorical_feature:** It denotes the index of categorical features. If categorical_features=0,1,2 then column 0, column 1 and column 2 are categorical variable

**ignore_column:** same as categorical_features just instead of considering specific columns as categorical, it will completely ignore them.

**save_binary:** If you are really dealing with the memory size of your data file then specify this parameter as ‘True’. Specifying parameter true will save the dataset to binary file, this binary file will speed your data reading time for the next time

#### Other Useful

**early_stopping_round:** reduces the number of iterations





## Feature Engineering

### Missing values

If you give np.nan to LGBM, then at each tree node split, it will split the non-NAN values and then send all the NANs to either the left child or right child depending on what’s best. Therefore NANs get special treatment at every node and can become overfit. By simply converting all NAN to a negative number lower than all non-NAN values (such as - 999) then LGBM will no longer overprocess NAN. Instead it will give it the same attention as other numbers. Try both ways and see which gives the highest CV.


### Categorical Features
LGBM can handle both categorical and numeric features. Use .as('category') to try both types


##### References
https://www.kaggle.com/c/ieee-fraud-detection/discussion/108575

https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html

https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc