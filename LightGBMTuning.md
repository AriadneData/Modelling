# LightGBM 

### Missing values
If you give np.nan to LGBM, then at each tree node split, it will split the non-NAN values and then send all the NANs to either the left child or right child depending on what’s best. Therefore NANs get special treatment at every node and can become overfit. By simply converting all NAN to a negative number lower than all non-NAN values (such as - 999) then LGBM will no longer overprocess NAN. Instead it will give it the same attention as other numbers. Try both ways and see which gives the highest CV.


### Categorical Features
LGBM can handle both categorical and numeric features. Use .as('category') to try both types


##### References
https://www.kaggle.com/c/ieee-fraud-detection/discussion/108575