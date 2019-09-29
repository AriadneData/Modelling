﻿﻿# Covariate Shift

Models maybe trained on data that has different distribution for validation datasets. This **dataset shift (or drifting)** can reduce the predictive power of models.


### Types of Datashift
Dataset shift could be divided into three types:

- Shift in the independent variables (Covariate Shift)
- Shift in the target variable (Prior probability shift)
- Shift in the relationship between the independent and the target variable (Concept Shift)

### Steps to identify drift
The basic steps that we will follow are:

1. Preprocessing: This step involves imputing all missing values and label encoding of all categorical variables.
2. Creating a random sample of your training and test data separately and adding a new feature origin which has value train or test depending on whether the observation comes from the training dataset or the test dataset.
3. Now combine these random samples into a single dataset. Note that the shape of both the samples of training and test dataset should be nearly equal, otherwise it can be a case of an unbalanced dataset.
4. Now create a model taking one feature at a time while having ‘origin’ as the target variable on a part of the dataset (say ~75%).
5. Now predict on the rest part(~25%) of the dataset and calculate the value of AUC-ROC.
6. Now if the value of AUC-ROC for a particular feature is greater than **0.80**, we classify that feature as drifting.
7. You can also manually check their difference in distribution through some visualisation or by using 1-way ANOVA test.

### Example Code
[Example Covariate Shift](CovariateShiftExample.md)



### Dropping of drifting features
Features having a drift value greater than 0.8 and are not important in our model, we drop them.
Before dropping features ensure there it isn't possible to create further features.




#### References
https://www.analyticsvidhya.com/blog/2017/07/covariate-shift-the-hidden-problem-of-real-world-data-science/


