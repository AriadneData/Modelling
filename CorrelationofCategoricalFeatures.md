# Correlation and Categorical Features



Contingency tables or cross tabulation display the multivariate frequency distribution of variable.

![TwoWayContingencyTable](C:\Users\AMAND\BT Cloud\gitModelling\images\TwoWayContingencyTable.jpg)

The chi-squared distribution can show whether interdependence according to a statistical significance. However, there are several tests that can give the strength of the association.



### Cramer's V

This is symmetrical , it is insensitive to swapping *x* and *y*.

````
import scipy.stats as ss
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))
````



### Theil's U (Uncertainty Coefficient)

This is asymmetrical and preserves information. Referring to the table below, if the value of *x* is known, the value of *y* still can’t be determined, but if the value of *y* is known — then the value of *x* is guaranteed. This valuable information is lost when using Cramer’s V due to its symmetry, so to preserve it we need an *asymmetric* measure of association between categorical features. 

![TheilsExplanation](C:\Users\AMAND\BT Cloud\gitModelling\images\TheilsExplanation.png)



[Theil’s U](https://en.wikipedia.org/wiki/Uncertainty_coefficient), also referred to as the Uncertainty Coefficient, is based on the *conditional entropy* between *x* and *y —* given the value of *x*, how many possible states does *y* have, and how often do they occur.



````python
def theils_u(x, y):
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x
````



### Categorical to Continuous feature

The Correlation Ratio is often marked by eta.  It is defined as the weighted variance of the mean of each category divided by the variance of all samples. *Given a continuous number, how well can we know to which category it belongs to?*

````python
def correlation_ratio(categories, measurements):
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator/denominator)
    return eta
````





#### References

https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9

