# Logistic Regression Implementation

The goal of this assignment is:
1. To demonstrate your understanding of classificationalgorithm -Logistic Regression.
2. Implementation of two versions–Binary (two possible outcomes) and Multi-variant (three or more possible outcomes) Logistic Regression.
3. Fitting into different combination of classification datasets provided.
4. 4.Using dataset of choice to compare between your implementation and version offered in scikit-learn.

## Implementation

See [LogisticRegression.py](./LogisticRegression.py)

## Restrictions

For simplicity's sake, I did not do some type check. When fitting the training and testing features(train_x and test_x), it only can accept the type of pandas.DataFrame. So before fitting and predicting a dataset, make sure the input parameters are a datafram.  

Another limitation is target's value, it can only accept the target value started from 0, and the classes have to be continuous numbers.  
For example:
* Targets containing [0,1] or [0,1,2,3,4,5,6] are okay.
* [1,2,3] is invalid because it's not started from 0
* [0,3,5,6] is also invalid because 4 classes should be [0,1,2,3]

## Comparison in 5 given datasets
See [5datasets.ipynb](./5datasets.ipynb)

## Performance Comparison
* [Binary classification](./Binary_Classification_Performace_Comparison.ipynb)
* [Multivariant classification](./Multivariant_Classification_Comparison.ipynb)
