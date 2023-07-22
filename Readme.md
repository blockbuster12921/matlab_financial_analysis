# Financial Analysis with MATLAB

## Convert categorical variables to dummy variables
Because some functions such as corr(), ridge() and lassoglm() cannot deal with categorical predictors

## GLS
Estimate a logistical model using the trainTbl (i.e., contains categorical variables)

## Stepwise regression models
Fit a stepwise regression model using only the top10 predictors you have found in the covariance analysis above.

## Lasso
Model estimation using training data lasso cannot handle categorical predictors

## Elastic Net
Do the same analysis as you did with the lasso, but use the elastic net model with alpha parameter set to 0.5.

## Bagging
Fit an ensemble model using bagging algo (using a tree as the weak learner. output the estimated model as 'bagMdl'

## Boosting
Fit a boosting ensemble model using a tree as the weak learner. It is very similar to the procedure above. Save MSE for the training and testing sample as boostMSE_train and boostMSE_test, respectively. Find the top 10 important predictors.