from datasetup import *
print(vif_train)
print(mi_train_score)

X_train = train.drop(columns='SalePrice')
Y_train = train[['SalePrice']]

from statsmodels.api import Logit
X_train_logit = X_train.select_dtypes(['int64','float64'])
logit_model = Logit(Y_train, X_train_logit).fit()

logit_model(X_train_logit,Y_train)