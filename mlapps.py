from datasetup import *
print(vif_train)
print(mi_train_score)

X_train = train.drop(columns='SalePrice')
Y_train = train['SalePrice']

from sklearn.ensemble import RandomForestRegressor
random_forest_model = RandomForestRegressor(n_estimators=200, max_depth=4, random_state=0)
random_forest_model.fit(X_train.select_dtypes(['int64', 'float64']), Y_train.values)
Y_test = random_forest_model.predict(test.select_dtypes(['int64', 'float64']))

Y_test = pd.DataFrame(Y_test, index=test.index, columns=['SalePrice'])
Y_test.to_csv('submission.csv')

