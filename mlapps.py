from datasetup import *
print(vif_train[vif_train<=2])
print(mi_train_score[mi_train_score<2])

X_train = train.drop(columns='SalePrice')
Y_train = train['SalePrice']

from sklearn.ensemble import RandomForestRegressor
random_forest_model = RandomForestRegressor(n_estimators=200, max_depth=4, random_state=0)
random_forest_model.fit(X_train.select_dtypes(['int64', 'float64']), Y_train.values)
Y_test = random_forest_model.predict(test.select_dtypes(['int64', 'float64']))

Y_test = pd.DataFrame(Y_test, index=test.index, columns=['SalePrice'])
Y_test.to_csv('submission.csv')


# Hyperparameter optimization (Manual with for loop)
rmse = []

for i,j,k in [(i,j,k) for i in range(1,201,50) for j in range(1,11) for k in range(2,11)]:
    
    random_forest_model = RandomForestRegressor(n_estimators=i, max_depth=j, max_leaf_nodes=k)
    random_forest_model.fit(X_train.select_dtypes(['int64', 'float64']), Y_train.values)
    
    Y_train_est = random_forest_model.predict(X_train.select_dtypes(['int64', 'float64']))
    
    rmse.append(
        {
            'rmse':(sum( (Y_train.values - Y_train_est)**2 ) / len(Y_train))**0.5,
            'i':i, 'j':j, 'k':k
        }
    )

rmse = pd.DataFrame(rmse)
[[r,i,j,k]] = rmse[rmse['rmse'] == rmse['rmse'].min()].values.tolist()
i,j,k = int(i),int(j),int(k)
random_forest_model = RandomForestRegressor(n_estimators=i, max_depth=j, max_leaf_nodes=k)
del rmse,r,i,j,k
random_forest_model.fit(X_train.select_dtypes(['int64', 'float64']), Y_train.values)
Y_test = random_forest_model.predict(test.select_dtypes(['int64', 'float64']))

Y_test = pd.DataFrame(Y_test, index=test.index, columns=['SalePrice'])
Y_test.to_csv('submission.csv')


# Gradient Boosting Regressor

from sklearn.ensemble import GradientBoostingRegressor
rmse = []

for i,j,k in [(i,j,k) for i in range(100,3100,500) for j in range(1,21,4) for k in range(3,21,3)]:
    
    gradient_boost_model = GradientBoostingRegressor(n_estimators=i, max_depth=j, min_samples_leaf=k)
    gradient_boost_model.fit(X_train.select_dtypes(['int64', 'float64']), Y_train.values)
    
    Y_train_est = gradient_boost_model.predict(X_train.select_dtypes(['int64', 'float64']))
    
    rmse.append(
        {
            'rmse':(sum( (Y_train.values - Y_train_est)**2 ) / len(Y_train))**0.5,
            'i':i, 'j':j, 'k':k
        }
    )

rmse = pd.DataFrame(rmse)
[[r,i,j,k]] = rmse[rmse['rmse'] == rmse['rmse'].min()].values.tolist()
i,j,k = int(i),int(j),int(k)
gradient_boost_model = GradientBoostingRegressor(n_estimators=i, max_depth=j, max_leaf_nodes=k)
del rmse,r,i,j,k
gradient_boost_model.fit(X_train.select_dtypes(['int64', 'float64']), Y_train.values)
Y_test = gradient_boost_model.predict(test.select_dtypes(['int64', 'float64']))

Y_test = pd.DataFrame(Y_test, index=test.index, columns=['SalePrice'])
Y_test.to_csv('submission.csv')


# One-hot encoding

# The more top frequency closer to the total frequency, the more the variable is useless.
print(X_train.select_dtypes(['category']).describe().iloc[:,:10])
X_train.drop(columns=['Street','Utilities','Condition2'], inplace=True)
print(X_train.select_dtypes(['category']).describe().iloc[:,10:20])
X_train.drop(columns=['RoofMatl'], inplace=True)
print(X_train.select_dtypes(['category']).describe().iloc[:,20:30])
X_train.drop(columns=['Heating'], inplace=True)
print(X_train.select_dtypes(['category']).describe().iloc[:,30:])

X_train_encoded = pd.get_dummies(X_train['ExterQual']).astype('int')