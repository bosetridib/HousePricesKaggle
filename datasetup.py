# The Data-Setup for house price prediction, from
# kaggle's House prices competition.

import pandas as pd
from snsplot import *

# Load the train and test data.
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# The sahpe shows both train and test have same
# number of observations and features.
print(train.shape)
print(test.shape)

# Function to check for NaN values.

def missing_values():
    return (
        train.isnull().sum()[train.isnull().sum() != 0],
        test.isnull().sum()[test.isnull().sum() != 0]
    )
missing_train, missing_test = missing_values()

# Plot for Train and Test
sns_plot(missing_train, missing_test, 'barplot')

# Alley, PoolQC, Fence, MiscFeature : these features
# should be dropped in both train and test.

train.drop(
    columns=['Alley', 'PoolQC', 'Fence', 'MiscFeature'],
    inplace=True
)
test.drop(
    columns=['Alley', 'PoolQC', 'Fence', 'MiscFeature'],
    inplace=True
)
# Update missing values
missing_train, missing_test = missing_values()

sns_plot(missing_train, missing_test, 'barplot')

# Lets check out the behavior of LotFrontage and
# FireplaceQu.

print(train[['LotFrontage', 'FireplaceQu']].info())

# LotFrontage can be imputed with KNN, while
# FireplaceQu would be transformed into a cateogory.
sns_plot(train['LotFrontage'], test['LotFrontage'], 'histplot')

# Both plots are seemingly normal distribution
# family, and hence 4 neighbours are used.

from sklearn.impute import KNNImputer
global_KNNimputer = KNNImputer(n_neighbors=4)
train['LotFrontage'] = global_KNNimputer.fit_transform(train[['LotFrontage']])
test['LotFrontage'] = global_KNNimputer.fit_transform(test[['LotFrontage']])

# Now we check for FireplaceQu.
sns_plot(train['FireplaceQu'], test['FireplaceQu'], 'histplot')

# We should add NA as a category for the missing data.
train['FireplaceQu'].fillna('NA', inplace = True)
test['FireplaceQu'].fillna('NA', inplace = True)

# Now we should convert the column to a categorical
train['FireplaceQu'] = train['FireplaceQu'].astype('category')
test['FireplaceQu'] = test['FireplaceQu'].astype('category')

# Update the missing values
missing_train, missing_test = missing_values()

sns_plot(missing_train, missing_test, 'barplot')

# Even after resolving the big missing values, we still
# can not perform dropna.
print(100 - (100*train.dropna().shape[0]/train.shape[0]), '%')
print(100 - (100*test.dropna().shape[0]/test.shape[0]), '%')
# Around 10% of dataloss will be caused with dropna.

# Print the datatypes, and convert strings to category.
print(train.dtypes.value_counts(), '\n\n', test.dtypes.value_counts())

# Let us check the mutual information scores and
# VIF of each feature.

from sklearn.metrics import mutual_info_score

X_train = train.drop(columns='SalePrice')
Y_train = train['SalePrice']

mutual_info_score(X_train, Y_train)