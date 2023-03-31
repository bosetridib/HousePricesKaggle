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

# Print the datatype comparisons.
print("\t Train \t Test",
    "".join(
        "\n{}\t{}\t{}".format(x,y,z) for x,y,z in zip(
            train.dtypes.value_counts().index,
            train.dtypes.value_counts(),
            test.dtypes.value_counts()
        )
    )
)

# Function to check for NaN values.

def missing_values():
    return (
        train.isnull().sum()[train.isnull().sum() != 0],
        test.isnull().sum()[test.isnull().sum() != 0]
    )
missing_train, missing_test = missing_values()

# Plot for Train and Test
# sns_plot(missing_train, missing_test, 'barplot')

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
# Set the Id as index
train.set_index('Id', inplace=True)
test.set_index('Id', inplace=True)

# Update missing values
missing_train, missing_test = missing_values()
# sns_plot(missing_train, missing_test, 'barplot')

# LotFrontage can be imputed with KNN, while
# FireplaceQu would be transformed into a cateogory.
# sns_plot(train['LotFrontage'], test['LotFrontage'], 'histplot')

# Both plots are seemingly normal distribution
# family, and hence 4 neighbours are used.

from sklearn.impute import KNNImputer
global_KNNimputer = KNNImputer(n_neighbors=4)
train['LotFrontage'] = global_KNNimputer.fit_transform(train[['LotFrontage']])
test['LotFrontage'] = global_KNNimputer.fit_transform(test[['LotFrontage']])

# We should add NA as a category for the missing data.
# Note that the inplace option gives warnings.
for i in train.select_dtypes('object').columns:
    train[i].fillna('NA', inplace=True)
missing_train, missing_test = missing_values()
print(missing_train, missing_test)
test[
    test.select_dtypes('object').columns
] = test[
    test.select_dtypes('object').columns
].fillna('NA')

# Now we should convert the column to a categorical
train[
    train.select_dtypes('object').columns
] = train[
    train.select_dtypes('object').columns
].astype('category')

test[
    test.select_dtypes('object').columns
] = test[
    test.select_dtypes('object').columns
].fillna('NA')

train[train.select_dtypes('object').columns] = train[train.select_dtypes('object').columns].astype('category')
test['FireplaceQu'] = test['FireplaceQu'].astype('category')

# Update the missing values
missing_train, missing_test = missing_values()

sns_plot(missing_train, missing_test, 'barplot')

# Even after resolving the big missing values, we still
# can not perform dropna.
print(100 - (100*train.dropna().shape[0]/train.shape[0]), '%')
print(100 - (100*test.dropna().shape[0]/test.shape[0]), '%')
# Around 10% of dataloss will be caused with dropna.

# Let us check the mutual information scores and
# VIF of each feature.

from sklearn.metrics import mutual_info_score

X_train = train.drop(columns='SalePrice')
Y_train = train['SalePrice']

mutual_info_score(X_train, Y_train)