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

# Function to plot for Train and Test

fig, axis = plt.subplots(1,2)
fig.suptitle("Missing values in test and train")

sns.barplot(
    x = missing_train.index,
    y = missing_train.values,
    ax=axis[0],     # The left side
    palette="bright"
)
axis[0].set_title("Train")
axis[0].tick_params(axis = 'x', rotation=90)
axis[0].bar_label(axis[0].containers[0])

sns.barplot(
    x = missing_test.index,
    y = missing_test.values,
    ax=axis[1],     # The right side
    palette="dark"
)
axis[1].set_title("Test")
axis[1].tick_params(axis = 'x', rotation=90)
axis[1].bar_label(axis[1].containers[0])

plt.show()

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

fig, axis = plt.subplots(1,2)
fig.suptitle("FireplaceQu in test and train")

sns.histplot(
    train['FireplaceQu'],
    ax=axis[0],     # The left side
    palette="bright"
)
axis[0].set_title("Train")

sns.histplot(
    test['FireplaceQu'],
    ax=axis[1],     # The left side
    palette="bright"
)
axis[1].set_title("Test")

plt.show()

# We should add for missing

# Let us check the mutual information scores and
# VIF of each feature.

from sklearn.metrics import mutual_info_score

X_train = train.drop(columns='SalePrice')
Y_train = train['SalePrice']

mutual_info_score(X_train, Y_train)