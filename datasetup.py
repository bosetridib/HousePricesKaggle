# The Data-Setup for house price prediction, from
# kaggle's House prices competition.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the train and test data.
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# The sahpe shows both train and test have same
# number of observations and features.
print(train.shape)
print(test.shape)

# Let us check for NaN values.

missing_train = train.isnull().sum()[train.isnull().sum() != 0]
missing_test = test.isnull().sum()[test.isnull().sum() != 0]

fig, axis = plt.subplots(1,2)
fig.suptitle("Missing values distribution in test and train")

sns.violinplot(missing_train, ax=axis[0])
sns.boxplot(ax=axis[1], data=missing_test)
plt.show()

# Let us check the mutual information scores and
# VIF of each feature.

from sklearn.metrics import mutual_info_score

X_train = train.drop(columns='SalePrice')
Y_train = train['SalePrice']

mutual_info_score(X_train, Y_train)