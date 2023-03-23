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
sns.scatterplot(missing_train, ax=axis[0])
axis[0].set_title("Train")
sns.scatterplot(missing_test, ax=axis[1])
axis[1].set_title("Test")
plt.xticks(90)
plt.show()

# Let us check the mutual information scores and
# VIF of each feature.

from sklearn.metrics import mutual_info_score

X_train = train.drop(columns='SalePrice')
Y_train = train['SalePrice']

mutual_info_score(X_train, Y_train)