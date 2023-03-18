# The Data-Setup for house price prediction, from
# kaggle's House prices competition.

import pandas as pd
import seaborn as sns

# Load the train and test data.
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.apply(len)
test.apply(len)