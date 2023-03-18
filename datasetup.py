import pandas as pd
import seaborn as sns

train = pd.read_csv('train.csv')
print(train.columns.value_counts())