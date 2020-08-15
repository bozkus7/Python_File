#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

df = pd.read_csv('unitedNation.csv', sep = ',')
df.head()

ax = sns.scatterplot(x="region", y="fertility", data=df)
ax
