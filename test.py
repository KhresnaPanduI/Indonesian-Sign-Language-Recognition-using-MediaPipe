import pandas as pd
df = pd.read_csv('coords.csv')
print(sum(df['class']=='V'))