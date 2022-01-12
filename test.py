import pandas as pd
df = pd.read_csv('coords.csv')
print(sum(df['class']=='V'))

dic = {
    'lr':1,
    'rf':2
}

print('contoh:', dic['lr'])