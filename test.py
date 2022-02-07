import pandas as pd
import json
df = pd.read_csv('coords.csv')
print(sum(df['class']=='V'))

dic = {
    'lr':1
}
dicstr = "{'lr':1}"
print(dicstr)
print(type(dicstr))
dicstr = dicstr.replace("'", "\"")
res = json.loads(dicstr)
print(res)
print(type(res))
print(res['lr'])
print(dic)
print(type(dic))
print('contoh:', dic['lr'])

dflr = pd.read_csv('akurasi logistic regression.csv')
print(dflr)
