import pandas as pd
df = pd.read_csv('C:/Users/Lenovo X260/Desktop/data/quest2.csv')
df1 = pd.DataFrame.drop_duplicates(df)
df2 = pd.DataFrame(df1)
df2['score'] = df2['score'].apply(lambda x : int(x * 2))
csv = 'C:/Users/Lenovo X260/Desktop/data/question2.csv'
df2.to_csv(csv,index=False)
print(len(df1))