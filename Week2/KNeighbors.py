import pandas as pd
#import sklearn

df = pd.read_csv('wine.csv', header=None)
classAlc = df[[0]]
features = df[df.columns[1:]]



