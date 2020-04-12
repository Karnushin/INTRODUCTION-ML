# -*- coding: utf-8 -*-
import pandas as pd
import re

data = pd.read_csv('titanic.csv', index_col='PassengerId')

print('\nMan and woman count')
print('men =',data.loc[data.Sex == 'male', 'Sex'].count())
print('women =', (data.Sex == 'female').sum())

print('\nPart of survived people')
print('part =', round(data.Survived.sum() / data.shape[0] * 100, 2))

print('\nPassengers of 1st class')
print('1st class =', round((data.Pclass == 1).sum() / data.shape[0] * 100,2))

print("\nMean and median of people's age")
print('mean =',round(data.Age.mean(), 2))
print('median =',data.Age.median())

print('\nСorrelation of Pearson between SibSp and Parch')
print('cor =', round(data['SibSp'].corr(data['Parch']), 2))

print('\nMost popular female name')
fn = data[data['Sex'] == 'female']['Name']
def extract_first_name(name):
    # первое слово в скобках
    m = re.search('\(([^\s]+).+\)', name)
    if m:
        return m.group(1)
    # первое слово после Mrs. or Miss. or else
    else:
        m = re.search(".*\.\s*([\w]*)", name)
        return m.group(1)

# получаем имя с максимальной частотой
print(fn.map(lambda full_name: extract_first_name(full_name)).value_counts().idxmax())
