# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

dfAll = pd.read_csv('titanic.csv', index_col='PassengerId')

#оставляем в выборке 4 интересующих нас признака + 1 - целевой
features = dfAll[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]

#выбрасывает те строки, в которых в колонке Age содержится Nan
features = features.dropna(subset=['Age'])
#features.reset_index(drop=True, inplace=True)

#выделяем целевую переменную
y = features['Survived']
#удаляем колонку, которая не относится к признакам
features.drop(columns=['Survived'], inplace=True)

#заменяем пол на числовые знаения, чтобы суметь обучить модель, так как со строками нельзя !!
features.loc[features.Sex == 'male', 'Sex'] = 1
features.loc[features.Sex == 'female', 'Sex'] = 0

#переходим к самому обучению
clf = DecisionTreeClassifier(random_state=241)
clf.fit(features, y)
#получим показатель важности признаков от 0 до 1
importance = clf.feature_importances_

print(importance)
