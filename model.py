# -*- coding: utf-8 -*-
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

    
train = pd.read_csv('train.csv')
test =  pd.read_csv('test.csv')
nt = test;
enc = LabelEncoder()
enc.fit(train['Sex'])
train['Sex'] = enc.transform(train['Sex'])

enc.fit(test['Sex'])
test['Sex'] = enc.transform(test['Sex'])

train = train.drop(['Cabin','PassengerId','Name','Ticket'], axis=1)
train = train.fillna(method='ffill')
enc.fit(train['Embarked'])
train['Embarked'] = enc.transform(train['Embarked'])
test = test.drop(['Cabin','PassengerId','Name','Ticket'], axis=1)
test = test.fillna(method='ffill')
enc.fit(test['Embarked'])
test['Embarked'] = enc.transform(test['Embarked'])

xgb = XGBClassifier()
xgb.fit(train.drop('Survived',axis=1), train['Survived'])

predictions = xgb.predict(test)

pd.DataFrame({
            'PassengerId': nt['PassengerId'],
            'Survived': predictions
        }).to_csv('result.csv', index=False)