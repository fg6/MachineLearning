import numpy as np	
from datetime import datetime
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split

from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

def get_data(file):
    df = pd.read_csv(file)
    df = df[['PassengerId','Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]
    df.Age = df.groupby(['Sex','Pclass', 'SibSp', 'Parch', 'Fare'])['Age'].apply(lambda x: x.fillna(x.mean()))
    df.Age = df.groupby(['Sex','Pclass', 'SibSp', 'Parch'])['Age'].apply(lambda x: x.fillna(x.mean()))
    df.Age = df.groupby(['Sex','Pclass', 'SibSp'])['Age'].apply(lambda x: x.fillna(x.mean()))
    df.Age = df.groupby(['Sex','Pclass'])['Age'].apply(lambda x: x.fillna(x.mean()))

    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    bins=[0,20,30,40,60,100]
    cAge = pd.cut(df.Age, bins, labels = ['0', '1', '2', '3', '4'])
    dummyAge = pd.get_dummies(cAge, prefix = "Age", drop_first = True)
    bins=[0,2,5,10,25,50,100,200,500]
    cFare = pd.cut(df.Fare, bins, labels = ['0', '1', '2', '3', '4','5','6','7'])
    dummyFare = pd.get_dummies(cFare, prefix = "Fare", drop_first = True)
    bins=[0,1,2,10]
    cParch = pd.cut(df.Parch, bins, labels = ['0','1','2'])
    dummyParch = pd.get_dummies(cParch, prefix = "Parch", drop_first = True)
    bins=[0,1,2,10]
    cSibSp = pd.cut(df.SibSp, bins, labels = ['0','1','2'])
    dummySibSp = pd.get_dummies(cSibSp, prefix = "SibSp", drop_first = True)

    #df = pd.concat([df, dummyAge, dummyFare, dummyParch,  dummySibSp ], axis=1)
    df = pd.concat([df, dummyAge, dummyParch,  dummySibSp ], axis=1)

    dummies = ['Pclass']
    df = pd.get_dummies(df, columns=dummies, prefix=dummies, drop_first = True) 

    
    ID = df['PassengerId']
    Y = df['Survived']

    df = df.drop(['PassengerId', 'Age', 'Fare', 'SibSp', 'Parch','Survived' ], axis=1 )
    X = df.as_matrix()
    
    return X, Y, ID, df

def kaggle_data(file):
    df = pd.read_csv(file)
    df = df[['PassengerId','Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
    df.Age = df.groupby(['Sex','Pclass', 'SibSp', 'Parch', 'Fare'])['Age'].apply(lambda x: x.fillna(x.mean()))
    df.Age = df.groupby(['Sex','Pclass', 'SibSp', 'Parch'])['Age'].apply(lambda x: x.fillna(x.mean()))
    df.Age = df.groupby(['Sex','Pclass', 'SibSp'])['Age'].apply(lambda x: x.fillna(x.mean()))
    df.Age = df.groupby(['Sex','Pclass'])['Age'].apply(lambda x: x.fillna(x.mean()))

    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    bins=[0,20,30,40,60,100]
    cAge = pd.cut(df.Age, bins, labels = ['0', '1', '2', '3', '4'])
    dummyAge = pd.get_dummies(cAge, prefix = "Age", drop_first = True)
    bins=[0,2,5,10,25,50,100,200,500]
    cFare = pd.cut(df.Fare, bins, labels = ['0', '1', '2', '3', '4','5','6','7'])
    dummyFare = pd.get_dummies(cFare, prefix = "Fare", drop_first = True)
    bins=[0,1,2,10]
    cParch = pd.cut(df.Parch, bins, labels = ['0','1','2'])
    dummyParch = pd.get_dummies(cParch, prefix = "Parch", drop_first = True)
    bins=[0,1,2,10]
    cSibSp = pd.cut(df.SibSp, bins, labels = ['0','1','2'])
    dummySibSp = pd.get_dummies(cSibSp, prefix = "SibSp", drop_first = True)

    df = pd.concat([df, dummyAge, dummyFare, dummyParch,  dummySibSp ], axis=1)
    
    dummies = ['Pclass']
    df = pd.get_dummies(df, columns=dummies, prefix=dummies, drop_first = True)

    ID = df['PassengerId']
    df = df.drop(['PassengerId', 'Age', 'Fare', 'SibSp', 'Parch' ], axis=1 )
    X = df.as_matrix()
    

    return X, ID


def find_best():

    clf =  AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1),learning_rate=1., algorithm= 'SAMME',  n_estimators=50)
 
    param_grid = {
        'algorithm': ['SAMME','SAMME.R'],
        'n_estimators': [50, 100, 300],
        'learning_rate': [0.1, 0.2, 0.5, 0.7, 0.8, 0.9],
        #'base_estimator': [DecisionTreeClassifier(max_depth=1)]
        }
 
    
    # Create a classifier with the parameter candidates
    CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 5)
    
    # Train the classifier on training data
    CV_rfc.fit(X_train, y_train)

    print '\n',CV_rfc.best_params_
    
    return CV_rfc.best_estimator_.n_estimators, CV_rfc.best_estimator_.algorithm, CV_rfc.best_estimator_.learning_rate


#### main ####
resize=1
findbest=0
dofit=1
predict=1
kaggle_predict=0

#del size
try: 
    size
except NameError:
    X, target, id, df = get_data('/Users/frangy/Documents/DataAna/kaggle_data/titanic_train.csv')
    kaggle_test, kid = kaggle_data('/Users/frangy/Documents/DataAna/kaggle_data/titanic_test.csv')
    size=len(target)

if resize:
   
    X_train, X_test, y_train, y_test = train_test_split(X, target, random_state=1, test_size=0.2)


if findbest:
    n_est, alg, rate = find_best()
    #{'n_estimators': 100, 'learning_rate': 0.2, 'algorithm': 'SAMME.R'}

else:
    n_est = 100
    alg = 'SAMME.R'
    rate=0.2


if dofit:
    # try next: Gradient Tree Boosting 
    clf =  AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), learning_rate=rate, algorithm= alg,  n_estimators=n_est)
    clf.fit(X_train,  y_train)
 
   
if predict:
    results = clf.predict(X_train) 
    score = metrics.accuracy_score(y_train,results)
    print("train:",score)
    scores = cross_val_score(clf, X_train,  y_train)
    print(scores.mean())
    
    results = clf.predict(X_test)
    score =  metrics.accuracy_score(y_test,results)
    print("test",score)
    scores = cross_val_score(clf, X_test,  y_test)
    print(scores.mean())

if kaggle_predict:

    results = clf.predict(kaggle_test)
    
    f = open('adaboost_prediction.txt', 'w')
    f.write('PassengerId,Survived\n')
    for i, r  in enumerate(results):
        k=kid[i]
        s = str(k)+str(',')+str(r)+str('\n')
        f.write(s)
    f.close()
