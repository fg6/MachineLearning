import numpy as np	
from datetime import datetime
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import Imputer

def get_data(file):
    df = pd.read_csv(file)
    df1 = df.copy()
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

    df = df.drop(['PassengerId', 'Age', 'Fare', 'SibSp', 'Parch', 'Survived' ], axis=1 )
    
    #X = df.drop('Survived', axis=1).as_matrix()
    X = df.as_matrix()

    print('train', df.head())
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

    #df = pd.concat([df, dummyAge, dummyFare, dummyParch,  dummySibSp ], axis=1)  
    df = pd.concat([df, dummyAge, dummyParch,  dummySibSp ], axis=1)  

    dummies = ['Pclass']
    df = pd.get_dummies(df, columns=dummies, prefix=dummies, drop_first = True) 

    ID = df['PassengerId']
    df = df.drop(['PassengerId', 'Age', 'Fare', 'SibSp', 'Parch' ], axis=1 )
    X = df.as_matrix()
    
    print('kaggle', df.head())
    return X, ID


def find_best():

    clf =  RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' , n_estimators=50, oob_score = True) 
    # min_sample_leaf >50?

    param_grid = {        
        'n_estimators': [200, 700, 1000],
        'max_features': ['auto', 'sqrt', 'log2'],
        'criterion': ['gini', 'entropy'],
        'min_samples_leaf': [1, 5, 10]
        }
 
    
    # Create a classifier with the parameter candidates
    CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 5)
    
    # Train the classifier on training data
    CV_rfc.fit(X_train, y_train)

    print '\n',CV_rfc.best_params_
    
    return CV_rfc.best_estimator_.n_estimators, CV_rfc.best_estimator_.max_features, CV_rfc.best_estimator_.criterion #,  CV_rfc.best_estimator_.min_sample_leaf



#### main ####
resize=1
findbest=0
dofit=1
predict=1
kaggle_predict=1

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
    n_est, maxf, crit, min_s_leaf = find_best()
    #{'max_features': 'auto', 'n_estimators': 200, 'criterion': 'gini', 'min_samples_leaf': 10}

else:
    n_est = 200 
    maxf = 'auto'
    crit = 'gini'
    min_s_leaf = 10


if dofit:
        
    clf =  RandomForestClassifier(n_jobs=-1, max_features= maxf, n_estimators=n_est, criterion = crit, min_samples_leaf = min_s_leaf, oob_score = True) 
    clf.fit(X_train,  y_train)
    
   
if predict:
    results = clf.predict(X_train) 
    score = metrics.accuracy_score(y_train,results)
    print("train:",score)

    results = clf.predict(X_test)
    score =  metrics.accuracy_score(y_test,results)
    print("test",score)


if kaggle_predict:

    results = clf.predict(kaggle_test)

    f = open('randforest_prediction.txt', 'w')
    f.write('PassengerId,Survived\n')
    for i, r  in enumerate(results):
        k=kid[i]
        s = str(k)+str(',')+str(r)+str('\n')
        f.write(s)
    f.close()
