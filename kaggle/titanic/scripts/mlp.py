import numpy as np	
from datetime import datetime
import utils


def titanic_data(file, train):
    import pandas as pd
    # columns:
    if train:
        t=0         # target column is first
    else:         
        t=-1        # no target column

    df = pd.read_csv(file)

    df['FamilySize'] =  df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 1 #initialize to yes/1 is alone
    df['IsAlone'] = df['IsAlone'].loc[df['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1
    
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    Embark_location = pd.get_dummies(df['Embarked'].fillna('0'), drop_first=True)
    df = pd.concat([df, Embark_location], axis=1)
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
    df['FareBin'] = pd.qcut(df['Fare'], 4)
    df.Age = df.groupby(['Sex','Pclass'])['Age'].apply(lambda x: x.fillna(x.mean()))
    df['AgeBin'] = pd.cut(df['Age'].astype(int), 5)

    stat_min = 10 
    df['Title'] = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    title_names = (df['Title'].value_counts() < stat_min) #this will create a true false series with title name as index
    df['Title'] = df['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
    title = pd.get_dummies(df['Title'], drop_first=True)
    df = pd.concat([df, title], axis=1)


    from sklearn.preprocessing import LabelEncoder
    label = LabelEncoder()
    df['Sex_Code'] = label.fit_transform(df['Sex'])
    df['Embarked_Code'] = label.fit_transform(df['Embarked'].fillna('0'))
    df['Title_Code'] = label.fit_transform(df['Title'])
    df['AgeBin_Code'] = label.fit_transform(df['AgeBin'])
    df['FareBin_Code'] = label.fit_transform(df['FareBin'])
 

    df = df.drop(['SibSp','Parch'], 1)
    df = df.drop(['Title'], axis=1)
    df = df.drop(['Name','Ticket','Cabin'], 1)
    df = df.drop(['Age','Fare'],1)
    df = df.drop(['Embarked', 'FareBin', 'AgeBin'],1)


    if train:
        y =  df['Survived']
        df.drop(['Survived','PassengerId'], axis=1, inplace=True)
        X = df.as_matrix()
        
        return X, y
    else:
        ID = df['PassengerId']
        X = df.as_matrix()
        return X, ID



#### main ####
findbest=0
kaggle_predict=0

#del size
#try: 
#    size
#except NameError:
X, y = titanic_data('/Users/frangy/Documents/DataAna/kaggle_data/titanic_train.csv', 1)
kaggle_test, kid = titanic_data('/Users/frangy/Documents/DataAna/kaggle_data/titanic_test.csv', 0)
size=len(y)

test_perc=0.2  # perc size of test sample
X_train, X_test, y_train, y_test = utils.resize(X, y, test_perc)
 
#if findbest:
    #n_est, maxf, crit, min_s_leaf = utils.best_randforest_cl(X, y)
#else:
    #n_est = 100 
    #maxf = 'sqrt'
    #crit = 'gini'
    #min_s_leaf = 3


clf_rf = utils. mlp_cl(X_train, y_train)

y_pred = clf_rf.predict(X_train)
print('Training score:')
utils.get_accuracy_cl(y_pred, y_train)
    
y_pred = clf_rf.predict(X_test)
print('Testing score:')
utils.get_accuracy_cl(y_pred, y_test)


if kaggle_predict:

    results = clf_rf.predict(kaggle_test)

    f = open('mlp.txt', 'w')
    f.write('PassengerId,Survived\n')
    for i, r  in enumerate(results):
        k=kid[i]
        s = str(k)+str(',')+str(r)+str('\n')
        f.write(s)
    f.close()

