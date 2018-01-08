import numpy as np	
from datetime import datetime
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import Imputer
from sklearn.metrics import accuracy_score, classification_report

# Create a custom method to assign the age
def age_approx(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return pclass_1_mean
        elif Pclass == 2:
            return pclass_2_mean
        else:
            return pclass_3_mean
    else:
        return float(Age)
    
def get_data(file):
    df = pd.read_csv(file)
   
    df = df.drop(['Name','Ticket','Cabin'], 1)

    pclass_means = df.groupby(['Sex', 'Pclass']).mean()
    pmv = pclass_means['Age'].values
    pclass_1F_mean = float(pmv[0])
    pclass_2F_mean = float(pmv[1])
    pclass_3F_mean = float(pmv[2])
    pclass_1M_mean = float(pmv[3])
    pclass_2M_mean = float(pmv[4])
    pclass_3M_mean = float(pmv[5])


    df['Age'] = df[['Age', 'Pclass', 'Sex']].apply(age_approx, axis=1)
    df.dropna(inplace=True)
    # Gender
    gender = pd.get_dummies(df['Sex'], drop_first=True)
    embark_location = pd.get_dummies(df['Embarked'], drop_first=True)
    df.drop(['Sex', 'Embarked'], axis=1, inplace=True)
    df = pd.concat([df, gender, embark_location], axis=1)
    
    ID = df['PassengerId']
    df.drop(['PassengerId', 'Fare', 'Pclass'], axis=1, inplace=True) # these are correlated, why drop both though?    
    Y = df['Survived']
    
    df.drop(['Survived'], axis=1, inplace=True)
    X = df.as_matrix()

    return X, Y, ID, df



def kaggle_data(file):
    df = pd.read_csv(file)
    df = df.drop(['Name','Ticket','Cabin'], 1)

    pclass_means = df.groupby(['Sex', 'Pclass']).mean()
    pmv = pclass_means['Age'].values
    pclass_1F_mean = float(pmv[0])
    pclass_2F_mean = float(pmv[1])
    pclass_3F_mean = float(pmv[2])
    pclass_1M_mean = float(pmv[3])
    pclass_2M_mean = float(pmv[4])
    pclass_3M_mean = float(pmv[5])


    df['Age'] = df[['Age', 'Pclass', 'Sex']].apply(age_approx, axis=1)
    #df.dropna(inplace=True)
    # Gender
    gender = pd.get_dummies(df['Sex'], drop_first=True)
    embark_location = pd.get_dummies(df['Embarked'], drop_first=True)
    df.drop(['Sex', 'Embarked'], axis=1, inplace=True)
    df = pd.concat([df, gender, embark_location], axis=1)
    
    ID = df['PassengerId']
    df.drop(['PassengerId', 'Fare', 'Pclass'], axis=1, inplace=True) # these are correlated, why drop both though?    
    X = df.as_matrix()

    
    return X, ID, df


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

    print('\n',CV_rfc.best_params_)
    
    return CV_rfc.best_estimator_.n_estimators, CV_rfc.best_estimator_.max_features,CV_rfc.best_estimator_.criterion #,  CV_rfc.best_estimator_.min_samples_leaf


    

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
    kaggle_test, kid, kdf = kaggle_data('/Users/frangy/Documents/DataAna/kaggle_data/titanic_test.csv')
    size=len(target)

if resize:
   
    X_train, X_test, y_train, y_test = train_test_split(X, target, random_state=1, test_size=0.3)


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
    print("Rand Forest Accuracy Score: %.3f" % accuracy_score(y_train, results))

    results = clf.predict(X_test)
    score =  metrics.accuracy_score(y_test,results)
    print("Rand Forest Accuracy Score: %.3f" % accuracy_score(y_test, results))
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


    # Plot the multi-label confusion matrix
cm = metrics.confusion_matrix(y_test,y_pred)
fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111)
ax.set_aspect(1)
col = [1,2,3,4]
imcm = ax.imshow(cm, cmap="Set2") #, origin='lower')

labels = ["Correct", "False Positive", "False Negative", "Correct"]

width, height = cm.shape

i=0
for x in range(width):
    for y in range(height):
        ax.annotate(str(cm[x][y]), xy=(y, x), 
                    horizontalalignment='center',
                    verticalalignment='center', color="white", size=30)
        ax.annotate(labels[i], xy=(y, x-0.3), 
                    horizontalalignment='center',
                    verticalalignment='center', color="white", size=14)
        i+=1
ax.set_xticks([0,1])
ax.set_yticks([0,1])
ax.set_xlabel("Predicted")
ax.set_ylabel("Truth")


#print(CV_rfc.cv_results_)
matrix = CV_rfc.cv_results_['mean_test_score'].reshape(len(param_grid["n_estimators"]),len(param_grid["min_samples_leaf"]))
plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.hot)
#plt.xlabel('n_estimators')
#plt.ylabel('min_samples_leaf')
plt.colorbar()
plt.xticks(np.arange(len(param_grid['n_estimators'])), param_grid['n_estimators'])
plt.yticks(np.arange(len(param_grid['min_samples_leaf'])), param_grid['min_samples_leaf'])
plt.title('Grid Search AUC Score')
plt.show()


