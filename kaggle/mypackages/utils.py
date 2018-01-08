class of(float):
    def __str__(self):
        return "%0.1f" % self.real
class tf(float):
    def __str__(self):
        return "%0.3f" % self.real
 

def resize(X, target, test_perc):
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(X, target, random_state=1, test_size = test_perc)   

    return X_train, X_test, y_train, y_test


def def_randforest_cl(x, y):
    from sklearn.ensemble import RandomForestClassifier
    print("\n Fitting with Random Forest")
    clf =  RandomForestClassifier() 
    clf.fit(x, y)
    return(clf)

def randforest_cl(x, y, n_est, maxf, crit, min_s_leaf):
    from sklearn.ensemble import RandomForestClassifier
    print("\n Fitting with Random Forest")
    clf =  RandomForestClassifier(n_jobs=-1, max_features= maxf, n_estimators=n_est, class_weight='balanced',
                                    criterion = crit, min_samples_leaf = min_s_leaf, oob_score = True) 
    clf.fit(x, y)
    return(clf)

def svm_rbf(x,y): # super slow
    from sklearn.svm import SVC
    print("\n Fitting with SVC")
    
    clf = SVC(kernel='rbf')
    clf.fit(x, y)
    return(clf)

def mlp_cl(x,y, hidden_layer_sizes, activation, solver, learning_rate):
    print("\n Fitting with MLP")
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes = hidden_layer_sizes, activation = activation, solver = solver,learning_rate = learning_rate)
    clf.fit(x, y)
    return(clf)
   

def best_mlp_cl(x,y):
    
    from sklearn.model_selection  import GridSearchCV
    from sklearn.neural_network import MLPClassifier

    hidden_layer_sizes = (100, )
    activation = "relu"
    solver = "sgd"
    learning_rate ="constant"
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver,learning_rate=learning_rate)


    param_grid = {
        'hidden_layer_sizes': [(100,),(200,)],
        'activation' : ['tanh'],  #'identity', 'logistic', 'tanh', 'relu'],
        'solver' : ['lbfgs', 'adam'],  #'sgd', 
        'learning_rate' : ['constant', 'adaptive']  # 'invscaling',
        }
    
    ### Create a classifier with the parameter candidates ###
    CV = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 5, scoring='accuracy', verbose=10)

    ### Train the classifier ### 
    CV.fit(x, y)

    ### Get the best parameters ###
    matrix = CV.cv_results_
    print("The best parameters are %s with a score of %0.2f"
        % (CV.best_params_, CV.best_score_))     
    
    return CV.best_estimator_.hidden_layer_sizes, CV.best_estimator_.activation,CV.best_estimator_.solver,CV.best_estimator_.learning_rate

    return(clf)

def best_randforest_cl(x, y):
    from sklearn.model_selection  import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    print(" Optimizing parameters for Random Forest")
    clf =  RandomForestClassifier(n_jobs=-1, max_features= 'sqrt', n_estimators=200, class_weight='balanced',
                                  criterion = 'gini', min_samples_leaf = 1, oob_score = True) 

    param_grid = {        
        'n_estimators': [100, 200, 700, 1000],
        'max_features': ['auto', 'sqrt', 'log2'],
        'criterion': ['gini', 'entropy'],
        'min_samples_leaf': [1, 3, 5, 10]
        }
    
    ### Create a classifier with the parameter candidates ###
    CV = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 5, scoring='accuracy', verbose=10)

    ### Train the classifier ### 
    CV.fit(x, y)

    ### Get the best parameters ###
    matrix = CV.cv_results_
    print("The best parameters are %s with a score of %0.2f"
        % (CV.best_params_, CV.best_score_))     

    
    return CV.best_estimator_.n_estimators, CV.best_estimator_.max_features, CV.best_estimator_.criterion, CV.best_estimator_.min_samples_leaf


    




def best_xgboost_cl(x, y):
    from sklearn.model_selection  import GridSearchCV
    from xgboost import XGBClassifier
    print(" Optimizing parameters for Xgboost")

    #clf = XGBClassifier(nthread=15, booster='gbtree', eta=0.1, max_depth=6, subsample=1)
    clf = XGBClassifier(nthread=15, max_depth=6, subsample=1)

    param_grid = {  
        #'booster': ['gbtree', 'gblinear', 'dart'],
        #'eta': [0.1, 0.5, 0.8],
        'max_depth': [5, 10, 30],
        'subsample': [0.5, 1],
        }
    
    ### Create a classifier with the parameter candidates ###
    CV = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 5, scoring='accuracy', verbose=10)

    ### Train the classifier ### 
    CV.fit(x, y)

    ### Get the best parameters ###
    print("The best parameters are %s with a score of %0.2f"
        % (CV.best_params_, CV.best_score_))     
    return CV.best_estimator_.max_depth,CV.best_estimator_.subsample
#CV.best_estimator_.booster, CV.best_estimator_.eta, CV.best_estimator_.max_depth,CV.best_estimator_.subsample



def xgboost_cl(x, y, booster, eta, max_depth, subsample):
    from xgboost import XGBClassifier
    print(" Fitting with XGBoost")
    
    # issue: the xgboost version I have does not support booster and eta
    #clf = XGBClassifier(booster = booster, eta = eta,  max_depth = max_depth, subsample = subsample)
    clf = XGBClassifier(  max_depth = max_depth, subsample = subsample)
    clf.fit(x, y)
    return(clf)


def get_accuracy_cl(y_pred, y):
    from sklearn import metrics
    print(" Getting Scores..")
             
    ### Estimate Accuracy ###
    score = metrics.accuracy_score(y, y_pred)     
    
    # testing score
    #f1score = metrics.f1_score(y, y_pred) 
    # for binary clf check this: sklearn.metrics.roc_curve

    ### Confusion Matrix ###
    conmat = metrics.confusion_matrix(y, y_pred)
    pos_ok = conmat[1][1]
    neg_ok = conmat[0][0]
    false_pos = conmat[0][1]
    false_neg = conmat[1][0]

    ### Print Score and Confusion Matrix ### 
    #print("  Score :", (tf(score)), " f1 score :", tf(f1score),
    print("  Score :", (tf(score)), #" f1 score :", tf(f1score),
          "\n  Pos_ok:", pos_ok, "False Neg:", false_neg, 
          " Pos error:", of(false_pos*100/(false_neg + pos_ok)),
          "%\n  Neg_ok:", neg_ok, "False_pos:", false_pos,           
          " Neg error:", of(false_neg*100/(false_pos + neg_ok)),"%")
    
