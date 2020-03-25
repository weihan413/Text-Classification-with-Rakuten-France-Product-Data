"""
    Python script to submit as a part of the project of ELTP 2020 course.
    
    This script serves as a template. Please use proper comments and meaningful variable names.
"""

"""
    Group Members:
        (1) Srijan Goyal
        (2) Nicolas Morel
        (3) Weihan Wang
"""

"""
    Import necessary packages
"""
import numpy as np
import pandas as pd

import spacy
import fr_core_news_sm
spacy_nlp = fr_core_news_sm.load()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from lightgbm import LGBMClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing

import warnings
warnings.filterwarnings("ignore")

#%%
"""
    Preprocessing
"""

def normalize_accent(string):
    string = string.replace('á', 'a')
    string = string.replace('â', 'a')

    string = string.replace('é', 'e')
    string = string.replace('è', 'e')
    string = string.replace('ê', 'e')
    string = string.replace('ë', 'e')

    string = string.replace('î', 'i')
    string = string.replace('ï', 'i')

    string = string.replace('ö', 'o')
    string = string.replace('ô', 'o')
    string = string.replace('ò', 'o')
    string = string.replace('ó', 'o')

    string = string.replace('ù', 'u')
    string = string.replace('û', 'u')
    string = string.replace('ü', 'u')

    string = string.replace('ç', 'c')
    
    return string

# Function to clean up the data
def raw_to_tokens(raw_string, spacy_nlp):
    # Write code for lower-casing
    string = raw_string.lower()
    
    # Write code to normalize the accents
    string = normalize_accent(string)
        
    # Write code to tokenize
    spacy_tokens = spacy_nlp(string)
    string_tokens = [token.orth_ for token in spacy_tokens]
    
    # Write code to remove punctuation tokens and create string tokens    
    string_tokens = [token.orth_ for token in spacy_tokens if not token.is_punct]
    string_tokens = [token.orth_ for token in spacy_tokens if not token.is_punct if not token.is_stop]
    
    # Write code to join the tokens back into a single string
    clean_string = " ".join(string_tokens)
    
    return clean_string

# Load train feature dataset
X_train = pd.read_csv('X_train_update.csv',index_col = 0)
docs_raw_train = X_train['designation']

# Preprocess and clean the train feature dataset
docs_clean_train = list()
for string in docs_raw_train:
    x = raw_to_tokens(string,spacy_nlp)
    docs_clean_train.append(x)
    
# Convert the train feature dataset into TF-IDF matrix form    
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(docs_clean_train)

# Load labels for training
Y_train = pd.read_csv('Y_train_CVw08PX.csv',index_col = 0)
Y_train = list(Y_train['prdtypecode'])

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, Y_train, test_size=0.2, random_state=42)

#%%
"""
    Your methods implementing the models.
    
    Each of your model should have a separate method. e.g. run_random_forest, run_decision_tree etc.
    
    Your method should:
        (1) create the proper instance of the model with the best hyperparameters you found
        (2) fit the model with a given training data
        (3) run the prediction on a given test data
        (4) return accuracy and F1 score
        
    Following is a sample method. Please note that the parameters given here are just examples.
"""
def model_random_forest(X_train, y_train, X_test, y_test):
    """
    @param: X_train - a numpy matrix containing features for training data (e.g. TF-IDF matrix)
    @param: y_train - a numpy array containing labels for each training sample
    @param: X_test - a numpy matrix containing features for test data (e.g. TF-IDF matrix)
    @param: y_test - a numpy array containing labels for each test sample
    """
    clf = RandomForestClassifier(
        n_estimators = 200
        , criterion = 'gini'
        , random_state = 42    
        , min_samples_leaf = 1
        , max_features = 'auto',
        min_samples_split = 2,
        max_depth = 1000,
        n_jobs = -1
    )
    clf.fit(X_train, y_train)

    y_predicted = clf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_predicted)
    rf_f1 = f1_score(y_test, y_predicted, average="weighted")

    return rf_accuracy, rf_f1

def model_gradient_boost(X_train, y_train, X_test, y_test):
    """
    @param: X_train - a numpy matrix containing features for training data (e.g. TF-IDF matrix)
    @param: y_train - a numpy array containing labels for each training sample
    @param: X_test - a numpy matrix containing features for test data (e.g. TF-IDF matrix)
    @param: y_test - a numpy array containing labels for each test sample
    """
    clf=GradientBoostingClassifier(learning_rate = 1.5, n_estimators = 500)
    clf.fit(X_train, y_train)

    y_predicted = clf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_predicted)
    rf_f1 = f1_score(y_test, y_predicted, average="weighted")

    return rf_accuracy, rf_f1

def model_ada_boost(X_train, y_train, X_test, y_test):
    """
    @param: X_train - a numpy matrix containing features for training data (e.g. TF-IDF matrix)
    @param: y_train - a numpy array containing labels for each training sample
    @param: X_test - a numpy matrix containing features for test data (e.g. TF-IDF matrix)
    @param: y_test - a numpy array containing labels for each test sample
    """
    clf=AdaBoostClassifier(learning_rate = 1.5,n_estimators = 500)
    clf.fit(X_train, y_train)

    y_predicted = clf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_predicted)
    rf_f1 = f1_score(y_test, y_predicted, average="weighted")

    return rf_accuracy, rf_f1

def model_xg_boost(X_train, y_train, X_test, y_test):
    """
    @param: X_train - a numpy matrix containing features for training data (e.g. TF-IDF matrix)
    @param: y_train - a numpy array containing labels for each training sample
    @param: X_test - a numpy matrix containing features for test data (e.g. TF-IDF matrix)
    @param: y_test - a numpy array containing labels for each test sample
    """
    #,tree_method = 'gpu_hist'
    clf=xgb.XGBClassifier(num_class=27
                          ,num_round = 500
                          , learning_rate = 0.01
                          , objective = 'multi:softmax'
                          )
    clf.fit(X_train, y_train)

    y_predicted = clf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_predicted)
    rf_f1 = f1_score(y_test, y_predicted, average="weighted")

    return rf_accuracy, rf_f1


def model_cat_boost(X_train, y_train, X_test, y_test):
    """
    @param: X_train - a numpy matrix containing features for training data (e.g. TF-IDF matrix)
    @param: y_train - a numpy array containing labels for each training sample
    @param: X_test - a numpy matrix containing features for test data (e.g. TF-IDF matrix)
    @param: y_test - a numpy array containing labels for each test sample
    """
    clf = CatBoostClassifier(iterations=2000
                             ,task_type="GPU"
                             ,devices='-1'
                             ,custom_metric='F1'
                             ,l2_leaf_reg = 1
                             , border_count = 254
                             , verbose=False
                             )
    clf.fit(X_train, y_train)

    y_predicted = clf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_predicted)
    rf_f1 = f1_score(y_test, y_predicted, average="weighted")

    return rf_accuracy, rf_f1


def model_light_GBM(X_train, y_train, X_test, y_test):
    """
    @param: X_train - a numpy matrix containing features for training data (e.g. TF-IDF matrix)
    @param: y_train - a numpy array containing labels for each training sample
    @param: X_test - a numpy matrix containing features for test data (e.g. TF-IDF matrix)
    @param: y_test - a numpy array containing labels for each test sample
    """
    le = preprocessing.LabelEncoder()
    y_train = le.fit_transform(y_train)
    
    #"device_type" : "gpu",
    params = {
          "objective" : "multiclass",
          "num_class" : 27,
          "num_leaves" : 80,
          "max_depth": -1,
          "learning_rate" : 0.001,
          "bagging_fraction" : 0.95,  # subsample
          "feature_fraction" : 0.95,  # colsample_bytree
          "bagging_freq" : 5,        # subsample_freq
          "bagging_seed" : 2018,
          "verbosity" : -1,
          'eval_metric' : 'evaluate().f1_score'}


    lgtrain = lgb.Dataset(X_train, y_train)
    clf = lgb.train(params, lgtrain, 200, verbose_eval=200)

    y_predicted = clf.predict(X_test)
    y_predicted=le.inverse_transform(y_predicted.argmax(axis=1))
    rf_accuracy = accuracy_score(y_test, y_predicted)
    rf_f1 = f1_score(y_test, y_predicted, average="weighted")

    return rf_accuracy, rf_f1


def model_stacking_clf(X_train, y_train, X_test, y_test):
    """
    @param: X_train - a numpy matrix containing features for training data (e.g. TF-IDF matrix)
    @param: y_train - a numpy array containing labels for each training sample
    @param: X_test - a numpy matrix containing features for test data (e.g. TF-IDF matrix)
    @param: y_test - a numpy array containing labels for each test sample
    """
    estimators = [
        ('rf', PassiveAggressiveClassifier(n_jobs = -1,C = 0.001,loss = 'squared_hinge',max_iter = 1000, tol = 1e-06)),
        ('svr', make_pipeline(RandomForestClassifier(n_estimators=1000, random_state=42,criterion = 'gini',bootstrap = True,max_features = 'auto')))]
        
    clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(penalty = 'l1',solver = 'saga',max_iter = 500))
    clf.fit(X_train, y_train)

    y_predicted = clf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_predicted)
    rf_f1 = f1_score(y_test, y_predicted, average="weighted")

    return rf_accuracy, rf_f1

#%%
"""
   The main function should print all the accuracies and F1 scores for all the models.
   
   The names of the models should be sklearn classnames, e.g. DecisionTreeClassifier, RandomForestClassifier etc.
   
   Please make sure that your code is outputting the performances in proper format, because your script will be run automatically by a meta-script.
"""
if __name__ == "__main__":
    
    
    model_rf_acc, model_rf_f1 = model_random_forest(X_train, y_train, X_test, y_test)
    model_gb_acc, model_gb_f1 = model_gradient_boost(X_train, y_train, X_test, y_test)
    model_ada_acc, model_ada_f1 = model_ada_boost(X_train, y_train, X_test, y_test)
    model_xgb_acc, model_xgb_f1 = model_xg_boost(X_train, y_train, X_test, y_test)
    model_cat_acc, model_cat_f1 = model_cat_boost(X_train, y_train, X_test, y_test)
    model_gbm_acc, model_gbm_f1 = model_light_GBM(X_train, y_train, X_test, y_test)
    model_stack_acc, model_stack_f1 = model_stacking_clf(X_train, y_train, X_test, y_test)
   

    # print the results
    print("RandomForestClassifier", model_rf_acc, model_rf_f1)
    print("GradientBoostingClassifier", model_gb_acc, model_gb_f1)
    print("AdaBoostClassifier", model_ada_acc, model_ada_f1)
    print("XGBClassifier", model_xgb_acc, model_xgb_f1)
    print("CatBoostClassifier", model_cat_acc, model_cat_f1)
    print("Light GBM", model_gbm_acc, model_gbm_f1)
    print("StackingClassifier", model_stack_acc, model_stack_f1)
    
 
