import pandas as pd
import numpy as np
import seaborn as sns
sns.set(style="ticks")
import matplotlib.pyplot as plt
import multiprocessing
from joblib import Parallel, delayed
from scipy.optimize import curve_fit 
from scipy import stats
import datetime as dt
import statsmodels.api as sm
from pandas.plotting import scatter_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
import pickle
from sklearn.model_selection import train_test_split
import xgboost as xgb


def main():
    #df_main = pd.read_pickle('/scratch/d.dasarathan/df_newdata_cleaned_500k_added_features.pkl')
    df_main = pd.read_pickle('/scratch/d.dasarathan/df_newdata_cleaned_600k_added_features.pkl')
    df=df_main[['ReqNodes', 'Priority', 'Partition', 'waitTimeHr', 'Req_totalMem', 'corehrs', 'QOD', 'QOY','exclusive','rnjbct']]
    df=df.sample(frac=1).reset_index(drop=True)
    # One-hot encode categorical columns
    df=pd.get_dummies(df)
    
    # Target is the value we want to predict
    target = np.array(df['waitTimeHr'])

    # Remove the target from the predictors
    X = df.drop('waitTimeHr', axis = 1)

    # Saving feature names for later use
    feature_list = list(X.columns)

    # Convert to numpy array
    X = np.array(X)

    # Using Skicit-learn to split data into training and testing sets
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, target, test_size = 0.20, random_state = 42)

    print('x_train Shape:', x_train.shape)
    print('y_train Shape:', y_train.shape)
    print('x_test Shape:', x_test.shape)
    print('y_test Shape:', y_test.shape)
   
    xg_reg = xgb.XGBRFRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 50, alpha = 10, n_estimators = 400)
    print('XGBoost model')
    print(xg_reg)
    xg_reg.fit(x_train, y_train)
    preds = xg_reg.predict(x_test)
    
    # Create a filename in which the trained model will be stored
    xgFile = "/home/d.dasarathan/project_sml/100Kdataset/trainedbasicxgb600_08_16.pkl"
    # Save the trained model to disk
    pickle.dump(xg_reg, open(xgFile, 'wb'))

    #rmse = np.sqrt(mean_squared_error(y_test, preds))
    #print("RMSE: %f" % (rmse))

    # Calculate the absolute errors
    errors = abs(y_test - preds)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'wait hours.')

    xgb = xgb.XGBRFRegressor(objective = 'reg:squarederror')
    param_dist = {'n_estimators': stats.randint(150, 1000),
                    'learning_rate': stats.uniform(0.01, 0.59),
                    'subsample': stats.uniform(0.3, 0.6),
                    'max_depth': [10, 18, 26, 34, 42, 50],
                    'colsample_bytree': stats.uniform(0.5, 0.4),
                    'min_child_weight': [1, 2, 3, 4]
                }

    xgb_random = RandomizedSearchCV(estimator = xgb, 
                                    param_distributions = param_dist,
                                    cv = 3,  
                                    n_iter = 10,
                                    random_state=42,
                                    error_score = 0, 
                                    verbose = 2, 
                                    n_jobs = -1)
    xgb_random.fit(x_train, y_train)

    print('XGBoost best Parameters: ',xgb_random.best_params_)
    xgb_random.best_params_

    #xg_reg_hyp = XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.89, learning_rate = 0.15, max_depth = 8, n_estimators = 712)
    #xg_reg_hyp.fit(x_train, y_train)

    #preds = xg_reg_hyp.predict(x_test)
    #rmse = np.sqrt(mean_squared_error(y_test, preds))
    #print("RMSE: %f" % (rmse))

    # Calculate the absolute errors
    #errors = abs(y_test - preds)
    #print('Mean Absolute Error:', round(np.mean(errors), 2), 'wait hours.')
    #print('R^2 Training Score: {:.2f} \nR^2 Test Score: {:.2f}'.format(xg_reg_hyp.score(x_train, y_train), xg_reg_hyp.score(x_test, y_test)))

if __name__ == '__main__':
    main()