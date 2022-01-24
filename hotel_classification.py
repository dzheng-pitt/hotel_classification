import pandas as pd
import numpy as np
import sys
from scipy.stats import skew
from scipy.stats import kurtosis

import random
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import optuna
import time
import xgboost as xgb



## Some data cleaning and EDA
data = pd.read_csv('data/hotel_bookings.csv')
data_types = np.unique(data.dtypes)

# function to check data
def eda(data, file):
    
    # write to log
    keep_stdout = sys.stdout
    log = open(file+".log","w")
    sys.stdout = log

    # for each variable
    for c in data.columns:
        
        print(c)
        data_col = data[c]
            
        # look at NAs
        print('NA count:', str(data_col.isna().sum()))
        
        # for continuous variables, get some distribution information
        if data_col.dtype == 'float64' or data_col.dtype == 'int64':
            
            print('Data percentiles [0 5 25 50 75 95 100]')
            print(np.round(np.nanpercentile(data_col,[0,5,25,50,75,95,100]),2))
            
            print('Data Moments')
            print('Mean:',
                  np.round(np.mean(data_col),2))
            print('Std Dev:',
                  np.round(np.std(data_col),2))
            print('Skewness:',
                  np.round(skew(data_col, nan_policy='omit'),2))
            print('Kurtosis:',
                  np.round(kurtosis(data_col, nan_policy='omit'),2))
            
            print('Data Top 5 Modes [item][count]')
            print(np.round(data_col.value_counts().index.to_list()[:5],2),
                  data_col.value_counts().to_list()[:5])
        
        # for categorical variables, get counts and unique levels
        elif data_col.dtype == 'O':
            
            print('Data Top 5 Modes [item][count]')
            print(data_col.value_counts().index.to_list()[:5],
                  data_col.value_counts().to_list()[:5])
            
            print('Data Unique Vals')
            print(pd.unique(data_col))
        
        # anything else, fix later iteration...
        else:
            print(data_col.dtype)
            
        print()

    sys.stdout = keep_stdout
    log.close()

eda(data, 'data/data_info_before')

# let's try to predict if the reservation is_canceled

# arrival_date_year, agent, company is categorical
def cat_this(data, col_to_cat):
    data[col_to_cat] = data[col_to_cat].round().astype(str)
    return data
data = cat_this(data, 'arrival_date_year')
# but the cardinality of agent and company seem difficult to deal with here
# let's remove to simplify modeling for this example
del data['agent']
# there doesn't seem to be much density for distinct companies either
del data['company']
# the cardinality of country also seems difficult to manage
del data['country']


# is_repeated_guest is an indicator
def ind_this(data, col_to_ind):
    data[col_to_ind].loc[data[col_to_ind] == 0] = 'no'
    data[col_to_ind].loc[data[col_to_ind] == 1] = 'yes'
    return data
data = ind_this(data, 'is_repeated_guest')

# days_in_waiting_list has a strong mode at 0
def mode_this(data, col_to_mode):
    mode = data[col_to_mode].value_counts().index.to_list()[0]
    data[col_to_mode+'_mode'] = 'no'
    data[col_to_mode+'_mode'].loc[data[col_to_mode] == mode] = 'yes'
    return data
data = mode_this(data, 'days_in_waiting_list')

# reservation_status has cancelled information in it
data.groupby(['reservation_status','is_canceled'])['hotel'].count()
del data['reservation_status']

# sometimes reservation_status_date comes after the arrival_date, sometimes 
# before. better to just delete for now
del data['reservation_status_date']

# clean continuous variables according to notes
for col in data.columns:

    if col == 'is_canceled':
        continue
    
    if data[col].dtype == 'float64' or data[col].dtype == 'int64':
        
        # standardize continuous variables for lasso
        mu = np.mean(data[col])
        sigma = np.std(data[col])
        data[col] = (data[col] - mu)/sigma
        
        # handle continuous nulls, impute median where missing
        if data[col].isna().sum() > 0:
            data[col+"_null"] = 'no'
            data[col+"_null"].loc[data[col].isna()] = 'yes'
            data[col].loc[data[col].isna()] = np.nanmedian(data[col])

# clean categorical variables according to notes
for col in data.columns:
    
    if col == 'is_canceled':
        continue
    
    if data[col].dtype == 'O':
        
        # handle categorical nulls
        if data[col].isna().sum() > 0:
            data[col].loc[data[col].isna()] = 'missing'
            
        # one hot encode categorical variables/indicators
        for level in np.unique(data[col]):
            data[col+'_'+level] = 0
            data[col+'_'+level].loc[data[col] == level] = 1
        
        data = data.drop(col,1)   
    
eda(data, 'data/data_info_after')

# imbalanced dataset on target, these rates should be okay for logit and trees
print(np.sum(data['is_canceled'])/len(data['is_canceled']))

# curious about correlations
correlations = data.corr().reset_index()
correlations_long = correlations.melt(id_vars = 'index')\
    .loc[correlations.melt(id_vars = 'index')['value'] != 1]\
    .sort_values('value').groupby(['value']).first().reset_index()

print(correlations_long.loc[correlations_long['value'] < -0.5])
print(correlations_long.loc[correlations_long['value'] > 0.5])

# there seem to be some dependencies between market segment, child nulls, and 
# distribution channel. Assigned and reserved rooms are fairly colinear






## Modeling
# arrival date week number and day of month probably has non linear 
# relationships with is_cancelled. other variables also have this potential.
# we can try these things. baseline logit knowing these flaws and xgboost. 
# Because I believe a tree based fitting algo
# is probably most convenient and applicable here, I won't delve too much into
# other feature engineerings for the lasso, but if lasso is the preferred model
# can try to clean some things better like log/box-cox transform heavily skewed 
# varables or to do an interaction search. Probably also helpful to do some 
# variable level groupings for similar types to provide sufficient density 
# where we can

# split up data for analysis, get test set for final test after modeling
random.seed(43)
train_ids = [(random.random() > 0.1) for i in range(data.shape[0])]
test_ids = [not i for i in train_ids]

train_split = data[train_ids]
test_split = data[test_ids]

# lasso grid
Cs = [0.001, 0.01, 0.1, 1, 10]
l1_ratios = [0, 0.25, 0.5, 0.75, 1.0]


# fit a penalized lasso with CV on training
lasso_model = LogisticRegressionCV(Cs = Cs, 
                                   penalty = 'elasticnet', 
                                   solver = 'saga',
                                   max_iter = 100,
                                   l1_ratios = l1_ratios,
                                   n_jobs = -1)
lasso_model.fit(train_split.iloc[:,1:].to_numpy(),
                train_split['is_canceled'].to_numpy())

# make training and held out testing predications
train_preds = lasso_model.predict_proba(train_split.iloc[:,1:].to_numpy())[:,1]
test_preds = lasso_model.predict_proba(test_split.iloc[:,1:].to_numpy())[:,1]

# evaluate ROC
train_fpr, train_tpr, _ = roc_curve(train_split['is_canceled'].to_numpy(), train_preds)
train_roc_auc = roc_auc_score(train_split['is_canceled'].to_numpy(), train_preds)

test_fpr, test_tpr, _ = roc_curve(test_split['is_canceled'].to_numpy(), test_preds)
test_roc_auc = roc_auc_score(test_split['is_canceled'].to_numpy(), test_preds)


plt.plot(train_fpr, train_tpr, label = 'training auc: '+str(round(train_roc_auc,4)))
plt.plot(test_fpr, test_tpr, label = 'testing auc: '+str(round(test_roc_auc,4)))
plt.plot([0,1], [0,1], label = 'null')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC: Penalized Logit')
plt.savefig('plots/roc_logit.png')
plt.show()

# evaluted Precision Recall
train_pre, train_rec, _ = precision_recall_curve(train_split['is_canceled'].to_numpy(), train_preds)
train_pr_auc = auc(train_rec, train_pre)

test_pre, test_rec, _ = precision_recall_curve(test_split['is_canceled'].to_numpy(), test_preds)
test_pr_auc = auc(test_rec, test_pre)

plt.plot(train_rec, train_pre, label = 'training auc: '+str(round(train_pr_auc,4)))
plt.plot(test_rec, test_pre, label = 'testing auc: '+str(round(test_pr_auc,4)))
plt.legend()
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall: Penalized Logit')
plt.savefig('plots/pr_logit.png')
plt.show()

# see what the logit picks out
for i in range(len(train_split.iloc[:,1:].columns)):
    print(train_split.iloc[:,1:].columns[i],
          round(lasso_model.coef_[0][i],3))

print('C:',lasso_model.C_, 'l1_ratio:', lasso_model.l1_ratio_)



# We could make interactions for logit, but XGBoost will make interactions for us

# let's tune with Optuna
dtrain = xgb.DMatrix(train_split.iloc[:,1:],label = train_split.iloc[:,0])

with open('data/xgb_optuna1.csv','w') as f:
    f.write('eta,max_depth,colsample_bytree,num_round,early_stopping_rounds,last_score,time\n')
    
def objective(trial):
    
    start = time.time()
    
    param = {"verbosity": 0,
              "objective": "binary:logistic",
              "colsample_bytree": trial.suggest_float('colsample_bytree', 0.1, 1),
              "max_depth": trial.suggest_int('max_depth',2,10),
              "eta": trial.suggest_float("eta", 0.001, 0.1)}

    num_round = trial.suggest_int("num_round", 10, 200)
    early_stopping_rounds = trial.suggest_int('early_stopping_rounds', 5, 20)
    
    bst = xgb.cv(param, dtrain, num_round, early_stopping_rounds=early_stopping_rounds)
    test_score = bst['test-logloss-mean'][bst.shape[0]-1]
    
    with open('data/xgb_optuna1.csv','a') as f:
        f.write(str(param['eta'])+','+
                str(param['max_depth'])+','+
                str(param['colsample_bytree'])+','+
                str(num_round)+','+
                str(early_stopping_rounds)+','+
                str(test_score)+','+
                str(round((time.time()-start)/60))+'\n')

    return test_score

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

xgb_optuna = pd.read_csv('data/xgb_optuna1.csv')

for c in xgb_optuna.columns:
    if c == 'last_score' or c == 'time':
        continue
    
    x = xgb_optuna[c]
    y = xgb_optuna['last_score']
    m,b = np.polyfit(x,y,1)
    plt.plot(x,y,'.')
    plt.plot(x,m*x+b)
    plt.ylabel('testing score')
    plt.xlabel(c)
    plt.savefig('plots/xgb_optuna1_'+c+'.png')
    plt.show()

# we want higher eta, more depth, more rounds
# lower colsample_bytree 
# early stopping seems indifferent as long as it's here


with open('data/xgb_optuna2.csv','w') as f:
    f.write('eta,max_depth,colsample_bytree,num_round,last_score,time\n')
def objective(trial):
    
    start = time.time()
    
    param = {"verbosity": 0,
              "objective": "binary:logistic",
              "colsample_bytree": trial.suggest_float('colsample_bytree', 0.2, 0.6),
              "max_depth": trial.suggest_int('max_depth',9,20),
              "eta": trial.suggest_float("eta", 0.08, 0.5)}

    num_round = trial.suggest_int("num_round", 175, 500)
    early_stopping_rounds = 15
    
    bst = xgb.cv(param, dtrain, num_round, early_stopping_rounds = early_stopping_rounds)
    test_score = bst['test-logloss-mean'][bst.shape[0]-1]
    
    with open('data/xgb_optuna2.csv','a') as f:
        f.write(str(param['eta'])+','+
                str(param['max_depth'])+','+
                str(param['colsample_bytree'])+','+
                str(num_round)+','+
                str(test_score)+','+
                str(round((time.time()-start)/60))+'\n')

    return test_score

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

xgb_optuna = pd.read_csv('data/xgb_optuna2.csv')

for c in xgb_optuna.columns:
    if c == 'last_score' or c == 'time':
        continue
    
    x = xgb_optuna[c]
    y = xgb_optuna['last_score']
    m,b = np.polyfit(x,y,1)
    plt.plot(x,y,'.')
    plt.plot(x,m*x+b)
    plt.ylabel('testing score')
    plt.xlabel(c)
    plt.savefig('plots/xgb_optuna2_'+c+'.png')
    plt.show()

# best seems like this
param = {'max_depth':16, 
         'eta':0.1, 
         'objective':'binary:logistic',
         'colsample_bytree': 0.55} 

num_round = 300
bst = xgb.train(param, dtrain, num_round)

# make prediction
train_preds = bst.predict(dtrain)
dtest = xgb.DMatrix(test_split.iloc[:,1:])
test_preds = bst.predict(dtest)

# evaluate ROC
train_fpr, train_tpr, _ = roc_curve(train_split['is_canceled'].to_numpy(), train_preds)
train_roc_auc = roc_auc_score(train_split['is_canceled'].to_numpy(), train_preds)

test_fpr, test_tpr, _ = roc_curve(test_split['is_canceled'].to_numpy(), test_preds)
test_roc_auc = roc_auc_score(test_split['is_canceled'].to_numpy(), test_preds)

plt.plot(train_fpr, train_tpr, label = 'training auc: '+str(round(train_roc_auc,4)))
plt.plot(test_fpr, test_tpr, label = 'testing auc: '+str(round(test_roc_auc,4)))
plt.plot([0,1], [0,1], label = 'null')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC: XGBoost')
plt.savefig('plots/roc_xgboost.png')
plt.show()

# evaluate Precision Recall
train_pre, train_rec, _ = precision_recall_curve(train_split['is_canceled'].to_numpy(), train_preds)
train_pr_auc = auc(train_rec, train_pre)

test_pre, test_rec, _ = precision_recall_curve(test_split['is_canceled'].to_numpy(), test_preds)
test_pr_auc = auc(test_rec, test_pre)

plt.plot(train_rec, train_pre, label = 'training auc: '+str(round(train_pr_auc,4)))
plt.plot(test_rec, test_pre, label = 'testing auc: '+str(round(test_pr_auc,4)))
plt.legend()
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall: XGBoost')
plt.savefig('plots/pr_xgboost.png')
plt.show()

# Curious about feature importance...
xgb.plot_importance(bst, max_num_features = 20)
plt.tight_layout()
plt.savefig('plots/xgboost_importance.png')
plt.show()
