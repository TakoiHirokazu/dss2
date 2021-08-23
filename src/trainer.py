import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import rankdata
from logger import LOGGER
from catboost import Pool
from catboost import CatBoostRegressor
from catboost import CatBoostClassifier 
import xgboost as xgb

def train_lgbm(X_train, y_train, X_valid, y_valid, X_test, categorical_features, feature_name, fold_id,lgb_params, fit_params, loss_func, calc_importances=True):
    
    train = lgb.Dataset(X_train, y_train,
                        categorical_feature=categorical_features,
                        feature_name=feature_name)
    if X_valid is not None:
        valid = lgb.Dataset(X_valid, y_valid,
                            categorical_feature=categorical_features,
                            feature_name=feature_name)
   
    if X_valid is not None:
        model = lgb.train(
            lgb_params,
            train,
            valid_sets=[train,valid],
            **fit_params
        )
    else:
        model = lgb.train(
            lgb_params,
            train,
            **fit_params
        )
    
    # train score
    if X_valid is not None:
        y_pred_valid = model.predict(X_valid)
        valid_loss = loss_func(y_valid, y_pred_valid)
    else:
        y_pred_valid = None
        valid_loss = None
    
    #test
    if X_test is not None:
        y_pred_test = model.predict(X_test)
    else:
        y_pred_test = None

    if calc_importances:
        importances = pd.DataFrame()
        importances['feature'] = feature_name
        importances['gain'] = model.feature_importance(importance_type='gain')
        importances['split'] = model.feature_importance(importance_type='split')
        importances['fold'] = fold_id
    else:
        importances = None

    return y_pred_valid, y_pred_test, valid_loss, importances, model.best_iteration, model


def train_cat_regressor(X_train, y_train, X_valid, y_valid, X_test, categorical_features, feature_name,cat_params , loss_func):

    train = Pool(X_train, y_train, cat_features=categorical_features, feature_names=feature_name)
    valid = Pool(X_valid, y_valid, cat_features=categorical_features, feature_names=feature_name)

    evals_result = {}
    model = CatBoostRegressor(**cat_params)
    model.fit(train,
              eval_set=valid,  # 検証用データ# 10回以上精度が改善しなければ中止
              use_best_model=True,  # 最も精度が高かったモデルを使用するかの設定
              verbose=200) 

    if X_valid is not None:
        # validation score
        y_pred_valid = model.predict(X_valid)
        valid_loss = loss_func(y_valid, y_pred_valid)
    else:
        y_pred_valid = None
        valid_loss = None

    if X_test is not None:
        # predict test
        y_pred_test = model.predict(X_test)
    else:
        y_pred_test = None

    
    return y_pred_valid, y_pred_test,valid_loss,model.get_best_iteration(),model


def train_cat_classifier(X_train, y_train, X_valid, y_valid, X_test, categorical_features, feature_name,cat_params , loss_func):
    
    train = Pool(X_train, y_train, cat_features=categorical_features, feature_names=feature_name)
    if X_valid is not None:
    
        valid = Pool(X_valid, y_valid, cat_features=categorical_features, feature_names=feature_name)
        model = CatBoostClassifier(**cat_params)
        model.fit(train,
                eval_set=valid,  # 検証用データ# 10回以上精度が改善しなければ中止
                use_best_model=True,  # 最も精度が高かったモデルを使用するかの設定
                verbose=200) 
        y_pred_valid = model.predict_proba(X_valid)
        valid_loss = loss_func(y_valid, y_pred_valid)
    else:
        model = CatBoostClassifier(**cat_params)
        model.fit(train)
        y_pred_valid = None
        valid_loss = None

    if X_test is not None:
        # predict test
        y_pred_test = model.predict_proba(X_test)
    else:
        y_pred_test = None

    
    return y_pred_valid, y_pred_test,valid_loss,model.get_best_iteration()



def train_xgb(X_train, y_train, X_valid, y_valid, X_test, feature_name,xgb_params,loss_func):
    
    train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=feature_name)
    
    
    if X_valid is not None:
        valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=feature_name)
        watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
        model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=500,
                          params=xgb_params)
        y_pred_valid = model.predict(valid_data, ntree_limit=model.best_ntree_limit)
        valid_loss = loss_func(y_valid, y_pred_valid)
    
    else:
        model = xgb.train(dtrain=train_data,params=xgb_params)
        y_pred_valid = None
        vali_loss = None
    
    if X_test is not None:
        y_pred_test = model.predict(xgb.DMatrix(X_test, feature_names=feature_name),ntree_limit=model.best_ntree_limit)
    else:
        y_pred_test = None

    return y_pred_valid, y_pred_test

