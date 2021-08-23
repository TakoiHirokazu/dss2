import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error,roc_auc_score
import matplotlib.pyplot as plt


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    print("column = ", len(df.columns))
    for i, col in enumerate(df.columns):
        if i % 50 == 0:
            print(i)
        try:
            col_type = df[col].dtype

            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int32)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float32)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float32)
        except:
            continue

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


def categorical_check(df_train, df_test, col):
    """
    カテゴリ変数のトレーニングデータとテストデータの違いを見る
    """
    #uniqueでみたとき
    print(col + " percentage unique_train coverd :" ,np.round(len(set(df_train[col]) & set(df_test[col])) / len(set(df_train[col])),3))
    print(col + " percentage unique_test coverd :" ,np.round(len(set(df_train[col]) & set(df_test[col])) / len(set(df_test[col])),3))
    print(col + " percentage all_train coverd :" ,np.round(len(df_train[df_train[col].isin(set(df_train[col]) & set(df_test[col]))]) / len(df_train),3))
    print(col + " percentage all_test coverd :" ,np.round(len(df_test[df_test[col].isin(set(df_train[col]) & set(df_test[col]))]) / len(df_test),3))
    
def train_test_hist(df_train, df_test, col, bins=50):
    """
   トレーニングデータとテストデータの分布の違いを見る
    """
    val_max = max(df_train[col].max(),df_test[col].max())
    val_min = min(df_train[col].min(),df_test[col].min())
    df_train[col].hist(range=(val_min-1, val_max+1),bins=bins, density=True)
    df_test[col].hist(range=(val_min-1, val_max+1),bins=bins, density=True,alpha=0.5,color="red")
    plt.title(col)
    plt.show()
    
def numerical_categorical(df_train,target):
    """
    数値データとカテゴリーデータを分ける
    """
    numerical = []
    categorical = []
    for i in df_train.columns:
        if i != target:
            if df_train[i].dtype == "O":
                categorical.append(i)
            else:
                numerical.append(i)
    return categorical,numerical
