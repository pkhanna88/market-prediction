import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# Impute missing values with median and scale features
def preprocess_data(df):
    imputer = SimpleImputer(strategy='median')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df.columns)
    return df_scaled


if __name__ == "__main__":
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    # Drop non-feature columns for preprocessing
    feature_cols = [col for col in train.columns if col not in ['date_id','forward_returns','risk_free_rate','market_forward_excess_returns']]
    X_train = preprocess_data(train[feature_cols])
    X_test = preprocess_data(test[feature_cols])
    X_train.to_csv('X_train_processed.csv', index=False)
    X_test.to_csv('X_test_processed.csv', index=False)