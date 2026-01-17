import os
import numpy as np
import pandas as pd
import polars as pl
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from next_training import SharpeOptimizedEnsemble

# Impute missing values with median and scale features
def preprocess_data(df, imputer=None, scaler=None, fit=False):
    if fit:
        imputer = SimpleImputer(strategy='median')
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df.columns)
        return df_scaled, imputer, scaler
    else:
        df_imputed = pd.DataFrame(imputer.transform(df), columns=df.columns)
        df_scaled = pd.DataFrame(scaler.transform(df_imputed), columns=df.columns)
        return df_scaled


train = pd.read_csv('train.csv')
feature_cols = [col for col in train.columns if col not in 
                ['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']]
X_train_raw = train[feature_cols]
X_train_processed, imputer, scaler = preprocess_data(X_train_raw, fit=True)


joblib.dump(imputer, 'preprocessor_imputer.pkl')
joblib.dump(scaler, 'preprocessor_scaler.pkl')
sharpe_ensemble = joblib.load('ensemble_sharpe_optimized.pkl')
imputer = joblib.load('preprocessor_imputer.pkl')
scaler = joblib.load('preprocessor_scaler.pkl')
print("Ensemble Weights:")
for key, weight in sharpe_ensemble.weights.items():
    print(f"  {key}: {weight:.4f}")


# Generate predictions for test set
def predict(test: pl.DataFrame) -> pl.DataFrame:
    test_df = test.to_pandas()
    feature_cols = [col for col in test_df.columns if col not in 
                    ['date_id', 'is_scored', 'lagged_forward_returns', 
                     'lagged_risk_free_rate', 'lagged_market_forward_excess_returns']]
    X_test_raw = test_df[feature_cols]
    X_test_processed = preprocess_data(X_test_raw, imputer=imputer, scaler=scaler)
    ensemble_pred = sharpe_ensemble.predict(X_test_processed)
    # 1 is neutral
    alloc_base = 1 + ensemble_pred
    pred_volatility = np.std(ensemble_pred) if len(ensemble_pred) > 1 else 0.1
    volatility_dampening = 1 / (1 + 0.5 * pred_volatility)
    alloc = np.clip(alloc_base * volatility_dampening, 0, 2)
    result = test.with_columns(
        prediction=pl.Series(alloc)
    )
    result.write_csv('final.csv')
    return result


test = pl.read_csv('test.csv')
predict(test)