import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


X_train = pd.read_csv('X_train_processed.csv')
train = pd.read_csv('train.csv')
y = train['market_forward_excess_returns']
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y, test_size=0.2, random_state=42)


# Estimate rolling volatility
def estimate_volatility(returns, window=20):
    vol = returns.rolling(window=window).std()
    return vol.fillna(vol.mean())


# Calculate Sharpe Ratio
def sharpe_score(y_true, y_pred):
    returns = y_pred
    # Assuming risk-free rate
    excess_returns = returns - 0
    if np.std(excess_returns) == 0:
        return 0
    sharpe = np.mean(excess_returns) / np.std(excess_returns)
    return sharpe


models_config = {
    'rf': {
        'model': RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'name': 'Random Forest'
    },
    'gbm': {
        'model': GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=6,
            min_samples_split=5,
            subsample=0.8,
            random_state=42
        ),
        'name': 'Gradient Boosting'
    },
    'ridge': {
        'model': Ridge(alpha=0.5),
        'name': 'Ridge Regression'
    },
    'mlp': {
        'model': MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            learning_rate_init=0.001,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            batch_size=32
        ),
        'name': 'MLP Regressor'
    }
}


print("Training Individual Models\n")
trained_models = {}
for model_key, config in models_config.items():
    print(f"Training {config['name']}...")
    model = config['model']
    model.fit(X_tr, y_tr)
    # Evaluate on validation set
    y_pred_val = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred_val)
    sharpe = sharpe_score(y_val, y_pred_val)
    print(f"MSE: {mse:.6f}")
    print(f"Sharpe Ratio: {sharpe:.6f}\n")
    trained_models[model_key] = model
    joblib.dump(model, f"model_{model_key}.pkl")