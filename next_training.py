import pandas as pd
import numpy as np
import joblib
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
    excess_returns = y_pred
    if np.std(excess_returns) == 0:
        return 0
    return np.mean(excess_returns) / np.std(excess_returns)


# Ensemble that combines models based on Sharpe Ratio
class SharpeOptimizedEnsemble:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights is not None else self._equal_weights()
    

    # Initialize equal weights
    def _equal_weights(self):
        n_models = len(self.models)
        return {key: 1.0 / n_models for key in self.models.keys()}
    

    # Compute weights based on Sharpe Ratio performance
    def compute_optimal_weights(self, X, y):
        preds = {}
        sharpes = {}
        for key, model in self.models.items():
            pred = model.predict(X)
            preds[key] = pred
            sharpes[key] = sharpe_score(y, pred)
        # Convert Sharpe scores to weights
        sharpe_values = np.array([max(0, s) for s in sharpes.values()])
        if sharpe_values.sum() == 0:
            # Fallback to equal weights if all Sharpes are negative
            self.weights = self._equal_weights()
        else:
            # Normalize Sharpes to weights
            self.weights = {key: sharpe_values[i] / sharpe_values.sum() 
                           for i, key in enumerate(self.models.keys())}
        return self.weights


    # Make weighted ensemble predictions
    def predict(self, X):
        preds = np.zeros(len(X))
        for key, model in self.models.items():
            weight = self.weights[key]
            pred = model.predict(X)
            preds += weight * pred
        return preds


if __name__ == "__main__":
    print("Trained Models")
    trained_models = {
        'rf': joblib.load('model_rf.pkl'),
        'gbm': joblib.load('model_gbm.pkl'),
        'ridge': joblib.load('model_ridge.pkl'),
        'mlp': joblib.load('model_mlp.pkl')
    }
    print("Computing Optimal Ensemble Weights")
    sharpe_ensemble = SharpeOptimizedEnsemble(trained_models)
    weights = sharpe_ensemble.compute_optimal_weights(X_val, y_val)
    print("\nEnsemble Weights")
    for key, weight in weights.items():
        print(f"  {key}: {weight:.4f}")
    print("\nEvaluating Ensemble")
    ensemble_pred = sharpe_ensemble.predict(X_val)
    ensemble_mse = mean_squared_error(y_val, ensemble_pred)
    ensemble_sharpe = sharpe_score(y_val, ensemble_pred)
    print(f"Ensemble MSE: {ensemble_mse:.6f}")
    print(f"Ensemble Sharpe Ratio: {ensemble_sharpe:.6f}")
    joblib.dump(sharpe_ensemble, "ensemble_sharpe_optimized.pkl")