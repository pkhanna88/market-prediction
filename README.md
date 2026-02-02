## S&P500 Market Prediction

The goal of this project was to build a daily fund allocation model that predicts S&P500 excess returns and assigns a leverage value within [0, 2] for each trading day while being bounded by a 120% volatility constraint. It tests whether repeatable predictive edges exist within noisy market data, and it optimizes for a Sharpe Ratio-like metric which penalizes excess volatility or strategies that fail to outperform the market return.


A blend of random forest, gradient boosting, ridge regression, and MLP was used to improve the predictive robustness of the project by minimizing the potential for overfitting. The ensemble weights were dynamically computed depending upon the individual model Sharpe Ratio performances. The final allocations involved volatility dampening to maximize the evaluation metric by reducing leverage when predictions were uncertain. 


The predictions were: 1.0002702288331613, 0.9999636641662607, 1.0006840971498958, 1.0005711711817962, 1.0003550374044616, 1.0000246212281763, 1.0004164834910272, 0.9992103801421452, 0.9999286497775485, 1.0000189547374465 - very strongly favoring the S&P500 as is.


The eda notebook includes plots which display different aspects of the train set. The sequence to run the .py files is: preprocessing, training, next_training, final.
