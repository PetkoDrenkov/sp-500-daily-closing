import pandas as pd
from src.a_lag_features import sp, returns_lagger
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

vol10 = returns_lagger(sp,10)**2

X_train = pd.DataFrame(vol10).iloc[:1116,1:]
X_test = pd.DataFrame(vol10).iloc[1116:,1:]
y_train = pd.DataFrame(vol10).iloc[:1116,0]
y_test = pd.DataFrame(vol10).iloc[1116:,0]

forest_reg = RandomForestRegressor()
forest_fit = forest_reg.fit(X_train,y_train)

with open("fit_10_forest_reg.pkl", "wb") as fit_10_vol:
    pickle.dump(forest_fit, fit_10_vol)

forest_mse_train = mean_squared_error(forest_reg.predict(X_train),y_train)
forest_r2_train = r2_score(forest_reg.predict(X_train),y_train)

forest_mse_test = mean_squared_error(forest_reg.predict(X_test),y_test)
forest_r2_test = r2_score(forest_reg.predict(X_test),y_test)
