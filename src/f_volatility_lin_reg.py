import pandas as pd
from src.a_lag_features import sp, returns_lagger
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

vol10 = returns_lagger(sp,10)**2

X_train = pd.DataFrame(vol10).iloc[:1116,1:]
X_test = pd.DataFrame(vol10).iloc[1116:,1:]
y_train = pd.DataFrame(vol10).iloc[:1116,0]
y_test = pd.DataFrame(vol10).iloc[1116:,0]

lin_reg = LinearRegression()
fit = lin_reg.fit(X_train,y_train)

with open("fit_10_vol.pkl", "wb") as fit_10_vol:
    pickle.dump(fit, fit_10_vol)

lin_reg.predict(X_train) # not real prediction, but a sanity check for possible underfitting
mse_train = mean_squared_error(y_train,lin_reg.predict(X_train))
r2_train = r2_score(y_train,lin_reg.predict(X_train))
lin_reg.predict(X_test) # actual prediction
mse_test = mean_squared_error(y_test,lin_reg.predict(X_test))
r2_test = r2_score(y_test,lin_reg.predict(X_test))
