import pandas as pd
from src.a_lag_features import sp, returns_lagger
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

vol10 = returns_lagger(sp,10)**2

X_train = pd.DataFrame(vol10).iloc[:1116,1:]
X_test = pd.DataFrame(vol10).iloc[1116:,1:]
y_train = pd.DataFrame(vol10).iloc[:1116,0]
y_test = pd.DataFrame(vol10).iloc[1116:,0]

tree_reg = DecisionTreeRegressor()
tree_fit = tree_reg.fit(X_train,y_train)

with open("fit_10_tree_reg.pkl", "wb") as fit_10_vol:
    pickle.dump(tree_fit, fit_10_vol)

tree_reg.predict(X_train)
tree_mse_train = mean_squared_error(y_train,tree_reg.predict(X_train))
tree_r2_train = r2_score(y_train,tree_reg.predict(X_train))
tree_reg.predict(X_test)
tree_mse_test = mean_squared_error(y_test,tree_reg.predict(X_test))
tree_r2_test = r2_score(y_test,tree_reg.predict(X_test))
