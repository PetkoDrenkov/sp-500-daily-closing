import pandas as pd
from src.lag_features import returns_lagger, sp
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


labels = pd.DataFrame(returns_lagger(sp,5)).loc[:,"returns"]
labels_retrain = labels[:1116]

feat_5 = pd.DataFrame(returns_lagger(sp,5).drop("returns",axis=1))
feat_5_retrain = feat_5[:1116]

lin_reg = LinearRegression()

fit_5_new = lin_reg.fit(feat_5_retrain,labels_retrain)
predict_5_again = lin_reg.predict(feat_5_retrain)
mse_5_retrain = mean_squared_error(labels_retrain, predict_5_again)

with open("fit_5_new.pkl", "wb") as reg_5_new:
    pickle.dump(fit_5_new, reg_5_new)


