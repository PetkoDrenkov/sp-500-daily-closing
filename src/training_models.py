from src.lag_features import sp, returns_lagger
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import pickle

feat_2 = pd.DataFrame(returns_lagger(sp,2).drop("returns",axis=1))
feat_3 = pd.DataFrame(returns_lagger(sp,3).drop("returns",axis=1))
feat_5 = pd.DataFrame(returns_lagger(sp,5).drop("returns",axis=1))
feat_7 = pd.DataFrame(returns_lagger(sp,7).drop("returns",axis=1))
feat_11 = pd.DataFrame(returns_lagger(sp,11).drop("returns",axis=1))
feat_15 = pd.DataFrame(returns_lagger(sp,15).drop("returns",axis=1))
labels = pd.DataFrame(returns_lagger(sp,1)["returns"])
with open("labels.pkl", 'wb') as labs:
    pickle.dump(labels, labs)

lin_reg = LinearRegression()

fit_2 = lin_reg.fit(feat_2.iloc[:992,:],labels[:992])
predict_2 = lin_reg.predict(feat_2.iloc[:992,:])
mse_2 = mean_squared_error(labels[:992], predict_2)
with open("fit_2.pkl", 'wb') as reg_2:
    pickle.dump(fit_2, reg_2)

fit_3 = lin_reg.fit(feat_3.iloc[:992,:],labels[:992])
predict_3 = lin_reg.predict(feat_3.iloc[:992,:])
mse_3 = mean_squared_error(labels[:992], predict_3)
with open("fit_3.pkl", "wb") as reg_3:
    pickle.dump(fit_3, reg_3)

fit_5 = lin_reg.fit(feat_5.iloc[:992,:],labels[:992])
predict_5 = lin_reg.predict(feat_5.iloc[:992,:])
mse_5 = mean_squared_error(labels[:992], predict_5)
with open("fit_5.pkl", "wb") as reg_5:
    pickle.dump(fit_5, reg_5)

fit_7 = lin_reg.fit(feat_7.iloc[:992,:],labels[:992])
predict_7 = lin_reg.predict(feat_7.iloc[:992,:])
mse_7 = mean_squared_error(labels[:992], predict_7)
with open("fit_7.pkl", "wb") as reg_7:
    pickle.dump(fit_7, reg_7)

fit_11 = lin_reg.fit(feat_11.iloc[:992,:],labels[:992])
predict_11 = lin_reg.predict(feat_11.iloc[:992,:])
mse_11 = mean_squared_error(labels[:992], predict_11)
with open("fit_11.pkl", "wb") as reg_11:
    pickle.dump(fit_11, reg_11)

fit_15 = lin_reg.fit(feat_15.iloc[:992,:],labels[:992])
predict_15 = lin_reg.predict(feat_15.iloc[:992,:])
mse_15 = mean_squared_error(labels[:992], predict_15)
with open("fit_15.pkl", "wb") as reg_15:
    pickle.dump(fit_15, reg_15)

# print(f"mse_2: {mse_2}\nmse_3: {mse_3}\nmse_5: {mse_5}\nmse_7: {mse_7}\nmse_11: {mse_11}\nmse_15: {mse_15}")

