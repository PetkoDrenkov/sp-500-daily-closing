import pandas as pd
import pickle
import numpy as np
from src.a_lag_features import returns_lagger, sp
from sklearn.metrics import mean_squared_error, r2_score

labels = pd.DataFrame(returns_lagger(sp,5)).loc[:,"returns"]
labels_test = labels[1116:]

feat_5 = pd.DataFrame(returns_lagger(sp,5).drop("returns",axis=1))
feat_5_test = feat_5[1116:]

with open("fit_5_new.pkl", 'rb') as f:
    fit_5_new = pickle.load(f)

mse_test = mean_squared_error(labels_test, fit_5_new.predict(feat_5_test))
r2_score_test = r2_score(labels_test, fit_5_new.predict(feat_5_test))
rms_test = np.sqrt(mse_test)
