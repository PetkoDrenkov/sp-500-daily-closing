import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import pickle
from src.b_training_models import feat_2, feat_3, feat_5, feat_7, feat_11, feat_15


with open("labels.pkl", 'rb') as labels:
    labels = pickle.load(labels)
labels = np.array(labels)

with open("fit_2.pkl", 'rb') as reg_2:
    fit_2 = pickle.load(reg_2)
df_2 = pd.DataFrame(feat_2)[992:1116]
mse_2_val = mean_squared_error(fit_2.predict(df_2),labels[992:1116])

with open("fit_3.pkl", 'rb') as reg_3:
    fit_3 = pickle.load(reg_3)
df_3 = pd.DataFrame(feat_3)[992:1116]
mse_3_val = mean_squared_error(fit_3.predict(df_3),labels[992:1116])

with open("fit_5.pkl", 'rb') as reg_5:
    fit_5 = pickle.load(reg_5)
df_5 = pd.DataFrame(feat_5)[992:1116]
mse_5_val = mean_squared_error(fit_5.predict(df_5),labels[992:1116])

with open("fit_7.pkl", 'rb') as reg_7:
    fit_7 = pickle.load(reg_7)
df_7 = pd.DataFrame(feat_7)[992:1116]
mse_7_val = mean_squared_error(fit_7.predict(df_7),labels[992:1116])

with open("fit_11.pkl", 'rb') as reg_11:
    fit_11 = pickle.load(reg_11)
df_11 = pd.DataFrame(feat_11)[992:1116]
mse_11_val = mean_squared_error(fit_11.predict(df_11),labels[992:1116])

with open("fit_15.pkl", 'rb') as reg_15:
    fit_15 = pickle.load(reg_15)
df_15 = pd.DataFrame(feat_15)[992:1116]
mse_15_val = mean_squared_error(fit_15.predict(df_15),labels[992:1116])

