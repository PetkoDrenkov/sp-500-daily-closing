import pandas as pd
import numpy as np

sp = pd.read_csv("https://github.com/PetkoDrenkov/sp-500-daily-closing/blob/main/data/SP_close_price_5Y.csv")
sp.drop("Date",axis=1,inplace=True)
sp["returns"] = np.log(sp["Close"]) - np.log(sp["Close"].shift(1))


""" max value for lim is 15,and I won't allow for
greater lag depth for replication purposes (identical size of sets)"""
def returns_lagger(df,lim):
    if lim > 15:
        raise ValueError("Limit should NOT be larger than 15")
    else:
        for x in range(1,lim+1):
            df[f"lag_{x}"] = df["returns"].shift(x)
        df = df.iloc[16:]
        return df.drop("Close",axis=1)

