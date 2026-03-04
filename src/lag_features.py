import pandas as pd
import numpy as np

sp = pd.read_csv("/home/petko/Desktop/SPY_close_price_5Y.csv")
sp.drop("Date",axis=1,inplace=True)
sp["returns"] = np.log(sp["Close"]) - np.log(sp["Close"].shift(1))
# print(sp)

""" max value for lim is 15, and we must remove lim+1 rows, and it should
be the same for all depths so we'll fix the slice at 16,and I won't allow for
greater lag depth than 15 for replication purposes (identical size of sets)"""

def returns_lagger(df,lim):
    if lim > 15:
        raise ValueError("Limit should NOT be larger than 15")
    else:
        for x in range(1,lim+1):
            df[f"lag_{x}"] = df["returns"].shift(x)
        df = df.iloc[16:]
        return df.drop("Close",axis=1)

