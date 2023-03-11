import os
import pandas as pd
import numpy as np
from FredMD import FredMD

from settings import DATA_UTILS_PATH, INPUTS_PATH

START_DATE = "1960-01-01"

def gen_fred_dataset(start_date):
    fredmd = FredMD()
    fredmd.apply_transforms()

    # load fredmd data from URL
    raw_fredmd_df = fredmd.rawseries
    transf_fredmd_df = fredmd.series

    # descriptions
    des_raw_fredmd_df = pd.DataFrame(raw_fredmd_df.iloc[0]).reset_index()
    des_raw_fredmd_df.columns = ["fred", "ttype"]
    des_raw_fredmd_df = des_raw_fredmd_df.drop([0], axis=0)
    des_fredmd_df = pd.read_csv(os.path.join(DATA_UTILS_PATH, "fredmd_description.csv"), delimiter=";")

    # select variables with description
    raw_fredmd_df = raw_fredmd_df[list(set(list(des_fredmd_df["fred"])) & set(list(raw_fredmd_df.columns)))]

    # select price data with logdiff transf
    des_prices = des_fredmd_df.loc[(des_fredmd_df["group"] == "Prices")&(des_fredmd_df["tcode"] == 6)]
    prices_var_names = des_prices["fred"]
    fredmd_prices_df = raw_fredmd_df[list(prices_var_names)]
    change_fredmd_prices_df = np.log(fredmd_prices_df).diff()

    # add log diff prices to original data
    selected_transf_fredmd_df = transf_fredmd_df.drop(list(prices_var_names), axis=1)
    target_df = pd.concat([selected_transf_fredmd_df, change_fredmd_prices_df], axis=1)

    # delete rows with NaN in all columns
    raw_fredmd_df = raw_fredmd_df.dropna(how="all")
    target_df = target_df.dropna(how="all")

    # fix index name
    target_df.index.name = "date"
    raw_fredmd_df.index.name = "date"

    # export
    target_df.loc[start_date:].to_csv(os.path.join(INPUTS_PATH,  "fredmd_transf_df.csv"))
    raw_fredmd_df.loc[start_date:].to_csv(os.path.join(INPUTS_PATH,  "fredmd_raw_df.csv"))


if __name__ == "__main__":
    gen_fred_dataset(start_date=START_DATE)
  



