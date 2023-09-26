import os
import pandas as pd

from settings import INPUTS_PATH

def gen_mixed_freq_dataset():
    df = pd.read_csv(os.path.join(INPUTS_PATH, "mixed_freq_raw_df.csv"))
    df.index = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df = df.sort_index()

    out_list = []
    for colname in df.columns:
        tmp_df = df[[colname]].dropna()
        tmp_df = tmp_df / (tmp_df).shift(1) - 1

        out_list.append(tmp_df)
    out_df = pd.concat(out_list, axis=1)

    out_df.to_csv(os.path.join(INPUTS_PATH, "mixed_freq_df.csv"))


if __name__ == "__main__":
    gen_mixed_freq_dataset()
  



