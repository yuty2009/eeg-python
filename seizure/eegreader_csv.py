# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


def read_csvdata(filepath):
    df = pd.read_csv(filepath, header=0, index_col=0)
    df1 = df.drop(["y"], axis=1)
    labels = df["y"].values - 1
    data = []
    for i, row in df1.iterrows():
        data.append(row.tolist())
    return np.array(data), labels


if __name__ == '__main__':

    datapath = 'e:/eegdata/seizure/Epileptic_Seizure_Recognition/data.csv'
    data, labels = read_csvdata(datapath)
    print(len(data))