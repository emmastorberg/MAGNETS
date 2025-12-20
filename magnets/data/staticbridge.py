import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset


class StaticBridgeDataset(Dataset):

    def __init__(
        self,
        path,
        bridges=[],
        target_scaling=1.0,
    ):

        self.target_scaling = target_scaling

        df = pd.read_csv(os.path.join(path, "static_bridge_data.csv"))
        window = 24*60//10  # 144
        stride = window // 4

        # cols = ["v_0.25"]  # "Temperature", "Load", "v_0.33", "v_0.50"
        cols = ["v_0.25", "Temperature", "Load"]
        target = "Damage"

        # scale features and target
        df["v_0.25"] = (df["v_0.25"] - (-4.140912e-03)) / 3.555861e-03
        df["Temperature"]  = (df["Temperature"] - (-7.269117)) / 8.279703
        df["Load"] = (df["Load"] - 3.933580e+02) / 3.319209e+02
        df[target] /= target_scaling

        # make windows for each bridge
        self.X = []
        self.Y = []
        for bridge in bridges:
            df_ = df[df["run_id"] == bridge]
            # damage_increments = df_[target].diff().fillna(0)
            for i in range((len(df_)-window)//stride):
                self.X.append(df_.iloc[i*stride:i*stride+window][cols].values.T)
                if i == 0:
                    cum_damage = df_.iloc[i*stride+window-1][target]
                else:
                    cum_damage = df_.iloc[i*stride+window-1][target] - df_.iloc[i*stride-1][target]
                self.Y.append(cum_damage)

        self.X = np.array(self.X)
        self.Y = np.array(self.Y)

        print(self.X.shape, self.Y.shape)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype=torch.float32), torch.tensor(self.Y[i], dtype=torch.float32)
