import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import find_peaks


def area_over_threshold(x):
    return x[x > 0.5].sum() / 5.16


def time_over_threshold(x):
    return (x > 0.5).mean()


def max_value(x):
    return x.max()


def max_loc(x):
    return np.argmax(x)/x.shape[1]


def loc_max_loc_min_area_over_threshold(x):
    A, B, C, D = 15, -9, 7, 1
    y = 0.0
    T = x.shape[1]
    threshold = 0.5
    for d in range(x.shape[0]):
        condition = x[d] > threshold
        y += A * np.argmax(x[d])/T + B * np.argmin(x[d])/T + C * x[d].max() + D * x[d, condition].sum()
    return y


def amplitude(x):
    return x.max() - x.min()


def area_over_threshold_bivariate(x):
    return x[0, x[1] > 0.5].sum() / 2.57


def area_over_threshold_trivariate_1(x):
    return x[0, x[1] > x[2]].sum() / 8.55


def area_over_threshold_trivariate_2(x):
    A, B, C = 1, 5, -2
    return (A * x[0, x[1] > x[2]].sum() + B * x[2, x[0] > x[1]].sum() + C * x[1, x[2] > x[0]].sum()) / 48.30


def find_peaks_(x):
    return find_peaks(x, height=0.25, distance=10)[0]


def distance_peaks(x):
    # distance between first and second peak; 0 if single peak
    peaks = find_peaks_(x[0])
    if len(peaks) < 2:
        return 0
    return (peaks[1] - peaks[0]) / x.shape[1]


def diff_height_peaks(x):
    # difference in height between first and second peak; 0 if single peak
    peaks = find_peaks_(x[0])
    if len(peaks) < 2:
        return 0
    return x[0, peaks[1]] - x[0, peaks[0]]


class SynthRegressionDataset(Dataset):

    def __init__(
        self,
        path,
        mode="train",
        gt_fun="max",
        ):

        self.mode = mode

        if gt_fun == "max":
            ground_truth_fun = max_value
        elif gt_fun == "max_loc":
            ground_truth_fun = max_loc
        elif gt_fun == "area_over_threshold":
            ground_truth_fun = area_over_threshold
        elif gt_fun == "time_over_threshold":
            ground_truth_fun = time_over_threshold
        elif gt_fun == "loc_max_loc_min_area_over_threshold":
            ground_truth_fun = loc_max_loc_min_area_over_threshold
        elif gt_fun == "amplitude":
            ground_truth_fun = amplitude
        elif gt_fun == "distance_peaks":
            ground_truth_fun = distance_peaks
        elif gt_fun == "diff_height_peaks":
            ground_truth_fun = diff_height_peaks
        elif gt_fun == "area_over_threshold_bivariate":
            ground_truth_fun = area_over_threshold_bivariate
        elif gt_fun == "area_over_threshold_trivariate_1":
            ground_truth_fun = area_over_threshold_trivariate_1
        elif gt_fun == "area_over_threshold_trivariate_2":
            ground_truth_fun = area_over_threshold_trivariate_2

        if gt_fun.endswith("bivariate"):
            self.X = torch.load(os.path.join(path, f"X_{mode}_2.pt"))
        elif gt_fun.endswith("trivariate_1") or gt_fun.endswith("trivariate_2"):
            self.X = torch.load(os.path.join(path, f"X_{mode}_3.pt"))
        else:
            self.X = torch.load(os.path.join(path, f"X_{mode}_large.pt"))
        self.Y = torch.Tensor([ground_truth_fun(x) for x in self.X])

        print(self.X.shape, self.Y.shape)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.Y[i]
