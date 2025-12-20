from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error
from sktime.regression.kernel_based import RocketRegressor
import numpy as np
import pandas as pd
import sys
import torch

sys.path.append("magnets")

from data.staticbridge import StaticBridgeDataset
from data.synth import SynthRegressionDataset
from data.tsregression import TSRegressionDataset


DATA_PATH_REAL = "/path/to/ts_regression_datasets"
DATA_PATH_SYNTH = "../datasets"
DATA_PATH_BRIDGE = "../datasets"

DATASETS = [
    "WindTurbinePower",
    "HouseholdPowerConsumption1_nmv",
    "BenzeneConcentration_nmv",
    "StaticBridge",
    "area_over_threshold",
    "area_over_threshold_bivariate",
    "area_over_threshold_trivariate_1",
    "area_over_threshold_trivariate_2",
]

DATASET_TYPES = [
    "Real",
    "Real",
    "Real",
    "StaticBridge",
    "Synthetic",
    "Synthetic",
    "Synthetic",
    "Synthetic",
]

TARGET_SCALING = [
    1000,
    1000,
    40.26,
    0.0039039507713784886,
    1,
    1,
    1,
    1,
]

LOGSCALE = [
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
]

SUBSAMPLE = [
    1,
    10,
    1,
    1,
    1,
    1,
    1,
    1,
]


def load_data(dataset_idx, mode="test"):
    if DATASET_TYPES[dataset_idx] == "Synthetic":
        return SynthRegressionDataset(
            path=DATA_PATH_SYNTH,
            mode=mode,
            gt_fun=DATASETS[dataset_idx],
        )
    elif DATASET_TYPES[dataset_idx] == "Real":
        return TSRegressionDataset(
            path=DATA_PATH_REAL,
            problem=DATASETS[dataset_idx],
            mode=mode,
            scaling="none",
            target_scaling=TARGET_SCALING[dataset_idx],
            logscale=LOGSCALE[dataset_idx],
            subsampling=SUBSAMPLE[dataset_idx],
        )
    elif DATASET_TYPES[dataset_idx] == "StaticBridge":
        return StaticBridgeDataset(
            path=DATA_PATH_BRIDGE,
            bridges=[0, 1, 2, 3, 4] if mode == "train" else [5],
            target_scaling=TARGET_SCALING[dataset_idx],
        )
    else:
        raise ValueError("Unsupported dataset type: ", DATASET_TYPES[dataset_idx])


def evaluate_baseline_mean(y_train, y_test):
    y_train_mean =y_train.mean()
    y_pred = np.full_like(y_test, y_train_mean)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred)) * TARGET_SCALING[DATASET_IDX]
    r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2))
    return {"RMSE": rmse, "R2": r2}


def evaluate_baseline_linear(X_train, y_train, X_test, y_test):
    reg = LinearRegression().fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred)) * TARGET_SCALING[DATASET_IDX]
    r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2))
    return {"RMSE": rmse, "R2": r2}


def evaluate_baseline_lasso(X_train, y_train, X_test, y_test):
    reg = LassoCV(alphas=np.logspace(-3, 3, 10)).fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred)) * TARGET_SCALING[DATASET_IDX]
    r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2))
    print("[LassoCV] Best alpha = ", reg.alpha_)
    return {"RMSE": rmse, "R2": r2}


def evaluate_baseline_ridge(X_train, y_train, X_test, y_test):
    reg = RidgeCV(alphas=np.logspace(-3, 3, 10)).fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred)) * TARGET_SCALING[DATASET_IDX]
    r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2))
    print("[RidgeCV] Best alpha = ", reg.alpha_)
    return {"RMSE": rmse, "R2": r2}


def evaluate_baseline_randomforest(X_train, y_train, X_test, y_test):
    reg = RandomForestRegressor(n_jobs=-1).fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred)) * TARGET_SCALING[DATASET_IDX]
    r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2))
    return {"RMSE": rmse, "R2": r2}


def evaluate_baseline_rocket(X_train, y_train, X_test, y_test):
    rocket = RocketRegressor(num_kernels=10000, rocket_transform="rocket", n_jobs=-1)
    rocket.fit(X_train, y_train)
    y_pred = rocket.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred)) * TARGET_SCALING[DATASET_IDX]
    r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2))
    return {"RMSE": rmse, "R2": r2}


def evaluate_baseline_multirocket(X_train, y_train, X_test, y_test):
    rocket = RocketRegressor(num_kernels=10000, rocket_transform="multirocket", n_jobs=-1)
    rocket.fit(X_train, y_train)
    y_pred = rocket.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred)) * TARGET_SCALING[DATASET_IDX]
    r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2))
    return {"RMSE": rmse, "R2": r2}


global DATASET_IDX

def main():

    for dataset_idx in range(len(DATASETS)):

        global DATASET_IDX
        DATASET_IDX = dataset_idx

        train_ds = load_data(DATASET_IDX, mode="train")
        test_ds = load_data(DATASET_IDX, mode="test")

        # Evaluate some baselines
        baselines = {}
        X_train = train_ds.X.reshape(train_ds.X.shape[0], -1)
        X_test = test_ds.X.reshape(test_ds.X.shape[0], -1)
        y_train = train_ds.Y
        y_test = test_ds.Y
        if isinstance(X_train, torch.Tensor):
            X_train = X_train.cpu().numpy()
            X_test = X_test.cpu().numpy()
            y_train = y_train.cpu().numpy()
            y_test = y_test.cpu().numpy()

        # Mean of training set
        baselines["Mean"] = evaluate_baseline_mean(y_train, y_test)

        # Linear regression
        baselines["Linear"] = evaluate_baseline_linear(X_train, y_train, X_test, y_test)
        baselines["Lasso"] = evaluate_baseline_lasso(X_train, y_train, X_test, y_test)
        baselines["Ridge"] = evaluate_baseline_ridge(X_train, y_train, X_test, y_test)

        # Non-linear baselines
        baselines["RandomForest"] = evaluate_baseline_randomforest(X_train, y_train, X_test, y_test)
        baselines["Rocket"] = evaluate_baseline_rocket(X_train, y_train, X_test, y_test)
        baselines["MultiRocket"] = evaluate_baseline_multirocket(X_train, y_train, X_test, y_test)

        # Print all baselines
        print(f"Baselines for {DATASETS[DATASET_IDX]}:")
        for name, result in baselines.items():
            print(f"{name}: RMSE = {result['RMSE']:.4f}, R2 = {result['R2']:.4f}")
        print("----------------------------------------")
        # Save results to CSV
        df = pd.DataFrame(baselines).T
        df.to_csv(f"baselines_{DATASETS[DATASET_IDX]}.csv")


if __name__ == '__main__':
    main()
