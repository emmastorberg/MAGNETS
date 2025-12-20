from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error
from sktime.regression.kernel_based import RocketRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import sys
import torch

sys.path.append("magnets")

from data.staticbridge import StaticBridgeDataset
from data.synth import SynthRegressionDataset
from data.tsregression import TSRegressionDataset
from models.cbm_discovery import ConceptDiscoveryModel
from models.cnn import CNN
from models.gatsm import GATSM
from models.natm import NATM, NATMTime, NATMFeature

import wandb

# Initialize API
api = wandb.Api()

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
    999,
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
    1,
]

if "DATASET_IDX" not in st.session_state:
    st.session_state.DATASET_IDX = 2

entity = "YOUR_USERNAME"  # Your WandB username or team name
project = "YOUR_PROJECT_NAME"  # Your WandB project name


@st.cache_data
def get_parameters(run_id):
    # Construct the full run path
    run_path = f"{entity}/{project}/{run_id}"

    # Fetch the run
    run = api.run(run_path)

    # Get the parameters (config)
    parameters = run.config

    return parameters


@st.cache_data
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


@st.cache_data
def evaluate_baseline_mean(y_train, y_test):
    y_train_mean =y_train.mean()
    y_pred = np.full_like(y_test, y_train_mean)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred)) * TARGET_SCALING[st.session_state.DATASET_IDX]
    r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2))
    return {"RMSE": rmse, "R2": r2}


@st.cache_data
def evaluate_baseline_linear(X_train, y_train, X_test, y_test):
    reg = LinearRegression().fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred)) * TARGET_SCALING[st.session_state.DATASET_IDX]
    r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2))
    return {"RMSE": rmse, "R2": r2}


@st.cache_data
def evaluate_baseline_lasso(X_train, y_train, X_test, y_test):
    reg = LassoCV(alphas=np.logspace(-3, 3, 10)).fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred)) * TARGET_SCALING[st.session_state.DATASET_IDX]
    r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2))
    print("[LassoCV] Best alpha = ", reg.alpha_)
    return {"RMSE": rmse, "R2": r2}


@st.cache_data
def evaluate_baseline_ridge(X_train, y_train, X_test, y_test):
    reg = RidgeCV(alphas=np.logspace(-3, 3, 10)).fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred)) * TARGET_SCALING[st.session_state.DATASET_IDX]
    r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2))
    print("[RidgeCV] Best alpha = ", reg.alpha_)
    return {"RMSE": rmse, "R2": r2}


@st.cache_data
def evaluate_baseline_randomforest(X_train, y_train, X_test, y_test):
    reg = RandomForestRegressor().fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred)) * TARGET_SCALING[st.session_state.DATASET_IDX]
    r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2))
    return {"RMSE": rmse, "R2": r2}


@st.cache_data
def evaluate_baseline_rocket(X_train, y_train, X_test, y_test):
    rocket = RocketRegressor(num_kernels=10000, rocket_transform="rocket")
    rocket.fit(X_train, y_train)
    y_pred = rocket.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred)) * TARGET_SCALING[st.session_state.DATASET_IDX]
    r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2))
    return {"RMSE": rmse, "R2": r2}


@st.cache_data
def evaluate_baseline_multirocket(X_train, y_train, X_test, y_test):
    rocket = RocketRegressor(num_kernels=10000, rocket_transform="multirocket")
    rocket.fit(X_train, y_train)
    y_pred = rocket.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred)) * TARGET_SCALING[st.session_state.DATASET_IDX]
    r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2))
    return {"RMSE": rmse, "R2": r2}


@st.cache_data
def evaluate_cnn(_model, _test_dl, run_id):
    _model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in _test_dl:
            y_true.append(y)
            y_pred.append(_model(x))
    y_true = torch.cat(y_true).squeeze().detach().cpu().numpy()
    y_pred = torch.cat(y_pred).squeeze().detach().cpu().numpy()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) * TARGET_SCALING[st.session_state.DATASET_IDX]
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))
    return {"RMSE": rmse, "R2": r2, "y_pred": y_pred}


@st.cache_data
def evaluate_gatsm(_model, _test_dl, input_length, run_id):
    _model.eval()
    y_true = []
    y_pred = []
    t = torch.tensor([input_length - 1], dtype=torch.int64)
    with torch.no_grad():
        for x, y in _test_dl:
            y_true.append(y)
            y_pred.append(_model(x, t).squeeze(dim=-1))
    y_true = torch.cat(y_true).squeeze().detach().cpu().numpy()
    y_pred = torch.cat(y_pred).squeeze().detach().cpu().numpy()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) * TARGET_SCALING[st.session_state.DATASET_IDX]
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))
    return {"RMSE": rmse, "R2": r2, "y_pred": y_pred}


@st.cache_data
def evaluate_model(_model, _test_dl, run_id):
    _model.eval()
    y_true = []
    y_pred = []
    masks = []
    with torch.no_grad():
        for x, y in _test_dl:
            y_true.append(y)
            out_dict = _model(x)
            y_pred.append(out_dict["y_pred"])
            masks.append(out_dict["masks"])
    y_true = torch.cat(y_true).squeeze().detach().cpu().numpy()
    y_pred = torch.cat(y_pred).squeeze().detach().cpu().numpy()
    masks = torch.cat(masks).detach().cpu().numpy()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) * TARGET_SCALING[st.session_state.DATASET_IDX]
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))
    return {"RMSE": rmse, "R2": r2, "y_pred": y_pred, "masks": masks}


def main():
    st.set_page_config(
        page_title="MAGNETS",
        page_icon="assets/logo_IMOS.png",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    dataset = st.sidebar.selectbox("Dataset", DATASETS, index=st.session_state.DATASET_IDX)
    st.session_state.DATASET_IDX = DATASETS.index(dataset)

    st.header("MAGNETS: Mask-and-AGgregate NEtworks for Time Series")

    if DATASET_TYPES[st.session_state.DATASET_IDX] == "Synthetic":
        runs = api.runs(project, filters={"config.dataset": "Synth", "config.gt_fun": dataset})
    else:
        runs = api.runs(project, filters={"config.dataset": dataset})

    if runs:
        run_idx = st.sidebar.selectbox("Run", range(len(runs)), format_func=lambda i: runs[i].name.removeprefix(f"{dataset}-"))
        run = runs[run_idx]

    train_ds = load_data(st.session_state.DATASET_IDX, mode="train")
    test_ds = load_data(st.session_state.DATASET_IDX, mode="test")

    input_dim = train_ds[0][0].shape[-2]
    input_length = train_ds[0][0].shape[-1]
    target_scaling = TARGET_SCALING[st.session_state.DATASET_IDX]

    cols = st.columns([3, 1, 1, 1, 1, 1])
    cols[0].metric("Dataset ", dataset)
    cols[1].metric("Dimensions", input_dim)
    cols[2].metric("Length", input_length)
    cols[3].metric("Target scaling", TARGET_SCALING[st.session_state.DATASET_IDX])
    cols[4].metric("Logscale", LOGSCALE[st.session_state.DATASET_IDX])
    cols[5].metric("Subsampling", SUBSAMPLE[st.session_state.DATASET_IDX])

    with st.expander("Data summary"):

        col0,col1 = st.columns(2)

        with col0:

            cols = st.columns(2)

            for col, ds, mode in zip(cols, [train_ds, test_ds], ["Train", "Test"]):
                with col:
                    st.write(f"## {mode}")
                    st.write("Samples", len(ds))

                    st.write("#### Y")
                    st.write("Min: {:.2f} / Max: {:.2f} / Mean: {:.2f} / Std: {:.2f}".format(ds.Y.min(), ds.Y.max(), ds.Y.mean(), ds.Y.std()))
                    fig, ax = plt.subplots()
                    ax.hist(ds.Y, bins=50)
                    # ax.set_title("Target distribution")
                    st.pyplot(fig)

                    for d in range(input_dim):
                        st.write(f"#### X dim {d}")
                        st.write("Min: {:.2f} / Max: {:.2f} / Mean: {:.2f} / Std: {:.2f}".format(ds.X[:, d, :].min(), ds.X[:, d, :].max(), ds.X[:, d, :].mean(), ds.X[:, d, :].std()))
                        if len(ds) > 1000:
                            st.write("Distribution not displayed for performance reasons")
                        else:
                            fig, ax = plt.subplots()
                            ax.hist(ds.X[:, d, :].flatten(), bins=50, density=True)
                            # ax.set_title(f"Feature {d} distribution")
                            st.pyplot(fig)

        # Plot random samples
        with col1:
            col11, col12, col13 = st.columns(3)
            mode = col11.radio("Mode", ["Test", "Train"])
            n_samples = col12.slider("Number of samples", min_value=1, max_value=100, value=1)
            color_by = col13.selectbox("Color by", ["ID", "Target"])

            ds = train_ds if mode == "Train" else test_ds

            samples_idx = np.random.choice(len(ds), n_samples, replace=False)
            fig = make_subplots(
                rows=input_dim,
                cols=1,
                subplot_titles=[f"Dim {d+1}" for d in range(input_dim)],
                vertical_spacing=0.01
            )

            cmap = plt.cm.get_cmap('viridis')

            for d in range(input_dim):
                for idx in samples_idx:
                    x, y = ds[idx]
                    if color_by == "Target":
                        rgba = cmap(y)
                        color = f"rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, {rgba[3]})"
                        fig.add_trace(go.Scatter(
                            x=np.arange(len(x[d])),
                            y=x[d],
                            mode='lines',
                            line=dict(color=color)
                        ), row=d+1, col=1)
                    else:
                        fig.add_trace(go.Scatter(
                            x=np.arange(len(x[d])),
                            y=x[d], mode='lines',
                            name=str(idx),
                            legendgroup=str(idx),
                        ), row=d+1, col=1)

                if color_by == "Target":
                    fig.update_layout(coloraxis=dict(colorscale='viridis', cmin=min(ds.Y[samples_idx]), cmax=max(ds.Y[samples_idx])), showlegend=False)
                else:
                    fig.update_layout(showlegend=True)

            fig.update_layout(
                height=300 * input_dim,
                # width=800,
                title_text="Random Samples"
            )
            st.plotly_chart(fig, use_container_width=True)

    if not runs:
        st.stop()

    with st.expander("Model parameters"):
        st.write("Run ID:", run.id)
        # fetch hyperparameters from wandb
        parameters = get_parameters(run.id)
        st.write(parameters)

    # with st.expander("Model analysis", expanded=True):

    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=8, num_workers=0, shuffle=False, drop_last=False)

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

    # Non-linear baselines (commented out for speed)
    # print("RandomForest")
    # baselines["RandomForest"] = evaluate_baseline_randomforest(X_train, y_train, X_test, y_test)
    # print(baselines["RandomForest"])
    # print("Rocket")
    # baselines["Rocket"] = evaluate_baseline_rocket(X_train, y_train, X_test, y_test)
    # print(baselines["Rocket"])
    # print("MultiRocket")
    # baselines["MultiRocket"] = evaluate_baseline_multirocket(X_train, y_train, X_test, y_test)
    # print(baselines["MultiRocket"])

    # Download model weights if needed
    if DATASET_TYPES[st.session_state.DATASET_IDX] == "Synthetic":
        output_dir = "output-synth"
    elif DATASET_TYPES[st.session_state.DATASET_IDX] == "Real":
        output_dir = "output-tsreg"
    elif DATASET_TYPES[st.session_state.DATASET_IDX] == "StaticBridge":
        output_dir = "output-bridge"
    else:
        raise ValueError("Unsupported dataset type: ", DATASET_TYPES[st.session_state.DATASET_IDX])

    # Load model
    if run.name.startswith(f"{dataset}-cnn"):
        model = CNN.load_from_checkpoint(
            f"{output_dir}/{run.name}/checkpoints/last.ckpt",
            input_dim=input_dim, lr=parameters["learning_rate"],
            target_scaling=target_scaling,
            input_length=input_length
        ).eval()

        # Evaluate model
        results = evaluate_cnn(model, test_dl, run.id)

    elif run.name.startswith(f"{dataset}-natm_time"):
        model = NATMTime.load_from_checkpoint(
            f"{output_dir}/{run.name}/checkpoints/last.ckpt",
            input_dim=input_dim,
            hidden_units=32,
            target_scaling=target_scaling,
        ).eval()

        # Evaluate model
        results = evaluate_cnn(model, test_dl, run.id)

    elif run.name.startswith(f"{dataset}-natm_feature"):
        model = NATMFeature.load_from_checkpoint(
            f"{output_dir}/{run.name}/checkpoints/last.ckpt",
            input_length=input_length,
            hidden_units=32,
            target_scaling=target_scaling,
        ).eval()

        # Evaluate model
        results = evaluate_cnn(model, test_dl, run.id)

    elif run.name.startswith(f"{dataset}-natm"):
        model = NATM.load_from_checkpoint(
            f"{output_dir}/{run.name}/checkpoints/last.ckpt",
            input_dim=input_dim,
            input_length=input_length,
            hidden_units=32,
            target_scaling=target_scaling,
        ).eval()

        # Evaluate model
        results = evaluate_cnn(model, test_dl, run.id)

    elif run.name.startswith(f"{dataset}-gatsm"):
        model = GATSM.load_from_checkpoint(
            f"{output_dir}/{run.name}/checkpoints/last.ckpt",
            task="m2o:reg",
            n_features=input_dim,
            n_outputs=1,
            nbm_hidden_dims=[256, 256, 128],
            nbm_n_bases=100,
            nbm_batchnorm=False,
            nbm_dropout=0.0,
            attn_emb_size=64,
            attn_n_heads=8,
            attn_dropout=0.0,
            lr=1e-3,
            weight_decay=1e-5,
            target_scaling=target_scaling,
        ).eval()

        # Evaluate model
        results = evaluate_gatsm(model, test_dl, input_length, run.id)

    elif run.name.startswith(f"{dataset}-magnets") or run.name.startswith(f"{dataset}-cdm"):
        model = ConceptDiscoveryModel.load_from_checkpoint(
            f"{output_dir}/{run.name}/checkpoints/last.ckpt",
            n_concepts=parameters["n_concepts"],
            feature_extraction="learned",
            mask_generator="unet",
            n_tasks=1,
            input_dim=input_dim,
            input_length=input_length,
            latent_dim=128,
            task_loss_weight=parameters["task_loss_weight"],
            sparsity_loss_weight=parameters["sparsity_loss_weight"],
            gsat_loss_weight=parameters["gsat_loss_weight"],
            gsat_r=parameters["gsat_r"],
            connect_loss_weight=parameters["connect_loss_weight"],
            ortho_loss_weight=parameters["ortho_loss_weight"],
            concept_ortho_loss_weight=parameters["concept_ortho_loss_weight"],
            mask_sparsity_loss_weight=parameters["mask_sparsity_loss_weight"],
            learning_rate=parameters["learning_rate"],
            n_masks=parameters["n_masks"],
            c2y_layers=parameters["c2y_layers"],
            aggs=parameters["aggs"],
            target_scaling=target_scaling,
        ).eval()

        # Evaluate model
        results = evaluate_model(model, test_dl, run.id)

    baselines["This model"] = {"RMSE": results["RMSE"], "R2": results["R2"]}

    # unscale all RMSE values if needed
    # for k in baselines:
    #     baselines[k]["RMSE"] /= TARGET_SCALING[st.session_state.DATASET_IDX]

    results_cols = st.columns([4, 1])
    # Display all results
    results_cols[0].table(pd.DataFrame(baselines).T)

    # Display scatter plots of true VS pred
    fig = px.scatter(x=test_ds.Y, y=results["y_pred"], hover_name=np.arange(len(test_ds)), labels={"x": "True", "y": "Predicted"}, height=250)
    fig.add_shape(
        type="line", line=dict(dash="dash"),
        x0=test_ds.Y.min(), y0=test_ds.Y.min(),
        x1=test_ds.Y.max(), y1=test_ds.Y.max()
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    if LOGSCALE[st.session_state.DATASET_IDX]:
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
    results_cols[1].plotly_chart(fig)


    # All the rest is specific to MAGNETS
    if not run.name.startswith(f"{dataset}-magnets") and not run.name.startswith(f"{dataset}-cdm"):
        st.stop()

    # st.write(model.c2y_model[0].weight, model.c2y_model[0].bias)

    col_masks, col_bottleneck, col_c2y = st.columns([3, 1.5, 2])

    with col_masks:
        st.write("## Mask generator")
        col_masks = st.columns(3)
        mode = col_masks[0].radio("Mode", ["Test", "Train"], key="masks_radio")
        ds = train_ds if mode == "Train" else test_ds
        soft = col_masks[1].checkbox("Soft mask", value=True)
        idx = col_masks[2].selectbox("Select a sample", range(len(ds)))

        st.write(f"True: {ds.Y[idx].item():.2f} / Predicted: {results['y_pred'][idx]:.2f}")

        fig = make_subplots(
            rows=input_dim,
            cols=1,
            # subplot_titles=[f"Dim {d+1}" for d in range(input_dim)],
            # vertical_spacing=0.05
        )

        for d in range(input_dim):
            fig.add_trace(go.Heatmap(
                x=np.arange(input_length),
                y=np.arange(model.n_masks),
                z=results["masks"][idx, :, d] if soft else (results["masks"][idx, :, d] > 0.5).astype(float),
                colorscale="Purples",
                showscale=False,
                zmin=0,
                zmax=1,
            ), row=d+1, col=1)
            fig.update_yaxes(title_text=f"Dim {d+1}", row=d+1, col=1)

        # Customize layout
        fig.update_layout(
            height=150 * input_dim,
            xaxis=dict(side="top"),  # Move x-axis labels to top
            yaxis=dict(
                tickvals=list(range(model.n_masks)),
                ticktext=[f"{m+1}" for m in range(model.n_masks)],
            ),
            margin=dict(l=0, r=0, t=0, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)
        # fig.write_image(f"{run.name}-masks.pdf", format="pdf")

        selected_mask = st.selectbox("Select a mask", range(model.n_masks), format_func=lambda i: i+1)

        fig = make_subplots(
            rows=input_dim,
            cols=1,
            # subplot_titles=[f"Dim {d+1}" for d in range(input_dim)],
            # vertical_spacing=0.05
        )

        Xnp = test_ds.X[idx].T
        T = Xnp.shape[0]
        x_range = np.arange(T)

        for d in range(input_dim):
            fig.add_trace(go.Scatter(
                x=x_range,
                y=Xnp[:, d],
                mode='lines',
                # line=dict(color='black'),
                showlegend=False
            ),
            row=d+1, col=1)
            fig.add_trace(go.Heatmap(
                x=np.linspace(min(x_range), max(x_range), len(x_range) + 1),
                y=[min(Xnp[:, d]), max(Xnp[:, d])],
                z=results["masks"][idx, selected_mask, d, None] if soft else (results["masks"][idx, selected_mask, d, None] > 0.5).astype(float),
                colorscale='Purples',
                showscale=False,
                zmin=0,
                zmax=1,
                opacity=0.5),
                row=d+1, col=1
            )
            fig.update_yaxes(title_text=f"Dim {d+1}", row=d+1, col=1)

        fig.update_layout(
            height=200 * input_dim,
            margin=dict(l=0, r=0, t=0, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)
        # fig.write_image(f"{run.name}-masks-overlay.pdf", format="pdf")

        mask_l1 = np.abs(results["masks"]).mean()
        thr = 1e-2
        mask_sparsity = (np.abs(results["masks"]) < thr).mean()
        st.write(
            f"""L1 norm: {mask_l1:.2e}<br/>
            Mask Sparsity @ {thr}: {mask_sparsity:.2f}""",
            unsafe_allow_html=True
        )

    with col_bottleneck:
        st.write("## Concept bottleneck")
        abs_value = st.checkbox("Absolute value", value=True)

        weights = model.bottleneck.weight.detach().numpy().T

        cell_size = 25

        fig = px.imshow(
            np.abs(weights) if abs_value else weights,
            origin="upper",
            labels=dict(x="Concept", y="Feature", color="Weight"),
            x=[f"Concept {c+1}" for c in range(model.n_concepts)],
            y=[f"Dim {d+1} - {agg} - {m+1}" for d in range(input_dim) for agg in model.aggs for m in range(model.n_masks)],
            # y=[m for d in range(input_dim) for agg in model.aggs for m in range(model.n_masks)],
            height=cell_size*weights.shape[0],
            width=cell_size*weights.shape[1],
            color_continuous_scale="Purples" if abs_value else "RdBu_r",
            zmin=-np.abs(weights).max() if not abs_value else 0,
            zmax=np.abs(weights).max() if not abs_value else np.abs(weights).max(),
            # aspect="auto",
        )

        for d in range(input_dim):
            for a in range(len(model.aggs)):
                fig.add_shape(
                    type="line",
                    x0=-0.5,
                    y0=d * model.n_masks * len(model.aggs) + a * model.n_masks - 0.5,
                    x1=model.n_concepts - 0.5,
                    y1=d * model.n_masks * len(model.aggs) + a * model.n_masks - 0.5,
                    line=dict(color="black", width=1.5),
                )
            fig.add_shape(
                type="line",
                x0=-0.5,
                y0=d * model.n_masks * len(model.aggs) - 0.5,
                x1=model.n_concepts - 0.5,
                y1=d * model.n_masks * len(model.aggs) - 0.5,
                line=dict(color="black", width=3),
            )
            fig.add_annotation(
                x=-0.1,
                y=d * model.n_masks * len(model.aggs) + (len(model.aggs) * model.n_masks) / 2 - 0.5,
                text=f"Dim {d+1}",
                showarrow=False,
                font=dict(size=18),
                xref="paper",
                yref="y",
                textangle=-90,
            )
        fig.add_shape(
                type="line",
                x0=-0.5,
                y0=input_dim * model.n_masks * len(model.aggs) - 0.5,
                x1=model.n_concepts - 0.5,
                y1=input_dim * model.n_masks * len(model.aggs) - 0.5,
                line=dict(color="black", width=3),
            )

        for i in range(-1, weights.shape[1]):
            fig.add_shape(type="line", x0=0.5 + i, y0=-0.5, x1=0.5 + i, y1=weights.shape[0] - 0.5, line=dict(color="gray", width=0.5))

        for i in range(weights.shape[0]):
            fig.add_shape(type="line", x0=-0.5, y0=0.5 + i, x1=weights.shape[1] - 0.5, y1=0.5 + i, line=dict(color="gray", width=0.5))

        # Customize layout
        fig.update_layout(
            # title="Bottleneck weights",
            xaxis=dict(side="top", title=None, automargin=True),  # Move x-axis labels to top
            # coloraxis_showscale=False,  # Remove colorbar
            coloraxis_colorbar=dict(
                len=128,
                lenmode="pixels",
                # yanchor="bottom",
                # y=0,
            ),
            yaxis=dict(
                title=None,
                tickvals=list(range(input_dim * model.n_masks * len(model.aggs))),
                ticktext=[f"{m+1}" for d in range(input_dim) for agg in model.aggs for m in range(model.n_masks)],
                automargin=True,
            ),
            margin=dict(l=50, r=0, t=0, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)
        # fig.write_image(f"{run.name}-bottleneck.pdf", format="pdf", scale=0.5)

        # Add colorbar
        # fig = px.imshow(
        #     [[0, np.abs(weights).max()]] if abs_value else [[-np.abs(weights).max(), 0, np.abs(weights).max()]],
        #     color_continuous_scale="Purples" if abs_value else "RdBu_r",
        #     zmin=-np.abs(weights).max() if not abs_value else 0,
        #     zmax=np.abs(weights).max() if not abs_value else np.abs(weights).max(),
        # )

        # st.plotly_chart(fig, use_container_width=True)

        l1 = torch.norm(model.bottleneck.weight, p=1) / model.bottleneck.weight.numel()
        thr = 1e-3
        sparsity = (torch.abs(model.bottleneck.weight) < thr).sum() / model.bottleneck.weight.numel()
        st.write(
            f"""max: {torch.max(model.bottleneck.weight):.2e} / min: {torch.min(model.bottleneck.weight):.2e} / mean: {torch.mean(model.bottleneck.weight):.2e}<br/>
            L1 norm: {l1.item():.2e}<br/>
            Sparsity @ {thr}: {sparsity:.2f}""",
            unsafe_allow_html=True
        )

    with col_c2y:
        st.write("## Final predictor")
        abs_value_c2y = st.checkbox("Absolute value", value=True, key="abs_value_c2y")

        weights_c2y = model.c2y_model[0].weight.detach().numpy()

        cell_size = 200

        fig = px.imshow(
            np.abs(weights_c2y) if abs_value_c2y else weights_c2y,
            x=[f"Concept {c+1}" for c in range(model.n_concepts)],
            y=["" for o in range(weights_c2y.shape[0])],
            origin="upper",
            labels=dict(x="Concept", y="Output", color="Weight"),
            height=cell_size*weights_c2y.shape[0],
            width=cell_size*weights_c2y.shape[1],
            color_continuous_scale="Purples" if abs_value_c2y else "RdBu_r",
            zmin=-np.abs(weights_c2y).max() if not abs_value_c2y else 0,
            zmax=np.abs(weights_c2y).max() if not abs_value_c2y else None,
        )

        # Customize layout
        fig.update_layout(
            # title="C2Y weights",
            xaxis=dict(side="top", title=None),  # Move x-axis labels to top
            yaxis=dict(title=None),
            # coloraxis_showscale=False,  # Remove colorbar
            coloraxis_colorbar=dict(
                len=128,
                lenmode="pixels",
                # yanchor="bottom",
                # y=0,
            ),
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show formula of computed
        formula_thr_w = st.slider("Threshold (weights)", min_value=0.0, max_value=float(np.abs(weights).max()), value=max(0.01, np.quantile(np.abs(weights), 0.95)))
        formula_thr_c2y = st.slider("Threshold (c2y)", min_value=0.0, max_value=float(np.abs(weights_c2y).max()), value=np.quantile(np.abs(weights_c2y), 0.95))

        formula = {}
        for c in range(model.n_concepts):
            if np.abs(weights_c2y[0, c]) >= formula_thr_c2y and (np.abs(weights[:, c]) >= formula_thr_w).any():
                formula[f"Concept {c+1}"] = {"weight": weights_c2y[0, c], "features": {}}
                for d in range(input_dim):
                    for a in range(len(model.aggs)):
                        for m in range(model.n_masks):
                            if np.abs(weights[d * len(model.aggs) * model.n_masks + a * model.n_masks + m, c]) >= formula_thr_w:
                                formula[f"Concept {c+1}"]["features"][f"{model.aggs[a]} of Dim {d+1} over Mask {m+1}"] = weights[d * len(model.aggs) * model.n_masks + a * model.n_masks + m, c]

        # Render the formula
        st.write("### Analytical formula")
        st.write(
            "#### Y = " + " + ".join([
                f"{v['weight']:.2f}" + "$\\times$" + f"{k}"
                for k, v in formula.items()
            ])
        )
        st.write(
            "#### &nbsp;&nbsp; = " + " + ".join([
                f"{v['weight']:.2f}" + "$\\times$(" + " + ".join([f"{v2:.2f}" + "$\\times$" + f"{k2}" for k2, v2 in v['features'].items()]) + ")"
                for k, v in formula.items()
            ])
        )


if __name__ == '__main__':
    main()
