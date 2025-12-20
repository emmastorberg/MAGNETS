from fire import Fire
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import os
import pytorch_lightning as pl
import torch

from data.staticbridge import StaticBridgeDataset
from data.synth import SynthRegressionDataset
from data.tsregression import TSRegressionDataset
from models.cbm_discovery import ConceptDiscoveryModel
from models.cnn import CNN
from models.gatsm import GATSM
from models.natm import NATM, NATMFeature, NATMTime


MODELS = ("linear", "cnn", "magnets", "natm", "natm_feature", "natm_time", "gatsm")


def main(
    output_dir: str,
    data_path: str,
    max_epochs: int,
    dataset: str = "Synth",
    eval: bool = False,
    model_type: str = "magnets",
    emb_size: int = 16,
    concepts: list[str] | None = None,
    n_concepts: int | None = None,
    supervised_concepts: bool = False,
    binary_concepts: bool = False,
    regressive_concepts: bool = False,
    n_masks: int = 10,
    mask_generator: str = "unet",
    task_loss_weight: float = 1.0,
    sparsity_loss_weight: float = 0.0,
    gsat_loss_weight: float = 0.0,
    gsat_r: float = 0.5,
    connect_loss_weight: float = 0.0,
    ortho_loss_weight: float = 0.0,
    concept_ortho_loss_weight: float = 0.0,
    mask_sparsity_loss_weight: float = 0.0,
    concept_entropy_loss_weight: float = 0.0,
    c2y_sparsity_loss_weight: float = 0.0,
    mask_smoothing: str | None = None,
    use_ste: bool = True,
    batch_size: int = 256,
    seed: int = 42,
    checkpoint: str = None,
    exclusive_concepts: bool = False,
    extra_dims: int = 0,
    boolean_cbm: bool = False,
    c2y_layers: list[int] = [],
    learning_rate: float = 1e-4,
    aggs: list[str] = ["sum"],
    kernel_size: int = 3,
    gt_fun: str = "area_over_threshold",  # ground-truth function for synthetic data
    subsampling: int = 1,
    target_scaling: float = 1.0,
    target_log: bool = False,
):

    if n_concepts is None:
        n_concepts = len(concepts)
    assert model_type in MODELS, f"model_type must be one of: ${MODELS}"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.set_num_threads(8)

    # tb_logger = pl_loggers.TensorBoardLogger(save_dir=output_dir)
    wandb_logger = pl_loggers.WandbLogger(project="MAGNETS", name=output_dir.split("/")[-1])
    wandb_logger.experiment.config.update({
        "dataset": dataset,
        "model_type": model_type,
        "emb_size": emb_size,
        "concepts": concepts,
        "n_concepts": n_concepts,
        "supervised_concepts": supervised_concepts,
        "binary_concepts": binary_concepts,
        "regressive_concepts": regressive_concepts,
        "n_masks": n_masks,
        "mask_generator": mask_generator,
        "task_loss_weight": task_loss_weight,
        "sparsity_loss_weight": sparsity_loss_weight,
        "gsat_loss_weight": gsat_loss_weight,
        "gsat_r": gsat_r,
        "connect_loss_weight": connect_loss_weight,
        "ortho_loss_weight": ortho_loss_weight,
        "concept_ortho_loss_weight": concept_ortho_loss_weight,
        "mask_sparsity_loss_weight": mask_sparsity_loss_weight,
        "concept_entropy_loss_weight": concept_entropy_loss_weight,
        "c2y_sparsity_loss_weight": c2y_sparsity_loss_weight,
        "mask_smoothing": mask_smoothing,
        "use_ste": use_ste,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "seed": seed,
        "exclusive_concepts": exclusive_concepts,
        "extra_dims": extra_dims,
        "boolean_cbm": boolean_cbm,
        "c2y_layers": c2y_layers,
        "learning_rate": learning_rate,
        "aggs": aggs,
        "kernel_size": kernel_size,
        "gt_fun": gt_fun,
        "subsampling": subsampling,
        "target_scaling": target_scaling,
    })

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    seed_everything(seed)

    if eval:
        if dataset == "Synth":
            test_ds = SynthRegressionDataset(data_path, mode="test", gt_fun=gt_fun)
            test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False)
        elif dataset == "StaticBridge":
            test_ds = StaticBridgeDataset(data_path, bridges=[5], target_scaling=target_scaling)
            test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=0, shuffle=False, drop_last=False)
        else :  # assume tsregression dataset
            test_ds = TSRegressionDataset(path=data_path, problem=dataset, mode="test", scaling="none", target_scaling=target_scaling, target_log=target_log, subsampling=subsampling)
            test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False)
        input_dim = test_ds.X.shape[1]
        input_length = test_ds.X.shape[2]
        print(f"Samples: {len(test_ds)} test, Input dim: {input_dim}, Input length: {input_length}")

        trainer = pl.Trainer(
            accelerator=device
        )

        if model_type == "cnn":
            model = CNN.load_from_checkpoint(checkpoint, input_dim=input_dim).eval()
        elif model_type == "magnets":
            model = ConceptDiscoveryModel.load_from_checkpoint(
                checkpoint,
                n_concepts=n_concepts,
                concepts=concepts if supervised_concepts else None,
                regressive_concepts=regressive_concepts,
                n_tasks=1,
                input_dim=input_dim,
                input_length=input_length,
                latent_dim=128,
                task_loss_weight=task_loss_weight,
                concept_loss_weight=1.0,
                sparsity_loss_weight=sparsity_loss_weight,
                gsat_loss_weight=gsat_loss_weight,
                gsat_r=gsat_r,
                connect_loss_weight=connect_loss_weight,
                ortho_loss_weight=ortho_loss_weight,
                concept_ortho_loss_weight=concept_ortho_loss_weight,
                mask_sparsity_loss_weight=mask_sparsity_loss_weight,
                concept_entropy_loss_weight=concept_entropy_loss_weight,
                c2y_sparsity_loss_weight=c2y_sparsity_loss_weight,
                mask_smoothing=mask_smoothing,
                use_ste=use_ste,
                optimizer="adam",
                learning_rate=learning_rate,
                n_masks=n_masks,
                mask_generator=mask_generator,
                c2y_layers=c2y_layers,
                aggs=aggs,
                target_scaling=target_scaling,
            ).eval()
        elif model_type == "natm":
            model = NATM.load_from_checkpoint(
                input_dim,
                input_length,
                hidden_units=32,
                lr=learning_rate,
                target_scaling=target_scaling,
            ).eval()
        elif model_type == "natm_feature":
            model = NATMFeature.load_from_checkpoint(
                input_length,
                hidden_units=32,
                lr=learning_rate,
                target_scaling=target_scaling,
            ).eval()
        elif model_type == "natm_time":
            model = NATMTime.load_from_checkpoint(
                input_dim,
                hidden_units=32,
                lr=learning_rate,
                target_scaling=target_scaling,
            ).eval()
        elif model_type == "gatsm":
            model = GATSM.load_from_checkpoint(
                "m2o:reg",
                input_dim,
                1,
                nbm_hidden_dims=[256, 256, 128],
                nbm_n_bases=100,
                nbm_batchnorm=False,
                nbm_dropout=0.0,
                attn_emb_size=64,
                attn_n_heads=8,
                attn_dropout=0.0,
                lr=learning_rate,
                weight_decay=1e-5,
                target_scaling=target_scaling,
            ).eval()

        return trainer.test(model=model, dataloaders=test_dl)

    if dataset == "Synth":
        train_ds = SynthRegressionDataset(data_path, mode="train", gt_fun=gt_fun)
        val_ds = SynthRegressionDataset(data_path, mode="val", gt_fun=gt_fun)
        train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=0, shuffle=True, drop_last=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=0, shuffle=False, drop_last=False)
    elif dataset == "StaticBridge":
        train_ds = StaticBridgeDataset(data_path, bridges=[0, 1, 2, 3, 4], target_scaling=target_scaling)
        val_ds = StaticBridgeDataset(data_path, bridges=[5], target_scaling=target_scaling)
        train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=0, shuffle=True, drop_last=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=0, shuffle=False, drop_last=False)
    else :  # assume tsregression dataset
        train_ds = TSRegressionDataset(path=data_path, problem=dataset, mode="train", scaling="none", target_scaling=target_scaling, target_log=target_log, subsampling=subsampling)
        val_ds = TSRegressionDataset(path=data_path, problem=dataset, mode="test", scaling="none", target_scaling=target_scaling, target_log=target_log, subsampling=subsampling)
        train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=0, shuffle=True, drop_last=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=0, shuffle=False, drop_last=False)
    input_dim = train_ds.X.shape[1]
    input_length = train_ds.X.shape[2]
    print(f"Samples: {len(train_ds)} train / {len(val_ds)} val, Input dim: {input_dim}, Input length: {input_length}")

    if model_type == "cnn":
        model = CNN(input_dim=input_dim, lr=learning_rate, target_scaling=target_scaling, input_length=input_length)
    elif model_type == "magnets":
        model = ConceptDiscoveryModel(
            n_concepts=n_concepts,
            concepts=concepts if supervised_concepts else None,
            regressive_concepts=regressive_concepts,
            n_tasks=1,
            input_dim=input_dim,
            input_length=input_length,
            latent_dim=128,
            task_loss_weight=task_loss_weight,
            concept_loss_weight=1.0,
            sparsity_loss_weight=sparsity_loss_weight,
            gsat_loss_weight=gsat_loss_weight,
            gsat_r=gsat_r,
            connect_loss_weight=connect_loss_weight,
            ortho_loss_weight=ortho_loss_weight,
            concept_ortho_loss_weight=concept_ortho_loss_weight,
            mask_sparsity_loss_weight=mask_sparsity_loss_weight,
            concept_entropy_loss_weight=concept_entropy_loss_weight,
            c2y_sparsity_loss_weight=c2y_sparsity_loss_weight,
            mask_smoothing=mask_smoothing,
            use_ste=use_ste,
            optimizer="adam",
            learning_rate=learning_rate,
            n_masks=n_masks,
            mask_generator=mask_generator,
            c2y_layers=c2y_layers,
            aggs=aggs,
            target_scaling=target_scaling,
        )
    elif model_type == "natm":
        model = NATM(
            input_dim,
            input_length,
            hidden_units=32,
            lr=learning_rate,
            target_scaling=target_scaling,
        )
    elif model_type == "natm_feature":
        model = NATMFeature(
            input_length,
            hidden_units=32,
            lr=learning_rate,
            target_scaling=target_scaling,
        )
    elif model_type == "natm_time":
        model = NATMTime(
            input_dim,
            hidden_units=32,
            lr=learning_rate,
            target_scaling=target_scaling,
        )
    elif model_type == "gatsm":
        model = GATSM(
            "m2o:reg",
            input_dim,
            1,
            nbm_hidden_dims=[256, 256, 128],
            nbm_n_bases=100,
            nbm_batchnorm=False,
            nbm_dropout=0.0,
            attn_emb_size=64,
            attn_n_heads=8,
            attn_dropout=0.0,
            lr=learning_rate,
            weight_decay=1e-5,
            target_scaling=target_scaling,
        )
        # model = GATSM(
        #     "m2o:reg",
        #     input_dim,
        #     1,
        #     nbm_hidden_dims=[256, 256, 128],
        #     nbm_n_bases=100,
        #     nbm_batchnorm=False,
        #     nbm_dropout=0.23148071468626916,
        #     attn_emb_size=110,
        #     attn_n_heads=8,
        #     attn_dropout=0.06924052436811544,
        #     lr=0.004950314086516403,
        #     weight_decay=0.0016794862055699246,
        #     target_scaling=target_scaling,
        # )

    checkpointer = ModelCheckpoint(
        dirpath=f"{output_dir}/checkpoints/",
        filename="{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    trainer = pl.Trainer(
        accelerator=device,
        max_epochs=max_epochs,
        logger=wandb_logger,
        callbacks=[checkpointer],
        log_every_n_steps=100,
        # resume_from_checkpoint=checkpoint,
        # log_every_n_steps=1,
        # val_check_interval=1,
        # limit_val_batches=None,# 1,
        # check_val_every_n_epoch=1,
        # gradient_clip_val=1.0,
    )

    return trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    Fire(main)
