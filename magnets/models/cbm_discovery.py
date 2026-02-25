import numpy as np
import pytorch_lightning as pl
import torch
import wandb

from utils.maskgen import SimpleMaskGenerator, UnetMaskGenerator
from utils.loss import (
    ConnectLoss,
    GSATLoss,
    OrthogonalityLoss,
    WeightOrthogonalityLoss,
    WeightEntropyLoss,
    ArgmaxSTE, ArgminSTE, softargmax, softargmin
)


class ConceptDiscoveryModel(pl.LightningModule):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        input_dim,
        input_length,
        latent_dim,

        concepts=None,
        regressive_concepts=False,
        concept_loss_weight=1.0,

        task_loss_weight=1.0,
        sparsity_loss_weight=0.0,
        gsat_loss_weight=0.0,
        gsat_r=0.5,
        connect_loss_weight=0.0,
        ortho_loss_weight=0.0,
        concept_ortho_loss_weight=0.0,
        mask_sparsity_loss_weight=0.0,
        concept_entropy_loss_weight=0.0,
        c2y_sparsity_loss_weight=0.0,
        mask_smoothing=None,
        use_ste=True,

        n_masks=10,
        aggs=["sum"],
        mask_generator="unet",

        c2y_model=None,
        c2y_layers=None,

        optimizer="adam",
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        target_scaling=1.0,
    ):
        """
        Automatic concept discovery from time series.

        :param int n_concepts: The number of concepts given at training time.
        :param int n_tasks: The number of output classes of the CBM.
        :param float sparsity_loss_weight: Weight to be used for the final loss'
            component corresponding to the concept classification loss. Default
            is 0.01.
        :param float task_loss_weight: Weight to be used for the final loss'
            component corresponding to the output task classification loss.
            Default is 1.


        :param str optimizer:  The name of the optimizer to use. Must be one of
            `adam` or `sgd`. Default is `adam`.
        :param float momentum: Momentum used for optimization. Default is 0.9.
        :param float learning_rate:  Learning rate used for optimization.
            Default is 0.01.
        :param float weight_decay: The weight decay factor used during
            optimization. Default is 4e-05.
        :param List[float] weight_loss: Either None or a list with n_concepts
            elements indicating the weights assigned to each predicted concept
            during the loss computation. Could be used to improve
            performance/fairness in imbalanced datasets.
        :param List[float] task_class_weights: Either None or a list with
            n_tasks elements indicating the weights assigned to each output
            class during the loss computation. Could be used to improve
            performance/fairness in imbalanced datasets.
        """
        super().__init__()
        self.n_concepts = n_concepts
        self.concepts = concepts
        self.n_tasks = n_tasks
        self.input_dim = input_dim
        self.n_masks = n_masks
        self.aggs = aggs
        self.n_agg = len(aggs)
        self.latent_dim = 128
        self.mask_generator = mask_generator

        # edit by Emma to compute number of features:
        self.n_features = input_dim * self.n_masks * self.n_agg

        self.tau = 1.0
        self.use_ste = use_ste

        if self.mask_generator == "simple":
            self.mask_generators = SimpleMaskGenerator(input_dim=input_dim, input_length=input_length, latent_dim=latent_dim, n_masks=n_masks, tau=self.tau, use_ste=self.use_ste, kernel_size=3, mask_smoothing=mask_smoothing)
        elif self.mask_generator == "unet":
            self.mask_generators = UnetMaskGenerator(input_dim=input_dim, input_length=input_length, n_masks=n_masks, tau=self.tau, use_ste=self.use_ste, kernel_size=2, mask_smoothing=mask_smoothing)
        else:
            raise ValueError("Invalid mask generator: available options are 'simple' and 'unet'")

        if n_concepts is not None:  # Use bottleneck layer with n_concepts
            # self.bn = torch.nn.BatchNorm1d(self.n_features)
            self.bottleneck = torch.nn.Linear(self.n_features, n_concepts)
            torch.nn.init.zeros_(self.bottleneck.weight)
            # self.bottleneck.weight.data.bernoulli_(0.5)
            # torch.nn.init.zeros_(self.bottleneck.bias)

            # Now construct the label prediction model
            if c2y_model is not None:
                # Then this method has been provided to us already
                self.c2y_model = c2y_model
            else:
                # Else we construct it here directly
                units = [n_concepts] + (c2y_layers or []) + [n_tasks]
                layers = []
                for i in range(1, len(units)):
                    layers.append(torch.nn.Linear(units[i-1], units[i]))
                    if i != len(units) - 1:
                        layers.append(torch.nn.LeakyReLU())
                self.c2y_model = torch.nn.Sequential(*layers)
                for layer in self.c2y_model:
                    if isinstance(layer, torch.nn.Linear):
                        torch.nn.init.zeros_(layer.weight)
                        # torch.nn.init.zeros_(layer.bias)
                        # layer.weight.data.bernoulli_(0.5)

        else:  # No bottleneck layer
            # Now construct the label prediction model
            if c2y_model is not None:
                # Then this method has been provided to us already
                self.c2y_model = c2y_model
            else:
                # Else we construct it here directly
                units = [self.n_features] + (c2y_layers or []) + [n_tasks]
                layers = []
                for i in range(1, len(units)):
                    layers.append(torch.nn.Linear(units[i-1], units[i]))
                    if i != len(units) - 1:
                        layers.append(torch.nn.LeakyReLU())
                self.c2y_model = torch.nn.Sequential(*layers)

        self.loss_task = torch.nn.MSELoss()
        self.loss_gsat = GSATLoss(r=gsat_r)
        self.loss_connect = ConnectLoss()
        self.loss_ortho = OrthogonalityLoss()
        self.loss_concept_ortho = WeightOrthogonalityLoss()
        self.loss_concept_entropy = WeightEntropyLoss()

        if concepts is not None:
            if regressive_concepts:
                self.loss_concept = torch.nn.MSELoss()
            else:
                self.loss_concept = torch.nn.BCELoss()

        self.task_loss_weight = task_loss_weight
        self.sparsity_loss_weight = sparsity_loss_weight
        self.gsat_loss_weight = gsat_loss_weight
        self.connect_loss_weight = connect_loss_weight
        self.ortho_loss_weight = ortho_loss_weight
        self.concept_ortho_loss_weight = concept_ortho_loss_weight
        self.mask_sparsity_loss_weight = mask_sparsity_loss_weight
        self.concept_loss_weight = concept_loss_weight
        self.concept_entropy_loss_weight = concept_entropy_loss_weight
        self.c2y_sparsity_loss_weight = c2y_sparsity_loss_weight
        self.mask_smoothing = mask_smoothing

        self.momentum = momentum
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer

        self.target_scaling = target_scaling

    def extract_features(self, batch, batch_idx=None):
        """Extract all curve features from the batch"""
        return self.extract_learned_features(batch, batch_idx)

    def apply_mask(self, x, ste_mask):
        # First apply mask directly on input:
        # baseline = self._get_baseline(b=x.shape[1])
        baseline = torch.zeros_like(x)  # NOTE: adapt baseline to different applications

        x_masked = torch.stack([
            torch.stack([
                x[:, i, :] * ste_mask[:, m, i, :] + baseline[:, i, :] * (1 - ste_mask[:, m, i, :])
                for i in range(self.input_dim)], dim=1)
            for m in range(self.n_masks)
        ], dim=1)

        return x_masked


    def mask_and_agg(self, agg: str, batch: torch.Tensor, mask: torch.Tensor, beta: float = 1.0):
        """
        Applies an aggregation function over the last dimension (T) of `batch`,
        considering only the elements where `mask` is True.

        Parameters:
            agg (str): Aggregation function ('mean', 'sum', 'min', 'max', 'std', 'argmax', 'argmin', 'amplitude', 'derivative', 'tv').
            batch (torch.Tensor): Input tensor of shape (B, C, T).
            mask (torch.Tensor): Binary mask of shape (B, C, T), where True indicates valid entries.

        Returns:
            torch.Tensor: Aggregated tensor of shape (B, C).
        """
        assert batch.shape == mask.shape, "batch and mask must have the same shape"

        MAX_VAL = 1e9

        if agg == "sum":
            return (batch * mask).sum(dim=-1)

        elif agg == "mean":
            sum_values = (batch * mask).sum(dim=-1)
            count = mask.sum(dim=-1).clamp(min=1)  # Avoid division by zero
            return sum_values / count

        elif agg == "min":
            masked_batch = torch.where(mask == 1.0, batch, torch.tensor(MAX_VAL, device=batch.device))
            return masked_batch.min(dim=-1)[0]

        elif agg == "max":
            masked_batch = torch.where(mask == 1.0, batch, torch.tensor(-MAX_VAL, device=batch.device))
            return masked_batch.max(dim=-1)[0]

        elif agg == "soft_max":
            masked_batch = torch.where(mask == 1.0, batch, torch.tensor(-MAX_VAL, device=batch.device))
            weights = torch.nn.functional.softmax(beta * masked_batch, dim=-1)
            return (weights * batch).sum(dim=-1)  # Weighted sum approximates max

        elif agg == "soft_min":
            masked_batch = torch.where(mask == 1.0, batch, torch.tensor(MAX_VAL, device=batch.device))
            weights = torch.nn.functional.softmax(-beta * masked_batch, dim=-1)
            return (weights * batch).sum(dim=-1)  # Weighted sum approximates min

        elif agg == "std":
            sum_values = (batch * mask).sum(dim=-1)
            count = mask.sum(dim=-1).clamp(min=1)  # Avoid division by zero
            mean_values = sum_values / count
            squared_diff = ((batch - mean_values.unsqueeze(-1)) ** 2) * mask
            variance = squared_diff.sum(dim=-1) / count
            return torch.sqrt(variance)

        elif agg == "argmax":
            masked_batch = torch.where(mask == 1.0, batch, torch.tensor(-MAX_VAL, device=batch.device))
            return masked_batch.argmax(dim=-1) / batch.shape[-1]

        elif agg == "argmin":
            masked_batch = torch.where(mask == 1.0, batch, torch.tensor(MAX_VAL, device=batch.device))
            return masked_batch.argmin(dim=-1) / batch.shape[-1]

        elif agg == "soft_argmax":
            masked_batch = torch.where(mask == 1.0, batch, torch.tensor(-MAX_VAL, device=batch.device))
            weights = torch.nn.functional.softmax(beta * masked_batch, dim=-1)
            indices = torch.arange(batch.shape[-1], device=batch.device).float()
            return (weights * indices).sum(dim=-1) / batch.shape[-1]  # Soft index approximation

        elif agg == "soft_argmin":
            masked_batch = torch.where(mask == 1.0, batch, torch.tensor(MAX_VAL, device=batch.device))
            weights = torch.nn.functional.softmax(-beta * masked_batch, dim=-1)
            indices = torch.arange(batch.shape[-1], device=batch.device).float()
            return (weights * indices).sum(dim=-1) / batch.shape[-1]  # Soft index approximation

        else:
            raise ValueError(f"Unsupported aggregation function: {agg}")


    def extract_learned_features(self, batch, batch_idx):
        """Extract all learned curve features from the batch"""
       
        mask, mask_ste = self.mask_generators(batch)
       
        features = torch.stack([
            torch.stack([
                self.mask_and_agg(agg, batch, mask_ste[:, m, :, :]) for agg in self.aggs
            ], dim=2)
            for m in range(self.n_masks)
        ], dim=3)

        # (batch, input_dim, n_agg, n_masks) -> (batch, n_agg * n_masks * input_dim)
        features = features.view(features.shape[0], -1)

        if batch_idx == 0:
            # Visualize masks of first sample
            for m in range(self.n_masks):
                for i in range(self.input_dim):
                    mask_i_m = mask[0, m, i, :].unsqueeze(0).unsqueeze(0).cpu().detach().numpy()
                    mask_ste_i_m = mask_ste[0, m, i, :].unsqueeze(0).unsqueeze(0).cpu().detach().numpy()
                    # self.logger.experiment.add_image(f"mask/dim_{i}_{m}", mask_i_m, global_step=self.current_epoch)
                    # self.logger.experiment.add_image(f"mask_ste/dim_{i}_{m}", mask_ste_i_m, global_step=self.current_epoch)
                    self.logger.experiment.log({f"mask/dim_{i}_{m}": wandb.Image(mask_i_m[0])})
                    self.logger.experiment.log({f"mask_ste/dim_{i}_{m}": wandb.Image(mask_ste_i_m[0])})

        return {
            "features": features,
            "masks": mask,
        }

    def forward(self, x, batch_idx=None):
        out_dict = self.extract_features(x, batch_idx)

        if self.n_concepts is not None:
            c_pred = self.bottleneck(out_dict["features"])
            y = self.c2y_model(c_pred)
        else:
            y = self.c2y_model(out_dict["features"])

        return {
            "c_pred": c_pred,
            "y_pred": y,
            "masks": out_dict["masks"],
        }

    def predict_step(
        self,
        batch,
        batch_idx,
    ):
        return self.forward(batch[0])

    def _run_step(
        self,
        batch,
        batch_idx,
        train=False,
    ):
        if len(batch) == 2:
            x, y = batch
            w = None
        elif len(batch) == 3:
            x, y, c = batch
            w = None
        else:
            x, y, c, w = batch
        # print("labels = ", y)
        out_dict = self.forward(x, batch_idx)

        if self.task_loss_weight != 0:
            if w is None:
                task_loss = self.loss_task(
                    out_dict["y_pred"] if out_dict["y_pred"].shape[-1] > 1 else out_dict["y_pred"].reshape(-1),
                    y,
                )
            else:
                task_loss = torch.mean(w * (out_dict["y_pred"] - y)**2)
            task_loss_scalar = task_loss.detach()
        else:
            task_loss = 0.0
            task_loss_scalar = 0.0

        if self.concepts is not None and self.concept_loss_weight != 0:
            concept_loss = self.loss_concept(out_dict["c_pred"], c)
            concept_loss_scalar = concept_loss.detach()
        else:
            concept_loss = 0.0
            concept_loss_scalar = 0.0

        if self.sparsity_loss_weight != 0:
            if self.n_concepts is not None:
                sparsity_loss = torch.norm(self.bottleneck.weight, p=1) / self.bottleneck.weight.numel()
            else:
                sparsity_loss = torch.norm(self.c2y_model[0].weight, p=1) / self.c2y_model[0].weight.numel()
            sparsity_loss_scalar = sparsity_loss.detach()
        else:
            sparsity_loss = 0.0
            sparsity_loss_scalar = 0.0

        if self.concept_ortho_loss_weight != 0:
            if self.n_concepts is not None:
                concept_ortho_loss = self.loss_concept_ortho(self.bottleneck.weight)
                concept_ortho_loss_scalar = concept_ortho_loss.detach()
        else:
            concept_ortho_loss = 0.0
            concept_ortho_loss_scalar = 0.0

        if self.concept_entropy_loss_weight != 0:
            if self.n_concepts is not None:
                concept_entropy_loss = self.loss_concept_entropy(self.bottleneck.weight)
            concept_entropy_loss_scalar = concept_entropy_loss.detach()
        else:
            concept_entropy_loss = 0.0
            concept_entropy_loss_scalar = 0.0

        if self.gsat_loss_weight != 0:
            gsat_loss = 0
            for m in range(self.n_masks):
                for i in range(self.input_dim):
                    gsat_loss += self.loss_gsat(out_dict["masks"][:, m, i, :])
            gsat_loss_scalar = gsat_loss.detach()
        else:
            gsat_loss = 0.0
            gsat_loss_scalar = 0.0

        if self.connect_loss_weight != 0:
            connect_loss = 0
            for m in range(self.n_masks):
                for i in range(self.input_dim):
                    connect_loss += self.loss_connect(out_dict["masks"][:, m, i, :])
            connect_loss_scalar = connect_loss.detach()
        else:
            connect_loss = 0.0
            connect_loss_scalar = 0.0

        if self.ortho_loss_weight != 0:
            ortho_loss = 0
            for i in range(self.input_dim):
                ortho_loss += self.loss_ortho(out_dict["masks"][:, :, i, :])
            ortho_loss_scalar = ortho_loss.detach()
        else:
            ortho_loss = 0.0
            ortho_loss_scalar = 0.0

        if self.mask_sparsity_loss_weight != 0:
            mask_sparsity_loss = 0
            for m in range(self.n_masks):
                for i in range(self.input_dim):
                    mask_sparsity_loss += torch.norm(out_dict["masks"][:, m, i, :], p=1) / out_dict["masks"][:, m, i, :].numel()
            mask_sparsity_loss_scalar = mask_sparsity_loss.detach()
        else:
            mask_sparsity_loss = 0.0
            mask_sparsity_loss_scalar = 0.0

        if self.c2y_sparsity_loss_weight != 0:
            c2y_sparsity_loss = torch.norm(self.c2y_model[0].weight, p=1) / self.c2y_model[0].weight.numel()  # TODO handle multiple layers
            c2y_sparsity_loss_scalar = c2y_sparsity_loss.detach()
        else:
            c2y_sparsity_loss = 0.0
            c2y_sparsity_loss_scalar = 0.0

        # mask_sparsity_loss_weight = min(1.0, self.global_step / (256 * 100)) * self.mask_sparsity_loss_weight
        loss = self.task_loss_weight * task_loss \
            + self.concept_loss_weight * concept_loss \
            + self.sparsity_loss_weight * sparsity_loss \
            + self.gsat_loss_weight * gsat_loss \
            + self.connect_loss_weight * connect_loss \
            + self.ortho_loss_weight * ortho_loss \
            + self.concept_ortho_loss_weight * concept_ortho_loss \
            + self.mask_sparsity_loss_weight * mask_sparsity_loss \
            + self.c2y_sparsity_loss_weight * c2y_sparsity_loss \
            + self.concept_entropy_loss_weight * concept_entropy_loss \

        result = {
            "task_loss": task_loss_scalar,
            "concept_loss": concept_loss_scalar,
            "sparsity_loss": sparsity_loss_scalar,
            "gsat_loss": gsat_loss_scalar,
            "connect_loss": connect_loss_scalar,
            "ortho_loss": ortho_loss_scalar,
            "concept_ortho_loss": concept_ortho_loss_scalar,
            "mask_sparsity_loss": mask_sparsity_loss_scalar,
            "concept_entropy_loss": concept_entropy_loss_scalar,
            "c2y_sparsity_loss": c2y_sparsity_loss_scalar,
            "rmse_unscaled": np.sqrt(task_loss_scalar.cpu().detach())*self.target_scaling,
            "loss": loss.detach(),
            "lr": self.optimizers().param_groups[0]['lr'],
        }

        if batch_idx == 0:
            # Visualize weights of bottleneck layer
            if self.n_concepts is not None:
                bottleneck_weights = self.bottleneck.weight.unsqueeze(0).cpu().detach().numpy()
                # self.logger.experiment.add_image("bottleneck_weights", bottleneck_weights, global_step=self.current_epoch)
                self.logger.experiment.log({"bottleneck_weights": wandb.Image(bottleneck_weights[0])})
            c2y_weights = self.c2y_model[0].weight.unsqueeze(0).cpu().detach().numpy()
            # self.logger.experiment.add_image("bottleneck_weights", c2y_weights, global_step=self.current_epoch)
            self.logger.experiment.log({"c2y_weights": wandb.Image(c2y_weights[0])})

        return loss, result

    def training_step(self, batch, batch_no):
        loss, result = self._run_step(batch, batch_no, train=True)
        for name, val in result.items():
            self.log(name, val, prog_bar=True, on_step=True, on_epoch=True)
        return {
            "loss": loss,
            "log": {
                "task_loss": result['task_loss'],
                "loss": result['loss'],
                "lr": result['lr'],
            },
        }

    def validation_step(self, batch, batch_no):
        _, result = self._run_step(batch, batch_no, train=False)
        for name, val in result.items():
            self.log("val_" + name, val, prog_bar=True, on_step=False, on_epoch=True)
        result = {
            "val_" + key: val
            for key, val in result.items()
        }
        return result

    def test_step(self, batch, batch_no):
        loss, result = self._run_step(batch, batch_no, train=False)
        for name, val in result.items():
            self.log("test_" + name, val, prog_bar=True, on_step=False, on_epoch=True)
        return result['loss']

    def configure_optimizers(self):
        if self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     verbose=False,
        # )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=0,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "loss",
        }
