import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-6


class ConnectLoss(nn.Module):
    """Connectivity loss for masks to encourage connectedness in the mask."""

    def __init__(self):
        super(ConnectLoss, self).__init__()

    def forward(self, mask):
        shift1 = mask[:,1:]
        shift2 = mask[:,:-1]

        # Also normalizes mask
        connect = torch.sum((shift1 - shift2).norm(p=2)) / shift1.flatten().shape[0]

        return connect


class GSATLoss(nn.Module):
    """Generalized Sigmoid Activation Transformation (GSAT) loss for masks to follow prior distribution."""

    def __init__(self, r):
        super(GSATLoss, self).__init__()
        self.r = r

    def forward(self, mask):
        if torch.any(torch.isnan(mask)):
            print('ALERT - mask has nans')
            exit()
        if torch.any(mask < 0):
            print('ALERT - mask less than 0')
            exit()
        assert (mask < 0).sum() == 0
        info_loss = (mask * torch.log(mask/self.r + EPS) + (1-mask) * torch.log((1-mask)/(1-self.r + EPS) + EPS)).mean()
        if torch.any(torch.isnan(info_loss)):
            print('INFO LOSS NAN')
            exit()
        return info_loss


class OrthogonalityLoss(nn.Module):
    """Orthogonality loss for masks to encourage non-overlapping masks for each dimension/aggregation."""

    def __init__(self):
        super(OrthogonalityLoss, self).__init__()

    def forward(self, mask):
        # input shape: (batch_size, num_masks, input_length)
        # compute pairwise matrix products between masks for each batch element
        mask = mask / (mask.norm(dim=-1, keepdim=True) + EPS)
        ortho = torch.matmul(mask, mask.transpose(1, 2))
        ortho_loss = torch.sum(torch.abs(ortho - torch.eye(ortho.shape[1], device=ortho.device).unsqueeze(0))) / ortho.shape[0]
        return ortho_loss


class WeightOrthogonalityLoss(nn.Module):
    """Orthogonality loss for weights to encourage non-overlapping weights for each dimension/aggregation."""

    def __init__(self):
        super(WeightOrthogonalityLoss, self).__init__()

    def forward(self, weight):
        # input shape: (n_concepts, n_features)
        # compute pairwise matrix products between weights
        weight = weight / (weight.norm(dim=-1, keepdim=True) + EPS)
        ortho = torch.matmul(weight, weight.T)
        ortho_loss = torch.sum(torch.abs(ortho - torch.eye(ortho.shape[1], device=ortho.device))) / ortho.shape[0]
        return ortho_loss


class WeightEntropyLoss(nn.Module):
    """Entropy loss for weights to encourage weights to be closer to binary for each dimension/aggregation."""

    def __init__(self):
        super(WeightEntropyLoss, self).__init__()

    def forward(self, weight):
        # input shape: (n_concepts, n_features)
        # compute entropy of weights
        # weight = weight / (weight.norm(dim=-1, keepdim=True) + EPS)
        weight = weight.softmax(dim=-1)
        entropy = -torch.sum(weight * torch.log(weight + EPS)) / weight.shape[0]
        return entropy


def softargmax(logits, tau=0.01):
    """
    Differentiable approximation of argmax. It returns a continuous approximation of the index of the maximum.

    Args:
    logits: Tensor of shape (..., num_classes) containing raw logits for each class.
    tau: Non-negative scalar temperature to control the "sharpness" of the softmax.

    Returns:
    A tensor containing the soft index (weighted average of indices based on softmax probabilities).
    """
    # Apply softmax with temperature
    # print(logits.shape, logits)
    probs = F.softmax(logits / tau, dim=-1)
    # print(probs.shape, probs)
    # Create a tensor of indices (0, 1, 2, ..., num_classes-1)
    indices = torch.linspace(0, 1, logits.size(-1), dtype=torch.float, device=logits.device)
    # print(indices.shape, indices)
    # Compute weighted sum of indices, weighted by the softmax probabilities
    soft_argmax = torch.sum(probs * indices, dim=-1)
    # print(soft_argmax.shape, soft_argmax)
    return soft_argmax


def softargmin(logits, tau=0.01):
    """
    Differentiable approximation of argmin. It returns a continuous approximation of the index of the minimum.

    Args:
    logits: Tensor of shape (..., num_classes) containing raw logits for each class.
    tau: Non-negative scalar temperature to control the "sharpness" of the softmax.

    Returns:
    A tensor containing the soft index (weighted average of indices based on softmax probabilities).
    """
    # Apply softmax with temperature
    probs = F.softmax(-logits / tau, dim=-1)

    # Create a tensor of indices (0, 1, 2, ..., num_classes-1)
    indices = torch.linspace(0, 1, logits.size(-1), dtype=torch.float, device=logits.device)

    # Compute weighted sum of indices, weighted by the softmax probabilities
    soft_argmin = torch.sum(probs * indices, dim=-1)

    return soft_argmin


class ArgmaxSTEFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.argmax(x, dim=-1).float() / x.shape[2]

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
        # return F.hardtanh(grad_output)


class ArgmaxSTE(nn.Module):
    """Argmax operation with straight-through estimator."""

    def __init__(self):
        super(ArgmaxSTE, self).__init__()

    def forward(self, x):
        return ArgmaxSTEFun.apply(x)


class ArgminSTEFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.argmin(x, dim=-1).float() / x.shape[2]

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
        # return F.hardtanh(grad_output)


class ArgminSTE(nn.Module):
    """Argmin operation with straight-through estimator."""

    def __init__(self):
        super(ArgminSTE, self).__init__()

    def forward(self, x):
        return ArgminSTEFun.apply(x)
