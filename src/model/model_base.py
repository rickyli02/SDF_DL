import abc
import glob
import os

import torch
import torch.nn as nn

from src.utils import deco_print


class ModelBase(nn.Module, abc.ABC):
    """Abstract base class for all models."""

    def __init__(self, model_params, mode):
        super().__init__()
        self._model_params = model_params
        self._mode = mode
        self.global_step = 0

    @abc.abstractmethod
    def forward(self, I_macro, I, R, mask):
        """Forward pass. All tensors on the model's device."""

    def build_optimizer(self):
        optimizer_name = self._model_params.get("optimizer", "Adam")
        lr = self._model_params["learning_rate"]

        if optimizer_name == "Momentum":
            return torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif optimizer_name == "AdaDelta":
            return torch.optim.Adadelta(self.parameters(), lr=lr, rho=0.95, eps=1e-8)
        elif optimizer_name == "Adam":
            return torch.optim.Adam(self.parameters(), lr=lr)
        elif optimizer_name == "RMSProp":
            return torch.optim.RMSprop(self.parameters(), lr=lr)
        elif optimizer_name == "GradientDescent":
            return torch.optim.SGD(self.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def build_scheduler(self, optimizer):
        if self._model_params.get("use_decay"):
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self._model_params["decay_steps"],
                gamma=self._model_params["decay_rate"],
            )
        return None

    def save(self, logdir, step=None):
        os.makedirs(logdir, exist_ok=True)
        step = step if step is not None else self.global_step
        path = os.path.join(logdir, f"model-{step:06d}.pt")
        torch.save({"step": step, "state_dict": self.state_dict()}, path)

    def load(self, logdir):
        checkpoints = sorted(glob.glob(os.path.join(logdir, "model-*.pt")))
        if not checkpoints:
            deco_print("WARNING: No checkpoint found. Using random initialization.")
            return
        path = checkpoints[-1]
        data = torch.load(path, map_location="cpu")
        self.load_state_dict(data["state_dict"])
        self.global_step = data.get("step", 0)
        deco_print(f"Restored checkpoint: {path}")

    @property
    def model_params(self):
        return self._model_params

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")