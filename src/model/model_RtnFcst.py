import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_base import ModelBase
from .model_utils import getFactor, calculateStatistics
from src.utils import deco_print, sharpe, construct_long_short_portfolio

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False


class FeedForwardModelWithNA_Return_Ensembled:
    """Ensemble wrapper for the return-forecasting beta network."""

    def __init__(self, logdirs, model_params, mode, device=None):
        self._logdirs = logdirs
        self._model = FeedForwardModelWithNA_Return(model_params, mode, device=device)

    def getPrediction(self, dl):
        preds = []
        for logdir in self._logdirs:
            self._model.load(logdir)
            preds.append(self._model.getPrediction(dl))
        return np.array(preds).mean(axis=0)

    def getSDFFactor(self, dl, normalized=False, norm=None):
        beta = self.getPrediction(dl)
        return getFactor(beta, dl, normalized=normalized, norm=norm)

    def calculateStatistics(self, dl):
        return calculateStatistics(self.getPrediction(dl), dl)

    def evaluate_sharpe(self, dl):
        R_pred = self.getPrediction(dl)
        for _, (_, _, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
            portfolio = construct_long_short_portfolio(R_pred, R[mask], mask)
        return sharpe(portfolio)


class FeedForwardModelWithNA_Return(ModelBase):
    """Return-forecasting beta network (PyTorch).

    Predicts next-period return for each (firm, time) pair via:
      [char_features ‖ macro_features] → FF layers → scalar prediction
    """

    def __init__(self, model_params, mode, device=None):
        super().__init__(model_params, mode)

        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self._device = torch.device(device) if isinstance(device, str) else device

        self._macro_dim = model_params["macro_feature_dim"]
        self._char_dim = model_params["individual_feature_dim"]
        self._dropout_rate = 1.0 - float(model_params.get("dropout", 1.0))

        layers, in_size = [], self._char_dim + self._macro_dim
        for h in model_params["hidden_dim"]:
            layers.append(nn.Linear(in_size, h))
            in_size = h
        self._ff = nn.ModuleList(layers)
        self._output = nn.Linear(in_size, 1)

        self.to(self._device)

    def _t(self, arr):
        if isinstance(arr, torch.Tensor):
            return arr.to(dtype=torch.float32, device=self._device)
        return torch.tensor(arr, dtype=torch.float32, device=self._device)

    def _to_bool(self, arr):
        if isinstance(arr, torch.Tensor):
            return arr.to(dtype=torch.bool, device=self._device)
        return torch.tensor(arr, dtype=torch.bool, device=self._device)

    def forward(self, I_macro, I, R, mask):
        I_macro = self._t(I_macro)
        I = self._t(I)
        mask = self._to_bool(mask)

        T, N, _ = I.shape
        macro_exp = I_macro.unsqueeze(1).expand(T, N, -1)
        concat = torch.cat([I[mask], macro_exp[mask]], dim=1)

        h = concat
        for layer in self._ff:
            h = F.relu(layer(h))
            if self.training:
                h = F.dropout(h, p=self._dropout_rate)
        return self._output(h).squeeze(-1)  # (num_valid,)

    def _mse_loss(self, R_pred, R, mask, loss_weight=None):
        R_valid = self._t(R)[mask]
        if loss_weight is not None:
            lw = self._t(loss_weight)[mask]
            lw = lw / lw.sum().clamp(min=1e-8)
            return ((R_valid - R_pred).pow(2) * lw).sum()
        return (R_valid - R_pred).pow(2).mean()

    def train_model(self, dl, dl_valid, logdir, loss_weight=None, loss_weight_valid=None,
                    dl_test=None, loss_weight_test=None,
                    printOnConsole=True, printFreq=128, saveLog=True):
        p = self._model_params
        os.makedirs(logdir, exist_ok=True)
        sw = SummaryWriter(logdir) if (saveLog and _TB_AVAILABLE) else None

        optimizer = self.build_optimizer()
        scheduler = self.build_scheduler(optimizer)

        best_valid_loss = float("inf")
        sharpe_train, sharpe_valid, sharpe_test = [], [], []
        has_test = dl_test is not None

        time_start = time.time()
        for epoch in range(p["num_epochs"]):
            self.train()
            for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=p["sub_epoch"])):
                mask_t = self._to_bool(mask)
                R_pred = self.forward(I_macro, I, R, mask)
                loss = self._mse_loss(R_pred, R, mask_t, loss_weight)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if scheduler:
                scheduler.step()
            self.global_step += 1

            tr_loss = self._eval_loss(dl, loss_weight)
            vl_loss = self._eval_loss(dl_valid, loss_weight_valid)
            tr_sr = self.evaluate_sharpe(dl)
            vl_sr = self.evaluate_sharpe(dl_valid)
            sharpe_train.append(tr_sr)
            sharpe_valid.append(vl_sr)

            te_loss = te_sr = None
            if has_test:
                te_loss = self._eval_loss(dl_test, loss_weight_test)
                te_sr = self.evaluate_sharpe(dl_test)
                sharpe_test.append(te_sr)

            if printOnConsole and epoch % printFreq == 0:
                if has_test:
                    deco_print(f"Epoch {epoch} loss {tr_loss:.4f}/{vl_loss:.4f}/{te_loss:.4f}  "
                               f"sharpe {tr_sr:.4f}/{vl_sr:.4f}/{te_sr:.4f}")
                else:
                    deco_print(f"Epoch {epoch} loss {tr_loss:.4f}/{vl_loss:.4f}  "
                               f"sharpe {tr_sr:.4f}/{vl_sr:.4f}")
                elapsed = time.time() - time_start
                deco_print(f"Epoch {epoch} {elapsed:.1f}s / {elapsed/(epoch+1)*p['num_epochs']:.1f}s est")

            if sw:
                sw.add_scalars("Loss", {"train": tr_loss, "valid": vl_loss}, epoch)
                sw.add_scalars("Sharpe", {"train": tr_sr, "valid": vl_sr}, epoch)

            if vl_loss < best_valid_loss:
                best_valid_loss = vl_loss
                self.save(logdir, epoch)

        if sw:
            sw.close()

        return (sharpe_train, sharpe_valid, sharpe_test) if has_test else (sharpe_train, sharpe_valid)

    def _eval_loss(self, dl, loss_weight):
        self.eval()
        with torch.no_grad():
            for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
                mask_t = self._to_bool(mask)
                R_pred = self.forward(I_macro, I, R, mask)
                return self._mse_loss(R_pred, R, mask_t, loss_weight).item()

    def getPrediction(self, dl):
        self.eval()
        with torch.no_grad():
            for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
                return self.forward(I_macro, I, R, mask).cpu().numpy()

    def evaluate_sharpe(self, dl):
        R_pred = self.getPrediction(dl)
        for _, (_, _, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
            portfolio = construct_long_short_portfolio(R_pred, R[mask], mask)
        return sharpe(portfolio)

    def calculateStatistics(self, dl):
        return calculateStatistics(self.getPrediction(dl), dl)
