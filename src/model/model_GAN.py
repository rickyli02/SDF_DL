import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_base import ModelBase
from .model_utils import build_rnn, calculateStatistics
from src.utils import deco_print, sharpe

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False


class FeedForwardModelWithNA_GAN_Ensembled:
    """Ensemble wrapper: averages outputs from multiple trained checkpoints."""

    def __init__(self, logdirs, model_params, mode, device=None):
        self._logdirs = logdirs
        self._model = FeedForwardModelWithNA_GAN(model_params, mode, device=device)

    def getZeroInitialState(self):
        return [self._model.getZeroInitialState() for _ in self._logdirs]

    def getNextInitialState(self, dl, initial_state):
        results = []
        for logdir, state in zip(self._logdirs, initial_state):
            self._model.load(logdir)
            results.append(self._model.getNextInitialState(dl, state))
        return results

    def getWeightWithData(self, dl, initial_state=None, normalized=False):
        if initial_state is None:
            initial_state = [None] * len(self._logdirs)
        w_all = []
        for logdir, state in zip(self._logdirs, initial_state):
            self._model.load(logdir)
            w_all.append(self._model.getWeightWithData(dl, initial_state=state, normalized=False))
        w = np.array(w_all).mean(axis=0)
        if normalized:
            for _, (_, _, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
                splits = np.sum(mask, axis=1).cumsum()[:-1]
                w_list = np.split(w, splits)
                w = np.concatenate([item / np.absolute(item).sum() for item in w_list])
        return w

    def getSDF(self, dl, initial_state=None):
        if initial_state is None:
            initial_state = [None] * len(self._logdirs)
        sdf_all = []
        for logdir, state in zip(self._logdirs, initial_state):
            self._model.load(logdir)
            sdf_all.append(self._model.getSDF(dl, initial_state=state))
        return np.array(sdf_all).mean(axis=0)

    def getNormalizedSDF(self, dl, initial_state=None):
        w = self.getWeightWithData(dl, initial_state=initial_state, normalized=True)
        for _, (_, _, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
            chunks = np.split(R[mask] * w, np.sum(mask, axis=1).cumsum()[:-1])
            SDF = np.array([[c.sum()] for c in chunks]) + 1
        return SDF

    def getSDFFactor(self, dl, initial_state=None):
        return 1 - self.getSDF(dl, initial_state=initial_state)

    def getNormalizedSDFFactor(self, dl, initial_state=None):
        return 1 - self.getNormalizedSDF(dl, initial_state=initial_state)

    def calculateStatistics(self, dl, initial_state=None):
        w = self.getWeightWithData(dl, initial_state=initial_state)
        return calculateStatistics(w, dl)


class FeedForwardModelWithNA_GAN(ModelBase):
    """GAN-based SDF model (PyTorch).

    Architecture:
      SDF network:    macro → LSTM → macro_hidden
                      [char_features ‖ macro_hidden] → FF → scalar weight w
                      SDF_t = 1 + Σ_i(w_i * R_i)

      Moment network: macro → LSTM → macro_hidden
                      [macro_hidden ‖ char_features] → FF → K-dim factor h
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
        self._residual_loss_factor = model_params.get("residual_loss_factor", 0.0)
        # TF uses keep_prob; convert to drop_prob
        self._dropout_rate = 1.0 - float(model_params.get("dropout", 1.0))

        use_rnn = model_params["use_rnn"]

        # --- SDF network ---
        if use_rnn:
            self._sdf_rnn = build_rnn(
                model_params["cell_type_rnn"],
                input_size=self._macro_dim,
                num_units=model_params["num_units_rnn"],
                num_layers=model_params["num_layers_rnn"],
                dropout=self._dropout_rate,
            )
            sdf_macro_out = model_params["num_units_rnn"][0]
        else:
            self._sdf_rnn = None
            sdf_macro_out = self._macro_dim

        sdf_layers, in_size = [], self._char_dim + sdf_macro_out
        for h in model_params["hidden_dim"]:
            sdf_layers.append(nn.Linear(in_size, h))
            in_size = h
        self._sdf_ff = nn.ModuleList(sdf_layers)
        self._sdf_output = nn.Linear(in_size, 1)

        # --- Moment network ---
        if use_rnn:
            self._moment_rnn = build_rnn(
                model_params["cell_type_rnn_moment"],
                input_size=self._macro_dim,
                num_units=model_params["num_units_rnn_moment"],
                num_layers=model_params["num_layers_rnn_moment"],
                dropout=self._dropout_rate,
            )
            moment_macro_out = model_params["num_units_rnn_moment"][0]
        else:
            self._moment_rnn = None
            moment_macro_out = self._macro_dim

        moment_layers, in_size = [], moment_macro_out + self._char_dim
        for h in model_params.get("hidden_dim_moment", []):
            moment_layers.append(nn.Linear(in_size, h))
            in_size = h
        self._moment_ff = nn.ModuleList(moment_layers)
        self._moment_output = nn.Linear(in_size, model_params["num_condition_moment"])

        self.to(self._device)

    # ------------------------------------------------------------------
    # Tensor helpers
    # ------------------------------------------------------------------

    def _t(self, arr):
        if isinstance(arr, torch.Tensor):
            return arr.to(dtype=torch.float32, device=self._device)
        return torch.tensor(arr, dtype=torch.float32, device=self._device)

    def _to_bool(self, arr):
        if isinstance(arr, torch.Tensor):
            return arr.to(dtype=torch.bool, device=self._device)
        return torch.tensor(arr, dtype=torch.bool, device=self._device)

    def _pack_state(self, rnn_state):
        """Pack RNN state → numpy (num_layers, hidden[*2]) for external API."""
        if rnn_state is None:
            return None
        if isinstance(rnn_state, tuple):  # LSTM (h, c): each (layers, 1, hidden)
            h, c = rnn_state
            return torch.cat([h, c], dim=2).squeeze(1).detach().cpu().numpy()
        return rnn_state.squeeze(1).detach().cpu().numpy()

    def _unpack_state(self, state_np, cell_type, num_units, num_layers):
        """Convert packed numpy state → PyTorch RNN initial state."""
        if state_np is None:
            return None
        t = torch.tensor(np.atleast_2d(state_np), dtype=torch.float32, device=self._device)
        if cell_type == "lstm":
            h, c = t.split(num_units[0], dim=-1)
            return (h.unsqueeze(1), c.unsqueeze(1))
        return t.unsqueeze(1)

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def _sdf_pass(self, I_macro, I, R, mask, initial_state_np=None):
        """SDF forward: returns (w, SDF, rnn_state, N_i)."""
        if self._model_params["use_rnn"]:
            h0 = self._unpack_state(
                initial_state_np,
                self._model_params["cell_type_rnn"],
                self._model_params["num_units_rnn"],
                self._model_params["num_layers_rnn"],
            )
            rnn_out, rnn_state = self._sdf_rnn(I_macro.unsqueeze(0), h0)
            macro_hidden = rnn_out.squeeze(0)  # (T, hidden)
        else:
            macro_hidden = I_macro
            rnn_state = None

        T, N = mask.shape
        macro_exp = macro_hidden.unsqueeze(1).expand(T, N, -1)
        concat = torch.cat([I[mask], macro_exp[mask]], dim=1)  # (num_valid, F_char+hidden)

        h = concat
        for layer in self._sdf_ff:
            h = F.relu(layer(h))
            if self.training:
                h = F.dropout(h, p=self._dropout_rate)
        w = self._sdf_output(h).squeeze(-1)  # (num_valid,)

        N_i = mask.sum(dim=1)
        splits = [int(s) for s in N_i.cumsum(0)[:-1].tolist()]
        wR_chunks = torch.tensor_split(w * R[mask], splits)
        SDF = torch.stack([c.sum() for c in wR_chunks]).unsqueeze(1) + 1  # (T, 1)

        return w, SDF, rnn_state, N_i

    def _moment_pass(self, I_macro, I):
        """Moment forward: returns h (K, T, N)."""
        if self._model_params["use_rnn"]:
            rnn_out, _ = self._moment_rnn(I_macro.unsqueeze(0))
            macro_hidden = rnn_out.squeeze(0)  # (T, hidden)
        else:
            macro_hidden = I_macro

        T, N, _ = I.shape
        macro_exp = macro_hidden.unsqueeze(1).expand(T, N, -1)
        concat = torch.cat([macro_exp, I], dim=2)

        h = concat
        for layer in self._moment_ff:
            h = F.relu(layer(h))
            if self.training:
                h = F.dropout(h, p=self._dropout_rate)
        h = torch.tanh(self._moment_output(h))  # (T, N, K)
        return h.permute(2, 0, 1)  # (K, T, N)

    def forward(self, I_macro, I, R, mask, initial_state_np=None):
        I_macro = self._t(I_macro)
        I = self._t(I)
        R = self._t(R)
        mask = self._to_bool(mask)
        w, SDF, rnn_state, N_i = self._sdf_pass(I_macro, I, R, mask, initial_state_np)
        h = self._moment_pass(I_macro, I)
        return {"w": w, "SDF": SDF, "h": h, "rnn_state": rnn_state,
                "mask": mask, "R": R, "N_i": N_i}

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------

    def _pricing_loss(self, SDF, R, mask, h, loss_weight=None):
        """mean_k,n of [mean_t(SDF_t * R_nt * h_knt * mask_nt)]^2."""
        mask_f = mask.float()
        T_i = mask_f.sum(dim=0).clamp(min=1)  # (N,)
        product = R.unsqueeze(0) * mask_f.unsqueeze(0) * SDF.unsqueeze(0) * h  # (K,T,N)
        emp_mean = product.sum(dim=1) / T_i.unsqueeze(0)  # (K, N)
        sq = emp_mean.pow(2)
        if loss_weight is not None:
            lw = self._t(loss_weight)
            sq = sq * (lw / lw.max()).unsqueeze(0)
        return sq.mean()

    def _loss_unc(self, SDF, R, mask, loss_weight=None):
        ones = torch.ones(1, *mask.shape, device=self._device)
        return self._pricing_loss(SDF, R, mask, ones, loss_weight)

    def _loss_cond(self, SDF, R, mask, h, loss_weight=None):
        return self._pricing_loss(SDF, R, mask, h, loss_weight)

    def _loss_residual(self, w, R, mask):
        N_i = mask.sum(dim=1)
        splits = [int(s) for s in N_i.cumsum(0)[:-1].tolist()]
        R_valid = R[mask]
        R_chunks = torch.tensor_split(R_valid, splits)
        w_chunks = torch.tensor_split(w, splits)
        res_sq, r_sq = [], []
        for R_t, w_t in zip(R_chunks, w_chunks):
            dot = (w_t * w_t).sum().clamp(min=1e-8)
            R_hat = (R_t * w_t).sum() / dot * w_t
            res_sq.append((R_t - R_hat).pow(2).mean())
            r_sq.append(R_t.pow(2).mean())
        return torch.stack(res_sq).mean() / torch.stack(r_sq).mean().clamp(min=1e-8)

    # ------------------------------------------------------------------
    # Parameter groups for separate optimizer steps
    # ------------------------------------------------------------------

    def sdf_parameters(self):
        params = []
        if self._sdf_rnn is not None:
            params += list(self._sdf_rnn.parameters())
        params += list(self._sdf_ff.parameters()) + list(self._sdf_output.parameters())
        return params

    def moment_parameters(self):
        params = []
        if self._moment_rnn is not None:
            params += list(self._moment_rnn.parameters())
        params += list(self._moment_ff.parameters()) + list(self._moment_output.parameters())
        return params

    # ------------------------------------------------------------------
    # Training  (3-stage GAN loop)
    # ------------------------------------------------------------------

    def train_model(self, dl, dl_valid, logdir, loss_weight=None, loss_weight_valid=None,
                    dl_test=None, loss_weight_test=None,
                    printOnConsole=True, printFreq=128, saveLog=True,
                    saveBestFreq=128, ignoreEpoch=64):
        p = self._model_params
        logdir_loss = os.path.join(logdir, "loss")
        logdir_sharpe = os.path.join(logdir, "sharpe")
        os.makedirs(logdir_loss, exist_ok=True)
        os.makedirs(logdir_sharpe, exist_ok=True)

        sw = SummaryWriter(logdir) if (saveLog and _TB_AVAILABLE) else None

        opt_sdf = torch.optim.Adam(self.sdf_parameters(), lr=p["learning_rate"])
        opt_moment = torch.optim.Adam(self.moment_parameters(), lr=p["learning_rate"])
        sched_sdf = self.build_scheduler(opt_sdf)

        best_loss_unc = best_loss = float("inf")
        best_sharpe_unc = best_sharpe = float("-inf")
        sharpe_train, sharpe_valid, sharpe_test = [], [], []
        has_test = dl_test is not None
        zero_state = self.getZeroInitialState() if p["use_rnn"] else None

        # ---- Stage 1: Unconditional ----
        deco_print("Start Training Unconditional Loss...")
        time_start = time.time()
        for epoch in range(p["num_epochs_unc"]):
            self.train()
            for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=p["sub_epoch"])):
                Im, It, Rt, mt = self._t(I_macro), self._t(I), self._t(R), self._to_bool(mask)
                w, SDF, _, _ = self._sdf_pass(Im, It, Rt, mt, zero_state)
                loss = self._loss_unc(SDF, Rt, mt, loss_weight) + \
                       self._residual_loss_factor * self._loss_residual(w, Rt, mt)
                opt_sdf.zero_grad(); loss.backward(); opt_sdf.step()
            if sched_sdf:
                sched_sdf.step()

            tr_loss = self._eval_loss(dl, zero_state, loss_weight)
            vl_loss, INIT_v = self._eval_loss_and_state(dl_valid, zero_state, loss_weight_valid)
            tr_sr, _ = self._eval_sharpe_and_state(dl, zero_state)
            vl_sr, INIT_t = self._eval_sharpe_and_state(dl_valid, zero_state)
            sharpe_train.append(tr_sr); sharpe_valid.append(vl_sr)

            te_loss = te_sr = None
            if has_test:
                te_loss = self._eval_loss(dl_test, INIT_v, loss_weight_test)
                te_sr, _ = self._eval_sharpe_and_state(dl_test, INIT_v)
                sharpe_test.append(te_sr)

            if printOnConsole and epoch % printFreq == 0:
                self._log(epoch, tr_loss, vl_loss, te_loss, tr_sr, vl_sr, te_sr, "UNC",
                          time.time() - time_start, p["num_epochs_unc"])
            if sw:
                sw.add_scalars("Loss_UNC", {"train": tr_loss, "valid": vl_loss}, epoch)
                sw.add_scalars("Sharpe_UNC", {"train": tr_sr, "valid": vl_sr}, epoch)

            if epoch > ignoreEpoch:
                if vl_loss < best_loss_unc:
                    best_loss_unc = vl_loss; self.save(logdir_loss, epoch)
                if vl_sr > best_sharpe_unc:
                    best_sharpe_unc = vl_sr; self.save(logdir_sharpe, epoch)
            if saveBestFreq > 0 and (epoch + 1) % saveBestFreq == 0:
                self._backup_best(logdir_loss, "UNC", epoch)
                self._backup_best(logdir_sharpe, "UNC", epoch)

        deco_print("Training Unconditional Loss Finished!")

        # ---- Stage 2: Moment update ----
        deco_print("Start Updating Moment Conditions...")
        self.load(logdir_loss)
        self.train()
        best_moment_loss = float("-inf")
        for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=p["sub_epoch"])):
            Im, It, Rt, mt = self._t(I_macro), self._t(I), self._t(R), self._to_bool(mask)
            for epoch in range(p["num_epochs_moment"]):
                w, SDF, _, _ = self._sdf_pass(Im, It, Rt, mt, zero_state)
                h = self._moment_pass(Im, It)
                loss_m = self._loss_cond(SDF.detach(), Rt, mt, h)
                opt_moment.zero_grad(); (-loss_m).backward(); opt_moment.step()
                val = loss_m.item()
                if val > best_moment_loss:
                    best_moment_loss = val
                    self.save(logdir_loss, p["num_epochs_unc"] + epoch)
        deco_print("Updating Moment Conditions Finished!")

        # ---- Stage 3: Conditional (GAN) ----
        deco_print("Start Training Conditional Loss...")
        self.load(logdir_loss)
        time_start = time.time()
        offset = p["num_epochs_unc"] + p["num_epochs_moment"]
        for epoch in range(p["num_epochs"]):
            self.train()
            for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=p["sub_epoch"])):
                Im, It, Rt, mt = self._t(I_macro), self._t(I), self._t(R), self._to_bool(mask)
                w, SDF, _, _ = self._sdf_pass(Im, It, Rt, mt, zero_state)
                h = self._moment_pass(Im, It).detach()
                loss = self._loss_cond(SDF, Rt, mt, h, loss_weight) + \
                       self._residual_loss_factor * self._loss_residual(w, Rt, mt)
                opt_sdf.zero_grad(); loss.backward(); opt_sdf.step()
            if sched_sdf:
                sched_sdf.step()
            self.global_step += 1

            tr_loss = self._eval_loss(dl, zero_state, loss_weight)
            vl_loss, INIT_v = self._eval_loss_and_state(dl_valid, zero_state, loss_weight_valid)
            tr_sr, _ = self._eval_sharpe_and_state(dl, zero_state)
            vl_sr, INIT_t = self._eval_sharpe_and_state(dl_valid, zero_state)
            sharpe_train.append(tr_sr); sharpe_valid.append(vl_sr)

            te_loss = te_sr = None
            if has_test:
                te_loss = self._eval_loss(dl_test, INIT_v, loss_weight_test)
                te_sr, _ = self._eval_sharpe_and_state(dl_test, INIT_v)
                sharpe_test.append(te_sr)

            if printOnConsole and epoch % printFreq == 0:
                self._log(epoch, tr_loss, vl_loss, te_loss, tr_sr, vl_sr, te_sr, "GAN",
                          time.time() - time_start, p["num_epochs"])
            if sw:
                sw.add_scalars("Loss", {"train": tr_loss, "valid": vl_loss}, offset + epoch)
                sw.add_scalars("Sharpe", {"train": tr_sr, "valid": vl_sr}, offset + epoch)

            if epoch > ignoreEpoch:
                if vl_loss < best_loss:
                    best_loss = vl_loss; self.save(logdir_loss, offset + epoch)
                if vl_sr > best_sharpe:
                    best_sharpe = vl_sr; self.save(logdir_sharpe, offset + epoch)
            if saveBestFreq > 0 and (epoch + 1) % saveBestFreq == 0:
                self._backup_best(logdir_loss, "GAN", epoch)
                self._backup_best(logdir_sharpe, "GAN", epoch)

        deco_print("Training Conditional Loss Finished!")
        self.save(logdir, self.global_step)
        if sw:
            sw.close()

        return (sharpe_train, sharpe_valid, sharpe_test) if has_test else (sharpe_train, sharpe_valid)

    def _eval_loss(self, dl, initial_state_np, loss_weight):
        self.eval()
        with torch.no_grad():
            for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
                Im, It, Rt, mt = self._t(I_macro), self._t(I), self._t(R), self._to_bool(mask)
                _, SDF, _, _ = self._sdf_pass(Im, It, Rt, mt, initial_state_np)
                return self._loss_unc(SDF, Rt, mt, loss_weight).item()

    def _eval_loss_and_state(self, dl, initial_state_np, loss_weight):
        self.eval()
        with torch.no_grad():
            for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
                Im, It, Rt, mt = self._t(I_macro), self._t(I), self._t(R), self._to_bool(mask)
                _, SDF, rnn_state, _ = self._sdf_pass(Im, It, Rt, mt, initial_state_np)
                loss = self._loss_unc(SDF, Rt, mt, loss_weight).item()
                return loss, self._pack_state(rnn_state)

    def _eval_sharpe_and_state(self, dl, initial_state_np):
        self.eval()
        with torch.no_grad():
            for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
                Im, It, Rt, mt = self._t(I_macro), self._t(I), self._t(R), self._to_bool(mask)
                _, SDF, rnn_state, _ = self._sdf_pass(Im, It, Rt, mt, initial_state_np)
                sr = sharpe((1 - SDF[:, 0]).cpu().numpy())
                return sr, self._pack_state(rnn_state)

    def _log(self, epoch, tr_loss, vl_loss, te_loss, tr_sr, vl_sr, te_sr, tag, elapsed, total):
        if te_loss is not None:
            deco_print(f"{tag} epoch {epoch} loss {tr_loss:.4f}/{vl_loss:.4f}/{te_loss:.4f}  "
                       f"sharpe {tr_sr:.4f}/{vl_sr:.4f}/{te_sr:.4f}")
        else:
            deco_print(f"{tag} epoch {epoch} loss {tr_loss:.4f}/{vl_loss:.4f}  "
                       f"sharpe {tr_sr:.4f}/{vl_sr:.4f}")
        deco_print(f"Epoch {epoch} elapsed {elapsed:.1f}s / {elapsed/(epoch+1)*total:.1f}s est")

    def _backup_best(self, logdir, stage, epoch):
        import shutil
        pts = [f for f in os.listdir(logdir) if f.endswith(".pt")]
        if not pts:
            return
        src = max(pts, key=lambda f: os.path.getmtime(os.path.join(logdir, f)))
        dst_dir = os.path.join(logdir, stage, str(epoch))
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(os.path.join(logdir, src), dst_dir)

    # ------------------------------------------------------------------
    # Inference API  (sess-free — compatible with notebook)
    # ------------------------------------------------------------------

    def _run_inference(self, dl, initial_state_np=None):
        self.eval()
        with torch.no_grad():
            for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
                Im, It, Rt, mt = self._t(I_macro), self._t(I), self._t(R), self._to_bool(mask)
                w, SDF, rnn_state, _ = self._sdf_pass(Im, It, Rt, mt, initial_state_np)
                return w.cpu().numpy(), SDF.cpu().numpy(), mask, R, self._pack_state(rnn_state)

    def getSDF(self, dl, initial_state=None):
        _, SDF, _, _, _ = self._run_inference(dl, initial_state)
        return SDF

    def getWeightWithData(self, dl, initial_state=None, normalized=False):
        w, _, mask, _, _ = self._run_inference(dl, initial_state)
        if normalized:
            splits = np.sum(mask, axis=1).cumsum()[:-1]
            w = np.concatenate([item / np.absolute(item).sum()
                                 for item in np.split(w, splits)])
        return w

    def getNormalizedSDF(self, dl, initial_state=None):
        w = self.getWeightWithData(dl, initial_state=initial_state, normalized=True)
        for _, (_, _, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
            chunks = np.split(R[mask] * w, np.sum(mask, axis=1).cumsum()[:-1])
            SDF = np.array([[c.sum()] for c in chunks]) + 1
        return SDF

    def getSDFFactor(self, dl, initial_state=None):
        return 1 - self.getSDF(dl, initial_state=initial_state)

    def getNormalizedSDFFactor(self, dl, initial_state=None):
        return 1 - self.getNormalizedSDF(dl, initial_state=initial_state)

    def getZeroInitialState(self):
        p = self._model_params
        hidden, layers = p["num_units_rnn"][0], p["num_layers_rnn"]
        if p["cell_type_rnn"] == "lstm":
            return np.zeros((layers, hidden * 2), dtype=np.float32)
        return np.zeros((layers, hidden), dtype=np.float32)

    def getNextInitialState(self, dl, initial_state):
        self.eval()
        with torch.no_grad():
            for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
                Im, It, Rt, mt = self._t(I_macro), self._t(I), self._t(R), self._to_bool(mask)
                _, _, rnn_state, _ = self._sdf_pass(Im, It, Rt, mt, initial_state)
        return self._pack_state(rnn_state)

    def getMomentWithData(self, dl, initial_state=None):
        self.eval()
        with torch.no_grad():
            for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
                return self._moment_pass(self._t(I_macro), self._t(I)).cpu().numpy()

    def calculateStatistics(self, dl, initial_state=None):
        return calculateStatistics(self.getWeightWithData(dl, initial_state=initial_state), dl)

    def evaluate_sharpe(self, dl, initial_state=None, normalized=False):
        SDF = self.getNormalizedSDF(dl, initial_state)[:, 0] if normalized \
              else self.getSDF(dl, initial_state)[:, 0]
        return sharpe(1 - SDF)
