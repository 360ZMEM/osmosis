"""@file eval/policy_loader.py
@brief Backend-agnostic policy loader for deployment / replay evaluation.

Supported model formats (auto-dispatched by file extension):
  .pt   -> torch.load (full nn.Module pickle, eval())
  .jit  -> torch.jit.load (TorchScript, eval())
  .onnx -> onnxruntime.InferenceSession (lazy import; CPU provider)

Contract:
  Policy(model_path).act(obs_np: np.ndarray (OBS_DIM,)) -> np.ndarray (4,) or (8,)

The wrapper deliberately does NOT depend on `omni.isaac.*`, `rsl_rl`, or
`gymnasium`; any host with `numpy + torch (or onnxruntime)` can run it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np


class Policy:
    """@brief Unified policy front-end.

    @details
      Internally caches a single backend handle (torch module or ORT session).
      Single-sample inference only; for batched evaluation prefer calling
      `act_batch` (broadcasts a (N, OBS_DIM) array).
    """

    def __init__(self, model_path: str | os.PathLike, *, device: str = "cpu"):
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"policy file not found: {self.model_path}")

        self.device = device
        suffix = self.model_path.suffix.lower()
        if suffix in (".pt", ".pth"):
            self._backend = "torch"
            self._handle = self._load_torch_pickle()
        elif suffix in (".jit", ".ts"):
            self._backend = "torchscript"
            self._handle = self._load_torchscript()
        elif suffix == ".onnx":
            self._backend = "onnx"
            self._handle = self._load_onnx()
        else:
            raise ValueError(
                f"unsupported policy extension '{suffix}' (expect .pt/.pth/.jit/.ts/.onnx)"
            )

    # ------------------------------------------------------------------ load

    def _load_torch_pickle(self):
        import torch  # local import keeps onnx-only deployment lean
        obj = torch.load(self.model_path, map_location=self.device, weights_only=False)
        if isinstance(obj, dict) and "model_state_dict" in obj:
            raise ValueError(
                "RSL-RL checkpoint dicts are not directly callable by the Isaac-independent "
                "Policy loader. Use the '*_deploy.jit' file exported by "
                "workflows/play_stdw_adapt.py --save_stdw_ckpt True --export_deploy_jit True."
            )
        if hasattr(obj, "eval"):
            obj.eval()
        self._torch = torch
        return obj

    def _load_torchscript(self):
        import torch
        module = torch.jit.load(str(self.model_path), map_location=self.device)
        module.eval()
        self._torch = torch
        return module

    def _load_onnx(self):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime not installed; pip install onnxruntime"
            ) from exc
        sess = ort.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )
        self._ort_input = sess.get_inputs()[0].name
        self._ort_output = sess.get_outputs()[0].name
        return sess

    # ------------------------------------------------------------------- act

    def act(self, obs: np.ndarray) -> np.ndarray:
        """@brief Single-sample inference.
        @param obs np.ndarray shape (OBS_DIM,) float32.
        @return np.ndarray shape (ACT_DIM,) float32.
        """
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        action = self._forward(obs)
        return action.reshape(-1)

    def act_batch(self, obs: np.ndarray) -> np.ndarray:
        """@brief Batched inference, obs shape (N, OBS_DIM)."""
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim != 2:
            raise ValueError(f"act_batch expects 2D obs, got shape {obs.shape}")
        return self._forward(obs)

    # ------------------------------------------------------------- internals

    def _forward(self, obs_2d: np.ndarray) -> np.ndarray:
        if self._backend in ("torch", "torchscript"):
            torch = self._torch
            with torch.no_grad():
                tensor = torch.from_numpy(obs_2d).to(self.device)
                out = self._handle(tensor)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                if not isinstance(out, torch.Tensor) and hasattr(out, "mean"):  # rsl_rl ActorCritic returns Distribution
                    try:
                        out = out.mean
                    except Exception:
                        pass
                return out.detach().cpu().numpy().astype(np.float32)
        # onnx
        result = self._handle.run([self._ort_output], {self._ort_input: obs_2d})
        return np.asarray(result[0], dtype=np.float32)

    # -------------------------------------------------------- introspection

    @property
    def backend(self) -> str:
        return self._backend

    def __repr__(self) -> str:  # pragma: no cover
        return f"Policy(path={self.model_path.name}, backend={self._backend})"
