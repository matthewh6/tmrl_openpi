from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from dsrl_openpi import transforms as _transforms
from dsrl_openpi.models import model as _model
from dsrl_openpi.shared import array_typing as at
from dsrl_openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self.action_dim = model.action_dim
        self.action_horizon = model.action_horizon
        self._get_prefix_rep = nnx_utils.module_jit(model.get_prefix_rep)

    @override
    def infer(
        self,
        obs: dict,
        *,
        action_noise: np.ndarray | None = None,
        cond_t: np.ndarray | None = None,
        prefix_noise: np.ndarray | None = None,
    ) -> dict:  # type: ignore[misc]
        noise = action_noise
        timestep_prefix = cond_t

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)

        # Batch handling -- same logic as openpi reference.
        if inputs["state"].ndim > 1:
            batched = True
            batch_size = inputs["state"].shape[0]

            def _add_batch_dim(x):
                return jnp.broadcast_to(
                    x[jnp.newaxis, ...],
                    (batch_size,) + x.shape,
                )

            inputs = jax.tree.map(lambda x: jnp.asarray(x), inputs)
            for key in inputs:
                if key not in ["image", "state"]:
                    inputs[key] = jax.tree.map(_add_batch_dim, inputs[key])
        else:
            batched = False
            batch_size = 1
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        # Prepare sample_kwargs
        sample_kwargs = dict(self._sample_kwargs)

        # Noise (action noise)
        if noise is None:
            self._rng, sample_rng = jax.random.split(self._rng)
            noise = jax.random.normal(sample_rng, (batch_size, self.action_horizon, self.action_dim))
        else:
            if isinstance(noise, torch.Tensor):
                noise = noise.detach().cpu().numpy()
            noise = np.asarray(noise)
            if noise.ndim == 2:
                noise = np.repeat(noise[:, None, :], self.action_horizon, axis=1)
            assert noise.ndim == 3
        sample_kwargs["noise"] = noise

        # Time prefix (cond_t for TMRL)
        if timestep_prefix is not None:
            timestep_prefix = np.reshape(np.asarray(timestep_prefix), -1)
            sample_kwargs["time_prefix"] = timestep_prefix

        # Prefix noise (for TMRL)
        if prefix_noise is not None:
            if isinstance(prefix_noise, torch.Tensor):
                prefix_noise = prefix_noise.detach().cpu().numpy()
            prefix_noise = np.asarray(prefix_noise)
            prefix_noise = np.repeat(prefix_noise[:, None, :], 816, axis=1)  # TODO: hardcoded
            sample_kwargs["noise_prefix"] = jnp.asarray(prefix_noise)

        # Sample actions
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(
                _model.Observation.from_dict(inputs),
                noise=noise,
                **{k: v for k, v in sample_kwargs.items() if k != "noise"},
            ),
        }
        model_time = time.monotonic() - start_time

        # Unbatch and convert to np.ndarray.
        if batch_size == 1:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x) if x is not None else None, outputs)
            
        outputs = self._output_transform(outputs)
        outputs["infer_ms"] = model_time * 1000
        return outputs

    @override
    def get_prefix_rep(self, obs: dict):
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x), inputs)

        if inputs["state"].ndim > 1:
            batch_size = inputs["state"].shape[0]

            def _add_batch_dim(x):
                return jnp.broadcast_to(
                    x[jnp.newaxis, ...],
                    (batch_size,) + x.shape,
                )

            for key in inputs:
                if key not in ["image", "state"]:
                    inputs[key] = jax.tree.map(_add_batch_dim, inputs[key])
        else:
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        return self._get_prefix_rep(_model.Observation.from_dict(inputs))

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
