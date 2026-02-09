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
        self._model = model
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
    # def infer(self, obs: dict, noise: jnp.ndarray | None = None) -> dict:  # type: ignore[misc]
    #     # Make a copy since transformations may modify the inputs in place.
    #     inputs = jax.tree.map(lambda x: x, obs)
    #     inputs = self._input_transform(inputs)
    #     # Make a batch and convert to jax.Array.
    #     if inputs["state"].ndim > 1:
    #         batch_size = inputs["state"].shape[0]
    #         def _add_batch_dim(x):
    #             return jnp.broadcast_to(
    #                 x[jnp.newaxis, ...],
    #                 (batch_size,) + x.shape
    #             )

    #         inputs = jax.tree.map(lambda x: jnp.asarray(x), inputs)
    #         for key in inputs:
    #             if key not in ["image", "state"]:
    #                 inputs[key] = jax.tree.map(lambda x: _add_batch_dim(x), inputs[key])
    #     else:
    #         batch_size = 1
    #         inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    #     # self._rng, sample_rng = jax.random.split(self._rng)
    #     if noise is None:
    #         self._rng, sample_rng = jax.random.split(self._rng)
    #         noise = jax.random.normal(sample_rng, (batch_size, self.action_horizon, self.action_dim))
    #     outputs = {
    #         "state": inputs["state"],
    #         "actions": self._sample_actions(_model.Observation.from_dict(inputs), noise=noise, **self._sample_kwargs),
    #     }

    #     # Unbatch and convert to np.ndarray.
    #     if batch_size == 1:
    #         outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

    #     return self._output_transform(outputs)
    def infer(
        self,
        obs: dict,
        *,
        action_noise: np.ndarray | None = None,
        cond_t: np.ndarray | None = None,
        prefix_noise: np.ndarray | None = None,
    ) -> dict:  # type: ignore[misc]
        # TODO: for now fitting the naming conventions
        noise = action_noise
        timestep_prefix = cond_t

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)

        batched = inputs["state"].ndim > 1
        batch_size = inputs["state"].shape[0] if batched else 1

        # if not self._is_pytorch_model:
        # Convert leaves to jax.Array and add batch dim if needed.
        def _to_jax_array(x):
            if x is None:
                return None
            arr = jnp.asarray(x)
            if not batched:
                arr = arr[np.newaxis, ...]
            return arr

        inputs = jax.tree.map(_to_jax_array, inputs)
        self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        # # else:
        # #     # Convert inputs to PyTorch tensors and move to correct device
        # #     def _to_torch_tensor(x):
        # #         if x is None:
        # #             return None
        # #         if isinstance(x, torch.Tensor):
        # #             tensor = x.to(self._pytorch_device)
        # #         else:
        # #             tensor = torch.from_numpy(np.asarray(x)).to(self._pytorch_device)
        # #         if not batched:
        # #             tensor = tensor.unsqueeze(0)
        # #         return tensor

        # #     inputs = jax.tree.map(_to_torch_tensor, inputs)
        # #     sample_rng_or_pytorch_device = self._pytorch_device

        # Add batch dim to masks
        if batched:
            for cam in inputs["image"].keys():
                if inputs["image_mask"][cam].ndim == 0:  # scalar
                    m = inputs["image_mask"][cam]
                    # if self._is_pytorch_model:
                    #     inputs["image_mask"][cam] = m.expand(batch_size)
                    # else:
                    inputs["image_mask"][cam] = jnp.broadcast_to(m, (batch_size,))

            # Explicitly handle tokenized_prompt and related keys to ensure they have batch dimension
            for key in ["tokenized_prompt", "tokenized_prompt_mask", "token_ar_mask", "token_loss_mask"]:
                if key in inputs and inputs[key] is not None:
                    val = jnp.asarray(inputs[key])
                    if val.ndim == 1:
                        # 1D array - add batch dim: [48] -> [batch_size, 48]
                        inputs[key] = jnp.broadcast_to(val[jnp.newaxis, ...], (batch_size,) + val.shape)
                    elif val.ndim == 0:
                        # Scalar - add batch dim
                        inputs[key] = jnp.broadcast_to(val[jnp.newaxis], (batch_size,))
                    elif val.shape[0] != batch_size:
                        # Has batch dim but wrong size - fix it
                        if val.shape[0] == 1:
                            inputs[key] = jnp.broadcast_to(val, (batch_size,) + val.shape[1:])
                        else:
                            inputs[key] = jnp.broadcast_to(val[jnp.newaxis, ...], (batch_size,) + val.shape)

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)

        if noise is None:
            self._rng, sample_rng = jax.random.split(self._rng)
            noise = jax.random.normal(sample_rng, (batch_size, self.action_horizon, self.action_dim))
            sample_kwargs["noise"] = noise
        else:
            if noise.ndim == 2:
                noise = np.repeat(noise[:, None, :], self.action_horizon, axis=1)
            assert noise.ndim == 3
            sample_kwargs["noise"] = noise

        if timestep_prefix is not None:
            # prefix_arr = _prepare_time_prefix(timestep_prefix)
            timestep_prefix = np.reshape(timestep_prefix, -1)
            sample_kwargs["time_prefix"] = timestep_prefix

        if prefix_noise is not None:
            # Convert prefix_noise to numpy array if it's a PyTorch tensor
            if isinstance(prefix_noise, torch.Tensor):
                prefix_noise_np = prefix_noise.detach().cpu().numpy()
            else:
                prefix_noise_np = np.asarray(prefix_noise)

            prefix_noise_arr = np.repeat(prefix_noise_np[:, None, :], 816, axis=1)  # TODO: hardcoded
            prefix_noise_tensor = jnp.asarray(prefix_noise_arr)
            sample_kwargs["noise_prefix"] = prefix_noise_tensor

        observation = _model.Observation.from_dict(inputs)
        

        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(observation=observation, rng=sample_rng_or_pytorch_device, **sample_kwargs),
        }
        
        model_time = time.monotonic() - start_time
        outputs = jax.tree.map(lambda x: np.asarray(x) if x is not None else None, outputs)

        if not batched:
            outputs = jax.tree.map(lambda x: x[0, ...] if hasattr(x, "ndim") and x.ndim > 0 else x, outputs)

        outputs = self._output_transform(outputs)
        outputs["infer_ms"] = model_time * 1000
        return outputs

    @override
    def get_prefix_rep(self, obs: dict):
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x), inputs)
        # add batch dim and broadcast for keys that are not "images" and "state"
        if inputs["state"].ndim > 1:
            batch_size = inputs["state"].shape[0]

            def _add_batch_dim(x):
                if x is None:
                    return None
                # Skip if not an array-like object
                if not hasattr(x, "ndim"):
                    return x
                if x.ndim == 0:
                    # Scalar - broadcast to batch
                    return jnp.broadcast_to(x[jnp.newaxis], (batch_size,))
                elif x.ndim == 1:
                    # 1D array (e.g., tokenized_prompt) - add batch dim and broadcast
                    return jnp.broadcast_to(x[jnp.newaxis, ...], (batch_size,) + x.shape)
                else:
                    # Already has batch dim or is higher dim - check if first dim matches
                    if x.shape[0] == 1:
                        return jnp.broadcast_to(x, (batch_size,) + x.shape[1:])
                    elif x.shape[0] == batch_size:
                        return x
                    else:
                        # Unexpected shape - try to add batch dim
                        return jnp.broadcast_to(x[jnp.newaxis, ...], (batch_size,) + x.shape)

            # Handle image_mask dict specially - add batch dim to each value
            if "image_mask" in inputs and isinstance(inputs["image_mask"], dict):
                inputs["image_mask"] = {
                    k: _add_batch_dim(v) if v is not None else v for k, v in inputs["image_mask"].items()
                }

            # Explicitly handle tokenized_prompt and tokenized_prompt_mask
            for key in ["tokenized_prompt", "tokenized_prompt_mask", "token_ar_mask", "token_loss_mask"]:
                if key in inputs and inputs[key] is not None:
                    # Ensure it's a JAX array and add batch dimension
                    val = jnp.asarray(inputs[key])
                    if val.ndim == 1:
                        # 1D array - add batch dim: [48] -> [1, 48]
                        inputs[key] = jnp.broadcast_to(val[jnp.newaxis, ...], (batch_size,) + val.shape)
                    elif val.ndim == 0:
                        # Scalar - add batch dim
                        inputs[key] = jnp.broadcast_to(val[jnp.newaxis], (batch_size,))
                    elif val.shape[0] != batch_size:
                        # Has batch dim but wrong size - fix it
                        if val.shape[0] == 1:
                            inputs[key] = jnp.broadcast_to(val, (batch_size,) + val.shape[1:])
                        else:
                            inputs[key] = jnp.broadcast_to(val[jnp.newaxis, ...], (batch_size,) + val.shape)
                    # If shape[0] == batch_size, it's already correct, leave it as is

            # Handle other keys (excluding "image" and "state" which are already handled)
            for key in inputs:
                if key not in [
                    "image",
                    "state",
                    "image_mask",
                    "tokenized_prompt",
                    "tokenized_prompt_mask",
                    "token_ar_mask",
                    "token_loss_mask",
                ]:
                    if key in inputs and inputs[key] is not None:
                        # Skip dicts (they should be handled separately)
                        if isinstance(inputs[key], dict):
                            inputs[key] = {
                                k: _add_batch_dim(v) if v is not None and hasattr(v, "ndim") else v
                                for k, v in inputs[key].items()
                            }
                        else:
                            inputs[key] = _add_batch_dim(inputs[key])
        else:
            # No batch dim yet - add it to all inputs
            def _add_single_batch_dim(x):
                if x is None:
                    return None
                # Skip dicts - handle them separately
                if isinstance(x, dict):
                    return {k: _add_single_batch_dim(v) for k, v in x.items()}
                x_arr = jnp.asarray(x)
                if x_arr.ndim == 0:
                    return x_arr[jnp.newaxis]
                return x_arr[jnp.newaxis, ...]

            inputs = jax.tree.map(_add_single_batch_dim, inputs)
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
