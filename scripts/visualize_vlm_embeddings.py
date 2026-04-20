"""Visualize raw VLM prefix embeddings and the effect of CSPi0 context noise at 4 linspaced levels.

Loads one real batch from whatever dataset the config points to, so embeddings reflect
actual observations. Requires norm stats to be computed for the config first.

Top row:    PCA projection of prefix tokens at noise levels t = 0.0, 0.33, 0.67, 1.0.
Middle row: Per-token cosine similarity to the clean (t=0) embedding, averaged over the batch.
Bottom row: Mean token L2 norm (±std) and total embedding variance vs noise level.

Usage:
    uv run scripts/visualize_vlm_embeddings.py \\
        --config_name cspi05_droid \\
        --checkpoint_path /path/to/checkpoint/params \\
        --output_path vlm_embeddings.png
"""

import argparse
import logging
import pathlib

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

import tmrl_openpi.models.cspi0 as cspi0
import tmrl_openpi.models.model as _model
import tmrl_openpi.training.config as _config
import tmrl_openpi.training.data_loader as _data_loader
import tmrl_openpi.training.weight_loaders as weight_loaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(config_name: str, checkpoint_path: str | None):
    config = _config.get_config(config_name)
    if not isinstance(config.model, cspi0.CSPi0Config):
        raise ValueError(
            f"Config '{config_name}' must use CSPi0Config (got {type(config.model).__name__})."
        )

    rng = jax.random.PRNGKey(0)
    model = config.model.create(rng)

    if checkpoint_path is not None:
        logger.info(f"Loading weights from {checkpoint_path}")
        loader = weight_loaders.CheckpointWeightLoader(checkpoint_path)
        params_shape = nnx.state(model).to_pure_dict()
        loaded = loader.load(params_shape)
        flat = {
            k: v
            for k, v in traverse_util.flatten_dict(loaded).items()
            if not isinstance(v, jax.ShapeDtypeStruct)
        }
        graphdef, state = nnx.split(model)
        state.replace_by_pure_dict(traverse_util.unflatten_dict(flat))
        model = nnx.merge(graphdef, state)
        logger.info("Weights loaded.")
    else:
        logger.warning(
            "No checkpoint — using random weights. Embeddings won't be semantically meaningful."
        )

    model.eval()
    return model, config


def get_prefix_tokens(model, config, batch_size: int = 8):
    """Pull one real batch from the config's dataset, embed the prefix, return tokens + mask."""
    import dataclasses

    # Override batch_size so we get exactly as many samples as requested.
    vis_config = dataclasses.replace(config, batch_size=batch_size)

    logger.info(f"Loading one batch (batch_size={batch_size}) from dataset...")
    loader = _data_loader.create_data_loader(
        vis_config,
        skip_norm_stats=False,
        shuffle=True,
        num_batches=1,
        num_workers=0,
    )
    obs, _ = next(iter(loader))

    obs = _model.preprocess_observation(None, obs, train=False)
    prefix_tokens, prefix_mask, _ = model.embed_prefix(obs)
    return np.array(prefix_tokens), np.array(prefix_mask)


def rms_normalize(tokens: np.ndarray) -> np.ndarray:
    """Per-token RMS normalization — matches CSPi0 model."""
    rms = np.sqrt(np.mean(np.square(tokens), axis=-1, keepdims=True))
    return tokens / (rms + 1e-6)


def apply_noise(
    prefix_tokens_np: np.ndarray, alpha_bars: list, time_frac: float, rng_key
) -> np.ndarray:
    """Apply DDIM-style marginal: noisy = sqrt(ᾱ_t) * tokens + sqrt(1-ᾱ_t) * ε."""
    if time_frac == 0.0:
        return prefix_tokens_np
    T = len(alpha_bars)
    t_idx = int(np.clip(time_frac * (T - 1), 0, T - 1))
    sqrt_ab = float(np.sqrt(alpha_bars[t_idx]))
    sqrt_bb = float(np.sqrt(1.0 - alpha_bars[t_idx]))
    noise = np.array(
        jax.random.normal(rng_key, prefix_tokens_np.shape, dtype=jnp.float32)
    )
    return (sqrt_ab * prefix_tokens_np + sqrt_bb * noise).astype(prefix_tokens_np.dtype)


def make_label(t: float, alpha_bars: list) -> str:
    T = len(alpha_bars)
    if t == 0.0:
        return "t = 0.00  (clean)"
    t_idx = int(np.clip(t * (T - 1), 0, T - 1))
    ab = alpha_bars[t_idx]
    return f"t = {t:.2f}  (ᾱ = {ab:.4f})"


def visualize(
    prefix_tokens: np.ndarray,
    prefix_mask: np.ndarray,
    alpha_bars: list,
    output_path: str,
    *,
    normalize: bool = False,
):
    """
    prefix_tokens : (B, S, D)
    prefix_mask   : (B, S)  boolean, True = valid token
    normalize     : if True, apply per-token RMS normalization before noise (matches CSPi0 model)
    """
    if normalize:
        prefix_tokens = rms_normalize(prefix_tokens)

    B, S, D = prefix_tokens.shape
    noise_fracs = np.linspace(0.0, 1.0, 4)

    rng = jax.random.PRNGKey(42)
    noised_list = []
    for t in noise_fracs:
        rng, sub = jax.random.split(rng)
        noised_list.append(apply_noise(prefix_tokens, alpha_bars, float(t), sub))

    # Fit PCA on the union of all noise levels (valid tokens only)
    all_valid = np.concatenate(
        [n[prefix_mask].astype(np.float32) for n in noised_list], axis=0
    )
    pca = PCA(n_components=2)
    pca.fit(all_valid)

    # Precompute per-noise-level stats over valid tokens
    def valid_tokens(arr):
        return arr[prefix_mask].astype(np.float32)  # (N_valid, D)

    stats = []
    for noised in noised_list:
        toks = valid_tokens(noised)  # (N, D)
        mean_vec = toks.mean(axis=0)  # (D,)
        mean_norm = float(np.linalg.norm(mean_vec))
        token_norms = np.linalg.norm(toks, axis=-1)  # (N,)
        mean_token_norm = float(token_norms.mean())
        variance = float(toks.var())  # scalar variance over all dims
        std_token_norm = float(token_norms.std())
        stats.append(
            dict(
                mean_norm=mean_norm,
                mean_token_norm=mean_token_norm,
                variance=variance,
                std_token_norm=std_token_norm,
            )
        )

    sample_colors = plt.cm.tab10(np.linspace(0, 1, B))
    line_colors = plt.cm.viridis(np.linspace(0.1, 0.9, 4))

    fig = plt.figure(figsize=(22, 14))
    gs = gridspec.GridSpec(
        3, 4, figure=fig, hspace=0.45, wspace=0.3, height_ratios=[1.2, 1, 1]
    )

    # ── Top row: PCA scatter per noise level ─────────────────────────────
    for col, (noised, t, st, lc) in enumerate(
        zip(noised_list, noise_fracs, stats, line_colors)
    ):
        ax = fig.add_subplot(gs[0, col])
        for b in range(B):
            valid = prefix_mask[b]
            proj = pca.transform(noised[b][valid].astype(np.float32))
            ax.scatter(
                proj[:, 0],
                proj[:, 1],
                s=8,
                alpha=0.55,
                color=sample_colors[b],
                label=f"obs {b}" if col == 0 else None,
            )

        # Annotate with mean and variance
        info = f"μ‖tok‖ = {st['mean_token_norm']:.3f}\nσ‖tok‖ = {st['std_token_norm']:.3f}\nvar     = {st['variance']:.3f}"
        ax.text(
            0.03,
            0.97,
            info,
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            ha="left",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )

        ax.set_title(make_label(t, alpha_bars), fontsize=9)
        ax.set_xlabel("PC 1", fontsize=8)
        ax.set_ylabel("PC 2", fontsize=8)
        ax.tick_params(labelsize=7)
        if col == 0:
            ax.legend(fontsize=6, markerscale=2, loc="lower right")

    norm_tag = "RMS-normalized" if normalize else "raw"
    fig.text(
        0.5,
        0.97,
        f"VLM Prefix Token Embeddings ({norm_tag}) — PCA at 4 Linspaced Noise Levels (CSPi0 Forward Process)",
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
    )

    # ── Middle row: cosine similarity to clean ───────────────────────────
    ax_sim = fig.add_subplot(gs[1, :])
    token_idx = np.arange(S)
    clean_tokens = noised_list[0]  # t=0, already normalized if normalize=True

    for noised, t, lc in zip(noised_list, noise_fracs, line_colors):
        dot = np.sum(clean_tokens * noised, axis=-1)
        nc = np.linalg.norm(clean_tokens, axis=-1)
        nn = np.linalg.norm(noised, axis=-1)
        cos = dot / (nc * nn + 1e-8)

        avg = np.full(S, np.nan)
        for s in range(S):
            valid_b = prefix_mask[:, s]
            if valid_b.any():
                avg[s] = cos[valid_b, s].mean()

        ax_sim.plot(
            token_idx,
            avg,
            label=make_label(t, alpha_bars),
            color=lc,
            alpha=0.9,
            linewidth=1.5,
        )

    ax_sim.set_xlabel("Token position", fontsize=9)
    ax_sim.set_ylabel("Cosine similarity to clean", fontsize=9)
    ax_sim.set_title(
        "Per-token cosine similarity to clean embedding (avg over batch)", fontsize=10
    )
    ax_sim.legend(fontsize=8)
    ax_sim.set_ylim(-0.15, 1.1)
    ax_sim.axhline(1.0, color="k", linestyle="--", linewidth=0.6, alpha=0.3)
    ax_sim.axhline(0.0, color="k", linestyle="--", linewidth=0.6, alpha=0.3)
    ax_sim.tick_params(labelsize=8)

    # ── Bottom row: mean token norm and variance vs noise level ──────────
    ax_mn = fig.add_subplot(gs[2, :2])
    ax_var = fig.add_subplot(gs[2, 2:])

    mean_norms = [st["mean_token_norm"] for st in stats]
    std_norms = [st["std_token_norm"] for st in stats]
    variances = [st["variance"] for st in stats]

    ax_mn.errorbar(
        noise_fracs,
        mean_norms,
        yerr=std_norms,
        marker="o",
        color="steelblue",
        linewidth=1.8,
        capsize=4,
        label="mean ± std of ‖token‖",
    )
    ax_mn.set_xlabel("Noise level t", fontsize=9)
    ax_mn.set_ylabel("Token L2 norm", fontsize=9)
    ax_mn.set_title("Mean (±std) token L2 norm vs noise level", fontsize=10)
    ax_mn.legend(fontsize=8)
    ax_mn.tick_params(labelsize=8)

    ax_var.plot(noise_fracs, variances, marker="o", color="darkorange", linewidth=1.8)
    ax_var.set_xlabel("Noise level t", fontsize=9)
    ax_var.set_ylabel("Variance", fontsize=9)
    ax_var.set_title("Embedding variance vs noise level", fontsize=10)
    ax_var.tick_params(labelsize=8)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved → {output_path}")
    plt.close(fig)


def _auto_output_path(base: str, normalize: bool) -> str:
    """Append _normalized before the extension when normalize=True."""
    if not normalize:
        return base
    p = pathlib.Path(base)
    return str(p.with_stem(p.stem + "_normalized"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_name",
        default="cspi05_droid",
        help="TrainConfig name (must use CSPi0Config).",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="Path to params checkpoint. Omit for random weights.",
    )
    parser.add_argument("--output_path", default="vlm_embeddings.png")
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Number of observations to embed."
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Apply per-token RMS normalization before noise (matches CSPi0 model).",
    )
    args = parser.parse_args()

    model, config = load_model(args.config_name, args.checkpoint_path)
    alpha_bars: list = model.alpha_bars
    T = len(alpha_bars)
    logger.info(
        f"Noise schedule: T={T}, ᾱ_0={alpha_bars[0]:.5f}, ᾱ_{T - 1}={alpha_bars[-1]:.7f}"
    )

    logger.info(f"Embedding {args.batch_size} observations...")
    prefix_tokens, prefix_mask = get_prefix_tokens(
        model, config, batch_size=args.batch_size
    )
    logger.info(f"prefix_tokens: {prefix_tokens.shape}  (B, S, D)")

    output_path = _auto_output_path(args.output_path, args.normalize)
    logger.info(f"normalize={args.normalize}  →  {output_path}")
    visualize(
        prefix_tokens, prefix_mask, alpha_bars, output_path, normalize=args.normalize
    )


if __name__ == "__main__":
    main()
