"""Krea2 Variance — the "Noise (sampler: Euler)" latent dial.

`sample_euler_variance` = k_diffusion.sampling.sample_euler with a seeded
re-noise after each Euler step in [var_start, var_end], scaled by the next
sigma (== remaining noise level). All noise is drawn through
k_diffusion.sampling.torch, which Sampler.initialize() reassigns to
TorchHijack(p) — i.e. from p.rng, the per-image seeded generators
(modules/rng.py) — so it is deterministic per seed.

scripts/krea2_variance.py installs it as a functools.partial with the
per-generation var_noise/var_start/var_end values.
"""

import torch
from tqdm.auto import trange


@torch.no_grad()
def sample_euler_variance(model, x, sigmas, extra_args=None, callback=None, disable=None,
                          s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0,
                          var_noise=0.0, var_start=0, var_end=None):
    """k_diffusion.sampling.sample_euler + seeded re-noise after each Euler step
    in [var_start, var_end], scaled by the next sigma (== remaining noise level)."""
    import k_diffusion.sampling as ks

    seeded = ks.torch  # TorchHijack(p) after Sampler.initialize()
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    n = len(sigmas) - 1
    end = n - 1 if var_end is None else max(0, min(int(var_end), n - 1))
    start = max(0, int(var_start))
    for i in trange(n, disable=disable):
        if s_churn > 0:
            gamma = min(s_churn / n, 2**0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
            sigma_hat = sigmas[i] * (gamma + 1)
        else:
            gamma = 0
            sigma_hat = sigmas[i]

        if gamma > 0:
            eps = seeded.randn_like(x) * s_noise
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = ks.to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        x = x + d * (sigmas[i + 1] - sigma_hat)
        if var_noise > 0 and start <= i <= end and sigmas[i + 1] > 0:
            x = x + seeded.randn_like(x) * (var_noise * sigmas[i + 1])
    return x
