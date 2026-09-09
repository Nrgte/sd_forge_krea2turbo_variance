"""Krea2 Variance — cond-noise machinery (positive-conditioning perturbation).

Pure tensor math for the "Cond noise" dial, no knowledge of the sampler or
the WebUI:

* `_CondStats`         — accumulated diagnostics for one noised pass
* `_noised_obj`        — recursive deep-copy noiser for the conditioning
* `_noised_conditioning` — one seeded CPU generator + the noised copy + stats
* `_kv_seed`           — base seed for a generation
* `_CondNoiseModel`    — step-counting stand-in for the CFG denoiser

scripts/krea2_variance.py wires this into the pipeline via
`_kv_cond_prepare` (which wraps `self.model_wrap_cfg`).

All noise is drawn from a single CPU torch.Generator seeded from the base
seed, so the perturbation is deterministic per seed and independent of the
latent RNG (TorchHijack/p.rng).
"""

import torch


class _CondStats:
    """Accumulated diagnostics for one noised-conditioning pass."""

    def __init__(self):
        self.std_sum_sq = 0.0
        self.clean_norm_sq = 0.0
        self.diff_norm_sq = 0.0
        self.values = 0
        self.skipped = 0

    @property
    def std(self):
        return (self.std_sum_sq / self.values) ** 0.5 if self.values else 0.0

    @property
    def rel_l2(self):
        return (self.diff_norm_sq / self.clean_norm_sq) ** 0.5 if self.clean_norm_sq > 0 else 0.0


def _noised_obj(obj, gen, strength, stats):
    """Deep-copy obj, adding seeded gaussian noise to every floating-point tensor
    it contains, scaled by that tensor's own measured std * strength.

    Structure is preserved (tensor / list / list-subclass / tuple / namedtuple /
    dict / dict-subclass / arbitrary objects via __dict__); int tensors (e.g.
    token ids) and scalars pass through untouched. Never mutates in place —
    the Qwen3-VL engine caches embeddings per prompt line and the same tensor
    objects are shared across schedule entries and generations.
    """
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point() and obj.numel() > 0:
            std = obj.std()
            if std is not None and torch.isfinite(std):
                # `gen` is always a CPU generator, so the noise is generated on
                # CPU and MOVED to the tensor's device. Two device traps:
                #   1. passing a CPU generator to torch.randn with
                #      device=obj.device raises "Expected a 'cuda' device type
                #      for generator but found 'cpu'";
                #   2. obj.std() is a 0-D TENSOR on obj.device — scaling the
                #      CPU noise with it raises "Expected all tensors to be on
                #      the same device ... cuda:N and cpu" (multi-GPU setups).
                # So the scale is a plain host float.
                scale = float(std) * strength
                noise = (torch.randn(obj.shape, dtype=torch.float32, generator=gen) * scale).to(obj.device, obj.dtype)
                stats.values += obj.numel()
                stats.std_sum_sq += float(std) ** 2 * obj.numel()
                stats.clean_norm_sq += float((obj * obj).sum())
                stats.diff_norm_sq += float((noise * noise).sum())
                return obj + noise
            stats.skipped += 1
        return obj
    if isinstance(obj, dict):
        return type(obj)({k: _noised_obj(v, gen, strength, stats) for k, v in obj.items()})
    if type(obj) is list:
        return [_noised_obj(v, gen, strength, stats) for v in obj]
    if isinstance(obj, tuple):
        items = [_noised_obj(v, gen, strength, stats) for v in obj]
        if type(obj) is tuple:
            return tuple(items)
        return obj.__class__(*items)  # namedtuple / tuple subclass
    if isinstance(obj, list):
        new = obj.__class__.__new__(obj.__class__)
        if getattr(obj, "__dict__", None):
            new.__dict__ = dict(obj.__dict__)
        new.extend(_noised_obj(v, gen, strength, stats) for v in obj)
        return new
    d = getattr(obj, "__dict__", None)
    if d:
        new = obj.__class__.__new__(obj.__class__)
        new.__dict__ = {k: _noised_obj(v, gen, strength, stats) for k, v in d.items()}
        return new
    return obj


def _noised_conditioning(conditioning, seed, strength):
    gen = torch.Generator().manual_seed(int(seed))
    stats = _CondStats()
    return _noised_obj(conditioning, gen, float(strength), stats), stats


def _kv_seed(p):
    """Base seed for this generation (always a fixed int by sampling time)."""
    seeds = getattr(p, "seeds", None) or getattr(p, "all_seeds", None) or []
    if seeds:
        return int(seeds[0])
    s = getattr(p, "seed", None)
    return int(s) if s is not None and int(s) > 0 else 0


class _CondNoiseModel:
    """Step-counting stand-in for the CFG denoiser (self.model_wrap_cfg).

    Swaps extra_args["cond"] between the noised and the clean positive
    conditioning: the first `steps` model calls get the noised copy, later
    calls get the original. All other attributes (launch_sampling/initialize
    set .steps/.total_steps/.p/.mask/... on model_wrap_cfg) are delegated to
    the wrapped denoiser on both get and set.
    """

    _OWN = ("_inner", "_noised", "_clean", "_steps", "_calls")

    def __init__(self, inner, noised_cond, clean_cond, steps):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_noised", noised_cond)
        object.__setattr__(self, "_clean", clean_cond)
        object.__setattr__(self, "_steps", int(steps))
        object.__setattr__(self, "_calls", 0)

    def __call__(self, x, t, **kwargs):
        kwargs["cond"] = self._noised if self._calls < self._steps else self._clean
        self._calls += 1
        return self._inner(x, t, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __setattr__(self, name, value):
        if name in _CondNoiseModel._OWN:
            object.__setattr__(self, name, value)
        else:
            setattr(self._inner, name, value)
