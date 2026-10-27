"""Krea2 Variance — cond-noise machinery (positive-conditioning perturbation).

Pure tensor math for the "Cond noise" dial, no knowledge of the sampler or
the WebUI:

* `_CondStats`         — accumulated diagnostics for one noised pass
* `_noised_obj`        — recursive deep-copy noiser for the conditioning
* `_noised_conditioning` — the noised copy + stats (per-image seeded)
* `_expand_seeds`      — map a per-image seed list onto the image count
* `_kv_seed`           — base seed for a generation (first image)
* `_kv_seeds`          — per-image base seeds for a generation
* `_CondNoiseModel`    — call-counting stand-in for the CFG denoiser

scripts/krea2_variance.py wires this into the pipeline via
`_kv_cond_prepare` (which wraps `self.model_wrap_cfg`).

Each image's conditioning is noised with its own CPU torch.Generator seeded
from that image's own seed (the same per-image seeds the latent noise uses),
so the perturbation is deterministic per image and independent of the latent
RNG (TorchHijack/p.rng). The baseline and the step-1 boost passes use the
SAME per-image seeds, hence the SAME random draw per image — only the scale
differs — so the perturbation direction is consistent across all steps.
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


def _expand_seeds(seeds, n):
    """Map a seed list onto n images, one seed per image.

    len == n: as-is; len == 1: seed + i (the same derivation the WebUI uses
    to turn one seed into per-image seeds); len > n: first n; 1 < len < n:
    wrap around (deterministic; only collides in degenerate setups).
    """
    seeds = [int(s) for s in seeds]
    if n <= 0:
        return []
    if not seeds:
        return [0] * n
    if len(seeds) == 1:
        return [seeds[0] + i for i in range(n)]
    if len(seeds) >= n:
        return seeds[:n]
    return [seeds[i % len(seeds)] for i in range(n)]


def _noised_conditioning(conditioning, seeds, strength):
    """Deep-copy the conditioning, noising each IMAGE with its own seeded
    generator. `seeds` is the per-image seed list (a scalar seed is tolerated
    and expanded to one per image via _expand_seeds).

    Per-image boundary:
    * an object with a `.batch` list (MulticondLearnedConditioning): each
      `.batch[i]` is image i's conditioning (a list of
      ComposableScheduledPromptConditioning whose .cond is the per-image
      [seq, dim] tensor); each row gets its own generator and the top-level
      object is rebuilt with the noised rows (other fields, e.g. the
      immutable .shape tuple, are shared as-is);
    * a top-level list/tuple whose length matches the image count: each
      element is one image and gets its own generator;
    * anything else: a single object noised with the first seed.
    All images accumulate into one _CondStats. Never mutates the input.
    """
    if not isinstance(seeds, (list, tuple)):
        seeds = [seeds]
    seeds = [int(s) for s in seeds]
    strength = float(strength)
    batch = getattr(conditioning, "batch", None)
    if isinstance(batch, (list, tuple)) and len(batch) > 0:
        img_seeds = _expand_seeds(seeds, len(batch))
        stats = _CondStats()
        noised = [_noised_obj(entry, torch.Generator().manual_seed(s), strength, stats) for s, entry in zip(img_seeds, batch)]
        new = conditioning.__class__.__new__(conditioning.__class__)
        if getattr(conditioning, "__dict__", None):
            new.__dict__ = dict(conditioning.__dict__)
        new.batch = noised
        return new, stats
    if isinstance(conditioning, (list, tuple)) and len(conditioning) == len(seeds) > 0:
        stats = _CondStats()
        noised = [_noised_obj(entry, torch.Generator().manual_seed(s), strength, stats) for s, entry in zip(seeds, conditioning)]
        return (tuple(noised) if isinstance(conditioning, tuple) else noised), stats
    stats = _CondStats()
    seed = seeds[0] if seeds else 0
    return _noised_obj(conditioning, torch.Generator().manual_seed(seed), strength, stats), stats


def _kv_seeds(p):
    """Per-image base seeds for this generation (one per image in the batch).

    p.seeds (sliced per batch right before p.sample) is the source of truth;
    p.all_seeds is a fallback for paths running before that slice (sliced by
    the current iteration); the scalar p.seed is the last resort (expanded
    to one per image by _expand_seeds).

    Subseed mode: with subseed_strength > 0 the WebUI gives EVERY slot the
    same seed (modules/processing.py: all_seeds = [seed] * N; the variation
    comes from p.subseeds). Using those identical seeds would give every
    image the SAME cond perturbation, so when all of p.seeds are equal the
    per-slot seeds are derived from p.subseeds instead (seed + offset — the
    same derivation the WebUI uses in the normal mode).
    """
    seeds = getattr(p, "seeds", None)
    if seeds:
        seeds = [int(s) for s in seeds]
        if len(seeds) > 1 and len(set(seeds)) == 1:
            subseeds = getattr(p, "subseeds", None)
            if subseeds and len(subseeds) == len(seeds):
                subseeds = [int(s) for s in subseeds]
                return [seeds[0] + (s - subseeds[0]) for s in subseeds]
        return seeds
    all_seeds = getattr(p, "all_seeds", None)
    if all_seeds:
        iteration = max(0, int(getattr(p, "iteration", 0) or 0))
        batch_size = max(1, int(getattr(p, "batch_size", len(all_seeds)) or 1))
        return [int(s) for s in all_seeds[iteration * batch_size : (iteration + 1) * batch_size] or all_seeds]
    s = getattr(p, "seed", None)
    return [int(s) if s is not None and int(s) > 0 else 0]


def _kv_seed(p):
    """Base seed for this generation (first image)."""
    return _kv_seeds(p)[0]


class _CondNoiseModel:
    """Call-counting stand-in for the CFG denoiser (self.model_wrap_cfg).

    Swaps extra_args["cond"] per model call: the FIRST call (the
    identity-forming step) gets the boosted conditioning (baseline + extra
    boost), calls 1..steps-1 get the baseline conditioning, and calls from
    `steps` onward get the clean (original) conditioning. All other
    attributes (launch_sampling / initialize set .steps/.total_steps/.p/
    .mask/... on model_wrap_cfg) are delegated to the wrapped denoiser on
    both get and set.
    """

    _OWN = ("_inner", "_first", "_rest", "_clean", "_steps", "_calls")

    def __init__(self, inner, first_cond, rest_cond, clean_cond, steps):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_first", first_cond)
        object.__setattr__(self, "_rest", rest_cond)
        object.__setattr__(self, "_clean", clean_cond)
        object.__setattr__(self, "_steps", max(1, int(steps)))
        object.__setattr__(self, "_calls", 0)

    def __call__(self, x, t, **kwargs):
        if self._calls == 0:
            kwargs["cond"] = self._first
        elif self._calls < self._steps:
            kwargs["cond"] = self._rest
        else:
            kwargs["cond"] = self._clean
        self._calls += 1
        return self._inner(x, t, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __setattr__(self, name, value):
        if name in _CondNoiseModel._OWN:
            object.__setattr__(self, name, value)
        else:
            setattr(self._inner, name, value)
