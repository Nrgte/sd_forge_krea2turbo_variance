"""Offline functional test for extensions/sd_forge_krea2_variance/scripts/krea2_variance.py.

Runs WITHOUT the full WebUI: stubs modules/backend/k_diffusion, loads the
extension (scripts/krea2_variance.py + the extension-root lib_krea2/ package,
via sys.path like Forge does), and verifies:
  1. _install() patches KDiffusionSampler.sample/sample_img2img (and its guard)
  2. _kv_prepare dispatch per tab sampler (Euler a / ER SDE / Euler / other)
  3. process() arg mapping + infotext + krea2/enabled/sampler gating
  4. ui() returns 9 components
  5. sample_euler_variance: matches plain Euler at var_noise=0, seeded and
     reproducible (same seed -> identical), varies with seed, window honored
  6. patched sample/img2img swap+restore (func and model_wrap_cfg)
  7. _noised_conditioning / _noised_obj: structure preservation, per-image
     seeded determinism (own generator per .batch row), no in-place
     mutation, strength 0 no-op, int pass-through
  8. _CondNoiseModel: per-call cond gating (step-1 boost, baseline window,
     clean after) + attribute delegation
  9. end-to-end cond noise through _patched_sample (sampler-independent +
     additive with the Euler latent dials)
  10. process() cond-noise mapping + infotext + gating
"""
import functools
import importlib.util
import io
import os
import sys
import types
from collections import namedtuple
from contextlib import redirect_stdout
from types import SimpleNamespace

import torch

# The test sits at the extension root, next to the code under test, so the
# paths resolve from __file__ (portable, no machine-specific configuration).
EXT_ROOT = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(EXT_ROOT, "scripts", "krea2_variance.py")

# ---------- stubs ----------
# gradio stub that tracks context nesting, so the UI test can assert the
# "RAW only (untested)" sub-accordion groups the latent dials (and that the
# cond-noise sliders sit outside it).
_gr_ctx_stack = []
_gr_accordions = []        # labels of gr.Accordion contexts, in creation order
_gr_slider_accordions = {}  # slider label -> list of enclosing accordion labels

class _GrCtx:
    def __init__(self, kind, label=None):
        self.kind = kind
        self.label = label
    def __enter__(self):
        if self.kind == "accordion":
            _gr_accordions.append(self.label)
        _gr_ctx_stack.append(self)
        return SimpleNamespace()
    def __exit__(self, *a):
        if _gr_ctx_stack:
            _gr_ctx_stack.pop()
        return False

def _gr_slider(*a, **kw):
    label = kw.get("label")
    enclosing = [c.label for c in _gr_ctx_stack if c.kind == "accordion"]
    if label is not None:
        _gr_slider_accordions[label] = enclosing
    return SimpleNamespace(**kw)

gr_mod = types.ModuleType("gradio")
gr_mod.Slider = _gr_slider
gr_mod.Row = lambda *a, **kw: _GrCtx("row")
gr_mod.HTML = lambda *a, **kw: SimpleNamespace(**kw)
gr_mod.Accordion = lambda *a, **kw: _GrCtx("accordion", kw.get("label"))
sys.modules["gradio"] = gr_mod

tqdm_pkg = types.ModuleType("tqdm")
tqdm_auto = types.ModuleType("tqdm.auto")
tqdm_auto.trange = lambda *args, **kw: range(args[0])
tqdm_pkg.auto = tqdm_auto
sys.modules["tqdm"] = tqdm_pkg
sys.modules["tqdm.auto"] = tqdm_auto

class Script:
    pass

class AlwaysVisible:
    pass

class InputAccordion:
    def __init__(self, value=None, setup=False, **kwargs):
        self.value = value
    def __enter__(self):
        return SimpleNamespace(value=self.value)
    def __exit__(self, *a):
        return False

class FakeSeededTorch:
    def __init__(self, seed):
        self.g = torch.Generator().manual_seed(seed)
    def randn_like(self, x):
        return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=self.g)

class FakeKDiffusionSampler:
    def __init__(self, funcname, func):
        self.funcname = funcname
        self.func = func
        self.model_wrap_cfg = object()
    def sample(self, p, x, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
        TESTS["recorded_func"] = self.func
        TESTS["sample_called"] = True
        return self.func(model=TESTS["model"], x=TESTS["x"], sigmas=TESTS["sigmas"], disable=True)
    def sample_img2img(self, p, x, noise, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
        TESTS["recorded_func_i2i"] = self.func
        TESTS["sample_i2i_called"] = True
        return self.func(model=TESTS["model"], x=TESTS["x"], sigmas=TESTS["sigmas"], disable=True)

class BadKDiffusionSampler:
    def sample(self, p, x, conditioning, unconditional_conditioning, steps=None, image_conditioning=None, extra=None):
        pass

modules_pkg = types.ModuleType("modules")
scripts_mod = types.ModuleType("modules.scripts")
scripts_mod.Script = Script
scripts_mod.AlwaysVisible = AlwaysVisible
ui_mod = types.ModuleType("modules.ui_components")
ui_mod.InputAccordion = InputAccordion
kds_mod = types.ModuleType("modules.sd_samplers_kdiffusion")
kds_mod.KDiffusionSampler = FakeKDiffusionSampler
modules_pkg.scripts = scripts_mod
modules_pkg.ui_components = ui_mod
modules_pkg.sd_samplers_kdiffusion = kds_mod
sys.modules["modules"] = modules_pkg
sys.modules["modules.scripts"] = scripts_mod
sys.modules["modules.ui_components"] = ui_mod
sys.modules["modules.sd_samplers_kdiffusion"] = kds_mod

backend_pkg = types.ModuleType("backend")
backend_args = types.ModuleType("backend.args")
backend_args.dynamic_args = SimpleNamespace(krea2=True)
backend_pkg.args = backend_args
sys.modules["backend"] = backend_pkg
sys.modules["backend.args"] = backend_args

calls = []
ks_pkg = types.ModuleType("k_diffusion")
ks_samp = types.ModuleType("k_diffusion.sampling")
def fake_er_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1.0, noise_sampler=None, noise_scaler=None, max_stage=3):
    calls.append(("er_sde", s_noise))
    return "ER_SDE"
ks_samp.sample_er_sde = fake_er_sde
ks_samp.to_d = lambda x, sigma, denoised: (x - denoised) / sigma
ks_samp.torch = FakeSeededTorch(123)
ks_pkg.sampling = ks_samp
sys.modules["k_diffusion"] = ks_pkg
sys.modules["k_diffusion.sampling"] = ks_samp

# ---------- cond-noise fakes (mimic the real KDiffusionSampler flow) ----------
class FakeDenoiser:
    """Stand-in for CFGDenoiserKDiffusion: records the cond it receives per call."""
    def __init__(self):
        self.conds = []
        self.p = None
    def __call__(self, x, t, **kw):
        self.conds.append(kw.get("cond"))
        return x * 0.5

def fake_euler_model(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0):
    """One model call per step, passing extra_args through (like k_diffusion funcs)."""
    extra_args = {} if extra_args is None else extra_args
    for i in range(len(sigmas) - 1):
        denoised = model(x, sigmas[i], **extra_args)
        x = x + (denoised - x)
    return x

class CondFakeSampler:
    """Mirrors KDiffusionSampler.sample: func(self.model_wrap_cfg, x, extra_args=...)."""
    def __init__(self, funcname, func, denoiser):
        self.funcname = funcname
        self.func = func
        self.model_wrap_cfg = denoiser
    def sample(self, p, x, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
        extra = {"cond": conditioning, "uncond": None, "cond_scale": 1.0, "s_min_uncond": 0.0}
        TESTS["recorded_func"] = self.func
        TESTS["sample_called"] = True
        return self.func(self.model_wrap_cfg, TESTS["x"], sigmas=TESTS["sigmas"], extra_args=extra, disable=True)
    def sample_img2img(self, p, x, noise, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
        extra = {"cond": conditioning, "uncond": None, "cond_scale": 1.0, "s_min_uncond": 0.0}
        TESTS["recorded_func_i2i"] = self.func
        TESTS["sample_i2i_called"] = True
        return self.func(self.model_wrap_cfg, TESTS["x"], sigmas=TESTS["sigmas"], extra_args=extra, disable=True)
SchedPC = namedtuple("ScheduledPromptConditioning", ["end_at_step", "cond"])

class ComposablePC:
    def __init__(self, schedules, weight=1.0):
        self.schedules = schedules
        self.weight = weight

class MultiCond:
    def __init__(self, shape, batch):
        self.shape = shape
        self.batch = batch

def make_cond():
    t1 = torch.randn(4, 6, 4)
    t2 = torch.randn(3, 6, 4)
    ids = torch.tensor([1, 2, 3])
    # 2 prompt rows: row0 single segment (t1); row1 weighted segment [t2 until step 4, then t1]
    return MultiCond((2,), [
        [ComposablePC([SchedPC(8, t1)], 1.0)],
        [ComposablePC([SchedPC(4, t2), SchedPC(8, t1)], 0.5)],
    ]), (t1, t2, ids)

TESTS = {}

# ---------- load extension ----------
# Forge prepends the extension root to sys.path while loading scripts/*.py
# (modules/scripts.py) so a script can import the lib_krea2/ package from the
# extension root; mirror that here.
sys.path.insert(0, EXT_ROOT)
import lib_krea2.cond_noise as kv_cn
import lib_krea2.euler_variance as kv_eul

buf = io.StringIO()
with redirect_stdout(buf):
    spec = importlib.util.spec_from_file_location("krea2_variance_test", EXT)
    kv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kv)
load_log = buf.getvalue()
assert "[Krea2Variance] sampler hooks installed" in load_log, f"install log missing: {load_log}"
assert FakeKDiffusionSampler.sample is kv._patched_sample
assert FakeKDiffusionSampler.sample_img2img is kv._patched_sample_img2img
print("1. install + patch: OK")

# guard: wrong signature -> skipped, not patched
kds_mod.KDiffusionSampler = BadKDiffusionSampler
buf2 = io.StringIO()
with redirect_stdout(buf2):
    kv._install()
assert "patch skipped" in buf2.getvalue(), buf2.getvalue()
assert BadKDiffusionSampler.sample.__name__ == "sample"  # untouched
print("2. install guard: OK")

# ---------- dispatch ----------
def mkp(**kw):
    base = dict(kv_active=False, kv_eta=1.0, kv_noise_er=1.0, kv_noise=0.1, kv_start=0, kv_end=49,
                kv_cond_active=False, kv_cond_noise=0.0, kv_cond_steps=2, kv_cond_boost=0.0)
    base.update(kw)
    return SimpleNamespace(**base)

s_off = FakeKDiffusionSampler("sample_euler", lambda **kw: "X")
assert kv._kv_prepare(s_off, mkp(kv_active=False, kv_noise=0.5)) is None
print("3a. not active -> None: OK")

s_anc = FakeKDiffusionSampler("sample_euler_ancestral", lambda **kw: "X")
p_anc = mkp(kv_active=True, kv_eta=0.55, eta=None)
assert kv._kv_prepare(s_anc, p_anc) is None
assert p_anc.eta == 0.55, f"p.eta not set: {p_anc.eta}"
print("3b. Euler a -> p.eta set (no func swap): OK")

s_er = FakeKDiffusionSampler("sample_er_sde", lambda **kw: "X")
p_er = mkp(kv_active=True, kv_noise_er=0.13)
fn = kv._kv_prepare(s_er, p_er)
assert fn is not None
r = fn(model="M", x=None, sigmas=None, s_noise=999, extra_args={"a": 1})
assert r == "ER_SDE"
assert calls[-1] == ("er_sde", 0.13), calls[-1]  # wrapper value wins over injected 999
print("3c. ER SDE wrapper binds s_noise (wins over injected): OK")

s_euler = FakeKDiffusionSampler("sample_euler", lambda **kw: "X")
p_st = mkp(kv_active=True, kv_noise=0.2, kv_start=1, kv_end=3)
fn = kv._kv_prepare(s_euler, p_st)
assert isinstance(fn, functools.partial) and fn.func is kv_eul.sample_euler_variance
assert fn.keywords == {"var_noise": 0.2, "var_start": 1, "var_end": 3}
p_st0 = mkp(kv_active=True, kv_noise=0.0)
assert kv._kv_prepare(s_euler, p_st0) is None  # zero noise -> leave plain Euler
s_other = FakeKDiffusionSampler("sample_dpmpp_2m", lambda **kw: "X")
assert kv._kv_prepare(s_other, mkp(kv_active=True)) is None  # other samplers untouched
print("3d. Euler -> partial(sample_euler_variance); 0.0 and other samplers -> None: OK")

# ---------- patched sample path (latent) ----------
TESTS.clear()
TESTS["sigmas"] = torch.tensor([1.0, 0.7, 0.4, 0.0])
TESTS["x"] = torch.zeros(1, 4, 8, 8)
TESTS["model"] = lambda x, sigma, **kw: x * 0.5 + 1.0
orig_func = lambda **kw: "ORIG_FUNC"

sampler = FakeKDiffusionSampler("sample_euler", orig_func)
out = kv._patched_sample(sampler, p_st, None, None, None)
assert TESTS["sample_called"] and isinstance(out, torch.Tensor) and out.shape == TESTS["x"].shape
assert isinstance(TESTS["recorded_func"], functools.partial) and TESTS["recorded_func"].func is kv_eul.sample_euler_variance
assert sampler.func is orig_func, "self.func not restored after swap"

out2 = kv._patched_sample(sampler, mkp(kv_active=True, kv_noise=0.0), None, None, None)  # active but zero noise
assert out2 == "ORIG_FUNC" and TESTS["recorded_func"] is orig_func

out3 = kv._patched_sample_img2img(sampler, p_st, None, None, None, None)
assert TESTS["sample_i2i_called"] and isinstance(TESTS["recorded_func_i2i"], functools.partial)
assert sampler.func is orig_func

# ER SDE end-to-end through the patched sampler
s_er2 = FakeKDiffusionSampler("sample_er_sde", orig_func)
out4 = kv._patched_sample(s_er2, p_er, None, None, None)
assert out4 == "ER_SDE" and calls[-1] == ("er_sde", 0.13)
assert s_er2.func is orig_func, "ER SDE func not restored"
print("4. patched sample/img2img swap+restore (Euler jitter, ER SDE, inert passthrough): OK")

# ---------- process() latent mapping ----------
def freshp(sampler_name):
    return SimpleNamespace(sampler_name=sampler_name, extra_generation_params={}, eta=None)

p = freshp("Euler a")
kv.Krea2Variance().process(p, True, 0.7, 1.0, 0.1, 0, 49, 0.0, 3, 0.0)
assert p.kv_active and p.kv_eta == 0.7
assert p.extra_generation_params["Krea2 Variance"] == "Euler a"
assert p.extra_generation_params["KV Eta"] == 0.7
print("5a. process() Euler a mapping + infotext: OK")

p = freshp("ER SDE")
kv.Krea2Variance().process(p, True, 1.0, 0.3, 0.1, 0, 49, 0.0, 3, 0.0)
assert p.kv_active and p.kv_noise_er == 0.3
assert p.extra_generation_params["KV Noise ER SDE"] == 0.3 and "KV Start" not in p.extra_generation_params
print("5b. process() ER SDE mapping: OK")

p = freshp("Euler")
kv.Krea2Variance().process(p, True, 1.0, 1.0, 0.2, 1, 3, 0.0, 3, 0.0)
assert p.kv_active and p.kv_noise == 0.2 and p.kv_start == 1 and p.kv_end == 3
assert p.extra_generation_params["KV Noise Euler"] == 0.2 and p.extra_generation_params["KV End"] == 3
print("5c. process() Euler mapping: OK")

backend_args.dynamic_args.krea2 = False
p = freshp("Euler a")
kv.Krea2Variance().process(p, True, 0.5, 0.1, 0.1, 0, 8, 0.0, 3, 0.0)
assert not p.kv_active and not p.kv_cond_active and not p.extra_generation_params  # inert
backend_args.dynamic_args.krea2 = True

p = freshp("Euler a")
kv.Krea2Variance().process(p, False, 0.5, 0.1, 0.1, 0, 8, 1.0, 3, 1.0)
assert not p.kv_active and not p.kv_cond_active

kv._warned.clear()
p = freshp("DPM++ 2M")
buf3 = io.StringIO()
with redirect_stdout(buf3):
    kv.Krea2Variance().process(p, True, 0.5, 0.1, 0.1, 0, 8, 0.0, 3, 0.0)
assert not p.kv_active and not p.extra_generation_params and "only Euler / Euler a / ER SDE" in buf3.getvalue()
print("5d. krea2 gate / disabled / other sampler -> inert: OK")

# ---------- ui() ----------
_gr_accordions.clear(); _gr_slider_accordions.clear()
inst = kv.Krea2Variance()
comps = inst.ui(False)
assert len(comps) == 9, f"ui returned {len(comps)} components"
assert inst.show(False) is AlwaysVisible
assert len(inst.infotext_fields) == 8
labels = [name for (_, name) in inst.infotext_fields]
assert labels == ["KV Eta", "KV Noise ER SDE", "KV Noise Euler", "KV Start", "KV End", "KV Cond Noise", "KV Cond Noise Steps", "KV Step 1 Boost"], labels
# the latent dials are tucked into ONE nested "RAW only (untested)" sub-accordion;
# the cond-noise sliders sit at the top level (outside it)
assert _gr_accordions == ["RAW only (untested)"], f"sub-accordions: {_gr_accordions}"
for _top in ("Cond noise (any sampler)", "Cond noise steps (any sampler)", "Step 1 boost (any sampler)"):
    assert _gr_slider_accordions[_top] == [], f"{_top} must be top-level: {_gr_slider_accordions.get(_top)}"
for _lat in ("Eta (sampler: Euler a)", "Noise (sampler: ER SDE)", "Noise (sampler: Euler)",
             "Start step (sampler: Euler)", "End step (sampler: Euler)"):
    assert _gr_slider_accordions[_lat] == ["RAW only (untested)"], f"{_lat} not in RAW-only sub-menu: {_gr_slider_accordions.get(_lat)}"
print("6. ui() 9 components, cond-noise on top + latent dials in RAW-only sub-accordion: OK")

# ---------- sample_euler_variance math ----------
sigmas = torch.tensor([1.0, 0.7, 0.4, 0.2, 0.0])
x0 = torch.zeros(2, 4, 8, 8)
model = lambda x, sigma, **kw: x * 0.5 + 1.0

def run_sampler(seed, **kw):
    ks_samp.torch = FakeSeededTorch(seed)
    return kv_eul.sample_euler_variance(model, x0.clone(), sigmas, disable=True, **kw)

# var_noise=0 must equal plain Euler
def plain_euler(x, sigmas, model):
    for i in range(len(sigmas) - 1):
        denoised = model(x, sigmas[i])
        x = x + ks_samp.to_d(x, sigmas[i], denoised) * (sigmas[i + 1] - sigmas[i])
    return x

a = run_sampler(42, var_noise=0.0, var_start=0, var_end=49)
b = plain_euler(x0.clone(), sigmas, model)
assert torch.allclose(a, b, atol=1e-6), "var_noise=0 deviates from plain Euler"
print("7a. var_noise=0 == plain Euler: OK")

# reproducibility
c1 = run_sampler(7, var_noise=0.2, var_start=0, var_end=49)
c2 = run_sampler(7, var_noise=0.2, var_start=0, var_end=49)
assert torch.equal(c1, c2), "same seed not reproducible"
d = run_sampler(8, var_noise=0.2, var_start=0, var_end=49)
assert not torch.equal(c1, d), "different seed produced identical output"
print("7b. seeded reproducibility (same seed equal, different seed differs): OK")

# window: early vs late steps change different ranges of the trajectory
e_early = run_sampler(9, var_noise=0.3, var_start=0, var_end=1)
e_late = run_sampler(9, var_noise=0.3, var_start=2, var_end=3)
e_off = run_sampler(9, var_noise=0.0, var_start=0, var_end=49)
assert not torch.equal(e_early, e_late) and not torch.equal(e_early, e_off)
# last-step injection must be skipped (sigma==0): var_end beyond last is clamped
f = run_sampler(10, var_noise=0.3, var_start=0, var_end=99)
assert f.shape == x0.shape
print("7c. step window + clamp: OK")

# churn path still works and is seeded
g1 = run_sampler(11, var_noise=0.0, s_churn=0.1)
g2 = run_sampler(11, var_noise=0.0, s_churn=0.1)
g3 = run_sampler(12, var_noise=0.0, s_churn=0.1)
assert torch.equal(g1, g2) and not torch.equal(g1, g3)
print("7d. s_churn path seeded: OK")

# ---------- _noised_conditioning / _noised_obj ----------
cond, (t1, t2, ids) = make_cond()
noised, stats = kv_cn._noised_conditioning(cond, 42, 1.0)
assert type(noised) is MultiCond and noised is not cond
assert noised.shape == (2,)
assert isinstance(noised.batch, list) and len(noised.batch) == 2
c0 = noised.batch[0][0]
assert type(c0) is ComposablePC and c0 is not cond.batch[0][0]
assert c0.weight == 1.0
s0 = c0.schedules[0]
assert isinstance(s0, SchedPC) and s0.end_at_step == 8
assert s0.cond.shape == t1.shape and s0.cond.dtype == t1.dtype
assert s0.cond is not t1, "cond tensor reused (in-place mutation risk)"
assert not torch.equal(s0.cond, t1), "float tensor not perturbed"
# t1 appears twice (schedule slots of both rows) -> counted twice
assert stats.values == 96 * 2 + 72, f"stats.values={stats.values}"
assert stats.std > 0 and stats.rel_l2 > 0
print("8a. structure preservation (MultiCond/list/namedtuple, shapes, dtypes, no reuse): OK")

# determinism
n1, _ = kv_cn._noised_conditioning(cond, 7, 1.0)
n2, _ = kv_cn._noised_conditioning(cond, 7, 1.0)
n3, _ = kv_cn._noised_conditioning(cond, 8, 1.0)
assert torch.equal(n1.batch[0][0].schedules[0].cond, n2.batch[0][0].schedules[0].cond), "same seed not reproducible"
assert not torch.equal(n1.batch[0][0].schedules[0].cond, n3.batch[0][0].schedules[0].cond), "different seed identical"
print("8b. seeded determinism (same seed equal, different seed differs): OK")

# per-image seeding: each .batch row is noised with its OWN generator
full, st_full = kv_cn._noised_conditioning(cond, [7, 8], 1.0)
full2, _ = kv_cn._noised_conditioning(cond, [7, 8], 1.0)
assert torch.equal(full.batch[0][0].schedules[0].cond, full2.batch[0][0].schedules[0].cond), "per-image row0 not reproducible"
assert torch.equal(full.batch[1][0].schedules[0].cond, full2.batch[1][0].schedules[0].cond), "per-image row1 not reproducible"
# row i with seed s == the same row ALONE noised with s
r0, _ = kv_cn._noised_conditioning(MultiCond((1,), [cond.batch[0]]), [7], 1.0)
r1, _ = kv_cn._noised_conditioning(MultiCond((1,), [cond.batch[1]]), [8], 1.0)
assert torch.equal(full.batch[0][0].schedules[0].cond, r0.batch[0][0].schedules[0].cond), "row0 must be noised with seed 7"
assert torch.equal(full.batch[1][0].schedules[0].cond, r1.batch[0][0].schedules[0].cond), "row1 must be noised with seed 8"
assert torch.equal(full.batch[1][0].schedules[1].cond, r1.batch[0][0].schedules[1].cond), "row1's 2nd schedule must share the row generator"
# a different seed list changes the rows
swp, _ = kv_cn._noised_conditioning(cond, [8, 7], 1.0)
assert not torch.equal(swp.batch[0][0].schedules[0].cond, full.batch[0][0].schedules[0].cond), "seed list must change the rows"
# a scalar seed expands to [s, s+1] (the WebUI per-image seed derivation)
sc, _ = kv_cn._noised_conditioning(cond, 7, 1.0)
assert torch.equal(sc.batch[0][0].schedules[0].cond, full.batch[0][0].schedules[0].cond), "scalar 7 row0 != [7, 8] row0"
assert torch.equal(sc.batch[1][0].schedules[0].cond, full.batch[1][0].schedules[0].cond), "scalar 7 row1 != [7, 8] row1"
# stats accumulate over all images (same total as the single-generator pass)
assert st_full.values == 96 * 2 + 72 and st_full.rel_l2 > 0
# top-level per-image list (length == image count) gets per-element seeding
lp, _ = kv_cn._noised_conditioning([cond.batch[0], cond.batch[1]], [7, 8], 1.0)
assert type(lp) is list and len(lp) == 2
assert torch.equal(lp[0][0].schedules[0].cond, r0.batch[0][0].schedules[0].cond) and torch.equal(lp[1][0].schedules[0].cond, r1.batch[0][0].schedules[0].cond)
# ...but a list whose length does NOT match the image count is noised as one object
lp1, _ = kv_cn._noised_conditioning(cond.batch, [7], 1.0)
whole = kv_cn._noised_obj(cond.batch, torch.Generator().manual_seed(7), 1.0, kv_cn._CondStats())  # returns a single object, not a tuple
def _conds(row):  # every .cond tensor in a row, in traversal order
    return [s.cond for c in row for s in c.schedules]
assert all(torch.equal(a, b) for i in range(2) for a, b in zip(_conds(lp1[i]), _conds(whole[i]))), "length mismatch must fall back to the single-object pass"
print("8c. per-image seeding (own generator per row, scalar expands to s+i, length-mismatch fallback): OK")

# seed plumbing: _expand_seeds / _kv_seeds / _kv_seed
assert kv_cn._expand_seeds([5], 3) == [5, 6, 7]
assert kv_cn._expand_seeds([5, 6, 7], 2) == [5, 6]
assert kv_cn._expand_seeds([5, 6], 3) == [5, 6, 5]
assert kv_cn._expand_seeds([], 2) == [0, 0]
assert kv_cn._kv_seeds(SimpleNamespace(seeds=[3, 4])) == [3, 4]
assert kv_cn._kv_seeds(SimpleNamespace(all_seeds=[9, 8, 7, 6], iteration=1, batch_size=2)) == [7, 6]
assert kv_cn._kv_seeds(SimpleNamespace(seed=11)) == [11]
assert kv_cn._kv_seeds(SimpleNamespace()) == [0]
assert kv_cn._kv_seed(SimpleNamespace(seeds=[3, 4])) == 3
# subseed mode: the WebUI gives every slot the SAME seed (subseed_strength > 0,
# processing.py: all_seeds = [seed] * N); per-slot cond seeds must come from p.subseeds
assert kv_cn._kv_seeds(SimpleNamespace(seeds=[7, 7, 7], subseeds=[100, 101, 102])) == [7, 8, 9]
assert kv_cn._kv_seeds(SimpleNamespace(seeds=[7, 7, 7], subseeds=[55, 58, 55])) == [7, 10, 7]  # arbitrary subseed offsets
assert kv_cn._kv_seeds(SimpleNamespace(seeds=[7, 8, 9], subseeds=[100, 101, 102])) == [7, 8, 9]  # distinct seeds: subseeds ignored
assert kv_cn._kv_seeds(SimpleNamespace(seeds=[7, 7])) == [7, 7]  # equal seeds, no subseeds -> as-is (degenerate)
assert kv_cn._kv_seeds(SimpleNamespace(seeds=[7, 7, 7], subseeds=[100, 101])) == [7, 7, 7]  # length mismatch -> as-is
print("8d. seed plumbing (_expand_seeds / _kv_seeds / _kv_seed, subseed fallback): OK")

# no in-place mutation of the cached embeddings
t1_snap, t2_snap = t1.clone(), t2.clone()
kv_cn._noised_conditioning(cond, 99, 2.0)
assert torch.equal(cond.batch[0][0].schedules[0].cond, t1_snap), "original t1 mutated"
assert torch.equal(cond.batch[1][0].schedules[0].cond, t2_snap), "original t2 mutated"
assert torch.equal(cond.batch[1][0].schedules[1].cond, t1_snap), "shared t1 occurrence mutated"
print("8e. no in-place mutation (cached tensors untouched): OK")

# strength 0 -> values unchanged
n0, stats0 = kv_cn._noised_conditioning(cond, 7, 0.0)
assert torch.equal(n0.batch[0][0].schedules[0].cond, t1)
assert stats0.values == 96 * 2 + 72 and stats0.rel_l2 == 0.0
print("8f. strength 0 -> no-op (values unchanged): OK")

# int tensor pass-through + dict/list structures
st = kv_cn._CondStats()
mixed = kv_cn._noised_obj({"ids": ids, "emb": t1}, torch.Generator().manual_seed(1), 1.0, st)
assert mixed["ids"] is ids, "int tensor was noised"
assert not torch.equal(mixed["emb"], t1)
assert st.values == t1.numel(), f"only the float tensor should be counted, got {st.values}"
dd = {"crossattn": torch.randn(2, 8, 4), "vector": torch.randn(2, 4)}
nd = kv_cn._noised_obj(dd, torch.Generator().manual_seed(5), 1.5, kv_cn._CondStats())
assert isinstance(nd, dict) and set(nd) == set(dd)
assert nd["crossattn"].shape == dd["crossattn"].shape and not torch.equal(nd["crossattn"], dd["crossattn"])
src = torch.randn(3, 4)
lst = kv_cn._noised_obj([src], torch.Generator().manual_seed(5), 1.5, kv_cn._CondStats())
assert type(lst) is list and lst[0].shape == src.shape and not torch.equal(lst[0], src)
print("8g. int pass-through + dict/list structures: OK")

# device robustness: the generator is ALWAYS a CPU generator (see _noised_conditioning).
# The noise must therefore be generated on CPU and MOVED to the tensor's device
# (obj.device, which is cuda on the real GPU run) — NOT generated with
# device=obj.device, which raises
#   "Expected a 'cuda' device type for generator but found 'cpu'"
# when obj is on cuda. And the scale must be a HOST FLOAT: obj.std() is a 0-D
# tensor on obj.device, so multiplying the CPU noise by it raises
#   "Expected all tensors to be on the same device ... cuda:N and cpu"
# in multi-GPU setups. No second device is available offline, so assert the
# invariants on the CPU path and exercise the fixed expression directly.
dev = torch.device("cpu")
src_dev = torch.randn(512, 64, device=dev)
out_dev = kv_cn._noised_obj(src_dev, torch.Generator().manual_seed(3), 1.0, kv_cn._CondStats())
assert out_dev.device == src_dev.device, f"noised output device {out_dev.device} != source {src_dev.device}"
# functional scale check: injected noise std ~= strength * std(source)
inj = out_dev - src_dev
std_src = float(src_dev.std())
assert abs(float(inj.std()) - 1.0 * std_src) / std_src < 0.05, "scale not applied as a host scalar"
# the exact fixed expression: std materialized to float (0-D tensor on the
# source device), CPU randn (no device arg) -> .to(tensor device, dtype)
cpu_gen = torch.Generator().manual_seed(3)  # identical to what _noised_conditioning builds
std_t = src_dev.std()  # 0-D tensor, like obj.std() in the real code
assert isinstance(std_t, torch.Tensor) and std_t.dim() == 0
noise = (torch.randn(src_dev.shape, dtype=torch.float32, generator=cpu_gen) * (float(std_t) * 1.0)).to(dev, src_dev.dtype)
assert noise.device == dev and noise.dtype == src_dev.dtype and torch.isfinite(noise).all()
print("8h. device robustness (CPU generator + host-float scale + .to(tensor device)): OK")

# ---------- _CondNoiseModel gating + delegation ----------
den = FakeDenoiser()
n_base, _ = kv_cn._noised_conditioning(cond, 7, 0.6)
n_step1, _ = kv_cn._noised_conditioning(cond, 7, 1.5)  # boosted draw (same seed)
w = kv_cn._CondNoiseModel(den, n_step1, n_base, cond, 3)
x = torch.zeros(1, 4, 8, 8)
for _ in range(8):
    w(x, torch.tensor([0.5]), cond=cond, uncond=None, cond_scale=1.0, s_min_uncond=0.0)
assert len(den.conds) == 8
assert den.conds[0] is n_step1, "first call must get the boosted cond"
assert all(c is n_base for c in den.conds[1:3]), "window calls must get the baseline cond"
assert all(c is cond for c in den.conds[3:]), "later calls must get the clean cond"
# steps=1 -> only the first call is noised, the rest is clean
den1 = FakeDenoiser()
w1 = kv_cn._CondNoiseModel(den1, n_step1, n_base, cond, 1)
for _ in range(4):
    w1(x, torch.tensor([0.5]), cond=cond, uncond=None, cond_scale=1.0, s_min_uncond=0.0)
assert den1.conds[0] is n_step1 and all(c is cond for c in den1.conds[1:])
# launch_sampling/initialize set public attrs on model_wrap_cfg -> delegate
w.steps = 8
w.total_steps = 8
w.p = "P"
assert den.steps == 8 and den.total_steps == 8 and den.p == "P"
print("9. _CondNoiseModel step-1 boost + baseline window + clean after + delegation: OK")

# ---------- end-to-end cond noise through _patched_sample ----------
# point _orig_sample at CondFakeSampler.sample (mirrors the real
# KDiffusionSampler.sample flow: func(self.model_wrap_cfg, x, extra_args=...))
kds_mod.KDiffusionSampler = CondFakeSampler
buf = io.StringIO()
with redirect_stdout(buf):
    kv._install()
assert CondFakeSampler.sample is kv._patched_sample, "_install did not re-patch for the E2E sampler"

den = FakeDenoiser()
sampler = CondFakeSampler("sample_dpmpp_2m", fake_euler_model, den)
p = freshp("DPM++ 2M")
p.seeds = [123]
kv.Krea2Variance().process(p, True, 1.0, 1.0, 0.1, 0, 49, 0.5, 3, 1.0)
assert p.kv_cond_active and not p.kv_active
cond, _ = make_cond()
TESTS["x"] = torch.zeros(1, 4, 8, 8)
TESTS["sigmas"] = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.0])  # 8 model calls
buf = io.StringIO()
with redirect_stdout(buf):
    out = kv._patched_sample(sampler, p, None, cond, None, None)
log = buf.getvalue()
assert "cond noise: seed=123" in log, f"diagnostics missing: {log}"
assert "L2_base" in log and "L2_step1" in log, f"log must carry both L2 values: {log}"
assert "only Euler / Euler a / ER SDE" not in log, "latent-dial warning should be suppressed when cond noise is on"
assert isinstance(out, torch.Tensor)
assert len(den.conds) == 8
assert den.conds[0] is not cond, "first call must be noised"
assert den.conds[0] is not den.conds[1], "step 1 (boosted) must differ from the baseline"
assert all(c is den.conds[1] for c in den.conds[1:3]), "same baseline object on the window calls"
assert all(c is cond for c in den.conds[3:]), "later calls must be clean"
assert sampler.model_wrap_cfg is den, "model_wrap_cfg not restored"
assert TESTS["recorded_func"] is fake_euler_model, "func must stay untouched (latent path inactive)"
print("10a. E2E cond noise: step-1 boost + baseline window + clean after, restore + diagnostics: OK")

# per-image seeding end-to-end: two seeds -> each row noised with its own seed
den4 = FakeDenoiser()
sampler4 = CondFakeSampler("sample_dpmpp_2m", fake_euler_model, den4)
p = freshp("DPM++ 2M")
p.seeds = [123, 456]
kv.Krea2Variance().process(p, True, 1.0, 1.0, 0.1, 0, 49, 0.5, 3, 1.0)
cond, _ = make_cond()
buf = io.StringIO()
with redirect_stdout(buf):
    kv._patched_sample(sampler4, p, None, cond, None, None)
assert "cond noise: seed=123" in buf.getvalue(), f"log must carry the first image seed: {buf.getvalue()}"
first_row0, _ = kv_cn._noised_conditioning(MultiCond((1,), [cond.batch[0]]), [123], 1.5)  # first call = boosted (0.5 + 1.0)
rest_row1, _ = kv_cn._noised_conditioning(MultiCond((1,), [cond.batch[1]]), [456], 0.5)   # window = base
assert torch.equal(den4.conds[0].batch[0][0].schedules[0].cond, first_row0.batch[0][0].schedules[0].cond), "E2E row0 must use its own seed (boosted pass)"
assert torch.equal(den4.conds[1].batch[1][0].schedules[0].cond, rest_row1.batch[0][0].schedules[0].cond), "E2E row1 must use its own seed (baseline pass)"
assert den4.conds[0] is not cond and den4.conds[1] is not cond and all(c is cond for c in den4.conds[3:])
assert sampler4.model_wrap_cfg is den4
print("10b. E2E per-image seeding: each batch row noised with its own seed: OK")

# additive with the Euler latent dials
den2 = FakeDenoiser()
sampler2 = CondFakeSampler("sample_euler", fake_euler_model, den2)
p = freshp("Euler")
p.seeds = [555]
kv.Krea2Variance().process(p, True, 1.0, 1.0, 0.2, 1, 3, 1.5, 2, 0.8)
assert p.kv_active and p.kv_cond_active
cond, _ = make_cond()
kv._patched_sample(sampler2, p, None, cond, None, None)
assert isinstance(TESTS["recorded_func"], functools.partial) and TESTS["recorded_func"].func is kv_eul.sample_euler_variance
assert den2.conds[0] is not cond and den2.conds[0] is not den2.conds[1]
assert den2.conds[1] is not cond and all(c is cond for c in den2.conds[2:])
assert sampler2.model_wrap_cfg is den2
print("10c. E2E additive: Euler latent jitter + cond noise both active: OK")

# boost-only (baseline 0): first call noised, the rest fully clean
den3 = FakeDenoiser()
sampler3 = CondFakeSampler("sample_dpmpp_2m", fake_euler_model, den3)
p = freshp("DPM++ 2M")
p.seeds = [777]
kv.Krea2Variance().process(p, True, 1.0, 1.0, 0.1, 0, 49, 0.0, 3, 1.0)
assert p.kv_cond_active and p.kv_cond_noise == 0.0 and p.kv_cond_boost == 1.0
cond, _ = make_cond()
buf = io.StringIO()
with redirect_stdout(buf):
    kv._patched_sample(sampler3, p, None, cond, None, None)
assert "L2_base=0.0%" in buf.getvalue(), "boost-only: baseline L2 must log as 0"
assert len(den3.conds) == 8
assert den3.conds[0] is not cond
assert all(c is cond for c in den3.conds[1:]), "boost-only: no baseline -> all later calls clean"
assert sampler3.model_wrap_cfg is den3
print("10d. E2E boost-only: first call noised, window clean: OK")

# subseed mode end-to-end: the WebUI gives every slot the SAME seed when
# subseed_strength > 0 (variation comes from p.subseeds); the per-slot cond
# seeds must be derived from the subseeds, so each row still gets its own
# perturbation instead of one shared draw
dens = FakeDenoiser()
sampler_s = CondFakeSampler("sample_dpmpp_2m", fake_euler_model, dens)
p = freshp("DPM++ 2M")
p.seeds = [900, 900]
p.subseeds = [7000, 7001]
kv.Krea2Variance().process(p, True, 1.0, 1.0, 0.1, 0, 49, 0.5, 3, 1.0)
cond_s, _ = make_cond()
buf = io.StringIO()
with redirect_stdout(buf):
    kv._patched_sample(sampler_s, p, None, cond_s, None, None)
assert "cond noise: seed=900" in buf.getvalue(), f"subseed mode must log the base seed: {buf.getvalue()}"
ref0, _ = kv_cn._noised_conditioning(MultiCond((1,), [cond_s.batch[0]]), [900], 1.5)  # boosted (0.5 + 1.0)
ref1, _ = kv_cn._noised_conditioning(MultiCond((1,), [cond_s.batch[1]]), [901], 0.5)   # derived seed 900 + (7001 - 7000)
old1, _ = kv_cn._noised_conditioning(MultiCond((1,), [cond_s.batch[1]]), [900], 0.5)   # pre-fix behavior: every row got seeds[0]
assert torch.equal(dens.conds[0].batch[0][0].schedules[0].cond, ref0.batch[0][0].schedules[0].cond), "subseed row0 must use seed 900 (boosted)"
assert torch.equal(dens.conds[1].batch[1][0].schedules[0].cond, ref1.batch[0][0].schedules[0].cond), "subseed row1 must use derived seed 901"
assert not torch.equal(dens.conds[1].batch[1][0].schedules[0].cond, old1.batch[0][0].schedules[0].cond), "row1 must NOT share row0's seed in subseed mode"
assert sampler_s.model_wrap_cfg is dens
print("10e. E2E subseed mode: identical p.seeds + p.subseeds -> distinct per-row cond seeds: OK")

# batch invariance of the noised conditioning: image 0 of a 4-row batch, noised
# with its own seed, must be BIT-IDENTICAL to the same row run as a 1-row batch
# with that seed (same seed + same cached base embedding). Any live
# batch1-vs-batch4 first-image difference can therefore only come from the
# model's forward numerics, not from this extension.
cond4, (t4_1, t4_2, ids4) = make_cond()
batch4 = MultiCond((4,), [cond4.batch[0], cond4.batch[1], cond4.batch[0], cond4.batch[1]])
row0_1row, _ = kv_cn._noised_conditioning(MultiCond((1,), [cond4.batch[0]]), [321], 0.5)
full4, st4 = kv_cn._noised_conditioning(batch4, [321, 322, 323, 324], 0.5)
assert torch.equal(full4.batch[0][0].schedules[0].cond, row0_1row.batch[0][0].schedules[0].cond), "row0 of the 4-row batch must be noised exactly like the 1-row run"
assert not torch.equal(full4.batch[2][0].schedules[0].cond, row0_1row.batch[0][0].schedules[0].cond), "row2 (same base tensor, seed 323) must differ from row0 (seed 321)"
assert not torch.equal(full4.batch[1][0].schedules[0].cond, full4.batch[3][0].schedules[0].cond), "rows with the same base tensor but different seeds must differ"
# stats still accumulate over all 4 images (each image's draw counted once per tensor)
assert st4.values == (96 * 2 + 72) * 2, f"4-row stats.values={st4.values}"
print("10f. batch invariance: row0(1-row, seed s) == row0(4-row, seeds s..s+3); rows distinct: OK")

# ---------- process() cond-noise mapping ----------
p = freshp("Euler")
kv.Krea2Variance().process(p, True, 1.0, 1.0, 0.1, 0, 49, 1.5, 4, 0.8)
assert p.kv_active and p.kv_cond_active
assert p.kv_cond_noise == 1.5 and p.kv_cond_steps == 4 and p.kv_cond_boost == 0.8
assert p.extra_generation_params["KV Cond Noise"] == 1.5
assert p.extra_generation_params["KV Cond Noise Steps"] == 4
assert p.extra_generation_params["KV Step 1 Boost"] == 0.8

p = freshp("Euler")
kv.Krea2Variance().process(p, True, 1.0, 1.0, 0.1, 0, 49, 0.0, 3, 0.0)
assert not p.kv_cond_active and "KV Cond Noise" not in p.extra_generation_params and "KV Step 1 Boost" not in p.extra_generation_params

kv._warned.clear()
p = freshp("DPM++ 2M")
buf = io.StringIO()
with redirect_stdout(buf):
    kv.Krea2Variance().process(p, True, 1.0, 1.0, 0.1, 0, 49, 0.0, 3, 1.0)
assert p.kv_cond_active and p.kv_cond_boost == 1.0 and "only Euler / Euler a / ER SDE" not in buf.getvalue()
print("11. process() cond mapping + infotext + sampler-independent gating: OK")

print("\nALL TESTS PASSED")
