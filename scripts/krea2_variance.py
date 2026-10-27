"""Krea2 Variance — per-run noise dials for Krea2: latent sampler noise +
seeded positive-conditioning noise.

Krea2 checkpoints are CFG-free few-step flow models. Two noise levers, both
seeded (same seed -> pixel-identical output, different seeds -> genuinely
different images) and both without any input image:

1. Latent dials — the sampler selected in the tab's Sampler dropdown chooses
   the behavior:

   * "Euler a"  -> the Eta slider sets p.eta for this generation
                   (overrides the global eta_ancestral setting)
   * "ER SDE"   -> the Noise slider binds s_noise for this generation
   * "Euler"    -> the Noise + Start/End step sliders add seeded per-step
                   re-noise on the chosen steps
   * other samplers -> the extension does nothing (latent)

   At CFG 1.0 the positive text conditioning dominates the trajectory and
   latent noise is often washed out — the face barely changes across seeds.

2. Cond noise (any sampler) — seeded gaussian noise on a COPY of the positive
    Qwen3-VL conditioning for the first N steps (Cond noise steps), where
    identity/composition form; later steps run on the clean conditioning so
    the image still refines. The **Cond noise** dial is the baseline strength
    for those steps; the **Step 1 boost** adds extra strength on top of the
    baseline for the first (identity-forming) step only. This is the main
    face-variety lever for Krea2 turbo. The copy is deep-copied recursively
    (list/dict/tuple/namedtuple/object), never mutated in place: the Qwen3-VL
    engine caches embeddings per prompt line and the same tensor objects are
    shared across schedule entries and generations. Baseline and boost use
    the same seed, hence the same random draw — only the scale differs.

All latent noise is drawn through k_diffusion.sampling.torch, which
Sampler.initialize() reassigns to TorchHijack(p) — i.e. from p.rng, the
per-image seeded generators (modules/rng.py). Cond noise uses its own local
torch.Generator seeded from the base seed, so the two noise streams are
independent of each other and of the latent RNG.

UI layout: the accordion shows **Cond noise** (the turbo-relevant lever) at the
top; the latent dials (Euler a / ER SDE / Euler) are tucked into a nested
"RAW only (untested)" sub-accordion. Both stay wired to the same process() args.

No core file is modified: a single guarded class-method monkeypatch on
KDiffusionSampler (modules/sd_samplers_kdiffusion.py) is installed at import
time with a structural guard and a logged fallback. process() only configures
p — the normal pipeline runs regardless.
"""

import functools
import inspect

import gradio as gr

from backend.args import dynamic_args
from modules import scripts
from modules.ui_components import InputAccordion

# The noise machinery lives in the extension-root lib_krea2/ package (the
# extension dir is on sys.path while this script loads — same pattern as
# sd_forge_spectrum importing lib_spectrum).
from lib_krea2.cond_noise import _CondNoiseModel, _kv_seeds, _noised_conditioning
from lib_krea2.euler_variance import sample_euler_variance

SAMPLER_ANCESTRAL = "Euler a"   # funcname sample_euler_ancestral
SAMPLER_ER_SDE = "ER SDE"       # funcname sample_er_sde
SAMPLER_EULER = "Euler"         # funcname sample_euler

_warned = {}

info = """
<b>Cond noise</b> &mdash; per-seed face variety for Krea2 turbo, no input
image. All noise is seeded: same seed &rarr; pixel-identical image,
different seeds &rarr; genuinely different images; with batch &gt; 1 each
image gets its own seed automatically. What each dial does: the tooltips
next to its label.
<ul>
<li>If the face barely moves: raise <b>Step 1 boost</b> first, then <b>Cond noise</b> / steps.</li>
<li>If it degrades the image: lower <b>Cond noise</b> first, then the boost.</li>
</ul>
The console logs one line per generation (field meanings in the README).
The <b>RAW only (untested)</b> sub-menu below holds the latent-noise dials,
which are usually washed out &mdash; Cond noise is the reliable lever.
"""

info_raw = """
Latent-noise dials. The sampler in the tab's Sampler dropdown selects the
behavior. At CFG 1.0 the positive text conditioning dominates the trajectory, so
these are often washed out &mdash; <b>Cond noise</b> is the reliable lever.
Unverified on the RAW (non-turbo) checkpoints, hence the label.
<ul>
<li><b>Euler a</b> &rarr; <b>Eta</b>: per-step ancestral re-noise (1.0 = full, 0 = plain Euler). Overrides the global <i>eta_ancestral</i> setting.</li>
<li><b>ER SDE</b> &rarr; <b>Noise (ER SDE)</b>: the solver's <i>s_noise</i>; 1.0 = exactly the tab sampler's behavior, lower = milder.</li>
<li><b>Euler</b> &rarr; <b>Noise (Euler)</b> + <b>Start/End step</b>: keeps the deterministic walk, adds seeded jitter on the chosen steps &mdash; early steps (0-3) change face/pose, late steps change detail. <b>Face diversity needs 0.3-1.0</b>; at 1.0 it injects as much as Euler a.</li>
</ul>
"""


def _warn(key, msg):
    if key in _warned:
        return
    _warned[key] = True
    print(f"[Krea2Variance] {msg}", flush=True)


def _kv_cond_prepare(self, p, conditioning):
    """Build the call-aware conditioning wrapper for this generation, or None.

    Sampler-independent: at CFG 1.0 the positive conditioning dominates, so
    the perturbation works the same for every k-diffusion sampler. Each
    image's conditioning row is noised with THAT image's own seed (from
    p.seeds), so a batch of N gets N independent perturbations instead of
    one shared draw. The first model call (identity-forming step) gets
    strength base + boost, calls 1..steps-1 get the base strength, calls
    from `steps` onward get the clean conditioning. Both noised passes are
    seeded with the same per-image seeds, so each image uses the same
    random draw (only the scale differs); a zero base leaves the clean
    conditioning in place for the later window calls.
    """
    if not getattr(p, "kv_cond_active", False) or conditioning is None:
        return None

    seeds = _kv_seeds(p)
    seed = seeds[0]
    base = float(p.kv_cond_noise)
    boost = float(p.kv_cond_boost)
    steps = max(1, int(p.kv_cond_steps))
    if base > 0:
        rest, rest_stats = _noised_conditioning(conditioning, seeds, base)
    else:
        rest, rest_stats = conditioning, None
    if boost > 0:
        first, first_stats = _noised_conditioning(conditioning, seeds, base + boost)
    else:
        first, first_stats = rest, rest_stats

    stats = first_stats if first_stats is not None else rest_stats
    if stats.values == 0:
        _warn("cond_none", "cond noise enabled but no float tensors found in the positive conditioning — nothing to perturb (check that the active model uses an embedding the recursive noiser can reach).")
        return None

    base_l2 = rest_stats.rel_l2 if rest_stats is not None else 0.0
    print(f"[Krea2Variance] cond noise: seed={seed} base={base} boost={boost} steps={steps} step1={base + boost} embed_std={stats.std:.4g} values={stats.values} L2_base={base_l2 * 100:.1f}% L2_step1={stats.rel_l2 * 100:.1f}%", flush=True)
    if stats.rel_l2 < 0.01:
        _warn("cond_weak", f"cond noise L2 perturbation {stats.rel_l2 * 100:.2f}% < 1% — likely too weak to register; raise Cond noise / Step 1 boost.")

    return _CondNoiseModel(self.model_wrap_cfg, first, rest, conditioning, steps)


def _kv_prepare(self, p):
    """Per-generation controls for the currently selected sampler.

    Returns a replacement func (bound to this generation's p), or None to
    leave self.func as-is. May also set p.eta for Euler a.
    """
    if not getattr(p, "kv_active", False):
        return None

    if self.funcname == "sample_euler_ancestral":
        # initialize() (sd_samplers_common.py:460) prefers p.eta over opts.eta_ancestral
        p.eta = p.kv_eta
        return None

    if self.funcname == "sample_er_sde":
        import k_diffusion.sampling as ks

        def _er_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
            # initialize() never passes s_noise for ER SDE (not in
            # sampler_extra_params); the explicit value always wins
            kwargs["s_noise"] = p.kv_noise_er
            return ks.sample_er_sde(model, x, sigmas, extra_args=extra_args, callback=callback,
                                    disable=disable, **kwargs)
        return _er_sde

    if self.funcname == "sample_euler" and p.kv_noise > 0:
        return functools.partial(sample_euler_variance,
                                 var_noise=p.kv_noise,
                                 var_start=p.kv_start,
                                 var_end=p.kv_end)

    return None


_orig_sample = None
_orig_sample_i2i = None


def _patched_sample(self, p, x, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
    fn = _kv_prepare(self, p)
    model = _kv_cond_prepare(self, p, conditioning)
    if fn is None and model is None:
        return _orig_sample(self, p, x, conditioning, unconditional_conditioning, steps, image_conditioning)
    orig_func = self.func
    orig_model = self.model_wrap_cfg
    try:
        if fn is not None:
            self.func = fn
        if model is not None:
            self.model_wrap_cfg = model
        return _orig_sample(self, p, x, conditioning, unconditional_conditioning, steps, image_conditioning)
    finally:
        self.func = orig_func
        self.model_wrap_cfg = orig_model


def _patched_sample_img2img(self, p, x, noise, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
    fn = _kv_prepare(self, p)
    model = _kv_cond_prepare(self, p, conditioning)
    if fn is None and model is None:
        return _orig_sample_i2i(self, p, x, noise, conditioning, unconditional_conditioning, steps, image_conditioning)
    orig_func = self.func
    orig_model = self.model_wrap_cfg
    try:
        if fn is not None:
            self.func = fn
        if model is not None:
            self.model_wrap_cfg = model
        return _orig_sample_i2i(self, p, x, noise, conditioning, unconditional_conditioning, steps, image_conditioning)
    finally:
        self.func = orig_func
        self.model_wrap_cfg = orig_model


def _install():
    global _orig_sample, _orig_sample_i2i

    from modules import sd_samplers_kdiffusion

    kds = sd_samplers_kdiffusion.KDiffusionSampler
    params = list(inspect.signature(kds.sample).parameters)
    if params != ["self", "p", "x", "conditioning", "unconditional_conditioning", "steps", "image_conditioning"]:
        print(f"[Krea2Variance] patch skipped: KDiffusionSampler.sample signature changed: {params}")
        return

    _orig_sample = kds.sample
    _orig_sample_i2i = kds.sample_img2img
    kds.sample = _patched_sample
    kds.sample_img2img = _patched_sample_img2img
    print("[Krea2Variance] sampler hooks installed (active only when the accordion is enabled)")


class Krea2Variance(scripts.Script):
    def title(self):
        return "Krea2 Variance"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(value=False, label=self.title()) as enabled:
            gr.HTML(info)
            with gr.Row():
                cond_noise = gr.Slider(
                    minimum=0.0,
                    maximum=3.0,
                    value=1.2,
                    step=0.05,
                    label="Cond noise (any sampler)",
                    info="how much seeded noise to add to the text prompt — this is what changes the face. 0 = off. Works with every sampler.",
                )
                cond_steps = gr.Slider(
                    minimum=1,
                    maximum=8,
                    value=1,
                    step=1,
                    label="Cond noise steps (any sampler)",
                    info="how many of the first steps the noise runs on. More steps = stronger effect, but more risk of degrading the image. Krea2 turbo runs 8 steps.",
                )
                boost = gr.Slider(
                    minimum=0.0,
                    maximum=3.0,
                    value=0.0,
                    step=0.05,
                    label="Step 1 boost (any sampler)",
                    info="extra noise for the very first step only, where the face forms — the most effective dial. 0 = off.",
                )
            with gr.Accordion(label="RAW only (untested)", open=False):
                gr.HTML(info_raw)
                with gr.Row():
                    eta = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=1.0,
                        step=0.05,
                        label="Eta (sampler: Euler a)",
                        info="per-step ancestral re-noise strength for this generation; 1.0 = full, 0.5 = half, 0 = plain Euler",
                    )
                    noise_er = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=1.0,
                        step=0.05,
                        label="Noise (sampler: ER SDE)",
                        info="the ER SDE solver's s_noise for this generation; 1.0 = the tab sampler's default behavior",
                    )
                with gr.Row():
                    noise_euler = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.3,
                        step=0.01,
                        label="Noise (sampler: Euler)",
                        info="extra per-step seeded jitter with plain Euler. Face/identity changes need 0.3-1.0; below ~0.2 it's usually too weak to beat the tab's Euler a. Use Start 0-3 to hit the identity-forming steps.",
                    )
                    start = gr.Slider(
                        minimum=0,
                        maximum=49,
                        value=0,
                        step=1,
                        label="Start step (sampler: Euler)",
                        info="first step (0-based) that receives the extra noise",
                    )
                    end = gr.Slider(
                        minimum=0,
                        maximum=49,
                        value=49,
                        step=1,
                        label="End step (sampler: Euler)",
                        info="last step (inclusive) that receives the extra noise; 49 = all steps",
                    )

        self.infotext_fields = [
            (eta, "KV Eta"),
            (noise_er, "KV Noise ER SDE"),
            (noise_euler, "KV Noise Euler"),
            (start, "KV Start"),
            (end, "KV End"),
            (cond_noise, "KV Cond Noise"),
            (cond_steps, "KV Cond Noise Steps"),
            (boost, "KV Step 1 Boost"),
        ]
        return [enabled, eta, noise_er, noise_euler, start, end, cond_noise, cond_steps, boost]

    def process(self, p, enabled, eta, noise_er, noise_euler, start, end, cond_noise, cond_steps, boost):
        # always-visible: the accordion checkbox is the on/off switch; only
        # configure p here, the normal pipeline runs regardless
        p.kv_active = False
        p.kv_eta = 1.0
        p.kv_noise_er = 1.0
        p.kv_noise = 0.3
        p.kv_start = 0
        p.kv_end = 49
        p.kv_cond_active = False
        p.kv_cond_noise = 1.2
        p.kv_cond_steps = 1
        p.kv_cond_boost = 0.0

        if not (enabled and getattr(dynamic_args, "krea2", False)):
            return

        # cond noise: sampler-independent, additive to the latent dials
        cond_on = float(cond_noise) > 0 or float(boost) > 0
        if cond_on:
            p.kv_cond_active = True
            p.kv_cond_noise = float(cond_noise)
            p.kv_cond_steps = max(1, int(cond_steps))
            p.kv_cond_boost = float(boost)
            p.extra_generation_params["KV Cond Noise"] = p.kv_cond_noise
            p.extra_generation_params["KV Cond Noise Steps"] = p.kv_cond_steps
            p.extra_generation_params["KV Step 1 Boost"] = p.kv_cond_boost

        sampler = p.sampler_name or ""
        if sampler == SAMPLER_ANCESTRAL:
            p.kv_active = True
            p.kv_eta = float(eta)
            p.extra_generation_params["Krea2 Variance"] = SAMPLER_ANCESTRAL
            p.extra_generation_params["KV Eta"] = p.kv_eta
        elif sampler == SAMPLER_ER_SDE:
            p.kv_active = True
            p.kv_noise_er = float(noise_er)
            p.extra_generation_params["Krea2 Variance"] = SAMPLER_ER_SDE
            p.extra_generation_params["KV Noise ER SDE"] = p.kv_noise_er
        elif sampler == SAMPLER_EULER:
            p.kv_active = True
            p.kv_noise = float(noise_euler)
            p.kv_start = int(start)
            p.kv_end = int(end)
            p.extra_generation_params["Krea2 Variance"] = SAMPLER_EULER
            p.extra_generation_params["KV Noise Euler"] = p.kv_noise
            p.extra_generation_params["KV Start"] = p.kv_start
            p.extra_generation_params["KV End"] = p.kv_end
        elif not cond_on:
            _warn("sampler", f"enabled with tab sampler '{sampler}' — only Euler / Euler a / ER SDE get the latent-noise dials; running the normal sampler (Cond noise still applies when > 0).")


_install()
