# Krea2 Variance

Krea2 checkpoints are CFG-free few-step flow models. At CFG 1.0 the positive
text conditioning dominates the sampling trajectory, so the two useful
variety levers are:

1. **Cond noise (any sampler)** — the main face-variety lever. Seeded gaussian
   noise on a copy of the positive Qwen3-VL conditioning for the first N
   steps (where identity/composition forms), then the clean conditioning so
   later steps still refine. The **Cond noise** dial is the baseline strength
   for those steps; the **Step 1 boost** adds extra strength on top of the
   baseline for the first (identity-forming) step only. No input image
   needed, works with every sampler.
2. **Latent dials (RAW only, untested)** — the sampler selected in the tab's
   Sampler dropdown adds noise to the *latent* instead (often washed out at
   CFG 1.0, weaker for faces, but still useful and additive to cond noise).
   Documented in the last section, since cond noise is the main lever.

The extension lives in a "Krea2 Variance" accordion in the main txt2img and
img2img layout. Tick the checkbox to enable, untick to disable. **Cond noise
is shown at the top** (the turbo-relevant lever). The **latent dials are
tucked into a "RAW only (untested)" sub-accordion** at the bottom of the
accordion — they only act for the three samplers listed in the last section
and are usually washed out at CFG 1.0.

All injected noise is **seeded from each image's own seed** (latent noise via
the WebUI's seeded RNG, the same mechanism the built-in ancestral samplers
use; cond noise via a separate generator seeded per image the same way the
WebUI derives per-image seeds), so the two streams are independent of each
other and of the latent RNG. Same seed → pixel-identical output, different
seeds → genuinely different images.

The settings are written to the infotext (`KV Eta` / `KV Noise ER SDE` /
`KV Noise Euler` / `KV Start` / `KV End` / `KV Cond Noise` /
`KV Cond Noise Steps` / `KV Step 1 Boost`), and pasting infotext restores the
sliders.

The extension only acts while a Krea2 model is loaded; otherwise the normal
sampler runs untouched.

## Cond noise (recommended first)

Three dials: **Cond noise** (baseline strength), **Cond noise steps** (how
many initial steps the baseline runs on), **Step 1 boost** (extra strength
added on top of the baseline for the first step only).

- **Defaults: Cond noise 1.2, steps 1, Step 1 boost 0.0** (Krea2 turbo runs
  8 steps): the enabled default run is the baseline on the first
  (identity-forming) step only — 1.2 on step 1, steps 2–8 clean. Raise
  **steps** to 2–3 for a broader window, but keep the baseline mild then —
  a full-strength second step can mangle the image. The boost is purely
  additive and off by default: add it when the face isn't varied enough
  (the identity forms on step 1, so it's the most effective dial).
- If the face barely moves: raise **Step 1 boost** first (the first step is
  where the identity forms). If it still isn't enough, raise **Cond noise**
  and/or **steps**.
- If the image degrades (smearing, unstable identity): lower **Cond noise**
  (the baseline) first, then the boost, then shorten the steps.
- Both noised passes for a given image use that image's seed, hence the same
  random draw — only the scale differs, so the perturbation direction is
  consistent across the window.
- The console logs one line per generation:
  `[Krea2Variance] cond noise: seed=… base=… boost=… steps=… step1=… embed_std=… values=… L2_base=…% L2_step1=…%`
   In plain terms:
   - `seed` — the first image's seed; with batch > 1 each image is noised with its own per-image seed (the same per-image seeds the WebUI uses for the latent noise), so one generation already spans different faces.
   - `base` / `boost` / `steps` / `step1` — just echo your settings; step 1 runs at base + boost.
   - `embed_std` — how strong the text signal itself is (reference only; the noise is sized relative to it).
   - `values` — how many numbers were noised; a sanity check that the text embedding was found (0 = nothing to perturb).
   - `L2_base` / `L2_step1` — the noise amount compared to the text signal: 50% means the noise is half as strong as the text itself. **Below ~1% it is too small to make a difference** — raise the strength.
- With **batch > 1** each image gets its own perturbation (each batch row is
  noised with its own per-image seed), so one generation already spans
  different faces — no need to run batch 1 with varying seeds.
- If **Variation strength** (subseed) is > 0, the WebUI gives every image the
  same seed (variation comes from the *Variation seed*); the per-image cond
  seeds are then derived from the variation seeds, so each image still gets
  its own distinct perturbation.
- The first image of a batch-N run can differ from a batch-1 run with the
  same seed: its seed, latent noise, and text perturbation are all
  bit-identical to the batch-1 run (verified: the per-image seeds and the
  per-slot latent RNG depend only on the seed + shape, and the noised
  embedding is a pure function of the seed), so the difference comes from
  the Krea2 model's forward pass running at a different batch size
  (floating-point arithmetic in the batched call), not from this extension.
  You can confirm it by running the same test with the accordion disabled —
  the first image still differs. If you need image 1 to match a solo run,
  generate with **Batch size 1 + Batch count N** instead (each image is
  then a solo-batch generation, still seeded s, s+1, …).

## If the face still doesn't change

1. Check the logged L2 — if it's < ~1%, raise Cond noise.
2. Make sure the prompt doesn't over-constrain identity (e.g. a named
   character or very specific descriptor can re-converge the face).
3. The remaining levers are the **Krea 2 Raw** model, identity-descriptive
   prompt terms, or a face-diversity LoRA.

## Latent dials (RAW only, untested — secondary, additive)

The tab sampler chooses which dial applies; all of them live in the
"RAW only (untested)" sub-accordion:

| Tab sampler | Latent dial |
|---|---|
| **Euler a** | **Eta** — per-step ancestral re-noise for this generation (otherwise only the global `eta_ancestral` setting exists; there is no per-tab eta control) |
| **ER SDE** | **Noise (ER SDE)** — the solver's `s_noise`, which is otherwise fixed at 1.0 and not configurable from the UI at all |
| **Euler** | **Noise (Euler)** + **Start/End step** — keeps the deterministic Euler walk and adds seeded per-step jitter on the steps you choose (the one thing no built-in sampler offers) |
| anything else | no effect (latent); cond noise still applies |

The starting points below are ordered roughly strongest → weakest face
change:

- **ER SDE + Noise 1.0** — the strongest per-step stochastic sampler. If it
  looks grainy, drop Noise to 0.7–0.8.
- **Euler + Noise 1.0, Start 0, End 3** — same strength but only on the early
  (identity-forming) steps, leaving detail steps (4–7) deterministic.
- **Euler a + Eta 1.0** — the standard ancestral sampler; weaker than the two
  above. (The accordion's Eta is the per-run version of the global
  `eta_ancestral` setting.)

Cond noise can be combined with any of these (they perturb different things).
