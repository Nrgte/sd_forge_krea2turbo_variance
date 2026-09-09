# Krea2 Variance

Krea2 checkpoints are CFG-free few-step flow models. At CFG 1.0 the positive
text conditioning dominates the sampling trajectory, so the two useful
variety levers are:

1. **Cond noise (any sampler)** — the main face-variety lever. Seeded gaussian
   noise on a copy of the positive Qwen3-VL conditioning for the first N
   steps (where identity/composition forms), then the clean conditioning so
   later steps still refine. No input image needed, works with every sampler.
2. **Latent dials** — the sampler selected in the tab's Sampler dropdown adds
   noise to the *latent* instead (often washed out at CFG 1.0, weaker for
   faces, but still useful and additive to cond noise).

The extension lives in a "Krea2 Variance" accordion in the main txt2img and
img2img layout. Tick the checkbox to enable, untick to disable. **Cond noise
is shown at the top** (the turbo-relevant lever). The **latent dials are
tucked into a "RAW only (untested)" sub-accordion** — they only act for the
three samplers below and are usually washed out at CFG 1.0:

| Tab sampler | Latent dial (in the RAW-only sub-menu) |
|---|---|
| **Euler a** | **Eta** — per-step ancestral re-noise for this generation (otherwise only the global `eta_ancestral` setting exists; there is no per-tab eta control) |
| **ER SDE** | **Noise (ER SDE)** — the solver's `s_noise`, which is otherwise fixed at 1.0 and not configurable from the UI at all |
| **Euler** | **Noise (Euler)** + **Start/End step** — keeps the deterministic Euler walk and adds seeded per-step jitter on the steps you choose (the one thing no built-in sampler offers) |
| anything else | no effect (latent); cond noise still applies |

All injected noise is **seeded from your seed** (latent noise via the WebUI's
seeded RNG, the same mechanism the built-in ancestral samplers use; cond
noise via a separate generator), so the two streams are independent of each
other and of the latent RNG. Same seed → pixel-identical output, different
seeds → genuinely different images.

The settings are written to the infotext (`KV Eta` / `KV Noise ER SDE` /
`KV Noise Euler` / `KV Start` / `KV End` / `KV Cond Noise` /
`KV Cond Noise Steps`), and pasting infotext restores the sliders.

The extension only acts while a Krea2 model is loaded; otherwise the normal
sampler runs untouched.

## Cond noise (recommended first)

- **Defaults: Cond noise 1.2, steps 1** (Krea2 turbo runs 8 steps; only the
  first step is perturbed, where the face/identity forms, so the remaining 7
  steps still refine on the clean conditioning). Raise steps to 2–3 for a
  broader window, or lower the strength if the image degrades.
- The console logs the measured perturbation each run:
  `[Krea2Variance] cond noise: seed=… strength=… steps=… embed_std=… L2=…% values=…`.
  **Below ~1% L2 it will not register** — raise the strength. The default
  1.2 gives a strong, face-changing perturbation; 2–3 is stronger still.
- If the image degrades (smearing, unstable identity): lower the strength or
  shorten the steps.
- With **batch > 1** the whole batch shares one perturbation (same face
  within the batch) — use batch 1 and vary the seed for variety.

## Latent dials (secondary, additive — in the RAW-only sub-menu)

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

## If the face still doesn't change

1. Check the logged L2 — if it's < ~1%, raise Cond noise.
2. Make sure the prompt doesn't over-constrain identity (e.g. a named
   character or very specific descriptor can re-converge the face).
3. The remaining levers are the **Krea 2 Raw** model, identity-descriptive
   prompt terms, or a face-diversity LoRA.
