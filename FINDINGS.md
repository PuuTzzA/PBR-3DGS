# GT-Prior Inverse Rendering on GIR — Findings through try_7 (2026-07-04)

This document summarizes what we have established so far about resolving the
albedo–lighting ambiguity in Gaussian Inverse Rendering (GIR) with ground-truth
priors, what the current best configuration is, and exactly how it differs from
the GIR baseline. Batches before `new_experiments_try_5_fixed_envlight` used a
broken envmap sampler and are not comparable; `try_6` established the reference
protocol (`--hdr_rotation`, random background); `try_7` introduced and stress-
tested light-linear indirect illumination (LLI).

---

## 1. Problem and setup

GIR decomposes multi-view images into gaussians carrying PBR materials
(albedo, metallic, roughness), geometry-derived normals, and one learned
environment map. Trained only photometrically, the decomposition is ambiguous:
the envmap and the albedo can trade off brightness and color freely (the
product is all the image constrains). We break the ambiguity with per-view
priors — currently **ground-truth** albedo/normal renders of the synthetic
lego scene (resolution 4, sunset HDRI base light, 37.5k iterations), later to
be replaced by **diffusion-model priors** (e.g. NVIDIA DiffusionRenderer).
Diffusion priors are multi-view **inconsistent**, which is why the albedo
prior uses correlation-style invariant losses (`zncc`, `zncc_grad`) rather
than a plain L1: a per-view inconsistent gain/shading must not be forced into
the shared 3D albedo.

The headline target metric is **relighting**: render the trained model under
six unseen HDRIs (fireplace, night, snow, city, courtyard, forest) for which
Blender-path-traced GT exists, and compare PSNR/SSIM — plus two calibration
metrics introduced in try_7/8 (see §4).

## 2. The current best configuration vs. baseline GIR

Best run so far: **`gt_zncc_grad_anchor_lli`** (try_7) — aligned relight
27.97 dB vs 26.95 for the best previous run and 25.40 for the no-prior
baseline. Its zncc twin `gt_zncc_anchor_lli` is nearly level on relight and
far better on albedo (31.2 dB vs 27.3). Everything below is a delta against
the reference GIR training (`t6_baseline_no_prior`, which itself already uses
`--random_background`, `--hdr_rotation`, reg_hdr 0.001-ish, densify-until-30k).

### 2.1 Three-phase schedule
`first_stage_step 5000`, `second_stage_step 25000`, 37.5k total:

* **Phase 1 (≤5k), geometry:** GIR's radiance warm-up (plain view-dependent
  color, densification active). With `--albedo_geometry_warmup` the rendered
  flat albedo is *additionally* supervised by the albedo prior in `direct`
  (plain Huber) mode already here, so densification places gaussians with
  sensible base colors.
* **Phase 2 (5k–25k), normal alignment:** the geometry-derived shading normal
  is made differentiable every iteration and supervised by the GT normal
  prior (`lambda_normal_gt 0.8`, masked 1−cosine in world space) — surface
  orientation locks in before any material decomposition.
* **Phase 3 (>25k), full PBR:** GIR's decomposition plus all priors under the
  weight scheduler.

Phases 1–2 depend on few parameters, so their stage-2 checkpoint is cached in
`outputs/warmup_buffer` (fingerprint-keyed) and every Phase-3 variant resumes
from it at ~1/3 of full run cost.

### 2.2 How the priors are added (Phase 3)
All prior losses are computed on the **rasterized** property maps (rendered
albedo/normal/metallic/roughness vs the per-view GT image), restricted to the
object silhouette via the GT alpha mask (critical for diffusion priors, whose
background pixels are hallucinated):

* **Albedo** — `lambda_albedo_gt 0.25`, mode per experiment:
  * `zncc`: per-channel standardize (subtract foreground mean, divide by
    std) both rendered and GT albedo, then Huber-compare. Invariant to a
    per-channel affine transform (gain+bias) — exactly the degrees of freedom
    a diffusion prior gets wrong globally.
  * `zncc_grad`: spatial gradients first (kills low-frequency baked shading),
    then per-channel standardized correlation of the gradient images. Even
    more invariant (any smooth shading field is ignored), which is why it was
    the most robust mode against inconsistent diffusion priors in try_5.
* **Albedo anchor** — `albedo_anchor_weight 0.05`: a *weak absolute* Huber
  term `0.05 · huber(rendered_albedo, gt_albedo)` added on top of the
  invariant mode. The invariant losses deliberately don't constrain global
  scale, so the optimizer used that freedom: in try_6 the unanchored albedo
  drifted ~1.2× too bright / blue-tinted with the envmap compensating. The
  anchor pins the absolute scale (albedo gain → 1.0) while being ~5× weaker
  than the structural term, so it cannot force per-view shading disagreements
  into the albedo. Verified effect (try_7): raw albedo PSNR 22.0 → 27.3
  (zncc_grad) / 31.2 (zncc). Note zncc_grad retains a ~3 dB raw-vs-aligned
  albedo gap (its gradient-domain loss leaves per-channel offsets that a 0.05
  anchor only partly fixes); plain zncc + anchor closes it completely.
* **Normal** — `lambda_normal_gt 0.8`, masked mean of (1 − cos) between the
  rendered normal and the world-space GT normal; backprops into gaussian
  rotation/scaling.
* **Metallic** — `lambda_metallic_gt 0.15` Huber against an all-zero GT
  (`metallic_simulated_zero`; lego is dielectric).
* **Scheduler** — `--use_prior_weight_scheduler`, ratio 0.15, final 1.0:
  priors ramp up over the first 15% of Phase 3 and then **hold** at full
  weight (the old decay-to-0.5 gave the late phase freedom to re-bake light).
* **Huber delta 0.2** for all prior terms.

### 2.3 Other deltas vs the reference launch config
* `tv_reduction_factor 0.75`: TV/smoothness regularizers are reduced to 75%
  for properties that have a GT prior (the prior already constrains them;
  artificial smoothness fights it).
* `reduce_geo_lr_third_stage 0.05` (cosine to `geo_lr_final_iter 40k`):
  geometry learning rates (xyz/scaling/rotation) are annealed to 5% during
  Phase 3 so the settled, normal-aligned geometry can't re-bake lighting.
  (Likely also part of why prior runs lose ~2 dB novel-view PSNR vs baseline
  — see open problems.)
* `reg_hdr_weight 0.001`: the envmap "neutrality" regularizer is a pure
  desaturation penalty; 0.01 forces a black-and-white envmap. **try_7
  falsified** the hypothesis that this mattered for relighting (§3.3).
* `lambda_dssim 0.4`, `densify_until_iter 30k`, `opacity_reset_interval 3k`,
  unlimited gaussians, `--random_background`, `--hdr_rotation` — shared with
  the baseline/reference protocol.

### 2.4 Light-linear indirect illumination (`--light_linear_indirect`, LLI)
Baseline GIR transports systematically less energy than Blender's
path-traced GT (measured ~0.6–0.9× foreground brightness at relight in
try_6):

* diffuse: per-gaussian **binary** occlusion; occluded sample directions
  contribute **zero** light (no bounce);
* specular "indirect": per-gaussian SH radiance that is **frozen** — it does
  not change when the light is swapped at relight.

During training the learned envmap silently absorbs this deficit by training
too bright; at relight the GT HDRI arrives at native intensity, the boost is
gone, and every render is uniformly too dark. LLI reparameterizes both baked
indirect terms as *reflectance × mean radiance of the current envmap*
(`env_mean = envlight.base.mean()`), and occluded diffuse directions receive
a per-gaussian SH bounce term instead of zero. At relight `env_mean` is
recomputed from the swapped HDRI, so the bounce energy tracks the new light.
No new parameters (reuses the existing SH features; densification untouched).

**try_8 fix:** `env_mean` is now **detached** (see §3.2 for why).

## 3. What try_7 established

Final-iteration summary (relight numbers averaged over 6 HDRIs; t6 runs had
no gain metric):

| run | test PSNR | relight PSNR | relight aligned | relight gain | albedo PSNR (raw/aligned) | normal err | envmap logPSNR |
|---|---|---|---|---|---|---|---|
| t6 baseline (no prior) | **32.87** | 24.49 | 25.40 | – | 23.2 / – | 35.2° | 28.7 |
| t6 gt_zncc_grad_ctrl | 30.76 | **25.12** | 26.95 | – | 22.0 / – | 11.6° | **30.3** |
| t7 gt_zncc_grad_anchor_lli | 30.58 | 24.82 | **27.97** | 1.32 | 27.3 / 30.4 | 11.3° | 26.1 |
| t7 gt_zncc_anchor_lli | 30.47 | 23.95 | 27.76 | 1.40 | **31.2 / 31.4** | **11.0°** | 26.1 |
| t7 gt_zncc_grad_plus_hdrfix | 30.23 | 22.66 | 26.36 | 1.51 | 25.3 / 25.6 | 11.2° | 27.7 |

(anchor-only, lli-only and diffusion arms OOMed on the 8 GB machine at ~1.2M
gaussians; anchor-only and diffusion are re-run in try_8.)

### 3.1 The gain question ("shouldn't the fitted gain be 1.0?")
The `*_gain` metric is the least-squares global gain g\* = ⟨render,GT⟩/⟨render,render⟩
fitted per image; g\* > 1 means the render is too dark. It is logged **only
for relight renders under unseen HDRIs** — no gain was computed under the
training light in try_7, so the ≥1.25 chart values are all relight gains.
Measured post-hoc from the saved eval visuals, the fitted gain under the
**training light is ≈1.09** — near 1.0 exactly as expected, because the
photometric loss pins the training-light render to the GT. So the metric is
not broken and "there was an error after all": the gap between 1.09
(training light) and 1.32–1.51 (relight) **is** the relight-specific energy
deficit. try_8 logs `test_gain` (train-light reference) alongside the relight
gains so this comparison is always in the report.

### 3.2 LLI works — but its first implementation opened a degeneracy
Anchor+LLI produced the best aligned relight ever (27.97 dB) and kept raw
relight near ctrl (24.82 vs 25.12) *while* fixing the albedo scale — i.e. it
genuinely replaced most of the accidental compensation that made ctrl look
good. But the relight gain stalled at 1.32–1.40 instead of → 1.0, and the
envmap recovery **collapsed** (logPSNR 30.3 → 26.1, log-correlation with the
GT sunset 0.69 → 0.28, the sunset core visually gone from
`train_process/hdr/37500.png`).

Root cause (verified from the checkpoints): the bounce term was
`reflectance × envlight.base.mean()` **with gradient flowing into the mean**.
The optimizer discovered it could pump bounce energy globally by inflating a
few tiny ultra-bright texels — the learned envmap's mean reached **3.3× the
GT sunset's mean** while the rest of the map went dark (least-squares
alignment factor ~90× confirms a misplaced bright spike). Because the SH
reflectance was calibrated against that inflated mean, swapping in a
native-intensity HDRI at relight under-scales the bounce by the same ~3.3×,
which is exactly the residual deficit the gain metric shows. **Fix (try_8):**
`env_mean` is detached — the envmap receives gradients only through the
direct light terms (as in ctrl, whose envmap stayed healthy), the SH
reflectance alone calibrates the bounce, and the light-linear rescaling at
relight is untouched.

Bookkeeping that supports this account: the learned envmap's brightness ratio
vs GT (1.43× for anchor_lli, 1.69× zncc, 3.4–3.6× for the _plus runs) tracks
each run's relight gain — the envmap absorbs whatever energy the transport
cannot deliver, and that absorbed factor is what relights are missing. try_8
logs this as `envmap_mean_ratio`.

### 3.3 The reg_hdr / black-and-white-envmap hypothesis is dead
`gt_zncc_grad_plus_hdrfix` = the try_6 `_plus` bundle with reg_hdr restored
to 0.001 (from 0.01). Every metric is within noise of try_6 `_plus`
(relight 22.66 vs 22.56, aligned 26.36 vs 26.22, envmap 27.67 vs 27.57). The
desaturated envmap was a symptom, not the cause of `_plus`'s weakness — its
deficit vs anchor_lli comes from its other ingredients (no LLI, the
`roughness_video` prior, `prior_geom_grad_scale 0.0`). The `_plus` bundle is
retired.

### 3.4 zncc is back in play
With the anchor pinning the scale, plain `zncc` recovers a *nearly perfect*
albedo (31.2 dB raw ≈ aligned — the ambiguity is actually resolved) and stays
within 0.2 dB of zncc_grad on aligned relight. zncc_grad's extra invariance
(needed for inconsistent diffusion priors) costs ~4 dB of albedo accuracy on
clean GT priors. Both stay in the batch; for GT priors zncc + anchor is
likely the headline once LLI is fixed.

### 3.5 Smaller observations
* The 30k-peak-then-decay of relight/envmap metrics from try_6 is much
  reduced; 35k→37.5k drift is now ≤0.2 dB.
* All prior runs sit ~2.1–2.4 dB below baseline on novel-view PSNR with ~8×
  the gaussians (1.0–1.2M vs 140k). >8 GB VRAM is required; this is what
  OOMed three try_7 arms.
* Per-HDRI gains within a run are fairly flat (deficit is global/multiplicative),
  with `forest` consistently worst (most ambient-driven HDRI → most sensitive
  to the bounce miscalibration).

## 4. Metrics guide (energy calibration)
* `relight_<hdri>_gain` — fitted gain render→GT at relight; >1 = too dark;
  goal → 1.0. Judge together with raw PSNR: raw relight rewards models whose
  albedo error happens to cancel the deficit (ctrl!), so "gain → 1 with
  aligned-level raw PSNR" is the success criterion.
* `test_gain` (new, try_8) — same fit under the training light; sanity
  reference ≈ 1.05–1.1. Relight gain above `test_gain` = real deficit.
* `envmap_mean_ratio` (new, try_8) — learned envmap brightness / GT HDRI
  brightness; tracks the relight gain when the envmap absorbs the deficit.
* `*_psnr_aligned` — PSNR after removing one global gain: decomposition
  quality independent of the energy calibration.

## 5. try_8 batch (prepared, `GIR/run_experiments.py`)
Four Phase-3-only runs (all warm-up-buffer HITs verified), relight eval capped
at 24 of 200 test views (`--relight_max_views`, deterministic evenly-spaced
subset; expect ±0.1–0.2 dB sampling difference vs try_6/7 numbers):

1. `gt_zncc_grad_anchor` — anchor without LLI (re-run of the OOMed arm): the
   clean no-LLI reference for gain / envmap_mean_ratio.
2. `gt_zncc_grad_anchor_lli2` — anchor + **fixed** LLI. Predictions: envmap
   logPSNR back to ~30 with visible sunset core, envmap_mean_ratio ≤ ~1.2,
   relight gain < 1.32, raw relight ≥ 25.
3. `gt_zncc_anchor_lli2` — zncc twin; if relight holds, new headline.
4. `diff_zncc_grad_lli2` — diffusion priors + fixed LLI, no anchor (diffusion
   albedo scale is unreliable; anchoring to it cost 5 dB in try_6).

Launch: `cd GIR && python run_experiments.py` (≥16 GB VRAM recommended).

## 6. Open problems / next candidates after try_8
* **Residual deficit after the LLI fix** — if the gain plateaus above ~1.1
  with a healthy envmap, the remaining energy is genuinely missing transport
  (GI on unoccluded surfaces gets no bounce; binary occlusion). Next step: a
  directional bounce normalizer (SH reflectance × diffuse irradiance at the
  normal instead of the global mean) or fractional occlusion.
* **zncc_grad albedo offsets** — the 3 dB raw-vs-aligned albedo gap suggests
  trying `albedo_anchor_weight 0.1–0.15` for zncc_grad only (zncc doesn't
  need it).
* **Novel-view gap** (−2.2 dB vs baseline): relax `reduce_geo_lr_third_stage`
  toward 0.3 now that the prior scheduler holds full weight, and/or cap
  gaussians (~8× baseline count also costs VRAM/eval time).
* **Diffusion-prior track**: re-establish the diffusion baseline with fixed
  LLI, then revisit per-view exposure/whitebalance handling for the albedo
  prior (`zncc` family) before scaling to real scenes.
