# GT-Prior Inverse Rendering on GIR — Findings through try_10 (2026-07-07)

This document summarizes what we have established so far about resolving the
albedo–lighting ambiguity in Gaussian Inverse Rendering (GIR) with ground-truth
priors, what the current best configuration is, and exactly how it differs from
the GIR baseline. Batches before `new_experiments_try_5_fixed_envlight` used a
broken envmap sampler and are not comparable; `try_6` established the reference
protocol (`--hdr_rotation`, random background); `try_7` introduced and stress-
tested light-linear indirect illumination (LLI); `try_8` fixed LLI's env_mean
degeneracy (detach); `try_9` pinned the albedo scale and thereby **proved the
residual relight darkness is genuine missing transport** — and its diffusion
arm showed the missing energy can be (illegitimately but effectively) parked
on the reflectance side. `try_10` validated the headline configs at the
paper's operating point (-r 2, 60k iterations) and **met the relight
success criterion** (§7.1).

A structured reference of every change vs. baseline GIR — what, why, how it
works, with code pointers — plus a paper-style method section lives in
[METHOD.md](METHOD.md). This document is the chronological experimental log.

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

Best **GT-prior** run: **`gt_zncc_grad_anchor_lli2`** (try_8) — LLI with
detached env_mean: raw relight 25.40 dB (above ctrl's 25.12 *with* a
near-correct albedo scale), relight SSIM 0.895, aligned relight 27.73 (level
with try_7's 27.97 within eval-subset noise), relight gain 1.22. (try_9's
**diffusion** arm beats every run on raw relight, 26.49, but through a
physically wrong decomposition — see §6.2; the best *honest* decomposition is
try_9's `gt_zncc_anchor_lli2`.)
Everything below is a delta against the reference GIR training
(`t6_baseline_no_prior`, which itself already uses `--random_background`,
`--hdr_rotation`, reg_hdr 0.001-ish, densify-until-30k).

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

### 2.5 Complete engine-code delta vs the reference repo (`GIR_Reference/`)

Everything our `GIR/` tree changes relative to the paper's code, and whether
it can affect a **baseline** run (no prior flags). Verified by full-tree diff
against `GIR_Reference/GIR` (2026-07-06); `utils/ir_utils.py` (envlight
sampling), `utils/sh_utils.py`, `utils/graphics_utils.py` and the CUDA
submodules are **byte-identical** to the reference.

| where | what | baseline effect |
|---|---|---|
| `arguments/__init__.py` | new dataset fields (`albedo_gt_dir`, `normal_gt_dir`, `metallic_gt_dir`, `roughness_gt_dir`, `normal_camera_convention`); `max_gaussians` (default 0 = unlimited) | none at defaults |
| `gaussian_renderer/__init__.py` | `prior_geom_grad_scale` (grad-scaled prior rasters; default 1.0 = identity); `albedo_geometry_warmup` path (flag-gated); Phase-2 normal map rasterized **with gradients** every iteration (reference: no-grad viz every 1000 it) so the normal prior can supervise it; stage-boundary albedo resets guarded by `is_train`; LLI threading (flag-gated) | extra compute only — no loss consumes the Phase-2 normal raster in a baseline run, so gradients are unaffected |
| `scene/gaussian_model.py` | `restore()` now loads the envlight state dict (reference *saved* it in checkpoints but never loaded it back — required for our warm-up-buffer resumes); occlusion-grid + diffuse-light computation chunked (numerically identical, removes OOM at >1M gaussians); LLI bounce (flag-gated); third-stage geo-LR schedule (inert at factor 1.0) | none for a fresh full run |
| `scene/__init__.py`, `scene/dataset_readers.py`, `scene/cameras.py`, `utils/camera_utils.py` | prior-image loading (Blender/COLMAP *WithPriors* variants), prior tensors carried on cameras, camera-space→world normal conversion | data plumbing only |
| `utils/loss_utils.py` | added losses: `huber_loss`, `albedo_prior_loss` (modes `direct`/`zncc`/`zncc_grad`/…), `decode_normal_to_world`; reference losses untouched | none unless a prior flag enables them |
| `train.py` | three-phase prior supervision + anchor + weight scheduler; `exclude_prior_loss`; `disable_reset_third_stage`; `max_gaussians` cap check; geo-LR anneal hookup; envmap-mean penalty `--reg_env_mean_weight` (default 0 = off, added for try_10); the whole eval stack (`periodic_evaluation`, relight eval + gains, envmap metrics, `metrics_log.json`, per-run PDF) | logging/eval only; the optimization path with all new flags at defaults is line-for-line the reference loop (same photometric loss, same Phase-3 TV/smoothness/reg_hdr terms, same densify/reset/optimizer logic) |
| new files | `run_experiments.py` (batch runner + warm-up buffer + comparison report), `generate_report.py`, `run_training.py` | not part of training |

**Baseline recipe (try_10, verified paper-exact):** `train.py -s <lego> --eval
--random_background --hdr_rotation --reg_hdr_weight 0.1 --reg_material_weight
0.05` with the engine defaults 60k iterations, stages 5k/30k, densify
500→45k, `lambda_dssim 0.4`, `opacity_reset_interval 3000` — the exact
`scripts/train_tensoir.sh` launch line of the reference repo. Two intentional
deviations in try_10, both applied to **all** runs equally: `-r 2` (400×400;
the paper trains native 800×800) and a `max_gaussians` 1.5M cap (the paper
densifies unlimited; the cap is OOM insurance at r2 and should only bind on
runaway densification — at r4 the baseline used 140k, prior runs ~1.2M).

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
* `test_albedo_gain` (new, try_9) — fitted gain rendered→GT albedo; < 1 =
  albedo too bright. Bookkeeping identity (verified in try_8):
  `relight_gain ≈ envmap_mean_ratio × test_albedo_gain`.
* `*_psnr_aligned` — PSNR after removing one global gain: decomposition
  quality independent of the energy calibration.

## 5. try_8: the detach fix works — and localizes the residual
Single run `gt_zncc_grad_anchor_lli2` (= try_7's headline with the detached
env_mean), `outputs/new_experiments_try_8_lli_detached`:

* **Blob exploit closed** (verified from the checkpoint): envmap base mean
  1.54 → 0.86 (was 3.3× the GT sunset mean), least-squares spike factor
  90× → 6.6×, log-correlation 0.28 → 0.50, envmap logPSNR 26.1 → 27.4. The
  HDR looks like a sunset again, though not yet at ctrl's level (30.3 /
  0.69) — the map is now *uniformly* ~1.43× too bright (the remaining
  deficit absorber), with moderate residual spikiness.
* **Best raw relight yet: 25.40 dB** (ctrl 25.12 — and unlike ctrl, with a
  near-correct albedo), best relight SSIM 0.895, relight gain 1.32 → 1.22,
  aligned 27.73 ≈ try_7's 27.97 (24-view eval subset, ±0.1–0.2 dB).
* **The residual gain 1.22 is fully bookkept** by two measured factors:
  `relight_gain ≈ envmap_mean_ratio × albedo_gain` = 1.43 × 0.87 ≈ 1.24.
  The zncc_grad albedo drifts ~1.15× too *bright* (fitted gain 0.87, blue
  channel 1.49× — the inverse-sunset tint) despite the 0.05 anchor: the
  gradient-domain loss leaves per-channel offsets that the weak anchor loses
  to. Plain zncc holds the albedo at 0.98 (pinned). The engine now logs
  `test_albedo_gain` so all three factors of this identity are in the report.
* Interpretation: **the transport question is (mostly) answered; the open
  lever is the albedo/envmap scale split.** If a truly pinned albedo (zncc)
  forces the envmap ratio down toward 1 (photometric pressure via the
  bounce ≥ 0 constraint on unoccluded surfaces), the gain follows and raw
  relight rises further. If the ratio stays ~1.4, the remainder is genuine
  missing transport energy.

## 6. try_9: the residual deficit is real transport — and the diffusion arm shows where it *can* live

`outputs/new_experiments_try_9_albedo_scale`, four Phase-3-only runs (all
warm-up-buffer HITs), relight eval on 24 views. Final-iteration summary
(t8 headline included for reference):

| run | test PSNR | relight raw | relight aligned | relight gain | relight SSIM | albedo raw/aligned | albedo gain | normal err | metallic MAE | envmap logPSNR | envmap ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| t8 gt_zncc_grad_anchor_lli2 | 30.46 | 25.40 | **27.73** | 1.22 | 0.895 | 26.1 / 30.3 | ~0.87 | 11.4° | 0.0011 | 27.4 | 1.44 |
| t9 gt_zncc_anchor_lli2 | 30.60 | 24.95 | 27.57 | 1.26 | 0.891 | 29.2 / 31.2 | 0.92 | **11.1°** | **0.0007** | 30.2 | 1.75 |
| t9 gt_zncc_grad_anchor15_lli2 | 30.49 | 24.47 | 27.70 | 1.32 | 0.887 | **29.7 / 31.3** | 0.93 | 11.4° | 0.0012 | 27.1 | 1.72 |
| t9 diff_zncc_grad_lli2 | **31.37** | **26.49** | 27.43 | **0.94** | **0.906** | 23.7 / 23.9 | 0.99 | 16.8° | 0.058 | **30.5** | 0.87 |
| t9 gt_zncc_grad_anchor (no LLI) | 30.23 | 23.31 | 26.84 | 1.40 | 0.872 | 26.7 / 29.7 | 0.88 | 11.5° | 0.0029 | 29.1 | 2.28 |

### 6.1 The §5 diagnostic is answered: the remainder is genuine missing transport

With the albedo truly pinned (`zncc` + anchor: albedo 29.2 dB raw, gain 0.92)
the envmap ratio did **not** drop toward 1 — it *rose* through Phase 3
(1.66 → 1.75 from iter 30k to 37.5k) while the envmap **structure** stayed
healthy (logPSNR 30.2, ≈ ctrl's 30.3, clearly a sunset in
`train_process/hdr/37500.png`). Raising the zncc_grad anchor to 0.15 told the
same story: it fixed the albedo drift as designed (gain 0.87 → 0.93, albedo
26.1 → 29.7 dB) but bought *nothing* at relight (24.47 raw, gain 1.32). So
the albedo scale was **not** the binding constraint — once albedo, normals
and metallic are all pinned, the photometric fit under a transport-deficient
renderer has exactly one knob left, the training envmap, and it inflates it
by the missing-interreflection factor. That factor stays home at relight.
The no-LLI anchor run is the cleanest attribution of what LLI buys: without
the bounce the envmap absorbs 2.28× (gain 1.40, worst raw relight 23.31);
with it, 1.4–1.75× (gain 1.22–1.32).

Caveat on the bookkeeping identity from §4: with LLI active,
`envmap_mean_ratio × albedo_gain` (1.75 × 0.92 ≈ 1.61) now *over*-predicts
the measured gain (1.26) — the mean ratio is a coarse proxy once part of the
transport rescales with the envmap mean itself, and the base-map mean weights
all directions equally regardless of how much they irradiate the object.
Directionally it still ranks the runs correctly.

### 6.2 Why the diffusion arm smokes everyone on raw relight (checkpoint forensics)

`diff_zncc_grad_lli2` is simultaneously the **best relighter** (raw 26.49,
+1.1 dB over t8; SSIM 0.906; per-HDRI gains 0.85–1.04, i.e. *calibrated*;
envmap ratio 0.87, logPSNR 30.5) and the **worst decomposition** (albedo
23.7 dB, normals 16.8°, metallic MAE 0.058, roughness MAE 0.10). From the
35k checkpoints: its opacity-weighted mean **metallic is 0.217** with 26% of
(opacity-weighted) gaussians above 0.5 — versus 0.008–0.025 in *every*
GT-prior run — and its roughness is 0.83 vs 0.68. The GT lego is fully
dielectric (metallic ≡ 0).

Mechanism: a high-roughness, moderate-metallic surface is a *pseudo-diffuse
transport channel through the specular path*. Its direct part samples the
(swapped) envmap through the roughness-mip lookup, and its indirect part is
`occ × SH × env_mean` — the light-linear term. Both **transfer to unseen
HDRIs**. The diffusion run's priors are too loose to forbid this (no albedo
anchor, metallic weight only 0.05 on noisy DiffusionRenderer metallic maps),
so the optimizer routed the missing interreflection energy into reflectance
— where it relights approximately correctly — instead of into the envmap —
where it doesn't. The GT runs' *better* priors close exactly this escape
hatch (metallic pinned to 0 at MAE ~0.001, albedo anchored), which is why
their honesty is punished at relight.

### 6.3 What this says about the metrics and the anchor

* **Raw vs aligned relight measure different failures.** Raw = structure ×
  energy calibration (and raw is what a deployed system delivers — there is
  no GT to scale-align against). Aligned = structure only. The GT anchor
  runs own the structure (27.6–27.7 aligned); the diffusion run owns the
  calibration (raw ≈ aligned, gain ≈ 1). Neither metric alone: the success
  criterion stays "**gain → 1.0 with aligned-level raw PSNR**" (§4).
* **The anchor is not a bad idea — it is doing precisely its job** (albedo
  gain 0.92–0.93 vs the 1.2× drift it was built against), and the diffusion
  run does *not* argue for dropping it: its win comes from unconstrained
  metallic, not from albedo freedom (its albedo gain is 0.99 anyway).
  Dropping the anchor on GT runs would just re-create ctrl's accidental-
  compensation regime that try_7 retired.
* **The actionable lesson**: ~1.5–2 dB of raw relight is on the table if the
  bounce energy gets a *legitimate* light-tracking home. The optimizer will
  find any such channel on its own (the diffusion run proves it) — the
  engine should offer one that doesn't corrupt the materials: strengthen the
  LLI bounce (per-gaussian, ≥ 0, light-linear) so it out-competes the envmap
  as deficit absorber, e.g. via a small penalty on envmap-mean inflation, an
  irradiance-normalized (directional) bounce, or fractional occlusion.

### 6.4 Smaller observations

* Per-HDRI gain ordering is identical in every run (fireplace ≈ best-
  calibrated → forest worst, spread ~0.3): the global env-mean normalizer
  over/under-scales depending on how the HDRI's energy is distributed
  relative to the object — a directional irradiance normalizer should
  flatten this.
* The diffusion run also has the best novel-view PSNR of all prior runs
  (31.37) and the fewest gaussians (854k vs 950k–1.2M) — looser priors churn
  the geometry less.
* Metrics still decay slightly 35k → 37.5k in the anchored GT runs (raw
  relight −0.15, envmap ratio +0.03–0.12) — the envmap keeps absorbing
  through the end of training.

## 7. try_10 batch (prepared, `GIR/run_experiments.py`)

Purpose: validate the three headline configs at the **paper's operating
point** before any new engine lever — resolution `-r 2` (400×400) and the
paper's 60k schedule (stages 5k/30k, densification 500→45k), replacing the
exploration setting (-r 4, 37.5k, stages 5k/25k, densify→30k).
`outputs/new_experiments_try_10_paper_scale`, four runs:

1. `baseline_no_prior` — **bit-for-bit the reference configuration** (§2.5
   baseline recipe; uncapped gaussians, reg_hdr 0.1, reg_material 0.05, no
   prior losses in the objective).
2. `gt_zncc_grad_anchor_lli2` — best GT raw-relight config (try_8 lineage);
   zncc_grad must stay viable for the diffusion end-goal.
3. `gt_zncc_anchor_lli2_envpen` — **challenger with the new engine lever**:
   try_9's best decomposition (zncc + anchor 0.05 + LLI) plus an
   **envmap-mean penalty** (`--reg_env_mean_weight 0.005`, scheduler-ramped,
   Phase 3 only). Rationale from §6: the only failure of the zncc+anchor run
   was the envmap absorbing the transport deficit (ratio 1.66→1.75, gain
   stuck at 1.26); constant downward pressure on the envmap's mean radiance
   makes the photometric fit re-home that energy into the ≥0, light-linear
   LLI bounce — the same transfer mechanism the diffusion run exploited via
   metallic, offered legitimately. The weight sits between the empirically
   mild (0.001) and drastic (0.01) values of the same-scale `reg_hdr`
   penalty; the diffusion run (ratio 0.87, gains 0.85–1.04) shows mild
   overshoot is benign. Success = envmap ratio → ~1–1.2, gain < 1.22, raw
   relight > 25.4, albedo still pinned.
4. `diff_zncc_grad_lli2` — the diffusion-prior arm.

Config notes: `geo_lr_final_iter` scaled to 66k (same relative anneal shape
as 40k was on the 37.5k schedule); **all** runs (baseline included) capped at
`max_gaussians` 1.5M (see §2.5 deviations); `EXTERNAL_RUNS` is empty because
**no earlier batch is comparable at this resolution**. Nothing can reuse the r4 warm-up buffer (resolution/schedule
are in the fingerprint): runs 2+3 share one fresh warm-up via the buffer,
baseline and diffusion each run all three phases — expect **days, not hours**
on a 16 GB GPU. The baseline and GT-prior command lines were smoke-tested end-to-end at r2
(120 iters + eval + relight + report, exit 0), and the envmap-mean penalty
was smoke-tested through Phase 3 with LLI active (tiny stage boundaries,
backward included, exit 0).

Launch: `cd GIR && python run_experiments.py`.

### 7.1 try_10 results (2026-07-07)

**Run status.** Baseline and envpen completed to 60k. The headline
(`gt_zncc_grad_anchor_lli2`) and diffusion runs OOMed at ~47k/~48k — *after*
their 45k checkpoints were saved — in `get_diffuse_occ`: a 1.8 GiB allocation
failed with ~7 GiB *reserved but unallocated* (caching-allocator
fragmentation, not true pressure; peak live memory was ~9.3 of 15.5 GiB). The
launcher now sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which
addresses exactly this. Densification ends at 45k, so resuming from
`chkpnt45000` replays the identical remaining schedule.

**Metric provenance for the crashed runs:** test/albedo/normal/envmap values
are the 45k eval; the relight values are the **40k** relight pass (relight
runs every 10k, so 45k was never relit — the resumed runs will log 50k/60k).

| run (last eval) | test PSNR | relight raw | aligned | gain | rel. SSIM | albedo raw/aligned | alb. gain | normal ° | env logPSNR | env ratio | #G |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline @60k | **34.45** | 25.79 | 26.87 | 1.10 | 0.891 | 23.7 / 25.2 | 0.91 | 31.4 | **29.1** | 1.51 | 133k |
| gt_zncc_grad_anchor_lli2 @45k | 32.21 | **27.62** | **28.65** | **1.002** | **0.910** | 24.7 / 31.6 | 0.815 | 8.6 | 28.9 | **1.14** | 1.28M |
| gt_zncc_anchor_lli2_envpen @60k | 31.12 | 26.89 | 27.98 | 0.977 | 0.903 | **27.8 / 33.7** | 0.867 | **8.2** | 26.7 | 2.00 | 929k |
| diff_zncc_grad_lli2 @45k | 33.22 | 24.58 | 28.62 | 0.802 | 0.906 | 24.2 / 24.7 | **0.95** | 16.4 | 28.9 | 0.71 | 1.17M |

**1. Paper scale unlocked the goal.** The headline run meets the try_7
success criterion: relight gain **1.002** at the best raw relight of any run
ever (27.62 dB, +1.8 over the paper-exact baseline), with per-HDRI gains
tightly clustered around 1 (0.89–1.14; at r4 they were 1.2–1.5). The r2/60k
operating point plus detached LLI resolves the energy calibration that the r4
batches could not: its envmap ratio *fell* through phase 3 (1.24 → 1.14),
opposite to every other run's rising trend. The remaining ~1 dB raw-vs-
aligned gap is per-HDRI gain spread (fireplace 0.89, forest 1.14), not a
global bias. Its **sole flaw**: albedo gain 0.815 — the known zncc_grad
per-channel drift (blue-tinted), which anchor 0.05 is too weak to hold
(raw albedo 24.7 vs 31.6 aligned).

**2. The envmap-mean penalty failed and is retired.** Despite the 0.005
penalty, envpen's envmap ratio rose 1.49 → **2.00** through phase 3, its
envmap logPSNR degraded to 26.7 (headline: 28.9), and raw relight (26.89)
stayed below the headline. The penalty did not re-home the deficit energy —
it degraded the envmap structure while the ratio *still* inflated (recall the
§6.4 caveat: with LLI, base-map mean is a coarse proxy — the relight gain
0.977 shows the *transferable* calibration was fine anyway). Confound to
note: this was also the batch's only plain-zncc run, and its albedo is the
best of the batch (27.8/33.7, in line with zncc's r4 pinning) — so it stays
in the report as the zncc reference, but `--reg_env_mean_weight` is retired.

**3. The diffusion arm's escape hatch overshoots at r2.** Aligned relight
28.62 *ties* the headline — the structure transfer is fine — but the envmap
trained 1.4× too **dark** (ratio 0.71) with all six per-HDRI gains < 1
(0.73–0.88): at paper scale the rough-metallic pseudo-bounce compensates
*past* the deficit, so renders come out too bright and raw relight (24.58)
falls below baseline. Same mechanism as r4 (metallic MAE 0.058, roughness
MAE 0.096), overcorrecting at the new operating point. Its test PSNR (33.22)
is the best of any prior run. Judge it at 60k after the resume.

**4. Baseline confirms the deficit is structural.** Even the paper-exact
reference inflates its envmap through phase 3 (ratio 1.03 → 1.51) and
relights 1.10× too dark — vanilla GIR has the same transport hole; it just
has no prior losses exposing it.

### 7.2 Repair batch (prepared, in `GIR/run_experiments.py`)

1. **Resume the headline 45k → 60k** (fragmentation fix is in the launcher;
   `train.py` keeps metric history up to and including the resume iteration):
   `python run_experiments.py --only lego_gt_zncc_grad_anchor_lli2 --resume-iter 45000` (~6 h).
2. **Resume the diffusion arm 45k → 60k**:
   `python run_experiments.py --only lego_diff_zncc_grad_lli2 --resume-iter 45000` (~5 h).
3. **New hyperparameter-only variant `gt_zncc_grad_anchor15_lli2`**: identical
   to the headline except `albedo_anchor_weight` 0.05 → **0.15** — the try_9
   r4 evidence (gain 0.87 → 0.93, albedo +5 dB raw, zero relight cost)
   aimed at the headline's only flaw. Verified warm-up-buffer HIT (fingerprint
   `37bf5b03f023cbb7`, the stage-2 checkpoint envpen buffered), so it costs
   Phase 3 only (~15 h). Watch: albedo gain → ~0.93–1.0 at 28–30 dB raw,
   relight raw/gain holding ~27.6 / ~1.0; envmap ratio may tick up from 1.14
   as the albedo stops absorbing scale.

`run_experiments.py` now **skips any run that already owns a checkpoint**
unless `--resume` is given, so a bare `python run_experiments.py` cannot
clobber the finished results. Watch on the resumes: in the two runs that
reached 60k, the envmap ratio *worsened* over the 45k→60k tail — if the
headline's calibration degrades likewise by 60k, that is a finding about the
phase-3 tail (and `chkpnt45000` stays on disk for a re-branch).

## 8. Open problems / next candidates after try_10

* **The transport deficit** — largely *dissolved at paper scale* for the GT
  arm (headline gain 1.002, §7.1), so this is no longer the top blocker. The
  envmap-mean penalty was tried (`--reg_env_mean_weight 0.005`) and **failed**
  (§7.1 #2 — ratio rose to 2.0, envmap structure degraded): retired. If
  calibration regresses on other scenes or at 60k, the remaining engine
  candidates are unchanged: irradiance-normalized directional bounce (SH ×
  diffuse irradiance at the normal instead of global mean; would also target
  the residual per-HDRI gain spread 0.89–1.14) or fractional (soft)
  occlusion.
* **Diffusion arm overshoots at r2** (§7.1 #3): the metallic pseudo-bounce
  now over-compensates (ratio 0.71, gains 0.73–0.88). Hyperparameter levers
  to try after the 60k resume: stronger metallic prior
  (`lambda_metallic_gt` 0.05 → 0.15) to shrink the hatch, and/or an anchor
  once per-view exposure handling exists (below).
* **Albedo scale for zncc_grad**: anchor 0.15 at paper scale is **queued**
  (`gt_zncc_grad_anchor15_lli2`, §7.2). If gain still < 0.9, next: per-channel
  anchor or anchor on the log-mean instead of the mean.
* **Novel-view gap** (−2.2 dB vs baseline at r2: 32.2 vs 34.4): relax
  `reduce_geo_lr_third_stage` toward 0.3, and/or revisit the gaussian-count
  blow-up of prior runs (~10× baseline at r2: 1.28M vs 133k).
* **Envmap structure for zncc_grad runs**: persists at r2 (28.9 logPSNR vs
  29.1 baseline — mild now; envpen's 26.7 was self-inflicted). Mild log-space
  TV on the envmap base only if it matters for a target scene.
* **Diffusion-prior track**: per-view exposure/whitebalance handling for the
  albedo prior (`zncc` family) before scaling to real scenes.
