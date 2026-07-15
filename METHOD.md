# Prior-Guided Gaussian Inverse Rendering — Change Reference & Method

*This document records **what** we changed relative to baseline GIR,
**why**, and **how it works** — precisely enough to be cited later.*

---

# Complete inventory of changes vs. baseline GIR

## 0. What "baseline" means

### 0.1 The reference recipe

The baseline is the reference GIR implementation (`GIR_Reference/`) trained
with the paper's own TensoIR-scene launch line
(`scripts/train_tensoir.sh`):

```
train.py -s <scene> --eval --random_background --hdr_rotation \
         --reg_hdr_weight 0.1 --reg_material_weight 0.05
```

with every other value at engine defaults: 60 000 iterations, stage
boundaries `first_stage_step 5000` / `second_stage_step 30000`, densification
from iter 500 to 45 000 every 100 iters, opacity reset every 3 000,
`percent_dense 0.01`, `densify_grad_threshold 2e-4`, `lambda_dssim 0.4`.

What each flag of that line means, and why the reference needs it:

* `--eval` — hold out the test cameras (Blender split: 100 train / 200
  test) instead of training on everything.
* `--random_background` — composite each training image over a random
  background colour per iteration. Prevents the model from "explaining"
  background pixels with gaussians (background baking): no fixed colour is
  ever reliable, so opacity outside the object is driven to zero.
* `--hdr_rotation` — remaps the direction vectors used for every
  environment-light query (diffuse samples, specular reflections, shading
  normals) between the engine's **Y-up lat-long** envmap convention and
  Blender's **Z-up** world. Reference behaviour for Blender/TensoIR scenes,
  not our addition. Without it every light query samples the HDRI sideways
  — we verified this empirically: the learned envmap correlates with the GT
  sunset only under GIR's rotation mapping (log-correlation +0.40 vs −0.21
  for identity).
* `--reg_hdr_weight 0.1` — weight of the envmap **desaturation** prior
  (§3.2). The reference sets it this high **because it has no other way to
  resolve the albedo–light colour ambiguity**: a global per-channel gain
  can move freely between albedo and envmap without changing any rendered
  pixel, so the reference imposes the heuristic "illumination is
  approximately white" by penalizing the envmap's distance from its own
  grayscale. That pushes colour out of the light and into the albedo — the
  right call when nothing else anchors albedo colour, and the wrong call
  once a GT albedo prior does (see §3.2 for why our prior runs lower it to
  0.001).
* `--reg_material_weight 0.05` — TV smoothness weight on the rasterized
  metallic and roughness maps (the launch scripts use 0.05; the engine
  default is 0.1).

### 0.2 The reference pipeline in brief

For the deltas below to make sense, this is what stock GIR optimizes
(details in [2], code in `scene/gaussian_model.py` and
`submodules/envlight`):

* **Representation.** A set of 3D gaussians [1] with standard geometry
  attributes (position, anisotropic scale, rotation, opacity) plus
  per-gaussian PBR attributes: albedo $a \in [0,1]^3$, metallic $m$,
  roughness $r$, and spherical-harmonics (SH) feature coefficients $f$ that
  double as a baked *indirect radiance* field. The shading normal is not a
  free parameter — it is the direction of the gaussian's smallest
  covariance eigenvector (`get_eigenvector`), i.e. geometry *is* the normal.
* **Environment light.** Not a free per-texel map: a trainable latent
  (1×128×64×128) is decoded by a small upsampling CNN (`hdr_net`) into a
  1024×2048 lat-long image, converted to a 6×512×512 cubemap, added to a
  trainable per-texel residual, and passed through a **softplus** — so
  `envlight.base` is a strictly positive cubemap
  ([light.py:35-107](GIR/submodules/envlight/envlight/light.py#L35)). Mip
  levels are prefiltered from it every iteration for roughness-dependent
  specular queries (split-sum [4,5], roughness range 0.08–0.5). The CNN
  parameterization acts as an implicit smoothness prior on the light.
* **Visibility.** A 128³ voxel occupancy grid is rasterized from the
  gaussian positions and covariance extents (`get_grid`). Each gaussian
  stores **binary** visibility for 128 fixed hemisphere directions
  (`diffuse_sample_num = 128`, cosine-weighted), refreshed periodically
  (`get_diffuse_occ`), plus a scalar reflection-direction occlusion for the
  specular path (`compute_occlusion`).
* **Shading.** Per gaussian: diffuse = $(1{-}m)\,a\,\cdot$ average of the
  envmap over the 128 directions with occluded directions contributing
  zero; specular = split-sum with $F_0 = 0.04(1{-}m) + m\,a$ and the
  pre-integrated BRDF LUT, mixing a prefiltered envmap query (unoccluded
  part) with the baked SH radiance evaluated at the reflection direction
  (occluded part). The result is tonemapped linear→sRGB and rasterized;
  everything trains end-to-end against the photometric loss
  $0.6\,L_1 + 0.4\,(1-\mathrm{SSIM})$.
* **Schedule.** Two stage boundaries (5 k / 30 k) switch the renderer from
  plain radiance (≤ 5 k) to PBR with direct light only (≤ 30 k) to the full
  model with occlusion + SH indirect (> 30 k). Densification runs from 500
  to 45 k; opacity is reset every 3 k iterations throughout.

### 0.3 Verified-identical code

Our engine tree (`GIR/`) was full-tree diffed against `GIR_Reference/GIR`
(2026-07-06): `utils/ir_utils.py` (envlight sampling, direction generation),
`utils/sh_utils.py`, `utils/graphics_utils.py` and the CUDA rasterizer
submodules are **byte-identical**. Every engine addition below is flag-gated
and verified inert at its default, so the run `lego_baseline_no_prior`
executes the reference optimization path line-for-line: same losses, same
gradient flow, same densification/reset/optimizer logic. The added prior
losses are still *computed* for logging under `--exclude_prior_loss`
([train.py:760-775](GIR/train.py#L760-L775)) — that is how the baseline's
albedo/normal error curves appear in the comparison reports — but they
never enter the objective.

### 0.4 Intentional deviations (applied to ALL runs equally)

Two deviations from the paper's exact protocol, kept identical across
baseline and prior runs so all comparisons remain internal:

* **`-r 2`** — training and evaluation at 400×400 instead of the native
  800×800 (cost: the r2 batch already takes multiple days on a 16 GB GPU;
  r1 would quadruple pixel work).
* **`--max_gaussians 1_500_000`** — the paper densifies without limit; the
  cap is OOM insurance that only binds on runaway densification (it gates
  *growth* during densification and never prunes). In practice it never
  bound: the r2 baseline settles at 133 k gaussians, prior runs at ~1.3 M.

### 0.5 Non-gated engine fixes (do not change training numerics)

* `GaussianModel.restore()` now loads the envlight state dict from
  checkpoints. The reference **saved** the envlight in checkpoints but
  never loaded it back on resume, so any `--start_checkpoint` run silently
  reinitialized the light while keeping the trained materials — harmless
  for the reference (which never resumes), fatal for our warm-up-buffer and
  crash-resume workflows. Fixed in `restore()`
  ([gaussian_model.py:116](GIR/scene/gaussian_model.py#L116)).
* Occlusion-grid and diffuse-light computation are chunked
  (`gaussian_model.py`, 4096-point chunks) — numerically identical output,
  removes the >1 M-gaussian OOM in `get_diffuse_occ`/`compute_color`.
* The Phase-2 normal map is rasterized **with gradients** every iteration
  (reference: no-grad visualization every 1000 iters). In a baseline run
  nothing consumes it, so gradients are unaffected (extra compute only); in
  prior runs it is what the Phase-2 normal loss backpropagates through.

## 1. The three-phase training schedule

GIR already has a two-boundary schedule: iterations ≤ 5 k render a plain
view-dependent radiance (geometry warm-up), iterations in (5 k, 30 k] render
the PBR model with direct environment light only, and iterations > 30 k
enable the full model (per-gaussian occlusion + baked SH indirect light). We
keep the machinery and boundaries, and turn the stages into an explicit
curriculum by *what is supervised* in each:

* **Phase 1 — geometry (iter ≤ 5 000).** Densification places gaussians.
  With `--albedo_geometry_warmup` (our flag, all prior runs) the warm-up
  render outputs the flat per-gaussian **albedo** instead of view-dependent
  radiance, and the photometric loss is **replaced** by the albedo prior in
  `direct` mode (Huber + DSSIM, see §2.2) against the GT albedo composited
  over the same random background ([train.py:686-703](GIR/train.py#L686-L703)).
  Rationale: PBR albedo is view-independent, so geometry can be fitted to a
  shading-free target from the start; densification then allocates gaussians
  with sensible base colors instead of colors polluted by baked shading.
* **Phase 2 — normal alignment (5 000 < iter ≤ 30 000).** The
  geometry-derived shading normal (smallest-eigenvector direction of the
  gaussian covariance) is rasterized differentiably every iteration and
  supervised with the GT normal prior: masked mean of `1 − cos(n_pred,
  n_gt)` in world space, weight `lambda_normal_gt` (0.8 GT / 0.4 diffusion),
  plus the reference edge-aware smoothness at reduced strength
  ([train.py:832-862](GIR/train.py#L832-L862)). This backpropagates into
  gaussian rotation and scaling — surface orientation locks in **before**
  any material decomposition exists. The albedo warm-up loss from Phase 1
  continues. For the baseline this whole block is skipped
  (`exclude_prior_loss`) to keep it exact.
* **Phase 3 — full PBR (iter > 30 000).** GIR's full decomposition
  (albedo, metallic, roughness, envmap, occlusion, indirect) optimized by
  the photometric loss, the reference regularizers (modified per §3), and
  all prior losses (§2) under the weight scheduler (§2.6).

Because Phases 1–2 depend only on a small parameter subset, their stage-2
checkpoint is cached in `outputs/warmup_buffer/` keyed by a config
fingerprint, and any Phase-3-only variant resumes from it, skipping the
first half of the schedule (30 k of 60 k iters; `run_experiments.py`, §8).

## 2. Prior losses (Phase 3)

All prior losses compare **rasterized full-image property maps** (the same
differentiable rasterization path as the RGB render) against per-view GT
images, restricted to the object silhouette: the mask is the non-zero-GT
foreground intersected with the dataset alpha mask
(`_prior_mask`, [loss_utils.py:129-137](GIR/utils/loss_utils.py#L129-L137)).
The alpha intersection is critical for diffusion priors, whose background
pixels contain hallucinated values. Rasterizing (instead of comparing
per-gaussian attributes) means the priors see exactly what the camera sees —
occlusion, blending and silhouettes included — at the price of prior
gradients also flowing into geometry (see §2.8 and the geometry LR anneal,
§4.1, which manage that channel).

All dense prior terms use the **Huber** loss (`huber_loss`,
[loss_utils.py:102-121](GIR/utils/loss_utils.py#L102-L121)): quadratic below
`delta = 0.2`, linear (slope 1, L1-matched) beyond — robust to the outlier
pixels diffusion priors produce.

### 2.1 Albedo prior — `lambda_albedo_gt 0.25`, mode selectable

The core difficulty is the **albedo–lighting ambiguity**: a per-channel gain
can move freely between albedo and envmap without changing the rendered
image ($a' = s \cdot a$, $E' = E / s$ renders identically). A *direct*
(absolute) albedo loss resolves the scale but transfers every local error of
the prior — including per-view exposure/white-balance jitter and baked
shading of diffusion priors — straight into the material, and (because the
priors are rasterized) into geometry. We therefore supervise structure
through an invariant loss, with $A$ the rasterized albedo, $\tilde A$
the per-view prior image, and $M$ the foreground mask:

* **`zncc`** (scale-and-shift-invariant / Pearson). Standardize both images
  per channel over the foreground and Huber-compare:

  $$\hat X_c = \frac{X_c - \mu_M(X_c)}{\sqrt{\sigma^2_M(X_c) + \varepsilon}},
  \qquad
  \mathcal{L}_{zncc} = \mathrm{Huber}_{\delta}\big(\hat A, \hat{\tilde A}\big)_M ,$$

  where $\mu_M/\sigma^2_M$ are the masked per-channel mean/variance
  ([loss_utils.py:289-305](GIR/utils/loss_utils.py#L289-L305)). The loss is
  invariant to any per-channel *affine* transform (gain + bias) of either
  input — exactly the global degrees of freedom the decomposition cannot
  determine and that a diffusion prior gets wrong per view (exposure /
  white balance). Anything beyond a global gain — spatially varying baked
  shading in the prior, wrong texture — is still penalized. This is the mode
  used in the final poster evaluation configurations (`gt_zncc_zncc_neu` and
  `diff_zncc_zncc`), as it yielded the best quantitative and qualitative
  material decomposition.

**What we also tried:**
* **`zncc_grad`** (gradient-domain ZNCC): Finite-differences both images first before ZNCC to remove smooth spatial fields (like baked illumination). While highly invariant, it permitted larger per-channel offsets that drifted without a strong absolute anchor.
* Other implemented modes (e.g. `lstsq`, `log_chroma`, `gradient`, `zncc_local`, `ssim_struct`, `si_ema`) are documented inline in `albedo_prior_loss` but are inactive.
* **Note on Warm-up Mode:** During the Phase 1/2 geometry warm-up, the absolute target is required (as no environment light is yet modeled), so we use a **`direct`** loss mode (`(1−λ_dssim)·Huber + λ_dssim·(1−SSIM)` on raw values).

### 2.2 Albedo anchor — `--albedo_anchor_weight` (0.0 in final configs)

We experimented with adding a weak **absolute** term on top of the invariant mode to pin the scale and prevent drift:
```
loss_prior += albedo_anchor_weight · Huber(A_render, A_gt, δ=0.2, mask)
```
Although anchors of 0.05 and 0.15 were tested to prevent drift under `zncc_grad`, they forced exposure mismatches and shading errors of the priors into the material. The final configs set this weight to `0.0` (no anchor), relying purely on the scale-invariant ZNCC.

### 2.3 Normal prior — `lambda_normal_gt 0.8` (GT) / 0.4 (diffusion)

Masked mean of `1 − cos(n_pred, n_gt)`
([train.py:800-817](GIR/train.py#L800-L817)). The rendered normal map
(encoded in [0,1]) is decoded to [−1,1]; the GT normal is converted to world
space when needed (`decode_normal_to_world` — for DiffusionRenderer priors
this is the camera-space **OpenGL** convention: flip Y/Z in camera axes,
then rotate by the camera-to-world rotation; verified at 22.9° mean angular
error against the lego GT, vs ~90° for a wrong convention). Active in both
Phase 2 (flat weight) and Phase 3 (scheduler-scaled; an optional
`--lambda_normal_third_stage_scale` can down-weight it in Phase 3
independently — 1.0, i.e. inert, in all final configs).

### 2.4 Metallic and roughness priors

Huber against the GT map, masked, like the others
([train.py:819-829](GIR/train.py#L819-L829)):

* **GT runs**: `lambda_metallic_gt 0.15` against **all-zero** maps
  (`metallic_simulated_zero`; lego is a dielectric scene). This is
  deliberately strong — try_9's forensics showed that with loose metallic
  the optimizer builds a rough-metallic "pseudo-bounce" energy channel
  (26 % of gaussians metallic > 0.5) that corrupts the decomposition.
  No roughness prior (`roughness_gt_dir ""` → the term never engages).
* **Diffusion runs**: `lambda_metallic_gt 0.05` against
  `metallic_video` and `lambda_roughness_gt 0.05` against `roughness_video`
  (the video-consistent DiffusionRenderer outputs) — weak, because these
  priors are themselves unreliable.

A weight is auto-forced to 0 whenever its GT folder is empty, so "no prior"
is expressed by data absence, not by remembering to zero a flag.

### 2.5 Prior weight scheduler — `--use_prior_weight_scheduler`

All Phase-3 prior terms (and `reg_hdr`, §3.2) are multiplied by a schedule
`s(t)` ([train.py:704-729](GIR/train.py#L704-L729)): linear ramp 0 → 1 over
the first `prior_weight_scheduler_ratio = 0.15` of Phase 3 (4.5 k of 30 k
iters), then linear interpolation from 1 to `prior_weight_final_ratio = 1.0`
— i.e. **hold at full strength**. The ramp lets the fresh PBR decomposition
settle before the priors pull on it; the hold (vs the earlier decay-to-0.5)
keeps the priors constraining the late phase, where the model otherwise
re-bakes the training light (verified in try_5/6).

### 2.6 Experimental features not in the final configs

We implemented several mechanisms during exploration that are disabled (set to 0.0/off/inert) in the final configuration:
* **Prior-to-geometry gradient scaling (`--prior_geom_grad_scale 1.0`)**: Intended to scale or block geometry gradients flowing through the prior rasters to prevent multi-view inconsistencies from warping geometry. Kept at default 1.0 (inert) as ZNCC and geometry LR annealing were sufficient.
* **Envmap mean penalty (`--reg_env_mean_weight 0.0`)**: Intended to force energy out of the training envmap into the LLI bounce term by penalizing envmap mean radiance. Retired as it degraded envmap structure and lowered relight performance.
* **Disable Stage-3 Opacity Reset (`--disable_reset_third_stage`)**: An option to skip opacity resets during Phase 3, left off (resets run normally every 3k iterations).

## 3. Regularizer changes

### 3.1 `tv_reduction_factor 0.75` (baseline: 1.0)

([train.py:520-529](GIR/train.py#L520-L529), applied at
[train.py:731-735](GIR/train.py#L731-L735).) The reference regularizes every
material map toward smoothness: albedo TV × 0.1, normal edge-aware
smoothness × 0.01, metallic and roughness TV × `reg_material_weight`. These
are *substitute priors* — with nothing else constraining the materials,
smoothness is the reference's way of suppressing noise and shading residue
in the maps. Once a property has a real per-view GT prior, the artificial
smoothness competes with it: TV actively fights legitimate high-frequency
prior detail (texture edges the zncc losses are trying to place). We
therefore scale each property's smoothness term to 75 % **only if that
property has a GT prior**; properties without one keep full TV (e.g.
roughness in the GT runs, whose `roughness_gt_dir` is empty). 0.75 rather
than something more aggressive: earlier batches showed removing TV entirely
lets prior noise through (the priors are per-view and imperfect; a little
smoothness still helps where views disagree).

### 3.2 `reg_hdr_weight 0.001` (baseline: 0.1) — and WHY they differ

The envmap "neutrality" regularizer is a pure **desaturation** penalty
(`regularizer_loss`, [loss_utils.py:80-82](GIR/utils/loss_utils.py#L80-L82)):

```
white = mean_c(base);   loss = mean |base − white|
```

i.e. the per-texel distance of the cubemap from its own grayscale — zero iff
the envmap is perfectly colourless. It is scheduler-scaled in Phase 3 like
the priors.

**Why the baseline needs it strong (0.1).** Without any albedo prior, the
albedo–light **colour** split is completely unconstrained: multiply the
albedo by any per-channel gain and divide the envmap by the same gain, and
every rendered pixel is unchanged. The reference resolves this with the
heuristic assumption *"illumination is approximately white"*: the strong
desaturation penalty makes envmap tint expensive, so the optimizer pushes
colour into the albedo. This is the reference's only mechanism against the
colour ambiguity — remove it and the baseline's decomposition can tint-swap
arbitrarily. That is why the paper's launch line carries `0.1` and why our
baseline run keeps it (paper-exactness).

**Why prior runs lower it to 0.001 (~off).** Our GT albedo prior pins the albedo's colour *directly from data*, which resolves the colour ambiguity from the material side — the white-light heuristic becomes redundant. Worse, it becomes actively wrong: our training light is a **sunset** (strongly tinted), so any meaningful desaturation pressure pushes the true orange tint out of the envmap and into the materials as an inverse-blue cast. Measured: at 0.01 the learned envmap is essentially black-and-white; at 0.001 it keeps the sunset tint. We keep 0.001 rather than 0 as a mild numerical stabilizer. Note the caveat from try_7: the *relight metrics* were insensitive to 0.001 vs 0.01 (< 0.15 dB everywhere — at relight the training envmap is swapped out anyway, and the albedo tint error partially cancels); the setting matters for the **decomposition quality** (envmap fidelity, albedo colour), which is a deliverable in its own right.

## 4. Optimizer / geometry changes

* **Third-stage geometry LR anneal — `reduce_geo_lr_third_stage 0.05`,
  `geo_lr_final_iter 66_000`** (`set_geo_lr_schedule` / `_geo_lr_factor`,
  [gaussian_model.py:616-651](GIR/scene/gaussian_model.py#L616-L651)): the
  learning rates of xyz, scaling and rotation are cosine-eased from 1.0 down
  to 5 % over [30 k, 66 k] and held there (on top of the reference xyz
  exponential decay). The window deliberately extends past the 60 k training
  end so the factor reached at the final iteration is ≈ 0.11, matching the
  anneal shape validated on the r4 schedule. Why: once Phase 2 has aligned
  the geometry to GT normals, cheap geometric "explanations" of lighting
  (surface micro-warping that bakes shading) must be throttled while the
  materials/light decompose. 1.0 = inert (baseline). Known cost: part of the
  ~2 dB novel-view PSNR gap of prior runs (open problem; relaxing toward 0.3
  is the queued A/B).
* **`--max_gaussians 1_500_000`**: hard cap enforced during densification
  only (it never prunes; inert after `densify_until_iter`). Batch-wide,
  baseline included (§0).

## 5. Renderer change: light-linear indirect illumination (LLI)

`--light_linear_indirect`, the one **rendering-model** change
([gaussian_model.py:387-443](GIR/scene/gaussian_model.py#L387-L443)); Phase 3
only.

**Problem.** GIR's transport has two baked terms that break relighting:
(i) diffuse: each gaussian holds binary occlusion values for its sampled
hemisphere directions, and occluded directions contribute **zero** light —
there is no bounce/interreflection; (ii) specular "indirect": occluded
reflection directions read a per-gaussian SH **radiance** field that is
frozen — it does not change when the light is swapped. Blender's GT is
path-traced with full interreflection, so GIR systematically transports only
~0.6–0.9× of the energy (measured on relight pairs in try_6). During
training the learned envmap silently absorbs the deficit (trains too
bright); at relight the GT HDRI arrives at native intensity and every render
is uniformly too dark — the "relight gain" (fitted brightness ratio
render→GT, §7) sat at 1.2–1.5 in every r4 GT run.

**Mechanism.** Reparameterize both baked terms as *reflectance ×
mean radiance of the current envmap*:

* `env_mean = envlight.base.detach().mean()` per channel (a 3-vector, read
  from whatever envmap is currently loaded);
* the specular SH indirect term becomes `I_SH(reflect_dir) · env_mean`;
* occluded diffuse directions receive a per-gaussian **bounce** term
  `b = clamp(SH(features, n) + 0.5) · env_mean` instead of zero:
  `chunk_light = (1−occ)·E(ω) + occ·b`.

During training this is a benign reparameterization — `env_mean` is just a
scalar factor the SH coefficients absorb. At **relight** the envmap is
swapped, `env_mean` is recomputed from the new HDRI, and all bounce energy
rescales linearly with the new light instead of staying frozen at
training-light levels. No new parameters (the existing per-gaussian SH
features double as bounce reflectance), so densification bookkeeping is
untouched.

**The detach (try_8 fix).** `env_mean` must be a pure read-out, never a
gradient path into the envmap. In the first implementation (try_7) the
bounce gradient flowed into the mean and the optimizer inflated it by
pumping a few ultra-bright texels (mean 3.3× the GT sunset's) while the rest
of the map went dark — envmap log-correlation collapsed and, because the
bounce was calibrated against the inflated mean, it under-scaled at relight.
With the detach, the envmap receives gradients only through the direct
terms (like the healthy no-LLI runs) and the SH reflectance alone calibrates
the bounce. Verified effect at paper scale (try_10): relight gain 1.002 with
per-HDRI gains 0.89–1.14, envmap mean ratio falling to 1.14.

## 6. Data and priors

The synthetic dataset (`data/datasets_with_priors/lego`) is Blender-rendered
with per-view GT buffers: `albedo_gt` and `normal_gt` (**world-space**),
relight GT renders of the test views under six unseen HDRIs (fireplace,
night, snow, city, courtyard, forest), and the base training light
(`hdris/sunset.hdr`) for envmap evaluation. `metallic_simulated_zero` is an
all-zero metallic map set (§2.4).

The diffusion arm replaces the GT folders with **DiffusionRenderer**
(NVIDIA) inverse-rendering outputs: `albedo`, `normal` (image-wise;
camera-space OpenGL — converted per §2.3), `metallic_video`,
`roughness_video` (video-consistent variants). These priors are multi-view
**inconsistent** (per-view exposure/white-balance jitter, hallucinated
backgrounds) — the reason the invariant loss family (§2.1) and the alpha
masking exist at all.

## 7. Evaluation additions (all logging-only)

The reference repo evaluates PSNR on renders. Our `periodic_evaluation`
([train.py:185+](GIR/train.py#L185)) adds, every `eval_interval = 5000`
iters (relight every 10 000):

* **Aligned PSNR + fitted gain** (`scale_aligned_psnr`,
  [train.py:85-100](GIR/train.py#L85-L100)): fits a single global gain
  `g* = ⟨I_render, I_gt⟩ / ⟨I_render, I_render⟩` before computing PSNR, and
  logs `g*` itself. Convention: for renders `g* > 1` = too dark (needs
  boosting); for albedo `g* < 1` = too bright. Raw PSNR measures
  structure × energy calibration (deployment-relevant — no GT to align
  against in practice); aligned PSNR isolates structure; the gain isolates
  calibration. Success criterion for relighting: **gain → 1.0 at
  aligned-level raw PSNR**.
* **Relight eval**: test cameras re-rendered under each unseen HDRI
  (envmap swapped, `env_mean` recomputed) vs the Blender GT —
  `relight_<hdri>_psnr / _psnr_aligned / _gain / _ssim` and their means.
  `--relight_max_views 24` caps it to 24 evenly-spaced, deterministic test
  cameras (±0.1–0.2 dB vs all 200, ~10× faster).
* **Decomposition metrics**: albedo PSNR raw/aligned + `test_albedo_gain`,
  normal mean angular error, metallic/roughness MAE.
* **Envmap metrics** vs the GT base HDRI: log-space PSNR after global gain
  alignment, relative L1, and `envmap_mean_ratio` (learned mean / GT mean) —
  the bookkeeping signal for where absorbed energy hides. (With LLI active
  it over-predicts the relight gain — treat as a ranking signal, not an
  exact factor.)
* `metrics_log.json` per run, a per-run training PDF, and a batch
  comparison PDF/CSV (`run_experiments.py`) including an "Energy
  Calibration" page that plots relight gain, albedo gain and envmap ratio
  side by side.

## 8. Infrastructure (not part of the method)

`GIR/run_experiments.py`: declarative batch runner (COMMON + per-variant
overrides → full `train.py` command lines), stage-2 **warm-up buffer**
(fingerprint of every Phase-1/2-relevant parameter; HIT → Phase 3 only),
`--only` / `--resume` / `--resume-iter` / `--set` repair tooling, a guard
that skips runs already owning checkpoints, archived script copy per batch,
and the comparison report. Training subprocesses get
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (fixes the caching-
allocator fragmentation OOM at >1 M gaussians). `train.py --start_checkpoint`
resumes preserve metric history up to and including the resumed iteration.

## 9. Final run configurations (try_10 batch, r2 / 60 k)

| parameter | baseline_no_prior | gt_zncc_zncc_neu | diff_zncc_zncc |
|---|---|---|---|
| prior source | — (logged only) | GT (world-space) | DiffusionRenderer |
| `albedo_prior_mode` | — | zncc | zncc |
| `warmup_albedo_prior_mode` | — | direct | direct |
| `albedo_geometry_warmup` | off | on | on |
| `lambda_albedo_gt` | 0 | 0.25 | 0.25 |
| `albedo_anchor_weight` | 0 | 0 | 0 |
| `lambda_normal_gt` | 0 | 0.8 | 0.4 |
| `lambda_metallic_gt` | 0 | 0.15 (vs zeros) | 0.05 (metallic_video) |
| `lambda_roughness_gt` | 0 | — (no GT) | 0.05 (roughness_video) |
| scheduler (ratio/final) | — | 0.15 / 1.0 | 0.15 / 1.0 |
| `huber_delta` | — | 0.2 | 0.2 |
| `tv_reduction_factor` | 1.0 | 0.75 | 0.75 |
| `reg_hdr_weight` | 0.1 | 0.001 | 0.001 |
| `reg_material_weight` | 0.05 | 0.05 | 0.05 |
| geo LR anneal (final ×, until) | off | 0.05, 66 k | 0.05, 66 k |
| `light_linear_indirect` | off | on | on |
| `normal_camera_convention` | — | — (world GT) | opengl |
| shared | `-r 2`, 60 k iters, stages 5 k/30 k, densify 500→45 k @100, reset 3 k, `lambda_dssim 0.4`, cap 1.5 M, `--eval --random_background --hdr_rotation`, relight 24 views | | |

---
