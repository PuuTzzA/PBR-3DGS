# Prior-Guided Gaussian Inverse Rendering — Change Reference & Method

*Status: 2026-07-14 (try_10 / final runs). Companion document to
[FINDINGS.md](FINDINGS.md), which records the experimental history and the
evidence behind every choice below. This document records **what** we changed
relative to baseline GIR, **why**, and **how it works** — precisely enough to
be cited later — followed by a paper-style method section (Part II).*

---

# Part I — Complete inventory of changes vs. baseline GIR

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
never enter the objective. The full per-file delta inventory is in
FINDINGS.md §2.5.

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
first half of the schedule (30 k of 60 k iters; `run_experiments.py`, §9).

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
gradients also flowing into geometry (see §2.6 and the geometry LR anneal,
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
priors are rasterized) into geometry. We therefore split the supervision:
**structure through an invariant loss** (this section) and **global scale
through a weak separate anchor** (§2.2). Modes implemented in
`albedo_prior_loss` ([loss_utils.py:178-418](GIR/utils/loss_utils.py#L178));
the two used in final configs, with $A$ the rasterized albedo, $\tilde A$
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
  shading in the prior, wrong texture — is still penalized. This is what the
  headline config uses.
 * **`zncc_grad`** (gradient-domain ZNCC). Finite-difference both images
  first, then maximize the masked Pearson correlation of the gradient
  images per channel and direction
  ([loss_utils.py:338-362](GIR/utils/loss_utils.py#L338-L362)):

  $$\mathcal{L}_{zncc\text{-}grad} = \sum_{d \in \{x,y\}} \frac{1}{3}\sum_{c}
  \Big(1 - \mathrm{corr}_{M_d}\big(\partial_d A_c,\; \partial_d \tilde A_c\big)\Big).$$

  Differentiating first removes *any smooth field*, not just a global one:
  low-frequency baked shading, soft shadows and brightness ramps in the
  prior all vanish from the target, and only the **placement and relative
  strength of texture edges** is supervised. The gradient masks $M_d$
  require both finite-difference neighbours to be foreground, so the
  object–background silhouette step never counts as a GT edge. This mode 
  was used in earlier batches (try_5) to handle multi-view inconsistency 
  of diffusion priors, but plain `zncc` proved superior in the final 
  paper-scale runs.

The invariance hierarchy matters: more invariance = more robustness to prior
errors, but also more residual freedom the anchor has to close (§2.2) —
`zncc` leaves one gain+bias per channel free; `zncc_grad` additionally
leaves *any* smooth multiplicative field free. That trade-off is visible in
the results: with the same 0.05 anchor, `zncc` pins the albedo scale
(fitted gain ≈ 0.98) while `zncc_grad` drifts (0.815 at paper scale) until the
anchor is raised to 0.15.

(Other implemented modes — `direct`, `lstsq`, `log_chroma`, `gradient`,
`zncc_local`, `ssim_struct`, `si_ema` — are documented inline in the same
function and selectable via `--albedo_prior_mode`; none are in the final
configs. `direct` = `(1−λ_dssim)·Huber + λ_dssim·(1−SSIM)` on raw values and
is what the Phase-1/2 geometry warm-up uses via
`--warmup_albedo_prior_mode direct` — during warm-up the *absolute* target
is wanted, since there is no envmap yet to be ambiguous against.)

### 2.2 Albedo anchor — `--albedo_anchor_weight` (0.0)

A weak **absolute** term added on top of the invariant mode
([train.py:791-798](GIR/train.py#L791-L798)):

```
loss_prior += albedo_anchor_weight · Huber(A_render, A_gt, δ=0.2, mask)
```

Why: the invariant losses deliberately leave global per-channel scale free,
and the optimizer *uses* that freedom — in try_6 the unanchored albedo
drifted ~1.2× too bright and blue-tinted, with the envmap compensating (the
inverse of the sunset training light). However, we found that without it we 
get better results: in the final paper-scale runs (try_10) using the `zncc` 
mode, setting the anchor weight to **0** proved superior as `zncc` pins 
the albedo scale sufficiently while avoiding the transfer of per-view 
shading errors that an absolute term would force. Earlier `zncc_grad` 
attempts required an anchor of 0.15 to hold the scale, but at the cost 
of raw relight quality. The diffusion arm likewise uses **no anchor** 
(weight 0).

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

### 2.6 Prior→geometry gradient scale — `--prior_geom_grad_scale` (available, off)

A gradient hook that scales (0 = blocks) the geometry gradients
(means/scales/rotations/opacity) flowing through the **albedo/material prior
rasters only**, leaving photometric and normal-prior geometry gradients
untouched (`gaussian_renderer/__init__.py`). Built for the try_5 failure
mode (per-view-inconsistent diffusion albedo restructuring geometry through
the raster). Default 1.0 = identity; **not** in the final configs — the
geometry LR anneal (§4.1) plus `zncc` proved sufficient.

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

**Why prior runs lower it to 0.001 (~off).** Our GT albedo prior (plus
anchor) pins the albedo's colour *directly from data*, which resolves the
colour ambiguity from the material side — the white-light heuristic becomes
redundant. Worse, it becomes actively wrong: our training light is a
**sunset** (strongly tinted), so any meaningful desaturation pressure pushes
the true orange tint out of the envmap and into the materials as an
inverse-blue cast. Measured: at 0.01 the learned envmap is essentially
black-and-white; at 0.001 it keeps the sunset tint. We keep 0.001 rather
than 0 as a mild numerical stabilizer. Note the honest caveat from try_7:
the *relight metrics* were insensitive to 0.001 vs 0.01 (< 0.15 dB
everywhere — at relight the training envmap is swapped out anyway, and the
albedo tint error partially cancels); the setting matters for the
**decomposition quality** (envmap fidelity, albedo colour), which is a
deliverable in its own right.

### 3.3 `--reg_env_mean_weight` (added for try_10, tried at 0.005, RETIRED)

A scheduler-scaled penalty on the learned envmap's mean radiance
([train.py:737-750](GIR/train.py#L737-L750)), intended to push the
transport-deficit energy out of the (non-transferable) training envmap into
the light-linear bounce (§5). try_10 falsified the mechanism: with the
penalty active the envmap mean ratio *rose* (1.49 → 2.00) while envmap
structure degraded (logPSNR 26.7 vs 28.9 without it), and raw relight ended
below the penalty-free headline. Default 0.0 = inert; not in any final
config. Kept in the code as a documented negative result.

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
* **`--disable_reset_third_stage`** (available, off in all final configs):
  skips opacity resets after Phase 3 starts; built for a reset A/B in
  earlier batches.

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
render→GT, §8) sat at 1.2–1.5 in every r4 GT run.

**Mechanism.** Reparameterize both baked terms as *reflectance ×
mean radiance of the current envmap*:

* `env_mean = envlight.base.detach().mean()` per channel (a 3-vector, read
  from whatever environment is currently loaded);
* the specular SH indirect term becomes $I_{SH}(\hat{r}) \cdot \mu_E$;
* occluded diffuse directions receive a per-gaussian **bounce** term
  $b = (\text{SH}(f, n) + 0.5)^+ \cdot \mu_E$ instead of zero:
  `chunk_light = (1−occ)·E(ω) + occ·b`.

**Why reflectance × mean radiance.** Standard GIR stores indirect radiance
as an absolute value tied to the training illumination. When relighting
under a novel environment map (e.g., a dark night or a bright sunny day),
this frozen "glow" does not rescale, leading to systematically dark or
blown-out interreflections. By reparameterizing the baked SH field as a
reflectance (unitless) that scales with the environment's mean radiance
$\mu_E$, we ensure the bounce energy adapts to the intensity of the
swapped HDRI.

**Notation and the $+0.5$ offset.** We use $I_{SH}(\omega)$ as a shorthand 
for the indirect radiance field $(\text{SH}(f, \omega) + 0.5)^+$. The 
$+0.5$ offset follows the standard Gaussian Splatting convention where 
the SH basis functions represent deltas around a neutral 0.5 baseline (DC 
component). This ensures that the learned reflectance remains in a stable 
range and that the final radiance is non-negative.

**SH Details & LLI.** Each gaussian stores its indirect/bounce field as 
spherical harmonic (SH) features. For the active degree $D=3$, this 
represents **48 dimensions** per gaussian (3 channels $\times$ 16 
coefficients). We derive two separate values from this single SH field 
to handle occlusions: **one for indirect reflections** (specular 
indirect radiance $I_{SH}(\text{reflect\_dir}) \cdot env\_mean$) 
and **one for occluded direct reflections** (diffuse bounce 
term $b$, evaluated at the shading normal $n$). The step is called 
**LLI (Light-Linear Indirect)** because it reparameterizes baked 
radiance (which is frozen) as a reflectance factor that scales 
**linearly** with the illumination ($env\_mean$) — ensuring 
interreflections adapt when the environment light is swapped.

During training this is a benign reparameterization — `env_mean` is just a
scalar factor the SH coefficients absorb. At **relight** the envmap is
swapped, `env_mean` is recomputed from the new HDRI, and all bounce energy
rescales linearly with the new light instead of staying frozen at
training-light levels. No new parameters (the existing per-gaussian SH
features double as bounce reflectance), so densification bookkeeping is
untouched.

## 6. Detailed Shading and Rasterization Pipeline

This section provides a formal, step-by-step breakdown of how the color of a single surface patch (represented by a 3D Gaussian) is rendered in our implementation.

### 6.1 Gaussian Attributes and Geometry
Each Gaussian $i$ is defined by its position $\mu_i$, anisotropic scale $s_i$, rotation $q_i$ (quaternion), and opacity $\alpha_i$. For inverse rendering, we extend these with per-Gaussian PBR material properties:
*   **Albedo** $a_i \in [0,1]^3$: The base color (diffuse reflectance).
*   **Metallic** $m_i \in [0,1]$ and **Roughness** $r_i \in [0,1]$.
*   **SH Features** $f_i \in \mathbb{R}^{48}$: Spherical Harmonic coefficients (degree 3, 16 per RGB channel) representing a baked radiance/reflectance field.

The **shading normal** $n_i$ is not an independent parameter; it is derived from the Gaussian's geometry as the direction of the smallest eigenvector of its covariance matrix $\Sigma_i = R_i S_i S_i^T R_i^T$. This ensures that the normal is intrinsically linked to the thin surface direction.

### 6.2 Visibility and Occlusion
Visibility is handled via a 128³ voxel occupancy grid rasterized from the Gaussians.
*   **Diffuse Visibility**: We pre-calculate binary occlusion $o_j \in \{0,1\}$ for $N=128$ directions $\omega_j$ sampled over the hemisphere defined by $n_i$.
*   **Specular Visibility**: We calculate a scalar occlusion $o \in [0,1]$ for the perfect reflection direction $\hat{r}$ (derived from the view direction $v$ and normal $n_i$).

### 6.3 PBR Shading Equation
The total outgoing radiance $C_i$ from Gaussian $i$ is the sum of diffuse and specular components:
$$C_i = (1 - m_i) \cdot a_i \cdot L_d + \rho_s \cdot L_s$$

#### 6.3.1 Diffuse Component ($L_d$)
The diffuse term averages the environment light over the hemisphere, accounting for self-occlusion and interreflections (LLI):
$$L_d = \frac{1}{N} \sum_{j=1}^{N} \left[ (1 - o_j) \cdot E(\omega_j) + o_j \cdot b \right]$$
*   **Direct Diffuse**: Unoccluded directions ($1-o_j$) sample the environment map $E$.
*   **Indirect Bounce ($b$)**: Occluded directions receive a bounce term derived from the SH field evaluated at the shading normal $n_i$:
    $$b = I_{SH}(n_i) \cdot \mu_E$$
    where $\mu_E$ is the mean radiance of the current environment light and $I_{SH}(\omega) = (\text{SH}(f_i, \omega) + 0.5)^+$.

#### 6.3.2 Specular Component ($L_s$)
The specular term uses the split-sum approximation to handle environment reflections:
$$L_s = (1 - o) \cdot E_r(\hat{r}, r_i) + o \cdot \left[ I_{SH}(\hat{r}) \cdot \mu_E \right]$$
*   **Direct Specular**: Unoccluded reflections sample the prefiltered environment map $E_r$ at the reflection direction $\hat{r}$ and roughness level $r_i$. **Crucially**, for a mirror ($r_i \approx 0$), this samples the high-resolution HDRI directly, preserving sharp details. The environment mean $\mu_E$ does NOT scale this term, ensuring mirrors work as expected.
*   **Indirect Specular**: Occluded reflections read the baked SH radiance field evaluated at the reflection direction $\hat{r}$, scaled by $\mu_E$ (LLI). $I_{SH}(\omega)$ is defined as $(\text{SH}(f_i, \omega) + 0.5)^+$.
*   **Specular Reflectance ($\rho_s$)**: Calculated via the split-sum BRDF:
    $$F_0 = 0.04(1 - m_i) + m_i \cdot a_i, \qquad \rho_s = F_0 \cdot A(n_i \cdot v, r_i) + B(n_i \cdot v, r_i)$$
    where $A$ and $B$ are pre-integrated LUT values.

### 6.4 The Single SH Field "Dual Evaluation"
A key detail of our LLI implementation is the derivation of two separate physical values from the single 48-dimensional SH feature vector $f_i$. By evaluating the same SH field at different directions, we extract:
1.  **Diffuse Bounce ($b$)**: Evaluated at the **normal** $n_i$. This represents the aggregate indirect energy arriving from the occluded hemisphere.
2.  **Specular Indirect Radiance**: Evaluated at the **reflection direction** $\hat{r}$. This represents the baked radiance reflected toward the viewer.

Both values are derived from the same underlying SH reflectance field and scale linearly with $\mu_E$, ensuring that interreflections adapt to light intensity changes while the high-frequency *direct* reflections from the environment map remain unscaled and sharp.

### 6.5 Alpha Blending and Rasterization
The final pixel color $C_{pixel}$ is computed by sorting all Gaussians overlapping the pixel by depth and performing front-to-back alpha blending:
$$C_{pixel} = \sum_{i \in \text{ray}} C_i \cdot (\alpha_i G_i(x)) \cdot \prod_{j=1}^{i-1} (1 - \alpha_j G_j(x))$$
where $G_i(x)$ is the 2D Gaussian evaluation at pixel coordinates $x$.

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

## 7. Data and priors

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

## 8. Evaluation additions (all logging-only)

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

## 9. Infrastructure (not part of the method)

`GIR/run_experiments.py`: declarative batch runner (COMMON + per-variant
overrides → full `train.py` command lines), stage-2 **warm-up buffer**
(fingerprint of every Phase-1/2-relevant parameter; HIT → Phase 3 only),
`--only` / `--resume` / `--resume-iter` / `--set` repair tooling, a guard
that skips runs already owning checkpoints, archived script copy per batch,
and the comparison report. Training subprocesses get
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (fixes the caching-
allocator fragmentation OOM at >1 M gaussians). `train.py --start_checkpoint`
resumes preserve metric history up to and including the resumed iteration.

## 10. Final run configurations (try_10 batch, r2 / 60 k)

| parameter | baseline_no_prior | gt_zncc_anchor0_lli2 | diff_zncc_lli2 |
|---|---|---|---|
| prior source | — (logged only) | GT (world-space) | DiffusionRenderer |
| `albedo_prior_mode` | — | zncc | zncc |
| `warmup_albedo_prior_mode` | — | direct | direct |
| `albedo_geometry_warmup` | off | on | on |
| `lambda_albedo_gt` | 0 | 0.25 | 0.25 |
| `albedo_anchor_weight` | 0 | **0.0** | 0 |
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

# Part II — Method (paper style)

## Prior-Guided Gaussian Inverse Rendering with Light-Linear Indirect Illumination

**Overview.** We build on GIR [2], which represents a scene as a set of 3D
Gaussians [1] with per-gaussian PBR attributes (albedo $a$, metallic $m$,
roughness $r$), a learned environment map $E$, per-gaussian binary
visibility, and a baked spherical-harmonics (SH) indirect-radiance field,
optimized end-to-end against posed images by differentiable rasterization.
Inverse rendering under a single unknown illumination is ill-posed — most
prominently through the *albedo–lighting ambiguity*, where per-channel gain
moves freely between albedo and light. We resolve it with dense per-view
material priors (ground-truth buffers on synthetic data; DiffusionRenderer
[6] estimates on real data), injected through *scale-invariant* losses with
a weak absolute anchor, on a curriculum that fixes geometry before
materials. We additionally identify a systematic energy deficit in GIR's
frozen indirect terms that breaks relighting, and repair it with a
*light-linear* reparameterization.

**Rendering model (preliminaries).** Following GIR, each gaussian is shaded
with a split-sum specular term [4,5]: $F_0 = 0.04(1-m) + m\,a$,
$\rho_s = F_0 A + B$ with $(A,B)$ from the pre-integrated BRDF LUT at
$(n\!\cdot\!v, r)$, and specular light mixed from a prefiltered environment
query and a per-gaussian baked SH radiance $I_{SH}$ gated by the reflection
visibility $o$. The diffuse term averages the environment over $N$
hemisphere directions $\omega_j$ with baked binary occlusion
$o_j\in\{0,1\}$:

$$L_d = \tfrac{1}{N}\textstyle\sum_j (1-o_j)\,E(\omega_j),\qquad
L_s = o\, I_{SH}(\hat r) + (1-o)\,E_r(\hat r; r).$$

**Three-phase curriculum.** Phase 1 (iter ≤ 5 k) fits geometry: with
*albedo geometry warm-up*, the rasterized flat albedo is fitted directly to
the albedo prior (robust Huber + DSSIM), replacing the photometric loss —
densification thus places gaussians against a shading-free target. Phase 2
(≤ 30 k) rasterizes the covariance-derived shading normal differentiably
each step and supervises it with the prior, $\mathcal{L}_n =
\langle 1 - \cos(n, \tilde n)\rangle_M$ (foreground mask $M$; camera-space
prior normals are rotated to world space), locking surface orientation
before any material exists. Phase 3 (> 30 k) optimizes the full
decomposition with all priors under a weight schedule $s(t)$ that ramps
linearly to 1 over the first 15 % of the phase and then holds — decaying
priors allowed late-phase light re-baking.

**Scale-invariant albedo supervision.** Absolute albedo losses import the
prior's per-view exposure and baked-shading errors into the material. We
instead supervise structure invariantly, in the spirit of scale-invariant
depth losses [7]. With $\hat x = (x-\mu_M(x))/\sigma_M(x)$ denoting
per-channel standardization over $M$:

$$\mathcal{L}_{zncc} = \mathrm{Huber}_\delta\!\big(\hat A, \hat{\tilde A}\big)_M,$$

which is invariant to any per-channel affine transform (gain + bias) of 
either input. While we also implemented a gradient-domain variant 
($\mathcal{L}_{zncc\text{-}grad}$) to handle smooth shading fields in 
the prior, plain $\mathcal{L}_{zncc}$ proved superior at paper scale 
when combined with the albedo anchor. A weak absolute **anchor**
$w_a\,\mathrm{Huber}_\delta(A,\tilde A)_M$ with $w_a < \lambda_{alb}$
(0.0 in the final configuration vs 0.25; we found that without it we 
get better results) re-pins the global scale without transferring 
local prior errors. Normal, metallic and roughness priors use masked robust losses;
on dielectric synthetic scenes the metallic prior (against zero) closes a
degenerate rough-metallic energy channel the optimizer otherwise exploits.
Prior gradients into geometry are throttled in Phase 3 by cosine-annealing
the geometry learning rates to 5 %.

**Light-linear indirect illumination (LLI).** GIR's occluded diffuse
directions receive zero light and its SH indirect radiance is frozen at
training-light levels, so the renderer transports systematically less
energy than path-traced references; the learned envmap absorbs the deficit
during training, and relit renders are uniformly too dark. We
reparameterize both baked terms as *reflectance × mean radiance*
$\mu_E = \mathrm{mean}(E)$ of the **currently loaded** environment
(gradient-detached):

$$L_d = \tfrac{1}{N}\textstyle\sum_j \big[(1-o_j)\,E(\omega_j) + o_j\, b\big],\quad
b = I_{SH}(n)\,\mu_E,\qquad
L_s = o\, I_{SH}(\hat r)\,\mu_E + (1-o)\,E_r(\hat r; r).$$

where $I_{SH}(\omega) = (\text{SH}(f, \omega) + 0.5)^+$. By scaling the 
baked terms by the current environment's mean radiance $\mu_E$, we 
transform them from absolute radiance into relative reflectances that 
dynamically respond to lighting changes.

Each gaussian stores its baked radiance field as 48-dimensional SH 
features (degree 3). We derive two separate values from this single SH 
field to handle occlusions by evaluating the same spherical harmonic 
function at different directions: **one for indirect reflections** 
(specular indirect light, evaluated at the reflection direction $\hat{r}$) 
and **one for occluded direct reflections** (diffuse bounce 
reflectance $b$, evaluated at the shading normal $n$). The name **LLI** 
(Light-Linear Indirect) refers to the reparameterization of baked 
radiance as a reflectance that scales linearly with the current 
environment's mean radiance $\mu_E$. Sharp reflections (mirrors) are 
preserved as they are handled by the *direct* specular term, which 
samples the high-resolution environment map directly and is not 
scaled by $\mu_E$. 
The detach is essential: allowing gradients into $\mu_E$
lets the optimizer inflate the mean via isolated bright texels, corrupting
both the envmap and the bounce calibration. LLI adds no parameters (the
existing SH features double as bounce reflectance).

**Objective.** With photometric loss $\mathcal{L}_{pho} =
0.6\,L_1 + 0.4\,(1-\mathrm{SSIM})$ and the (prior-reduced) TV/smoothness
regularizers $\mathcal{L}_{reg}$ of [2], Phase 3 minimizes

$$\mathcal{L} = \mathcal{L}_{pho} + \mathcal{L}_{reg}
+ s(t)\big[\lambda_{alb}\mathcal{L}_{alb} + w_a\mathcal{L}_{anchor}
+ \lambda_n\mathcal{L}_n + \lambda_m\mathcal{L}_m + \lambda_r\mathcal{L}_r\big].$$

**Evaluation protocol.** Besides raw PSNR under novel light, we report
scale-*aligned* PSNR (after a fitted global gain $g^\ast = \langle I,
\tilde I\rangle / \langle I, I\rangle$) and $g^\ast$ itself: raw couples
structure with energy calibration, aligned isolates structure, and the gain
isolates calibration ($g^\ast{>}1$: too dark). A decomposition is only
considered resolved when $g^\ast \to 1$ at aligned-level raw PSNR — a
criterion that unmasks configurations whose material errors accidentally
cancel their transport errors.

### References

[1] B. Kerbl, G. Kopanas, T. Leimkühler, G. Drettakis. *3D Gaussian
Splatting for Real-Time Radiance Field Rendering.* ACM TOG (SIGGRAPH) 2023.

[2] Y. Shi, Y. Wu, C. Wu, X. Liu, C. Zhao, H. Feng, J. Liu, L. Zhang,
J. Zhang, B. Zhou, E. Ding, J. Wang. *GIR: 3D Gaussian Inverse Rendering
for Relightable Scene Factorization.* arXiv:2312.05133, 2023.

[3] H. Jin, I. Liu, P. Xu, X. Zhang, S. Han, S. Bi, X. Zhou, Z. Xu, H. Su.
*TensoIR: Tensorial Inverse Rendering.* CVPR 2023. (Dataset protocol /
relighting benchmark style.)

[4] B. Karis. *Real Shading in Unreal Engine 4.* SIGGRAPH Courses, 2013.
(Split-sum approximation.)

[5] J. Munkberg, J. Hasselgren, T. Shen, J. Gao, W. Chen, A. Evans,
T. Müller, S. Fidler. *Extracting Triangular 3D Models, Materials, and
Lighting From Images.* CVPR 2022. (Split-sum in inverse rendering.)

[6] R. Liang et al. *DiffusionRenderer: Neural Inverse and Forward
Rendering with Video Diffusion Models.* NVIDIA, CVPR 2025. (Source of the
estimated albedo/normal/metallic/roughness priors.)

[7] R. Ranftl, K. Lasinger, D. Hafner, K. Schindler, V. Koltun. *Towards
Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-
Dataset Transfer.* IEEE TPAMI 2020. (Scale-/shift-invariant and
gradient-matching losses.)

[8] P. J. Huber. *Robust Estimation of a Location Parameter.* Annals of
Mathematical Statistics, 1964.
