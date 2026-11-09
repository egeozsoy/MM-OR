# MM-OR modernization: changes and verification

This change ports the MM2SG scene-graph pipeline from its original
2023 stack to a current one, without changing what the model computes:

| | original (baseline) | modernized |
|---|---|---|
| Python | 3.10 | 3.12 |
| torch | 2.0.1 + cu118 | 2.9.1 + **cu128** (covers sm_80 … sm_120 / Blackwell) |
| transformers | 4.31.0 | 4.57.6 |
| numpy | 1.26 | 2.2.6 |
| flash-attn | 2.3.4 | 2.8.3 (release wheel `cu12torch2.9cxx11abiTRUE-cp312`) |
| spconv | `spconv-cu117` 2.3.6 | `spconv-cu126` 2.3.8 |
| torch-scatter | 2.1.2 | **dropped** (native ATen `segment_csr`) |
| open3d | 0.17 | **dropped** (built-in PCD reader) |

The source tree is **bi-compatible**: it still imports and runs on the original stack. That
is what allows the original environment to serve as a parity oracle for every change below.

The panoptic-segmentation part is ported to the same torch in its own environment; §4 covers
its changes and verification (VPQ of all three released checkpoints within 0.16 of the
original stack).

**Result: the modernized stack reproduces the original pipeline's metrics on the full test
set (combined Macro-F1 0.691 vs 0.689, every take within ±0.007), and its training loss
tracks the original within run-to-run noise over 200 steps (see
[Verification](#3-verification)).**

---

## Installation

```bash
cd scene_graph_generation
./setup_env_modern.sh            # --help for options; --locked for the exact verified set
source .venv-mmor/bin/activate
```

Everything comes from prebuilt wheels, so no CUDA toolkit or compiler is needed. The
requirements are Linux x86_64, Python 3.12 and a driver for CUDA ≥ 12.8. The README's
[evaluation walkthrough](README.md#evaluating-the-published-checkpoints-step-by-step-modernized-stack)
covers data, checkpoints and the eval command.

| file | contents |
|---|---|
| `requirements-modern/torch.txt` | torch 2.9.1 / torchvision 0.24.1, installed from the cu128 index |
| `requirements.txt` | pinned top-level packages (transformers 4.57.6, peft 0.21.2, bitsandbytes 0.50.2, spconv-cu126 2.3.8, …) |
| `requirements-modern/lock.txt` | `pip freeze` of the environment the metrics below were produced with |
| `setup_env_modern.sh` | installs the above, plus the flash-attn 2.8.3 release wheel, LLaVA (editable, `--no-deps`) and the PTv3 init weights |

The original 2023 pins (torch 2.0.1, transformers 4.31) moved to `requirements-original.txt`;
use it only to rebuild the original stack. `LLaVA/pyproject.toml` deliberately declares no
versions.

**The install recipe is verified.**
- **Fresh install.** `setup_env_modern.sh` builds a fresh venv in a clean checkout of this
  branch with nothing but prebuilt wheels. Its package set matches `lock.txt` apart from a
  few transitive patch releases (certifi, tifffile, …).
- **Same scores.** The README eval command on the released stage-1 checkpoint scores the
  same as the reference environment, within run-to-run noise.
  - On the full test set, combined Macro-F1 was 0.687 on an RTX A6000, against 0.691 on an
    A40 in the reference environment (§3).
  - Over 20 batches, both environments land in the same 0.880–0.885 spread.
- **The PTv3 init file does not matter for evaluation.** Its content is random, but the
  checkpoint contains all 288 PointTransformerV3 parameters and buffers and overwrites
  them.

**Evaluation is not bit-reproducible run to run.** Repeating the same command in the same
venv on the same GPU gave Macro-F1 0.882, 0.882 and 0.885 over 20 batches, and even the two
0.882 runs differ in precision. Expect a few thousandths of variation on the full test set
as well. Compare runs by take-level numbers within about ±0.01, not by exact equality.

### Troubleshooting

| symptom | cause / fix |
|---|---|
| `AssertionError: No weights found at .../pointtransformerv3_weights.pth` | The PTv3 init file is required even for evaluation. Run `cd LLaVA && python -m llava.model.multimodal_projector.pointtransformerv3` once, on a GPU. |
| `RuntimeWarning: flash_attn not installed: PointTransformerV3 is falling back …` | flash-attn is missing, so results deviate from the verified numbers. Install the release wheel (see `setup_env_modern.sh`). Never `pip install flash-attn`: that compiles from source. |
| pip tries to downgrade torch, or installs transformers 4.31 / numpy 1.26 | You installed `requirements-original.txt` (or the `requirements.txt` of a pre-modernization checkout), or ran a plain `pip install -e LLaVA`. Use `setup_env_modern.sh`, or `requirements.txt` and `pip install --no-deps -e LLaVA`. |
| `ModuleNotFoundError: torch_scatter` / `open3d` | The code is from before the modernization; update the checkout. |
| `FileNotFoundError: .../mm_projector.bin` (or another loading error) | The checkpoint directory was renamed. The loader needs `llava` and `lora` in the directory name to load it as a LoRA adapter on top of `liuhaotian/llava-v1.5-7b`. |
| Metrics far below §3, quietly | Missing modality folders are skipped without an error (§2). Check that the per-timepoint audio, transcript and segmask folders exist for the test takes. |
| Temporal eval of a self-trained checkpoint crashes with a negative-dimension error after a few hundred samples, with garbled triplets (`anaesthetistmulticolor`, `Constra…`) in the output | Evaluate with `--load-4bit`. Merged into bf16, the QLoRA model can degenerate, and the temporal prompt (built from its own past predictions) then outgrows 2048 tokens. Starting stage 2 from a clean stage-1 checkpoint (the released one, or a retrained one with low 4-bit/bf16 divergence) avoids it. See §3, "Evaluating in 4-bit" and "Cause: the stage-1 checkpoint". |
| `CUDA driver version is insufficient` / `no kernel image is available` | The driver is too old for cu128 (`nvidia-smi` must report CUDA ≥ 12.8), or the GPU is older than the cu128 wheels support. Verified on A40 and RTX A6000 (sm_86). |

---

## 1. Code changes

The full rationale for each change is in its commit message. Summary:

### transformers 4.57 / torch 2.9 compatibility

- **`model_type` `llava` → `llava_mm2sg`.** transformers ships its own LLaVA under
  `"llava"`, so `AutoConfig.register` raised. Released checkpoints still declare `"llava"`
  and still load, because every load site now constructs `LlavaConfig` explicitly. Going
  through `AutoConfig` would return transformers' own config class and silently drop
  MM-OR's custom fields. In the other direction, checkpoints saved by this code declare
  `"llava_mm2sg"`; code from before the modernization loads them only after changing
  `model_type` in their `config.json` back to `"llava"`.
- **Cache API.** `llava_arch.py` read the KV-cache length with the pre-4.36 tuple layout.
  It now uses `Cache.get_seq_length()`, with the old indexing kept as a fallback.
- **`forward()` accepts `cache_position` and `**kwargs`**, which `generate()` passes since 4.38.
- **Batched generation with padding (`llava_llama.py`).** This was the most consequential
  fix, and it was invisible to every unit-level check. `generate()` derives
  `cache_position` from the text-only input length, but the KV cache holds the spliced
  sequence (text plus modality embeddings), which is longer. Forwarding it built a causal
  mask that hid most of the context from each new token. The effect showed only when an
  explicit mask was materialized, i.e. when a batch contained padding, and then **every
  row of that batch** came out as garbage:
  - 4D-OR prompts have a fixed length, so 4D-OR matched exactly.
  - MM-OR prompts vary in length (speech/metadata), so MM-OR fell to Macro-F1 0.106.
  - Fix: never forward `cache_position` when images are present.
- **QLoRA checkpoint loading (`builder.py`).** The released checkpoints carry their
  training-time bnb nf4 `quantization_config` in `config.json`.
  - transformers 4.31 ignored it, so **the original pipeline evaluates in bf16**.
  - 4.57 honors it and builds 4-bit layers that the bf16 base never fills, which fails
    with `Cannot copy out of meta tensor`.
  - It is now always dropped. The default therefore keeps the original bf16 behaviour,
    and `load_4bit`/`load_8bit` pass their own quantization config.
- **4-bit evaluation (`builder.py`, `main.py --load-4bit`).** The original `load_4bit`
  path did not work on QLoRA checkpoints. It now loads the base exactly as `train.py`
  does (NF4, bf16 compute, projector/pooler/vision tower unquantized) and keeps the LoRA
  adapters unmerged, since they cannot be merged into a 4-bit base. Embeddings are resized
  only when the tokenizer actually grew. See §3, "Evaluating in 4-bit".
- **Trainer.** `ALL_LAYERNORM_LAYERS` moved to `transformers.pytorch_utils`. The
  `ShardedDDPOption` / fairscale branches were removed upstream and were never taken here,
  since training uses DeepSpeed ZeRO-2.
- **`torch.load`.** All 13 call sites now pass an explicit `weights_only=True`, because
  the default flipped in torch 2.6. Every site was audited against the released artifacts:
  all of them load tensors only.
- **`pyproject.toml`** no longer pins the 2023 stack. `pip install -e .` used to try to
  downgrade torch 2.9 → 2.0. The web-demo extras moved to an optional `demo` extra.

### Training path (transformers 4.57 / peft 0.21 / DeepSpeed 0.19)

- **Unfrozen CLIP layers received no gradient (silent).** transformers >= 4.35 enables
  gradient checkpointing in *every* submodule that supports it, including the nested CLIP
  vision encoder; 4.31 only checkpointed Llama. Under the default *reentrant* checkpointing,
  a block whose input does not require grad yields no parameter gradients, and CLIP's input
  comes from its frozen embeddings. So `--unfreeze_n_vision_tower_layers 12` trained
  nothing. `train.py` now selects non-reentrant checkpointing (numerically equivalent).
- **Gradient-accumulation normalization (silent).** transformers >= 4.46 skips the
  `/ gradient_accumulation_steps` division when the model's forward takes `**kwargs`
  (ours does), assuming the model normalizes itself. `compute_loss` returns a per-batch
  mean as in 4.31, so `LLaVATrainer` restores the original normalization; otherwise every
  gradient would be scaled by 4.
- **fp32 loss.** 4.31's Llama always returned fp32 logits; 4.57 keeps bf16. Logits are
  upcast before the token-weighted cross-entropy, as before.
- **Seeding.** The Trainer seeds only in its `__init__`, after the image pooler and LoRA
  weights were randomly initialized, so no two original runs started from the same weights.
  `train.py` seeds first. `--data_seed` now seeds the length-grouped sampler's own
  generator. Without it, as in the original code, the order is drawn from the global torch
  RNG at the point the sampler is built, so it depends on everything that consumed random
  numbers before (`--seed`, model construction).
- **Attention in training.** The original `train_mem.py` patched Llama attention with a
  hand-written flash-attn forward. Training now uses transformers' default attention (SDPA).
  The training parity below covers this change.
- **API drift:** `load_in_4bit` only via `quantization_config`; new signatures of
  `_get_train_sampler`, `compute_loss`, `_save_checkpoint`.

### Dead code removed (~2780 lines)

- **The MPT model subtree.** MM2SG is Llama-based and never reaches it, and it imported
  symbols that no longer exist in transformers.
- **The three attention monkeypatches** (flash-attn, xformers, `llama_patch`) and
  `train_xformers.py`.
  - Attention is selected natively now.
  - On 4.57 the patches would still import cleanly while replacing `LlamaAttention.forward`
    with a 4.31-signature function, so they would fail *silently*.
  - The one non-attention helper they contained, `upcast_layer_for_flash_attention`, is
    preserved byte-identical in `llava/train/dtype_utils.py`.

### Compiled dependencies dropped

- **torch-scatter.** `multimodal_projector/compat.py` uses it if it is installed;
  otherwise it falls back to native ATen `segment_csr` in `native_ops.py`. The fallback is
  exact for max and within ~1e-7 for mean.
- **open3d.** Replaced by `llava/utils_pcd.py`, a dependency-free PCD reader. It is
  bit-exact against `open3d.io.read_point_cloud` on 12 released files, including the
  mm → m scaling.
- **flash-attn in PointTransformerV3.** If flash-attn is missing, PTv3 falls back to its
  non-flash path with a warning. `MMOR_REQUIRE_FLASH_ATTN=1` turns the fallback into an
  error. This matters because the non-flash path groups points differently from the
  trained model, so it is **not** numerically equivalent.

### Usability and robustness

- **Former hardcoded literals are CLI flags now.** Every default reproduces the old value,
  and the derived dataset filenames are verified byte-identical. With `--no-4dor`, the
  dataset generators append `_no4dor` to the filename, so an MM-OR-only json does not
  overwrite the default one.
  - `main.py`: `--mode`, `--batch-size`, `--no-4dor`, `--limit-val-batches`.
  - `generate_dataset_format_for_llava.py`: `--n-perm`, `--add-temporal`, `--split`,
    `--drop-history`, `--no-4dor`, and others.
  - Same for `generate_downstream_dataset_format_for_llava.py`.
- **Weight caches no longer depend on the working directory.** They were addressed
  relative to the process CWD, so `_init_weights` reported "No weights found" when run
  from anywhere other than `LLaVA/`. They are now anchored to their module.
- **Generated artifacts are no longer tracked.** `__pycache__/*.pyc` were committed
  upstream, and importing the package dirtied 16 files. The PTv3 weight cache and the
  dataset symlink roots are gitignored too.

---

## 2. Dataset: deviations from the release and what has to be generated

The release does **not** contain everything the README says it does. These are the steps
and fixes needed to get from the published download to a dataset that trains and
evaluates.

### Layout

- `MM-OR_data` must point at the directory that **directly** contains the take folders
  (`001_PKA`, …, `take_jsons`). The official download script unzips one level deeper, into
  `MM-OR_data/MM-OR_processed/`, so the nested directory is the correct target.
- **4D-OR is effectively mandatory.** The published labels are *HybridOR*: about 30 % of
  each split comes from 4D-OR, and `ORDataset` defaults to `load_4dor=True`. Use
  `--no-4dor` for MM-OR-only runs, but those are not paper-comparable.
  - Scene graphs need only part of 4D-OR, per take: `colorimage/` plus
    `timestamp_to_pcd_and_frames_list.json` from `4D-OR.zip`, `pcds_sparse/` (shipped with
    the MM-OR release) and GT `segmentation_export_{1,2,5}/` (HF `egeozsoy/MM-OR`).
  - Depth, fused clouds, 3D boxes and 6D poses are not needed.
- An older 338 GB MM-OR export exists that lacks `relation_labels`, which is the
  scene-graph ground truth, along with `dino_features_l`, `panoptic_seg_*` and `pose`.
  It fails silently rather than loudly. Use the current release.

### Per-timepoint modality folders: generated, not shipped

The README says the preprocessed versions are included. They are not, so they have to be
generated with these scripts (run from `scene_graph_generation/`):

| folder | script |
|---|---|
| `take_audio_per_timepoint` | `create_take_sample_audios.py` |
| `take_audio_embeddings_per_timepoint` | `create_take_sample_audio_embeddings.py` |
| `take_transcripts_per_timepoint` | `create_take_sample_speech_transcripts.py` |
| `take_segmasks_per_timepoint` (MM-OR and 4D-OR) | `create_take_sample_segmasks.py` |

Fixes made to these scripts:

- **`create_take_sample_segmasks.py`** used bare, CWD-relative paths (`Path('MM-OR_data')`,
  `Path('4D-OR_data')`, `Path('panoptic_segmentation/...')`) for three of its paths. That only resolves with CWD at the repo root,
  while its imports need CWD at `scene_graph_generation/`, so both cannot hold. It now
  uses the configured roots and the script's own location.
- **`create_take_sample_audios.py`** called `mkdir` without `parents=True`, so it failed
  on a fresh dataset where the parent folder does not exist.
- **Predicted segmasks (`PREDSEGMASKS`) need the panoptic-segmentation stack.** The script
  reads predictions from a local DVIS++/CTVIS inference run (`pan_pred`), as before. The
  release does **not** contain predictions: its `panoptic_seg_<cam>_for_val/` folders are
  renderings of the ground-truth annotations. The panoptic data loader writes them from
  `segmentation_export_<cam>/` (`or4d_dataset.py`, `mmor_dataset.py`). Their file names
  match `segmentation_export_<cam>/` one to one, including the interpolated masks, and their
  pixels match the annotation exactly. `SEGMASK_FROM_RELEASED_GT=1` makes the script use
  them anyway, with a warning, to reproduce the runs below (see "Segmask inputs in this
  verification").
- **Sharding and resuming.** Both the segmask and the audio-embedding scripts accept take
  names on argv.
  - Segmasks split work into (take, timestamp-shard) items (`SEGMASK_WORKERS`,
    `SEGMASK_SHARDS_PER_TAKE`) and skip timestamps that are already written.
  - Embeddings take `EMB_NUM_WORKERS` and skip undecodable clips with a warning instead of
    aborting.

### Known gaps in the release (affect both stacks equally)

- **`008_PKA/003512.mp3`** is a truncated 555-byte final clip and cannot be decoded. It is
  the only one: every other take has full audio-embedding coverage, and this one timestamp
  has no audio embedding.
- **`009_TKA`** has no segmentation source at all, so it has 0 segmasks, which is correct.
- **No predicted segmasks are released.** The checkpoint is a `PREDSEGMASKS` model, but
  predicted masks require running the released DVIS++/CTVIS checkpoints (panoptic
  segmentation part of this repository) over all frames.

#### Segmask inputs in this verification

All scene-graph runs in §3 used segmasks generated with the behaviour that is now
`SEGMASK_FROM_RELEASED_GT=1`:

- **MM-OR:** every test sample (2,960 of 2,960) and 10,350 of 11,591 training samples got
  **ground-truth** masks in place of predicted ones.
- **4D-OR:** no segmasks at all (the released `panoptic_seg_*_for_val` folders exist only for
  MM-OR), in training and in testing.

This was identical for both stacks, so the parity results hold. The absolute scores are
**not** a predicted-segmask setting, however: MM-OR saw ground-truth segmentation, and
4D-OR saw none. They are not comparable to the paper's numbers.
- **Missing modalities are skipped silently.** `or_dataset.py` adds audio/segmask inputs
  only `if path.exists()`, so any run started before preprocessing finishes is a run with
  those modalities dropped. Evaluate only on a complete, frozen dataset.
- **`extract_metalabels_for_MMOR.py` was never published.** It is imported by
  `downstream_prediction_model.py`, so the three downstream tasks (sterility breach, next
  action, robot phase) cannot run. Core scene-graph generation is unaffected.

---

## 3. Verification

The method throughout is to run the original environment (frozen pins, untouched
semantics) and the modernized one on the same code, data and hardware (NVIDIA A40,
sm_86), and compare outputs. Comparisons are made against a measured run-to-run noise
floor rather than bit-equality, because the sparse-conv and attention kernels accumulate
with atomics.

### Component level

The component checks used test scripts that are not part of the repository.

| check | result |
|---|---|
| Module imports | 12/12 baseline, 15/15 modern |
| Model smoke test: forward, loss, cache, `generate()`, modality passthrough | 7/7 in both stacks; **identical loss 4.9313** |
| PointTransformerV3 outputs (feat / pooled / projected) | **cos = 1.000000**; every delta below the in-env noise floor |
| `spconv` `SubMConv3d` | bit-identical between stacks |
| native `segment_csr` vs torch-scatter | max: exact; mean: 6e-8 |
| PCD reader vs open3d | bit-exact on 12 released files |
| flash-attn 2.8.3 vs torch SDPA | max diff 2.4e-4 (fp16) |
| **PTv3 on Blackwell** (RTX PRO 6000, sm_120, PTX JIT) | **6/6, cos ≥ 0.999999** vs A40; `spconv` bit-identical to A40 (max_abs 3.3e-7 vs a dense conv3d reference on both) |

### End-to-end: full test set

Setup:
- Checkpoint `llava-v1.5-7b-task-lora_hybridor_qlora_20perm_AZURE_SIMSTATION_TRACKERCAM_PC_AUDIO_SPEECH_ROBOTMETA_TRACKINGMETA_PREDSEGMASKS_0.50drop_MIXED`
  on `liuhaotian/llava-v1.5-7b`.
- Evaluated with `main.py --mode evaluate`, batch size 8, on one A40 per stack.
- Same frozen dataset for both. Modern run with `MMOR_REQUIRE_FLASH_ATTN=1`.
- Segmasks: ground truth for MM-OR, none for 4D-OR (see §2, "Segmask inputs in this
  verification").
- Runtime: baseline 2 h 16 min, modern 2 h 12 min.

Relation Macro-F1:

| take | original stack | modernized | Δ |
|---|---|---|---|
| 002_4DOR | 0.907 | 0.908 | +0.001 |
| 006_4DOR | 0.880 | 0.887 | +0.007 |
| 004_PKA | 0.469 | 0.464 | −0.005 |
| 011_TKA | 0.502 | 0.498 | −0.004 |
| 036_PKA | 0.538 | 0.538 | 0.000 |
| 038_TKA | 0.516 | 0.513 | −0.003 |
| **4D-OR overall** | **0.895** | **0.899** | +0.004 |
| **MM-OR overall** | **0.501** | **0.500** | −0.001 |
| **combined** | **0.689** | **0.691** | +0.002 |

Other metrics on the combined test set:

| metric | original | modernized |
|---|---|---|
| Macro precision / recall | 0.800 / 0.635 | 0.803 / 0.636 |
| Binary interaction macro-F1 | 0.901 | 0.902 |
| Entity micro-F1 | 0.936 | 0.936 |

**Verdict: parity.** The deltas have no consistent sign and are the size expected from
greedy decoding resolving near-ties differently under different kernels (flash-attn 2.3.4
→ 2.8.3, a different bf16 GEMM stack). Inspecting raw generations on a padded batch
confirms this: 7 of 8 rows produced the same triplet set as the original, and the eighth
produced one extra triplet.

For reference, before the padding fix in §1 the same run scored MM-OR 0.106, while 4D-OR
was unchanged at 0.895. It also ran about 20 % slower, because the broken generations ran
to `max_new_tokens` instead of stopping at `</SG>`.

### Training: 200-step QLoRA runs

Setup: the README recipe (QLoRA r=128, bs 4 × GA 4, DeepSpeed ZeRO-2, bf16, gradient
checkpointing, image-order and modality-drop augmentation), one A40 per run, 200 optimizer
steps, identical batch order (`--data_seed 42`) in both stacks, two init seeds (42, 7).
The modern runs use `--max_grad_norm 0` (see the clipping note below).

Training loss, modern relative to original, mean per window:

| steps | seed 42 | seed 7 | original: seed 7 vs seed 42 |
|---|---|---|---|
| 11–50 | −0.4% | −0.5% | +0.4% |
| 51–100 | −0.3% | −0.5% | +0.4% |
| 101–150 | −0.5% | −0.3% | +0.6% |
| 151–200 | −0.4% | −0.4% | +1.3% |

The median per-step difference between stacks is 0.4–0.5%, against 1.0% between two seeds
of the original stack. **Verdict: parity.** The very first steps agree to 0.2–0.4%, and
Adam's state after step 1 shows matching gradients per parameter group (cosine 0.89–0.998).
The exception is LoRA B, because the two peft versions draw different random LoRA A
matrices; that is an init difference, not a computational one.

Before the CLIP-gradient fix, modern ran 2–5% *below* the original in every window, for
both seeds. That gap is how the bug was found.

Throughput on an A40: 14.8–16.4 s per step (modern) vs 17.7 s (original); peak GPU
memory 22 GB vs 40 GB. A full stage-1 epoch (312,300 samples, 19,518 steps) takes about
3.3 days on one A40.

**Gradient clipping never ran in the original training.** DeepSpeed 0.9.5's ZeRO-1/2
optimizer computes the gradient norm, and therefore clips, only for fp16
(`if self.dtype == torch.float16`). Under bf16, `--max_grad_norm 0.1` was a no-op, and
gradient norms of 2–4 went to Adam unclipped. DeepSpeed 0.19 does clip. The README recipe still
passes `--max_grad_norm 0.1`. To reproduce the original
training behaviour, pass `--max_grad_norm 0`. Clipping had no measurable effect on the 200-step
curves, which is expected: Adam is nearly invariant to a uniform gradient scale.

### Full training: stage 1 and temporal stage 2

Setup: the README recipe in the modern stack, stage 1 followed by the temporal stage 2,
one epoch per stage on one A40 (stage 1: 82 h, stage 2: 90 h), `--max_grad_norm 0` (no
clipping, which is what the original bf16 training effectively did), `--seed 42`. Compared
with the released checkpoints on the full test set, all modern stack:

| model | eval load mode | combined Macro-F1 | 4D-OR | MM-OR |
|---|---|---|---|---|
| stage 1, released | bf16, merged | 0.686–0.691 | 0.894–0.899 | 0.498–0.500 |
| stage 1, first retraining | bf16, merged | 0.689 | 0.895 | 0.508 |
| temporal, released | bf16, merged | 0.696 (original stack: 0.702; one run each) | | |
| temporal, released | 4-bit, unmerged | 0.707 | 0.896 | 0.532 |
| temporal, first retraining | bf16, merged | crashes (see below) | | |
| temporal, first retraining | **4-bit, unmerged** | **0.703** | 0.911 | 0.522 |
| temporal, retrained from a clean stage 1 (run D below) | bf16, merged | 0.707 | 0.912 | 0.540 |
| temporal, retrained from a clean stage 1 (run D below) | 4-bit, unmerged | 0.705 | 0.910 | 0.537 |

**Verdict: the modern stack reproduces the released checkpoints**, within the run-to-run
spread of the eval (0.686–0.691 over five runs of the same checkpoint, see "Installation"
and the README). The first retrained temporal model has to be evaluated with
`--load-4bit`, however. The cause is its stage-1 checkpoint, not the stack (see below).

#### Evaluating in 4-bit

The standard eval path loads the base in bf16 and merges the LoRA weights into it. The
model was trained with the LoRA on top of an NF4-quantized base, so the merged model is
close to, but not the same as, the trained one.

- **Released checkpoints:** the difference does not matter. Both modes run cleanly, and
  the temporal checkpoint scores 0.696 (bf16) and 0.707 (4-bit).
- **First retrained temporal checkpoint:** in bf16 it occasionally emits garbled entity names
  (`anaesthetistmulticolor`, `Constra…`), 29 of the first 150 temporal predictions. The
  temporal config feeds the model's own past predictions back into the prompt, so the
  errors compound. Once the prompt exceeds 2048 tokens, it is truncated and the eval
  crashes with a negative-dimension error.
- **With `--load-4bit`:** 0 garbled predictions out of 4338, for both checkpoints.

Recommendation: evaluate self-trained QLoRA checkpoints with `--load-4bit`, at least with
the temporal config. Keep the default (bf16, merged) for comparisons with the published
numbers, which were produced that way. 4-bit is slower: the full temporal test set took
11–12 h per checkpoint on an A40, against 5 h in bf16.

#### Cause: the stage-1 checkpoint

To find out why the first retrained temporal model breaks in bf16 and the released one does not,
we measured how much a checkpoint's output changes between the two load modes. The
checkpoint is loaded in 4-bit and generates on 109 test samples (every 40th). Teacher-forced
on exactly those outputs, the token distributions of the 4-bit model and the bf16-merged
model are then compared:

- **Mean KL divergence** KL(4-bit ‖ bf16) per generated token.
- **Noise floor:** the 4-bit model compared with itself. It is 0.0001 everywhere.
- **Reference:** the released checkpoints, at 0.005 (temporal) and 0.006 (stage 1).

The first retrained temporal model's divergence over its training run:

| stage-2 step | 500 | 2,000 | 3,500 | 5,000 | 7,500 | 10,000 | 12,500 | 15,000 |
|---|---|---|---|---|---|---|---|---|
| mean KL | 0.041 | 0.009 | 0.006 | 0.111 | 0.223 | 0.223 | 0.386 | 0.540 |
| bf16 garbled outputs (of 300) | 0 | | | 0 | | 9 | | 85 |

The jump between steps 3,500 and 5,000 never reverses, and garbled output follows once the
divergence passes about 0.25. To isolate the cause, we ran three more stage-2 trainings and
stopped them at about step 5,500 (`--max_grad_norm 0`, data and recipe as above):

| run | stack | stage-1 start | seed | mean KL at stage-2 step 2,000 / 3,500 / 5,000 / 5,500 |
|---|---|---|---|---|
| first retraining | modern | first retrained | 42 | 0.009 / 0.006 / **0.111** / – |
| A | modern | first retrained | 7 | 0.011 / **0.087** / 0.086 / – |
| B | modern | **released** | 42 | 0.004 / 0.004 / 0.004 / 0.005 |
| C | **original (2023)** | first retrained | 42 | 0.028 / 0.026 / 0.057 / **0.112** |

Every run that started from the first retrained stage-1 checkpoint drifts, on either stack and with either
seed. Only the onset varies, from about step 2,500 to 5,000. The run from the released
stage-1 stays at the released model's level, on the modern stack and with the regenerated
stage-2 data. This rules out:

- the modern stack (C drifts on the original one),
- the regenerated stage-2 data (B uses it and stays clean),
- the seed (A drifts too),
- gradient clipping (neither training clipped).

The difference is already visible in the stage-1 checkpoints themselves, on single-frame
inputs (`mmor.json`), even though both score the same Macro-F1:

| stage-1 checkpoint | mean KL, single-frame inputs | mean KL, temporal inputs |
|---|---|---|
| released | 0.006 | 0.014 |
| first retrained (modern stack, 1 epoch) | 0.046 | 0.089 |

The 200-step stage-1 runs (§3, "Training: 200-step QLoRA runs") show no stack difference
on single-frame inputs: 0.026 and 0.032 (original stack, seeds 42 and 7) against 0.026 and
0.030 (modern). These early checkpoints all sit above the released final value, as does
stage 2 at step 500 (0.041), so they say nothing about where a full run ends up. They only
show that the two stacks do not diverge in the first 200 steps.

So the property that makes stage 2 drift away from what a bf16 merge can reproduce is already
present in the first retrained stage-1 checkpoint.

#### Stage 1: an unlucky seed, not the stack or the data

We then retrained stage 1 three more times and measured the checkpoints along the way
(mean KL, single-frame inputs; released: 0.006):

| stage-1 run | stack | `--seed` | batch order | 7,500 | 10,000 | 15,000 | final |
|---|---|---|---|---|---|---|---|
| first retraining | modern | 42 | global RNG after seed 42 | 0.005 | 0.019 | 0.055 | 0.046 |
| S1_baseline | **original** | 42 | global RNG after seed 42 | 0.005 | 0.033 | stopped at 11,000 (0.041) | – |
| S1_modern_seed7 | modern | **7** | `--data_seed 7` | 0.004 (8,500) | 0.005 | 0.008 | **0.0075** |
| S1_seed7_order42 | modern | **7** | **replayed from the seed-42 runs** | 0.006 (7,000) | 0.010 | 0.011 | 0.012 |

- **The stack is ruled out:** seed 42 drifts on the original stack too, from the same point.
  (Both seed-42 runs saw the identical batch order; their training losses correlate at 0.99.)
- **The regenerated stage-1 data is ruled out:** seed 7 on the same data ends at 0.0075,
  close to the released 0.006.
- **Batch order or starting weights:** the last run takes its starting weights from seed 7
  and replays the seed-42 batch order. (The seed-42 runs drew their order from the
  global torch RNG, without `--data_seed`, so `--data_seed 42` does not reproduce it; the
  RNG state was captured and replayed with a temporary hook.) It
  rises a little where the seed-42 runs take off, between steps 9,000 and 10,000, then stays flat at
  about 0.012. So the seed-42 order contributes a small bump. Most of the drift comes from
  the seed-42 starting weights interacting with it.

In practice, this is chance: some initialisations end up in a state that a bf16 merge cannot
reproduce, and the Macro-F1 does not show it.

**Stage 2 from the clean seed-7 stage-1** (run D: modern stack, `--seed 42`, `--max_grad_norm 0`;
mean KL, temporal inputs; released: 0.005):

| stage-2 step | 2,000 | 4,000 | 5,000 | 6,000 | 6,500 | 7,500 | 8,500 | 9,000 | 9,500 | 10,000 | 10,500 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| first retraining | 0.009 | | 0.111 | | | 0.223 | | | | 0.223 | |
| D | 0.005 | 0.006 | 0.013 | 0.012 | 0.007 | 0.007 | 0.009 | 0.043 | 0.011 | 0.034 | 0.015 |

D does not drift like the first retraining: the divergence stays around 0.01, with single
checkpoints spiking and returning, and remains an order of magnitude below the level where
garbled bf16 output appeared (about 0.25). It still sits about 2× above the released model.

The fully trained D (19,519 steps) on the full test set (4,338 samples):

| temporal model | load mode | Macro-F1 | garbled entity names | completed |
|---|---|---|---|---|
| released | bf16 | 0.696 | 0 | yes |
| released | 4-bit | 0.707 | 0 | yes |
| first retraining | bf16 | – | 413 by sample 1,537 | no, crashed |
| first retraining | 4-bit | 0.703 | 0 | yes |
| **D** | **bf16** | **0.707** | 8 | yes |
| D | 4-bit | 0.705 | 0 | yes |

So with a clean stage 1, the modernized stack trains a temporal model that evaluates
correctly in bf16, at the released model's score. The 8 garbled names
(`nurseconstrain`, `head_surgeonmulticolitor`, …) are of the same kind as before, but
about 100× rarer. They do not affect the score and do not derail the temporal prompt. The
4-bit evaluation stays the cleaner choice for self-trained checkpoints.

Practical consequences:

- To train a temporal model that also evaluates in bf16, start stage 2 from a clean
  stage-1 checkpoint: the released one, passed as `--curriculum_learning_weights` (run B
  above), or a retrained one that passes the check below (run D).
- Otherwise evaluate with `--load-4bit`.
- When retraining stage 1, check the checkpoint before spending 90 h on stage 2. A mean KL near the
  released 0.006 is a good sign; one well above it predicts the bf16 drift. In that case,
  retrain stage 1 with another `--seed`. The rise shows by step 10,000, about halfway through stage 1.
  (The measurement script is not part of the repository; the method is described at the
  start of this section.)

### Not yet verified

- **Comparison against the paper's published table.** This verification establishes
  *original pipeline == modernized pipeline*. It does not establish *either == paper*.
  The segmask inputs deviate from the paper's setup: ground truth for MM-OR, none for 4D-OR
  (see §2, "Segmask inputs in this verification").
- **Downstream tasks.** Blocked by the missing `extract_metalabels_for_MMOR.py`.

---

## 4. Panoptic segmentation

The panoptic-segmentation part (`panoptic_segmentation/`: DVIS++ and CTVIS on detectron2 and
Mask2Former) is ported to the same torch as the scene-graph part, in its own environment:

| | original (`requirements.txt`) | modernized (`requirements-modern/`) |
|---|---|---|
| Python | 3.10 | 3.12 |
| torch | 1.13.1 + cu117 | 2.9.1 + cu128 |
| detectron2 | v0.6 | main at `247691a` (reports version 0.6) |
| numpy / Pillow | 1.23 / 9.1 | 2.5 / 13.0 |
| MSDeformAttn op | compiled with nvcc 11.x | compiled with nvcc 12.x |

As for the scene-graph part, the source stays usable on the original stack, which again served
as the reference for all comparisons below.

### Installation

```bash
cd panoptic_segmentation
./setup_env_modern.sh                    # venv in .venv-panoptic
./setup_env_modern.sh --locked           # exact verified package set
```

Unlike the scene-graph environment, this one compiles detectron2 and the MSDeformAttn op, so
it needs a CUDA 12.x toolkit (12.8 or newer for Blackwell; `--cuda-home`). `MAX_JOBS` (default
4) limits parallel compilation, `TORCH_CUDA_ARCH_LIST` selects the GPU architectures.

### Code changes

- **MSDeformAttn CUDA op.** `Tensor::type()` (removed from the dispatch macros),
  `Tensor::data<T>()` and the THC header are replaced by `scalar_type()` / `is_cuda()`,
  `data_ptr<T>()` and `ATen/cuda/Atomic.cuh`. All exist in torch 1.13 as well: the changed source
  still compiles there, and on identical inputs its forward pass is bit-identical to the
  original op's. The gradients differ by at most 2.4e-7, less than the original op differs
  from itself between two runs (3.0e-7; the backward pass accumulates with `atomicAdd`).
- **numpy 2.** `np.bool` / `np.float` (removed in numpy 1.24) become `np.bool_` / `np.float64`.
- **A frame listed twice in a video crashed the DVIS++ evaluation (bug in the original code).**
  In the test split, a few synchronisation timestamps map to the same image as a neighbouring
  one (`004_PKA` cameras 1/4/5, 4D-OR take 2 camera 5): 12 frames appeared twice, in 8 of the 90
  test videos. `VPSEvaluator` saves predictions as `pan_pred/<video>/<frame>.png`, from a thread
  pool, so the second copy overwrote the first. With DVIS++ offline this left corrupt PNGs; with
  DVIS++ online, whose track ids differ between the two copies, scoring failed with
  `Segment with ID ... is presented in PNG and not presented in JSON`. CTVIS happened to give
  both copies the same ids. Both loaders now keep the first timestamp of each frame, and the
  evaluator asserts unique frames per video. This removes those 12 entries from the test split
  (16,243 → 16,231; the 200-frame video parts are cut after deduplication). The train
  and val splits contain no repeats and are unchanged. On the original stack, CTVIS moves from
  VPQ 58.25 to 58.22.
- **The VPQ ground truth is not released.** The evaluator reads
  `datasets/<dataset>_ground_truth_<split>.json`, which only the `__main__` blocks of the
  dataset modules wrote (for all splits at once, train included). `eval_from_preds.py
  --build-gt-only` builds one split, and `eval_from_preds.py <OUTPUT_DIR>/inference` scores saved
  predictions without a GPU. A missing file now raises an error that names both commands.
- `VPQ_NUM_PROCESSES` sets the size of the VPQ scoring pool (default 12, as before).

Nothing else had to change: the current detectron2 runs the code as is, and all three released
checkpoints load completely (no missing or unexpected keys, every tensor as stored).

### Verification

Both environments ran this branch's code on the same data (the HybridOR test split: MM-OR takes
004/011/036/038 and 4D-OR takes 2/6, 90 videos) on the same GPU model (NVIDIA A40), each with its
own copy of the folders the loaders write.

- **Ground truth.** The test split's ground-truth json and all 32,030 `panoptic_seg_*` PNGs the
  loaders render are byte-identical between the two stacks.
- **Released checkpoints, HybridOR test split.** VPQ is the mean over the 15- and 35-frame
  windows that `VPSEvaluator` computes; precision and recall likewise.

  | model | original stack | modernized stack | Δ VPQ |
  |---|---|---|---|
  | CTVIS | 58.22 (P 0.637, R 0.644) | 58.31 (P 0.642, R 0.642) | +0.08 |
  | DVIS++ online | 56.97 (P 0.659, R 0.606) | 57.13 (P 0.659, R 0.608) | +0.16 |
  | DVIS++ offline | 60.50 (P 0.680, R 0.663) | 60.57 (P 0.680, R 0.660) | +0.07 |

  The differences are at most 0.16 VPQ (0.3 % relative), with precision and recall moving in
  both directions. Run-to-run noise of this evaluation was not measured; the per-class scores
  agree closely (e.g. CTVIS instrument table 0.948 on both stacks).
- **Runtime.** CTVIS inference over the test split took 48 min on the modernized stack and
  56 min on the original one; VPQ scoring takes about 2 hours on the CPU on either.

### Not yet verified

- **Training** the panoptic models on the modernized stack (only the released checkpoints were
  evaluated).
- **The train and val splits**, which are needed to generate predicted segmasks for scene-graph
  training (see §2, "Segmask inputs in this verification").
- **Comparison with the paper's table**, as for the scene-graph part.
