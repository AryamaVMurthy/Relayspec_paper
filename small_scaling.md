# Small-data scaling extension — 16 September 2026

Authorized: **20 fits and 35 full evaluation cohorts**, at 16,128,512,1,024,2,048 examples. No new native/AR benchmarks.

| Phase | Architecture / loss | Domains | Epochs | LR | Supervision | Fits | Eval cohorts |
|---|---|---|---:|---:|---|---:|---:|
| Phase1 | Five identity square maps / AUF | GSM8K,KiCad,NanoCoder | 1 | 1e-4 | 32 anchors/example | 15 | 15 |
| Phase2 T1 | Five Xavier rectangular maps / MSE | Four existing tasks | 3 | 1e-3 | 25% prompt+response | 5 | 20 |

Nested original training subsets; saved rollouts/features reused, no fresh generation. Final checkpoint, seed42, global batch8, unchanged schedules. Evaluation uses Transformers and original128-prompt cohorts/caps. More examples also mean more optimizer updates; this is not a fixed-compute comparison. No monotonic outcome is assumed.

Implementation and provenance: `experiments/dflash_small_scaling_20260916/README.md`. Launch receipts: `launches.json` in that directory. The16-example checkpoints are reused after their small inference checks. The 512-example manifests are the first 512 unique items in each validated 1,024-example manifest, including the first 128 unchanged. Failed jobs hold dependent evaluations rather than silently skipping failures.

Requested concurrency: four two-GPU lanes, eight GPUs total across node08/node09. Slurm's shared group limit may restrict simultaneous use. No unrelated jobs were stopped.

The 512-point extension is queued as fits 34859–34862, followed by T1 evaluations 34863–34866. Check the live table below for completed results; a queued job is not counted as evidence.

<!-- small-scaling-live-begin -->

### Live results (35/35 complete evaluation cohorts)

A result is shown only after its full 128-prompt cohort completes. Fit time excludes feature collection and evaluation. TPS is the arithmetic mean of each request's generated tokens divided by its decode seconds. Mean accepted length pools all verifier steps, including the bonus token. These are Transformers single-request measurements; the same existing native/AR controls are reused.

**LoRA target adapters, AUF**

| Domain/task | Examples | Fit time (s) | Evaluated | TPS | Mean accepted length | Mean output tokens |
|---|---:|---:|---:|---:|---:|---:|
| math | 16 | 1.6 | 128/128 | 142.83 | 4.91 | 152.76 |
| math | 128 | 5.9 | 128/128 | 160.54 | 5.36 | 152.76 |
| math | 512 | 19.9 | 128/128 | 168.46 | 5.66 | 152.76 |
| math | 1,024 | 39.0 | 128/128 | 175.36 | 5.72 | 152.76 |
| math | 2,048 | 78.2 | 128/128 | 178.58 | 5.87 | 152.76 |
| kicad | 16 | 2.3 | 128/128 | 176.91 | 6.42 | 5040.14 |
| kicad | 128 | 11.3 | 128/128 | 207.30 | 7.55 | 5040.14 |
| kicad | 512 | 37.7 | 128/128 | 236.12 | 8.58 | 5040.14 |
| kicad | 1,024 | 85.2 | 128/128 | 262.02 | 9.27 | 5040.14 |
| kicad | 2,048 | 163.7 | 128/128 | 290.60 | 10.27 | 5040.14 |
| nanocoder | 16 | 1.8 | 128/128 | 151.60 | 5.12 | 217.69 |
| nanocoder | 128 | 6.1 | 128/128 | 152.57 | 5.17 | 217.69 |
| nanocoder | 512 | 20.4 | 128/128 | 159.15 | 5.30 | 217.69 |
| nanocoder | 1,024 | 40.4 | 128/128 | 157.87 | 5.37 | 217.69 |
| nanocoder | 2,048 | 80.2 | 128/128 | 162.89 | 5.44 | 217.69 |

**T1 cross-size transfer, MSE**

| Domain/task | Examples | Fit time (s) | Evaluated | TPS | Mean accepted length | Mean output tokens |
|---|---:|---:|---:|---:|---:|---:|
| math | 16 | 1.8 | 128/128 | 34.39 | 1.19 | 687.76 |
| gsm | 16 | 1.8 | 128/128 | 34.47 | 1.22 | 282.12 |
| code | 16 | 1.8 | 128/128 | 28.65 | 1.06 | 552.18 |
| chat | 16 | 1.8 | 128/128 | 29.06 | 1.06 | 709.50 |
| math | 128 | 10.6 | 128/128 | 113.74 | 3.84 | 687.76 |
| gsm | 128 | 10.6 | 128/128 | 76.39 | 2.68 | 282.12 |
| code | 128 | 10.6 | 128/128 | 36.82 | 1.44 | 552.18 |
| chat | 128 | 10.6 | 128/128 | 34.98 | 1.29 | 709.50 |
| math | 512 | 41.1 | 128/128 | 199.42 | 7.22 | 687.76 |
| gsm | 512 | 41.1 | 128/128 | 137.48 | 4.90 | 282.12 |
| code | 512 | 41.1 | 128/128 | 58.38 | 2.32 | 552.18 |
| chat | 512 | 41.1 | 128/128 | 47.72 | 1.69 | 709.50 |
| math | 1,024 | 79.9 | 128/128 | 213.25 | 7.72 | 687.76 |
| gsm | 1,024 | 79.9 | 128/128 | 160.88 | 5.70 | 282.12 |
| code | 1,024 | 79.9 | 128/128 | 78.04 | 3.10 | 552.18 |
| chat | 1,024 | 79.9 | 128/128 | 57.52 | 2.03 | 709.50 |
| math | 2,048 | 157.4 | 128/128 | 216.01 | 7.88 | 687.76 |
| gsm | 2,048 | 157.4 | 128/128 | 170.40 | 6.02 | 282.12 |
| code | 2,048 | 157.4 | 128/128 | 97.45 | 3.79 | 552.18 |
| chat | 2,048 | 157.4 | 128/128 | 67.56 | 2.35 | 709.50 |

Pending rows remain blank until complete; the raw per-request measurements and fit summaries are retained in the experiment directory on Turing.
<!-- small-scaling-live-end -->
