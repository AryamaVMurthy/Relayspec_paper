# Phase 2 — new MLP and EAGLE mapper results

Snapshot: 16 September 2026. Only complete 128-request cohorts are included. All eight new evaluations are complete.

For layer slot $i$, the MLP is

$$
g_i(h)=W_i h+B_i\operatorname{SiLU}(A_i h),\qquad z=\sum_i F_i g_i(h_i),\qquad c=N(z).
$$

$F_i$ is the original frozen fusion block, and $N$ is the original frozen RMSNorm. $A_i$ reduces to 256 coordinates; $B_i$ expands back to the source width. The draft body, target, source embedding/head, fusion and normalization remain frozen. $B_i$ starts at zero. At deployment $F_iW_i$ and $F_iB_i$ are precomputed; the nonlinear branch remains active. EAGLE instead uses three **linear** maps $g_i(h)=W_i h$, folded completely into its original fusion. It has no MLP branch.

## Training recipe

Target Qwen3-8B; source drafter originally trained for Qwen3-4B. Map width is **4,096 to 2,560**. MLP uses five maps; EAGLE uses three taps (HF hidden-state indices 2, 18, 33). All $W_i$ and MLP $A_i$ start at Xavier; $B_i=0$. These are trained from scratch, not refinements of the old mapper.

| Setting | New MLP / EAGLE and selected old 5W MSE |
| --- | --- |
| Training data | 16,384 existing T1 NuminaMath trajectories; saved response cap 4,096 |
| MSE positions | Deterministic 25% of prompt and response tokens separately, nested in the saved 50% sample; fixed across epochs |
| Objective | Normalized per-layer MSE plus post-frozen-RMSNorm context MSE |
| Schedule | Three epochs, 6,144 updates, final checkpoint |
| Optimizer | AdamW, LR 0.001, weight decay 0, clip norm 1; 5% warmup then cosine |
| Batch | Two GPUs × two examples × two accumulation steps = global batch 8 |
| Precision | FP32 trainable maps/loss, TF32 disabled, BF16 cached features/export; seed 42 |
| Parameters | 5W 52,428,800; MLP 60,948,480; EAGLE 31,457,280 |

The old AUF comparator is explicitly **8,192 examples, three epochs, LR 0.0001, 32 response anchors/example**, five Xavier linear maps. Its AUF loss supervises the correct prefix and first greedy error in each block. It is included to show the earlier token-loss recipe; unlike the selected 5W MSE control it is not a matched-data, matched-loss MLP ablation.

For reconstruction, define $D(u,v)=\lVert u-v\rVert_2^2/(\lVert v\rVert_2^2+10^{-6})$. With $K$ maps and selected token positions $S_b$ in example $b$:

$$
L_{MSE}=\frac1B\sum_b\frac1{|S_b|}\sum_{p\in S_b}\left[\frac1K\sum_{i=1}^K D(g_i(h^t_{bpi}),h^s_{bpi})+D\left(N\left(\sum_iF_i g_i(h^t_{bpi})\right),N\left(\sum_iF_i h^s_{bpi}\right)\right)\right].
$$

The target and source process the **same saved tokens**. Selected positions retain their full causal context; sampling removes supervised positions, not preceding context.

## Evaluation recipe

Transformers, one L40S, BF16, greedy nonthinking (temperature 0, top-p 1, top-k 0), 128 original prompts each from Math, GSM8K, Code and Chat; response cap 2,048. Two 64-token warmups are excluded. CUDA synchronization brackets each complete generation call. DFlash uses block size 16; EAGLE uses the pinned tree settings (60 tokens, depth 7, top-k 10). The EAGLE raw tail is preserved in artifacts but returned output is truncated by the existing T1 response-prefix contract; throughput uses returned tokens, while acceptance uses upstream emitted counters. Native and AR controls are reused; no baselines were rerun. MLP and selected 5W use the same training recipe and heldout cohorts.

## Metric definitions

All result rows contain 128 requests. Mean TPS is the arithmetic mean of each request's returned-token count divided by its full request latency. Pooled TPS is total returned tokens divided by summed request latency. Native-relative speedup is the ratio of mean TPS within the same drafter family. Latency saving is $100(1-T/T_{native})$ on the cohort; it need not equal $100(1-1/S)$ for a mean-TPS speedup $S$. Positive changes mean improvement. Acceptance length is pooled emitted verification tokens divided by verification steps, including correction/bonus tokens; DFlash excludes its initial prefill token. EAGLE uses its saved upstream emitted-token counters. These are not percentages of proposed tokens accepted. EAGLE and DFlash use different proposal structures, so equal acceptance lengths do not imply equal cost. Timings are single passes, not statistical proof of small differences. Fit time excludes feature extraction and model setup.

## Results: grouped by dataset and drafter family, descending throughput

| Dataset | Method | Mean TPS | Pooled TPS | vs family native | Acceptance length | Total latency (s) | Latency saving vs native |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Math | 5W MSE25 | 220.26 | 217.27 | 1.0022× | 7.984 | 405.17 | 0.57% |
| Math | DFlash native | 219.79 | 216.03 | 1.0000× | 8.211 | 407.50 | 0.00% |
| Math | MLP MSE25 | 215.71 | 213.26 | 0.9815× | 7.980 | 412.80 | -1.30% |
| Math | 5W AUF (8k) | 194.22 | 186.76 | 0.8837× | 6.892 | 471.37 | -15.67% |
| Math | EAGLE MSE25 | 77.99 | 76.98 | 1.0805× | 3.638 | 1179.74 | 0.97% |
| Math | EAGLE native | 72.18 | 72.57 | 1.0000× | 3.627 | 1191.32 | 0.00% |
| GSM8K | DFlash native | 177.94 | 173.86 | 1.0000× | 6.479 | 207.70 | 0.00% |
| GSM8K | 5W MSE25 | 176.87 | 173.75 | 0.9940× | 6.242 | 207.84 | -0.07% |
| GSM8K | MLP MSE25 | 174.56 | 171.57 | 0.9810× | 6.257 | 210.48 | -1.34% |
| GSM8K | 5W AUF (8k) | 162.05 | 157.39 | 0.9107× | 5.677 | 229.43 | -10.46% |
| GSM8K | EAGLE MSE25 | 80.59 | 80.78 | 1.0663× | 3.840 | 436.21 | 5.66% |
| GSM8K | EAGLE native | 75.58 | 75.98 | 1.0000× | 3.713 | 462.36 | 0.00% |
| Code | DFlash native | 151.63 | 144.52 | 1.0000× | 5.644 | 489.07 | 0.00% |
| Code | MLP MSE25 | 128.41 | 125.63 | 0.8469× | 4.782 | 562.58 | -15.03% |
| Code | 5W MSE25 | 127.61 | 125.46 | 0.8416× | 4.739 | 563.34 | -15.19% |
| Code | 5W AUF (8k) | 58.37 | 61.90 | 0.3850× | 2.346 | 1141.85 | -133.47% |
| Code | EAGLE MSE25 | 65.53 | 68.18 | 1.0257× | 3.293 | 1072.23 | -2.08% |
| Code | EAGLE native | 63.88 | 66.99 | 1.0000× | 3.351 | 1050.36 | 0.00% |
| Chat | DFlash native | 89.50 | 87.64 | 1.0000× | 3.324 | 1036.24 | 0.00% |
| Chat | 5W MSE25 | 80.93 | 78.42 | 0.9043× | 2.887 | 1158.06 | -11.76% |
| Chat | MLP MSE25 | 80.78 | 78.37 | 0.9025× | 2.902 | 1158.80 | -11.83% |
| Chat | 5W AUF (8k) | 49.93 | 47.84 | 0.5578× | 1.767 | 1898.26 | -83.19% |
| Chat | EAGLE MSE25 | 63.41 | 64.87 | 1.0257× | 3.136 | 1438.58 | 0.42% |
| Chat | EAGLE native | 61.82 | 64.45 | 1.0000× | 3.186 | 1444.63 | 0.00% |

## Explicit old 5W comparison

The EAGLE-to-5W rows compare complete pipelines, not matched drafter architectures. Acceptance counters also follow their respective upstream conventions.

| Dataset | New method | Reference | Old TPS | New TPS | TPS change | Old length | New length | Length change | Latency saving vs reference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Math | MLP MSE25 | 5W MSE25 | 220.26 | 215.71 | -2.07% | 7.984 | 7.980 | -0.004 | -1.88% |
| Math | MLP MSE25 | 5W AUF (8k) | 194.22 | 215.71 | 11.06% | 6.892 | 7.980 | 1.088 | 12.43% |
| Math | EAGLE MSE25 | 5W MSE25 | 220.26 | 77.99 | -64.59% | 7.984 | 3.638 | -4.346 | -191.17% |
| Math | EAGLE MSE25 | 5W AUF (8k) | 194.22 | 77.99 | -59.84% | 6.892 | 3.638 | -3.254 | -150.28% |
| Math | EAGLE MSE25 | EAGLE native | 72.18 | 77.99 | 8.05% | 3.627 | 3.638 | 0.011 | 0.97% |
| GSM8K | MLP MSE25 | 5W MSE25 | 176.87 | 174.56 | -1.31% | 6.242 | 6.257 | 0.015 | -1.27% |
| GSM8K | MLP MSE25 | 5W AUF (8k) | 162.05 | 174.56 | 7.72% | 5.677 | 6.257 | 0.579 | 8.26% |
| GSM8K | EAGLE MSE25 | 5W MSE25 | 176.87 | 80.59 | -54.44% | 6.242 | 3.840 | -2.402 | -109.88% |
| GSM8K | EAGLE MSE25 | 5W AUF (8k) | 162.05 | 80.59 | -50.27% | 5.677 | 3.840 | -1.837 | -90.13% |
| GSM8K | EAGLE MSE25 | EAGLE native | 75.58 | 80.59 | 6.63% | 3.713 | 3.840 | 0.127 | 5.66% |
| Code | MLP MSE25 | 5W MSE25 | 127.61 | 128.41 | 0.63% | 4.739 | 4.782 | 0.043 | 0.14% |
| Code | MLP MSE25 | 5W AUF (8k) | 58.37 | 128.41 | 120.00% | 2.346 | 4.782 | 2.436 | 50.73% |
| Code | EAGLE MSE25 | 5W MSE25 | 127.61 | 65.53 | -48.65% | 4.739 | 3.293 | -1.446 | -90.33% |
| Code | EAGLE MSE25 | 5W AUF (8k) | 58.37 | 65.53 | 12.26% | 2.346 | 3.293 | 0.948 | 6.10% |
| Code | EAGLE MSE25 | EAGLE native | 63.88 | 65.53 | 2.57% | 3.351 | 3.293 | -0.058 | -2.08% |
| Chat | MLP MSE25 | 5W MSE25 | 80.93 | 80.78 | -0.20% | 2.887 | 2.902 | 0.015 | -0.06% |
| Chat | MLP MSE25 | 5W AUF (8k) | 49.93 | 80.78 | 61.79% | 1.767 | 2.902 | 1.135 | 38.95% |
| Chat | EAGLE MSE25 | 5W MSE25 | 80.93 | 63.41 | -21.65% | 2.887 | 3.136 | 0.249 | -24.22% |
| Chat | EAGLE MSE25 | 5W AUF (8k) | 49.93 | 63.41 | 27.01% | 1.767 | 3.136 | 1.369 | 24.22% |
| Chat | EAGLE MSE25 | EAGLE native | 61.82 | 63.41 | 2.57% | 3.186 | 3.136 | -0.051 | 0.42% |

## Saved AR comparison

| Dataset | Method | AR mean TPS | Method mean TPS | vs AR |
| --- | --- | --- | --- | --- |
| Math | DFlash native | 37.84 | 219.79 | 5.8087× |
| Math | 5W MSE25 | 37.84 | 220.26 | 5.8212× |
| Math | 5W AUF (8k) | 37.84 | 194.22 | 5.1330× |
| Math | MLP MSE25 | 37.84 | 215.71 | 5.7010× |
| GSM8K | DFlash native | 37.76 | 177.94 | 4.7121× |
| GSM8K | 5W MSE25 | 37.76 | 176.87 | 4.6838× |
| GSM8K | 5W AUF (8k) | 37.76 | 162.05 | 4.2913× |
| GSM8K | MLP MSE25 | 37.76 | 174.56 | 4.6227× |
| Code | DFlash native | 37.50 | 151.63 | 4.0437× |
| Code | 5W MSE25 | 37.50 | 127.61 | 3.4032× |
| Code | 5W AUF (8k) | 37.50 | 58.37 | 1.5566× |
| Code | MLP MSE25 | 37.50 | 128.41 | 3.4246× |
| Chat | DFlash native | 35.09 | 89.50 | 2.5508× |
| Chat | 5W MSE25 | 35.09 | 80.93 | 2.3066× |
| Chat | 5W AUF (8k) | 35.09 | 49.93 | 1.4229× |
| Chat | MLP MSE25 | 35.09 | 80.78 | 2.3021× |

AR records use the matching saved target and evaluation backend. No new AR runs were launched. Phase2 EAGLE has no separately saved EAGLE-runtime AR control in this extension table, so no such ratio is inferred.

## Token counts and request latency

| Dataset | Method | Returned tokens | Verification steps | Emitted counter | Mean latency (s) | Median (s) | P95 (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Math | 5W MSE25 | 88033 | 11010 | 87905 | 3.17 | 2.59 | 7.94 |
| Math | DFlash native | 88033 | 10706 | 87905 | 3.18 | 2.54 | 8.10 |
| Math | MLP MSE25 | 88033 | 11015 | 87905 | 3.23 | 2.64 | 8.22 |
| Math | 5W AUF (8k) | 88033 | 12754 | 87905 | 3.68 | 2.95 | 9.47 |
| Math | EAGLE MSE25 | 90816 | 24986 | 90906 | 9.22 | 7.04 | 26.23 |
| Math | EAGLE native | 86454 | 23883 | 86630 | 9.31 | 7.51 | 24.76 |
| GSM8K | DFlash native | 36111 | 5554 | 35983 | 1.62 | 1.52 | 2.63 |
| GSM8K | 5W MSE25 | 36111 | 5765 | 35983 | 1.62 | 1.49 | 2.65 |
| GSM8K | MLP MSE25 | 36111 | 5751 | 35983 | 1.64 | 1.49 | 2.72 |
| GSM8K | 5W AUF (8k) | 36111 | 6338 | 35983 | 1.79 | 1.64 | 2.95 |
| GSM8K | EAGLE MSE25 | 35239 | 9185 | 35270 | 3.41 | 3.32 | 5.22 |
| GSM8K | EAGLE native | 35131 | 9514 | 35322 | 3.61 | 3.42 | 5.81 |
| Code | DFlash native | 70679 | 12500 | 70551 | 3.82 | 3.50 | 8.99 |
| Code | MLP MSE25 | 70679 | 14754 | 70551 | 4.40 | 4.08 | 9.73 |
| Code | 5W MSE25 | 70679 | 14887 | 70551 | 4.40 | 4.22 | 9.79 |
| Code | 5W AUF (8k) | 70679 | 30078 | 70551 | 8.92 | 8.51 | 20.00 |
| Code | EAGLE MSE25 | 73100 | 22214 | 73160 | 8.38 | 7.26 | 23.94 |
| Code | EAGLE native | 70364 | 21077 | 70633 | 8.21 | 7.50 | 18.28 |
| Chat | DFlash native | 90816 | 27286 | 90688 | 8.10 | 7.11 | 19.90 |
| Chat | 5W MSE25 | 90816 | 31415 | 90688 | 9.05 | 7.79 | 23.14 |
| Chat | MLP MSE25 | 90816 | 31254 | 90688 | 9.05 | 8.11 | 23.16 |
| Chat | 5W AUF (8k) | 90816 | 51331 | 90688 | 14.83 | 12.97 | 35.05 |
| Chat | EAGLE MSE25 | 93326 | 29798 | 93432 | 11.24 | 10.14 | 26.00 |
| Chat | EAGLE native | 93102 | 29293 | 93337 | 11.29 | 10.14 | 26.15 |

## Fitting cost

| Dataset | Method | Trainable parameters | Fit minutes | Fit GPU-hours | Last logged loss |
| --- | --- | --- | --- | --- | --- |
| All four | 5W MSE25 | 52428800 | 24.9 | 0.8300 | See original fit record |
| All four | MLP MSE25 | 60948480 | 26.667 | 0.8889 | 0.229555 |
| All four | EAGLE MSE25 | 31457280 | 13.132 | 0.4377 | 0.221323 |
| All four | 5W AUF (8k) | 52428800 | 31.643 | 1.0548 | 0.395707 |

Last logged MSE loss is a sampled final-batch diagnostic; AUF summaries report mean microbatch loss over the epoch. These values are not directly comparable across objectives or datasets. GPU-hours here count the fitting loop only, not allocation time.

## Interpretation

The matched comparison is MLP MSE25 versus 5W MSE25. Small or workload-dependent acceptance changes do not establish an advantage for the nonlinear interface, and its residual branches add inference work. The EAGLE transfer must be assessed against native EAGLE as well as DFlash: different draft bodies and tree verification costs prevent attribution of absolute throughput differences to the mapper alone. These are architecture/performance extensions, not new causal mechanistic probes.

## Provenance

Metrics are regenerated from saved per-request JSON and fit summaries by `scripts/report_extensions_20260916.py`. Exact source paths and SHA-256 hashes are saved in `experiments/dflash_extensions_20260916/report_sources.json`; machine-readable metrics are in `report_metrics.json`. The phase-specific experiment manifests retain model paths and pinned implementation hashes.
