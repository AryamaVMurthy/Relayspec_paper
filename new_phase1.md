# Phase 1 — new MLP and EAGLE mapper results

Snapshot: 16 September 2026. Only complete 128-request cohorts are included. All six new evaluations are complete.

For layer slot $i$, the MLP is

$$
g_i(h)=W_i h+B_i\operatorname{SiLU}(A_i h),\qquad z=\sum_i F_i g_i(h_i),\qquad c=N(z).
$$

$F_i$ is the original frozen fusion block, and $N$ is the original frozen RMSNorm. $A_i$ reduces to 256 coordinates; $B_i$ expands back to the source width. The draft body, target, source embedding/head, fusion and normalization remain frozen. $B_i$ starts at zero. At deployment $F_iW_i$ and $F_iB_i$ are precomputed; the nonlinear branch remains active. EAGLE instead uses three **linear** maps $g_i(h)=W_i h$, folded completely into its original fusion. It has no MLP branch.

## Training recipe

The target is **Qwen3-4B with the corresponding existing GSM8K, KiCad or NanoCoder LoRA active**, both in training trajectories and evaluation. Source features for EAGLE reconstruction come from base Qwen3-4B. All maps are square, 2,560 to 2,560; $W_i$ starts at identity, $A_i$ at Xavier and $B_i=0$.

| Setting | DFlash MLP and old 5W AUF | EAGLE three linear maps |
| --- | --- | --- |
| Data | Same 4,096 saved domain-specific trajectories, response cap 4,096 | Same saved token sequences; new paired EAGLE features |
| Trainable parameters | MLP 39,321,600; old 5W 32,768,000 | 19,660,800 |
| Loss | AUF, first greedy error included; 32 response anchors/example | Normalized layer + post-RMSNorm context MSE |
| Coverage | Up to 15 supervised positions per block; AUF masks positions after first error | Every prompt and response position (100%) |
| Schedule | One epoch, 512 optimizer updates, final checkpoint | Same |
| Optimizer | AdamW, LR 0.0001, weight decay 0, clip norm 1; 5% warmup then cosine | Same |
| Batch | Two GPUs, two examples/GPU, two accumulation steps: global batch 8 | Same |
| Precision/seed | FP32 trainable maps; frozen BF16 model; seed 42 | FP32 maps/loss, BF16 cached features/export; seed 42 |

AUF uses the unchanged parent block construction: one clean anchor plus 15 masked positions, bidirectional within-block attention and only prior target context. For valid position $j$, the detached weight is one if every earlier proposal was correct; the first wrong token is included. With CE $\ell_j=-\log q_j(x_j)$, the local loss is $\sum_j w_j\ell_j/\sum_j w_j$, masking invalid positions. The optimizer averages the four locally normalized microbatch losses from two GPUs and two accumulation steps. There is no positional decay in this pilot. EAGLE MSE is a different objective and feature workload: the comparison is not an isolated architecture ablation.

For reconstruction, define $D(u,v)=\lVert u-v\rVert_2^2/(\lVert v\rVert_2^2+10^{-6})$. With $K$ maps and selected token positions $S_b$ in example $b$:

$$
L_{MSE}=\frac1B\sum_b\frac1{|S_b|}\sum_{p\in S_b}\left[\frac1K\sum_{i=1}^K D(g_i(h^t_{bpi}),h^s_{bpi})+D\left(N\left(\sum_iF_i g_i(h^t_{bpi})\right),N\left(\sum_iF_i h^s_{bpi}\right)\right)\right].
$$

The target and source process the **same saved tokens**. Selected positions retain their full causal context; sampling removes supervised positions, not preceding context.

## Evaluation recipe

One L40S GPU/request stream; Transformers BF16, greedy nonthinking, temperature 0, top-p 1, top-k 0. The same 128 saved domain prompts and original chat templates/tokenizers are used. Response cap is 2,048 for GSM8K/NanoCoder and 8,192 for KiCad. Two 64-token warmups are excluded; CUDA synchronization brackets the complete generation call. The LoRA is merged into the frozen target. DFlash uses block size 16; EAGLE uses its existing tree (60 tokens, depth 7, top-k 10). DFlash keeps its canonical stopping contract; EAGLE retains the upstream raw output tail, including possible cap overshoot, matching its saved Phase1 control. KiCad EAGLE evaluation uses eight disjoint 16-prompt shards across eight L40S GPUs (four on each of node07 and node09); each GPU processes requests serially, with two excluded warmups per shard. Summed request latency is not the eight-GPU job wall time. Old 5W results below are the Transformers evaluations of the original AUF32 checkpoints, not their vLLM results.

## Metric definitions

All result rows contain 128 requests. Mean TPS is the arithmetic mean of each request's returned-token count divided by its full request latency. Pooled TPS is total returned tokens divided by summed request latency. Native-relative speedup is the ratio of mean TPS within the same drafter family. Latency saving is $100(1-T/T_{native})$ on the cohort; it need not equal $100(1-1/S)$ for a mean-TPS speedup $S$. Positive changes mean improvement. Acceptance length is pooled emitted verification tokens divided by verification steps, including correction/bonus tokens; DFlash excludes its initial prefill token. EAGLE uses its saved upstream emitted-token counters. These are not percentages of proposed tokens accepted. EAGLE and DFlash use different proposal structures, so equal acceptance lengths do not imply equal cost. Timings are single passes, not statistical proof of small differences. Fit time excludes feature extraction and model setup.

## Results: grouped by dataset and drafter family, descending throughput

| Dataset | Method | Mean TPS | Pooled TPS | vs family native | Acceptance length | Total latency (s) | Latency saving vs native |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GSM8K | 5W AUF | 179.83 | 169.22 | 1.3951× | 5.988 | 115.55 | 24.32% |
| GSM8K | MLP AUF | 177.94 | 166.56 | 1.3805× | 5.953 | 117.39 | 23.12% |
| GSM8K | DFlash native | 128.90 | 128.06 | 1.0000× | 4.472 | 152.69 | 0.00% |
| GSM8K | EAGLE MSE | 71.89 | 73.76 | 1.0099× | 3.429 | 281.92 | -6.50% |
| GSM8K | EAGLE native | 71.19 | 71.82 | 1.0000× | 3.347 | 264.72 | 0.00% |
| KiCad | MLP AUF | 321.76 | 320.43 | 1.9098× | 11.411 | 2013.36 | 47.16% |
| KiCad | 5W AUF | 312.47 | 310.91 | 1.8547× | 11.256 | 2074.97 | 45.54% |
| KiCad | DFlash native | 168.48 | 169.33 | 1.0000× | 5.991 | 3810.01 | 0.00% |
| KiCad | EAGLE MSE | 78.70 | 79.45 | 1.0660× | 3.853 | 7892.17 | 8.35% |
| KiCad | EAGLE native | 73.82 | 74.68 | 1.0000× | 3.794 | 8611.21 | 0.00% |
| NanoCoder | 5W AUF | 171.73 | 162.25 | 1.1389× | 5.552 | 171.73 | 10.61% |
| NanoCoder | MLP AUF | 169.74 | 159.78 | 1.1257× | 5.520 | 174.39 | 9.22% |
| NanoCoder | DFlash native | 150.79 | 145.04 | 1.0000× | 5.111 | 192.11 | 0.00% |
| NanoCoder | EAGLE native | 81.77 | 83.71 | 1.0000× | 3.852 | 347.37 | 0.00% |
| NanoCoder | EAGLE MSE | 75.34 | 77.03 | 0.9213× | 3.701 | 375.87 | -8.20% |

## Explicit old 5W comparison

The EAGLE-to-5W rows compare complete pipelines, not matched drafter architectures. Acceptance counters also follow their respective upstream conventions.

| Dataset | New method | Reference | Old TPS | New TPS | TPS change | Old length | New length | Length change | Latency saving vs reference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GSM8K | MLP AUF | 5W AUF | 179.83 | 177.94 | -1.05% | 5.988 | 5.953 | -0.035 | -1.60% |
| GSM8K | EAGLE MSE | 5W AUF | 179.83 | 71.89 | -60.02% | 5.988 | 3.429 | -2.559 | -143.99% |
| GSM8K | EAGLE MSE | EAGLE native | 71.19 | 71.89 | 0.99% | 3.347 | 3.429 | 0.082 | -6.50% |
| KiCad | MLP AUF | 5W AUF | 312.47 | 321.76 | 2.97% | 11.256 | 11.411 | 0.154 | 2.97% |
| KiCad | EAGLE MSE | 5W AUF | 312.47 | 78.70 | -74.81% | 11.256 | 3.853 | -7.403 | -280.35% |
| KiCad | EAGLE MSE | EAGLE native | 73.82 | 78.70 | 6.60% | 3.794 | 3.853 | 0.059 | 8.35% |
| NanoCoder | MLP AUF | 5W AUF | 171.73 | 169.74 | -1.16% | 5.552 | 5.520 | -0.032 | -1.55% |
| NanoCoder | EAGLE MSE | 5W AUF | 171.73 | 75.34 | -56.13% | 5.552 | 3.701 | -1.851 | -118.87% |
| NanoCoder | EAGLE MSE | EAGLE native | 81.77 | 75.34 | -7.87% | 3.852 | 3.701 | -0.151 | -8.20% |

## Saved AR comparison

| Dataset | Method | AR mean TPS | Method mean TPS | vs AR |
| --- | --- | --- | --- | --- |
| GSM8K | DFlash native | 38.81 | 128.90 | 3.3212× |
| GSM8K | 5W AUF | 38.81 | 179.83 | 4.6334× |
| GSM8K | EAGLE native | 31.49 | 71.19 | 2.2605× |
| GSM8K | MLP AUF | 38.81 | 177.94 | 4.5848× |
| GSM8K | EAGLE MSE | 31.49 | 71.89 | 2.2828× |
| KiCad | DFlash native | 38.42 | 168.48 | 4.3850× |
| KiCad | 5W AUF | 38.42 | 312.47 | 8.1327× |
| KiCad | EAGLE native | 30.37 | 73.82 | 2.4305× |
| KiCad | MLP AUF | 38.42 | 321.76 | 8.3745× |
| KiCad | EAGLE MSE | 30.37 | 78.70 | 2.5910× |
| NanoCoder | DFlash native | 39.01 | 150.79 | 3.8656× |
| NanoCoder | 5W AUF | 39.01 | 171.73 | 4.4025× |
| NanoCoder | EAGLE native | 31.80 | 81.77 | 2.5717× |
| NanoCoder | MLP AUF | 39.01 | 169.74 | 4.3514× |
| NanoCoder | EAGLE MSE | 31.80 | 75.34 | 2.3692× |

AR records use the matching saved target and evaluation backend. No new AR runs were launched. Phase2 EAGLE has no separately saved EAGLE-runtime AR control in this extension table, so no such ratio is inferred.

## Token counts and request latency

| Dataset | Method | Returned tokens | Verification steps | Emitted counter | Mean latency (s) | Median (s) | P95 (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GSM8K | 5W AUF | 19553 | 3244 | 19425 | 0.90 | 0.76 | 1.92 |
| GSM8K | MLP AUF | 19553 | 3263 | 19425 | 0.92 | 0.76 | 1.89 |
| GSM8K | DFlash native | 19553 | 4344 | 19425 | 1.19 | 1.02 | 2.11 |
| GSM8K | EAGLE MSE | 20795 | 6065 | 20795 | 2.20 | 1.89 | 3.66 |
| GSM8K | EAGLE native | 19012 | 5681 | 19012 | 2.07 | 1.89 | 3.57 |
| KiCad | MLP AUF | 645138 | 56527 | 645010 | 15.73 | 15.56 | 26.93 |
| KiCad | 5W AUF | 645138 | 57302 | 645010 | 16.21 | 15.77 | 27.81 |
| KiCad | DFlash native | 645138 | 107660 | 645010 | 29.77 | 28.69 | 52.84 |
| KiCad | EAGLE MSE | 627006 | 162723 | 627006 | 61.66 | 59.48 | 106.24 |
| KiCad | EAGLE native | 643055 | 169475 | 643055 | 67.28 | 66.59 | 111.31 |
| NanoCoder | 5W AUF | 27864 | 4996 | 27736 | 1.34 | 1.09 | 3.21 |
| NanoCoder | MLP AUF | 27864 | 5025 | 27736 | 1.36 | 1.14 | 3.16 |
| NanoCoder | DFlash native | 27864 | 5427 | 27736 | 1.50 | 1.25 | 3.73 |
| NanoCoder | EAGLE native | 29078 | 7549 | 29078 | 2.71 | 2.27 | 5.51 |
| NanoCoder | EAGLE MSE | 28953 | 7824 | 28953 | 2.94 | 2.43 | 6.36 |

## Fitting cost

| Dataset | Method | Trainable parameters | Fit minutes | Fit GPU-hours | Last logged loss |
| --- | --- | --- | --- | --- | --- |
| GSM8K | 5W AUF | 32768000 | 2.663 | 0.0888 | 0.401822 |
| GSM8K | MLP AUF | 39321600 | 2.647 | 0.0882 | 0.403498 |
| GSM8K | EAGLE MSE | 19660800 | 0.247 | 0.0082 | 0.000795 |
| KiCad | 5W AUF | 32768000 | 5.321 | 0.1774 | 0.123163 |
| KiCad | MLP AUF | 39321600 | 6.110 | 0.2037 | 0.121155 |
| KiCad | EAGLE MSE | 19660800 | 3.734 | 0.1245 | 1.069542 |
| NanoCoder | 5W AUF | 32768000 | 2.756 | 0.0919 | 0.452404 |
| NanoCoder | MLP AUF | 39321600 | 2.749 | 0.0916 | 0.454443 |
| NanoCoder | EAGLE MSE | 19660800 | 0.330 | 0.0110 | 0.078077 |

Last logged MSE loss is a sampled final-batch diagnostic; AUF summaries report mean microbatch loss over the epoch. These values are not directly comparable across objectives or datasets. GPU-hours here count the fitting loop only, not allocation time.

EAGLE requires fresh paired feature extraction; DFlash reuses its existing feature cache. Completed capture receipts report:

| Dataset | Examples | Teacher passes (min) | Capture wall time (min) |
| --- | --- | --- | --- |
| GSM8K | 4096 | 7.20 | 8.43 |
| KiCad | 4096 | 45.78 | 57.88 |
| NanoCoder | 4096 | 7.47 | 9.00 |

## Interpretation

MLP versus old 5W AUF is the direct architecture comparison: same data, AUF32 loss and fitting schedule. The three-map EAGLE fit tests feature reconstruction for an active LoRA target; it does not test AUF-trained EAGLE. Its fitting loop omits draft-body/vocabulary-head backpropagation, explaining why fit-only time is much smaller; feature capture must still be counted. Small throughput differences without acceptance gains should not be called a robust architectural improvement.

## Provenance

Metrics are regenerated from saved per-request JSON and fit summaries by `scripts/report_extensions_20260916.py`. Exact source paths and SHA-256 hashes are saved in `experiments/dflash_extensions_20260916/report_sources.json`; machine-readable metrics are in `report_metrics.json`. The phase-specific experiment manifests retain model paths and pinned implementation hashes.
