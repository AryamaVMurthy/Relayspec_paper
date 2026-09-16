# EAGLE3 AUF adaptation for three LoRA targets

Completed 16 September 2026. **All three fits and all 384 evaluation requests are complete.** This report covers only the new EAGLE AUF run; saved native, AR and previous EAGLE MSE measurements are included solely as comparison controls.

## Main results

Each row uses 128 heldout requests with the corresponding Qwen3-4B LoRA active. The native reference is the original unmodified EAGLE3 drafter serving that same adapted target.

| Domain | Mean TPS | Pooled TPS | Avg acceptance length | vs native EAGLE | TPS gain vs native | vs saved AR | Cohort latency reduction vs native |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GSM8K | 120.35 | 118.46 | 5.6951 | 1.6906× | 69.06% | 3.8217× | 38.83% |
| NanoCoder | 119.30 | 117.70 | 5.6677 | 1.4588× | 45.88% | 3.7517× | 32.71% |
| KiCad | 151.86 | 150.13 | 7.3632 | 2.0570× | 105.70% | 4.9997× | 51.37% |

## Saved controls and acceptance improvement

| Domain | AR mean TPS | Native EAGLE TPS | AUF EAGLE TPS | Native acceptance | AUF acceptance | Acceptance increase |
| --- | --- | --- | --- | --- | --- | --- |
| GSM8K | 31.49 | 71.19 | 120.35 | 3.3466 | 5.6951 | 70.18% |
| NanoCoder | 31.80 | 81.77 | 119.30 | 3.8519 | 5.6677 | 47.14% |
| KiCad | 30.37 | 73.82 | 151.86 | 3.7944 | 7.3632 | 94.05% |

## Comparison with the previous EAGLE MSE fit

These are different objectives and supervision workloads, not a controlled change of only one algebraic loss term. MSE reconstructed features at every saved position; AUF backpropagates token prediction loss through the frozen drafter on sampled response anchors.

| Domain | Previous MSE TPS | New AUF TPS | TPS gain vs MSE | MSE acceptance | AUF acceptance |
| --- | --- | --- | --- | --- | --- |
| GSM8K | 71.89 | 120.35 | 67.41% | 3.4287 | 5.6951 |
| NanoCoder | 75.34 | 119.30 | 58.35% | 3.7005 | 5.6677 |
| KiCad | 78.70 | 151.86 | 92.96% | 3.8532 | 7.3632 |

## Token counts and latency

| Domain | Returned tokens | Upstream emitted-token counter | Verification steps | Summed latency (s) | Mean request (s) | Median request (s) | P95 request (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GSM8K | 19181 | 19181 | 3368 | 161.92 | 1.265 | 1.110 | 2.295 |
| NanoCoder | 27511 | 27511 | 4854 | 233.74 | 1.826 | 1.578 | 4.174 |
| KiCad | 628639 | 628639 | 85376 | 4187.31 | 32.713 | 32.562 | 57.942 |

TPS means returned tokens per second. **Mean TPS** averages the 128 individual request rates; **pooled TPS** divides all returned tokens by summed request latency. Speedup divides mean TPS by the matched control's mean TPS. Cohort latency reduction is one minus the ratio of summed latencies; it is not necessarily one minus inverse mean-TPS speedup.

Acceptance length is the pooled upstream emitted-token counter divided by verification steps, including the correction/bonus token. The counter is not merely accepted draft proposals. Returned-token and emitted counters need not coincide under EAGLE's original raw-tail stopping convention. P95 is the nearest-rank 95th percentile. Summed latency is serial request time, not parallel evaluation wall time.

## Architecture

The target is frozen Qwen3-4B with the existing domain LoRA merged into it. The EAGLE3 drafter receives three target feature vectors, each of width 2,560. The only trainable weights are three independent square linear maps, initialized to identity:

$$
W_i\in\mathbb{R}^{2560\times2560},\quad W_i^{(0)}=I,\qquad i=1,2,3.
$$

Writing the original frozen fusion as three column blocks, its adapted output is

$$
z=F_1W_1h_1+F_2W_2h_2+F_3W_3h_3.
$$

All original draft-body weights, fusion weights, normalizations, embedding and vocabulary head remain frozen. This trains **19,660,800 parameters**, with no MLP. At export, combine each product into the fusion matrix:

$$
F_{\mathrm{export}}=[F_1W_1\;F_2W_2\;F_3W_3].
$$

Thus inference uses the original EAGLE architecture with an updated fusion weight; there are no extra mapper operations.

## Data and training recipe

| Setting | Used in this run |
|---|---|
| Domains | GSM8K Math LoRA, KiCad LoRA, NanoCoder LoRA |
| Examples | 4,096 saved domain-specific training trajectories per LoRA |
| Generation cap | Original maximum 4,096 response tokens; early stopping allowed |
| Features | Reused dense active-LoRA EAGLE target features on the same saved tokens; no new generation or extraction |
| Target taps | Existing zero-based saved target taps [2, 18, 33] |
| Epochs / updates | One epoch / 512 updates; final checkpoint evaluated |
| Batch | Global eight examples; two GPUs, four sequential examples per GPU |
| Anchors | Up to 32 response anchors per example, sampled without replacement |
| Draft path | Up to eight recurrent predictions per anchor |
| Optimizer | AdamW, peak LR 1e-4, weight decay 0, clipping norm 1 |
| Schedule | 5% warmup then cosine; seed 42 |
| Precision | FP32 trainable maps/fusion arithmetic, frozen BF16 drafter; BF16 folded export; TF32 disabled |
| Hardware | Two L40S GPUs per fit |

The full saved prefix is processed causally. Each anchor can access only its preceding target-feature context; future keys are masked. The draft then follows its own recurrent hidden states. Teacher tokens are fed only on surviving paths where they equal greedy predictions. Prompt tokens provide context but are not direct CE labels.

## Exact AUF objective

This is a **custom recurrent EAGLE adaptation of AUF**, not DFlash's masked-block training implementation. For one example, let $a$ identify an anchor and $j=1,\ldots,8$ the next draft position. Let $y_{a,j}$ be its saved target token, and $\hat y_{a,j}$ the draft's greedy token. A position is eligible only if all earlier predictions were correct:

$$
w_{a,j}=\prod_{k<j}\mathbf{1}[\hat y_{a,k}=y_{a,k}].
$$

The empty product at $j=1$ is one. Therefore the first wrong prediction **does receive loss**, and everything after it receives none. The correctness mask is detached from gradients.

Let $v_{a,j}$ mark a valid saved position and $r_{a,j}$ mark a token represented by the frozen reduced vocabulary head. With $\ell_{a,j}=-\log q_{a,j}(y_{a,j})$, the example loss is

$$
L_{\mathrm{example}}=
\frac{\sum_{a,j}w_{a,j}v_{a,j}r_{a,j}\ell_{a,j}}
{\max(1,\sum_{a,j}w_{a,j}v_{a,j}r_{a,j})}.
$$

An out-of-head-vocabulary token cannot be predicted by this fixed head: it ends the surviving path but contributes no trainable CE term. Counts of these events are recorded. Each optimizer update averages the eight example-normalized losses, rather than globally normalizing all tokens. No MSE, positional decay or soft-logit KL is added.

## Fitting time and training diagnostics

| Domain | Trainable parameters | Fit seconds | Fit minutes | Fit GPU-hours | Last logged loss | Rank-0 supervised positions | Rank-0 out-of-head events |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GSM8K | 19660800 | 163.92 | 2.732 | 0.0911 | 0.895063 | 261136 | 3003 |
| NanoCoder | 19660800 | 184.23 | 3.070 | 0.1023 | 0.866957 | 251000 | 6032 |
| KiCad | 19660800 | 1149.30 | 19.155 | 0.6385 | 0.145231 | 394676 | 12115 |

Fit times cover the fitting/export interval measured by the script, excluding initial model setup, earlier feature extraction and evaluation. GPU-hours multiply this interval by two. The loss is the last rank-0 local batch mean, **not an epoch-average or heldout loss**. Position/event counters are cumulative rank-0 counts, not global totals or unique tokens.

## Evaluation recipe

- Transformers, one L40S per serial request stream; BF16, greedy temperature 0, top-p 1, top-k 0, thinking disabled.
- Same 128 original heldout prompts per domain and same chat templates/tokenizers as the saved controls.
- Maximum output tokens: 2,048 for GSM8K and NanoCoder; 8,192 for KiCad. These are caps, not forced lengths.
- Pinned existing EAGLE compatibility implementation; tree token budget 60, depth 7, top-k 10. Preserve its raw-tail/count convention, including possible cap overshoot.
- Two excluded 64-token warmups per worker. CUDA synchronization brackets each complete request. Model loading and warmup are excluded.
- Math and NanoCoder: two 64-request shards each. KiCad: four 32-request shards. Each merged cohort has 128 unique IDs, matching its controls.
- Saved native EAGLE and EAGLE-runtime AR results are reused; no controls were rerun.

These are single timing passes. No repeated-run confidence interval, acceptance-length histogram, task-accuracy measurement or concurrent-serving result was collected in this run. The large throughput improvements accompany substantial acceptance increases; small numerical timing differences should not be treated as statistically established.

## Artifacts

The run is in `experiments/phase1_eagle_auf_20260916/`: `train.py`, `evaluate.py`, `README.md`, per-domain `runs/*/summary.json`, and merged `measurements/*_eagle.json`. Remote per-domain run directories also retain the raw mapper and folded EAGLE export. Launch IDs: fits 34672–34674; evaluations 34675–34682. Saved control artifacts are under `experiments/dflash_ready_20260913/measurements/`; previous MSE outputs are under `experiments/phase1_extensions_20260916/measurements/`.
