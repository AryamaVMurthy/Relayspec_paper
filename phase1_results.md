# Phase 1 — LoRA results

<!-- AGREEMENT_NOTE -->
In the verified selected Transformers comparisons, native and adapted drafters matched all 128 output sequences per domain. In vLLM, token sequences diverged, consistent with documented numerical and batching variability. The authors completed a manual audit of all divergent answers and found them semantically similar. Semantic similarity does not imply identical token counts. [vLLM documentation](https://docs.vllm.ai/en/stable/features/batch_invariance/).
<!-- /AGREEMENT_NOTE -->





This is the consolidated numerical record for [Phase 1](papers/phase1.pdf). It contains the current matched comparisons and all completed, in-scope historical studies, including negative results. [Phase 1 probes](phase1_probe.md) develops the mechanistic interpretation and preserves its quantitative evidence. [Phase 2 results](phase2_results.md) and [Phase 2 probes](phase2_probe.md) cover cross-model transfer separately.

## How to compare the measurements

- The target remains the task-adapted model unless a row explicitly names a base-target control. A draft-body LoRA is distinct from the target LoRA.
- Current Transformers TPS is the arithmetic mean of per-request rates. Historical serial vLLM rates generally pool output tokens over summed request time. Concurrent TPS divides all returned tokens by workload wall time, including drain. Compare only matched definitions and cohorts.
- Live acceptance includes the verifier contribution where specified: $1+\text{accepted draft tokens}/\text{verification steps}$. Offline correct-prefix length excludes that contribution. They are different measurements.
- Native-relative throughput is not AR-relative speedup. A missing measurement is not inferred from CE, matrix energy or another backend.
- Screening and heldout use different evaluation cohorts. Reused controls and compressed checkpoints are not independent new training runs; fitting time remains inherited where stated.
- Earlier recipes remain separately labelled. The original 25% paired-feature MSE and later MSE50/MSE100 controls are distinct. The excluded sparse MSE configuration, Rust/CS work, stopped repeats and uncompleted probes are not restored as paper evidence.

## Coverage

41 scoped reports; 42 architecture/objective cells; 96 Transformers supervision-budget cells; 24 verified vLLM size/anchor cells; six corrected MSE coverage controls; 52 optimized serving measurements; ten 8B standalone entries; 76 live five-map mechanism rows. Historical controls recur across tables and must not be counted as independent experiments.

## Study index

- [2026-09-07_dflash_lora_objective](#study-1)
- [2026-09-07_dflash_lora_pilot](#study-2)
- [2026-09-07_dflash_lora_rank](#study-3)
- [2026-09-08_dflash_fusion_mechanism](#study-4)
- [2026-09-08_dflash_interface_architecture](#study-5)
- [2026-09-08_dflash_kicad_mapping](#study-6)
- [2026-09-08_dflash_lora_decisive_controls](#study-7)
- [2026-09-08_dflash_workload_transfer](#study-8)
- [2026-09-09_dflash_fusion_interpretability](#study-9)
- [2026-09-09_dflash_moe_deployment](#study-10)
- [2026-09-09_dflash_nanocoder](#study-11)
- [2026-09-09_dflash_nanocoder_dense](#study-12)
- [2026-09-09_dflash_nanocoder_diagnosis](#study-13)
- [2026-09-09_dflash_nanocoder_objectives](#study-14)
- [2026-09-09_dflash_nanocoder_scaling](#study-15)
- [2026-09-09_dflash_steering](#study-16)
- [2026-09-09_dflash_steering_causality](#study-17)
- [2026-09-10_dflash_auf_domains](#study-18)
- [2026-09-10_dflash_auf_scale](#study-19)
- [2026-09-10_dflash_kicad_auf_data_scale](#study-20)
- [2026-09-10_dflash_nanocoder_direct](#study-21)
- [2026-09-11_dflash_auf_mapper_probes](#study-22)
- [2026-09-11_dflash_auf_mixture](#study-23)
- [2026-09-11_dflash_auf_target_router](#study-24)
- [2026-09-11_dflash_body_decay](#study-25)
- [2026-09-11_dflash_joint_auf](#study-26)
- [2026-09-11_dflash_nanocoder_sampled_auf](#study-27)
- [2026-09-11_dflash_pooled_mixture](#study-28)
- [2026-09-11_dflash_pooled_target_router](#study-29)
- [2026-09-12_decay32](#study-30)
- [2026-09-12_dflash_auf_scaling](#study-31)
- [2026-09-12_dflash_optimized_baselines](#study-32)
- [2026-09-12_dflash_transformers_ablation](#study-33)
- [2026-09-12_dflash_transformers_gsm8k](#study-34)
- [2026-09-12_jointw32](#study-35)
- [2026-09-13_dflash_eagle3](#study-36)
- [2026-09-13_dflash_fivew_mechanism](#study-37)
- [2026-09-13_dflash_matched_scaling](#study-38)
- [2026-09-13_dflash_qwen8b](#study-39)
- [2026-09-13_dflash_reconstruction](#study-40)
- [2026-09-13_eagle3_transformers](#study-41)

<a id="study-1"></a>


## 1. 2026-09-07_dflash_lora_objective

Source: [2026-09-07_dflash_lora_objective.md](paper/data/history/2026-09-07_dflash_lora_objective.md).

### GSM8K LoRA: token-trained interface versus drafter LoRA

The token-trained mapper improves mean acceptance length from **4.3605 to 5.6448** and gives **1.2733× speedup over the unchanged drafter**. Direct drafter LoRA reaches 5.6808 and 1.2847×. All four speculative pipelines match the saved adapted-target autoregressive token IDs and finish reasons on **128/128 prompts**, with **17,128 tokens each**.

Acceptance is the pooled vLLM convention, **1 + accepted draft tokens / verification steps**, including the bonus token. Original/MSE/CE-mapper/draft-LoRA counters are respectively 13,227/3,936; 13,213/3,951; 14,097/3,035; and 14,108/3,014. Terminal clipping means these counters need not sum to the retained output count. CE mapper acceptance is 29.45% higher than the original control; request latency is 21.46% lower. Its measured throughput is 99.11% of direct drafter LoRA's throughput.

The mapper can improve acceptance while the drafter backbone stays frozen. This supports token-level adaptation as a viable diagnostic direction. It does **not** yet isolate a LoRA-specific advantage: GSM8K domain adaptation may also help a drafter serving the unadapted target. The MSE fit uses different sampling and optimizer settings, so this is not a controlled loss-only ablation. The approximately 0.9% difference between the two new methods is not a repeated-run significance result. Matched base-target controls are reported separately in the training-target study; this comparison alone does not isolate target-specific adaptation.

#### Parameterization

For five adapted-target features $h_j$, identity-initialized $M_j$ map2,560→2,560 and only these matrices train. With frozen fusion blocks $F_{0,j}$ and frozen RMSNorm $N_0$,

$$
c=N_0\left(\sum_{j=1}^5F_{0,j}M_jh_j\right),\qquad
F_{export}=[F_{0,1}M_1,\ldots,F_{0,5}M_5].
$$

The mapper has32,768,000 parameters. The comparator instead trains rank 112/alpha 112 updates $W=W_0+BA$ in q/k/v/o/gate/up/down of all five draft blocks,32,112,640 parameters, random A/zero B initialization, no LoRA dropout or bias. Both retain frozen original embedding, head and normalization. Matrices are merged into BF16 exports for inference.

#### Data and training

Reused the existing 4,096 unique GSM8K training rollouts and 128 disjoint test prompts from the [LoRA pilot](paper/data/history/2026-09-07_dflash_lora_pilot.md). Target: `Qwen/Qwen3-4B` plus `witcheer/qwen3-4b-gsm8k-grpo`, adapter revision `f52e1ed1b3accd404d37a9621db0536dd99bdfc5`. Dataset `openai/gsm8k` revision `740312add88f781978c0658806c59bc2815b9866`; author prompt source revision `816390766c9251cb0e8a1699b145ec56dd83bf67`. Exact selected messages and IDs remain in the parent family's `setup/math_{train,eval}.json` and `setup/math_contract.json`.

Training/test selection uses Python Random(42), unique stripped-question SHA256 IDs and disjoint train/test splits. The user message is `Solve the math problem. Show brief reasoning, then end with '#### <answer>'.\n\nProblem: {q}\nSolution:`, rendered with the base Qwen chat template and thinking disabled.

The cached target revision is `1cfa9a7208912126459214e8b04321603b3df60c`; original `z-lab/Qwen3-4B-DFlash-b16` drafter revision is `b74e3a329c4d963783143b1e970d95b002be72bd`. Download metadata and reused runtime-source hashes are recorded in `results/model_provenance.json`.

The 4,096-token training response cap is unchanged. Saved responses total 544,585 tokens; 4,094 stop naturally and two hit the cap. New native-vLLM capture persists all five adapted-target context vectors at every token on these saved sequences: 930,941 prompt-plus-response positions, 128 shards, **23,840,147,904 bytes (23.84 GB)**. Original paired 25% feature caches remain intact. Dense context supports the block attention objective; supervision still samples eight response anchors per example rather than training on every output position.

Training uses unchanged concrete DFlash model, mask, block construction and hard-token CE from [SpecForge revision 953d43a](https://github.com/sgl-project/SpecForge/tree/953d43a0c1c0f5e32989dc43f91ce5fc2d9ddfef), with PyTorch SDPA. For a microbatch, eight valid anchors per example are sampled uniformly without replacement. Each block has one clean anchor token and 15 mask tokens. Context attention admits only target hidden states strictly before the anchor; attention inside a draft block is bidirectional and blocks cannot attend to each other. For anchor `a`, position `j` predicts saved target token `x[a+j]`, not a shifted label. The clean anchor, padding and nonsupervised positions receive no loss.

$$
\mathcal L_{\mathcal B}
=-\frac{\sum_{i,a}\sum_{j=1}^{15}m_{iaj}\log q_\theta(x_{i,a+j}\mid H^{\mathrm{adapted}}_{i,<a},x_{i,a},\mathrm{MASK}^{15})_j}
{\sum_{i,a}\sum_{j=1}^{15}m_{iaj}}.
$$

The vocabulary head is frozen. Full-vocabulary CE is reduced in two-block chunks to bound memory; no position loss decay, soft-logit KL or acceptance-specific auxiliary term is used. Each optimizer step averages four locally token-normalized losses: $L_{\mathrm{step}}=(1/4)\sum_{g=1..2,\ b=1..2}L_{g,b}$. This differs from one global token normalization when valid-token counts vary. Eligible anchors have both $x[a]$ and $x[a+1]$ inside the response; mask token ID151669. The five selected layers are zero-based[1,9,17,25,33], each width2,560. RMSNorm retains its frozen gain and epsilon1e-6.

| Variant | Trainable parameters | Initialization | Export |
| --- | ---: | --- | --- |
| Five context maps, each 2,560 → 2,560 | 32,768,000 | Identity; original fusion and backbone frozen | Fold each map into its original fusion weight block |
| Drafter LoRA on q/k/v/o/gate/up/down in all five layers | 32,112,640 | Rank 112, alpha 112, zero initial update | Merge into the original BF16 draft checkpoint schema |

Both use fused AdamW with betas(0.9,0.999), epsilon1e-8, learning rate1e-4, weight decay 0, gradient clipping1,5%warmup/cosine, seed 42, FP32 trainables and BF16 forward autocast. Two examples per GPU, two GPUs per variant, two gradient-accumulation steps: effective batch eight. Same example ordering and anchor seeds; **three epochs, 1,536 optimizer updates, final checkpoint, no dev selection**. Original embedding/head weights are frozen and tied; the target LoRA does not modify them. Both exports use the existing native DFlash serving architecture without additional mapper or draft-LoRA kernels.

Mapper epoch mean microbatch CE: 1.8940 → 1.6415 → 1.4558. Drafter LoRA: 1.8573 → 1.6174 → 1.5541. These are training losses, not held-out loss estimates.

Declared upstream deviations: eight anchors rather than the default 512; custom trainable-parameter selection and export; this existing GSM8K data and prompt; dense vLLM pooling for context capture; concrete-module imports bypass unrelated strategy registration. This is not an official DFlash paper reproduction. The adapter is deployed on BF16 Qwen rather than its author's Unsloth 4-bit setup, as already recorded in the parent pilot.

#### Evaluation and execution cost

Native **vLLM 0.28.0+cu129**, PyTorch 2.13.0+cu129, Transformers 5.16.1, PEFT 0.20.0; node12 L40S GPUs. Greedy, thinking disabled, 2,048-token evaluation cap, block size 16 / 15 speculative tokens. Batch invariance, synchronous scheduling, no torch.compile, FULL_DECODE_ONLY CUDA graphs, FlashAttention. One request at a time per worker. Two disjoint 64-prompt workers per new variant use four GPUs total.

Timings sum measured request wall time, excluding setup, warmup and separately retained first-use compilation retries. Acceptance counters are read outside the timer. The draft uses frozen embedding/head parameters; target LoRA wrapper state is isolated. The shared AR reference takes 543.265 seconds; native/MSE controls use the same fixed prompt contract. All sequences finish before the cap. There is one measured generation per prompt and no claim about saturated server throughput or repeated-run variance.

Manual review includes test indices 0, 7 and 47 (the longest response). Index 7 retains the target's incorrect answer of 8 instead of 16; equality is an inference-equivalence result, not a task-accuracy score. Mask, gradient, export, boundary-case and small-runtime checks are retained as JSON within the run family.

The study consumes0.6797 GPU-hours including validation; the collection/training/evaluation stages consume0.5458 GPU-hours. Fitting loops use 0.1113 and0.1249 GPU-hours. These costs exclude shared training rollout generation, MSE fitting and reference measurements. Four-GPU aggregate limit; per-prompt summed time is not multiworker elapsed time.

Charging448 GPU-seconds of shared dense capture and468 GPU-seconds of mapper fitting allocation gives 916 GPU-seconds. At0.2421 seconds saved per comparable request, capture-plus-fit amortizes after approximately 3,784 requests; this excludes generation and deployment costs.

#### Complete results and fitting records

| Cohort / target | Draft | Tokens | Seconds | Tok/s | Tau | vs native | vs AR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GSM8K/test128 | AR | 17,128 | 543.265 | 31.53 | — | 0.2658× | 1.0000× |
| GSM8K/test128 | Original | 17,128 | 144.389 | 118.62 | 4.3605 | 1.0000× | 3.7625× |
| GSM8K/test128 | MSE mapper | 17,128 | 144.949 | 118.17 | 4.3442 | 0.9961× | 3.7480× |
| GSM8K/test128 | CE mapper | 17,128 | 113.401 | 151.04 | 5.6448 | 1.2733× | 4.7907× |
| GSM8K/test128 | Body LoRA r112 | 17,128 | 112.395 | 152.39 | 5.6808 | 1.2847× | 4.8335× |

AR has no speculative acceptance statistic. Counts, draft-only means, acceptance rates, finish counts, allocation memory and setup times are in the machine-readable table. TTFT/TPOT were not recorded for these LoRA studies.

| Fit | Parameters | Steps | GPUs | Fit s | Fit GPU-h | Epoch-average loss |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| full/lora | 32,112,640 | 1536 | 2 | 224.801 | 0.1249 | 1.857326 → 1.617403 → 1.554068 |
| full/mapper | 32,768,000 | 1536 | 2 | 200.402 | 0.1113 | 1.894004 → 1.641520 → 1.455785 |

#### Reproduction and metric records

The [canonical code collection](paper/data/experiments/dflash_lora/README.md) groups this study's method sources under `src/objective/` and the shared vLLM implementation under `src/runtime/`. Source hashes, dataset/prompt contracts and raw-measurement identities are pinned in `provenance/`; `artifacts/objective/` links the executable backing directory and retained tensors/checkpoints.

[Complete metric CSV](paper/data/experiments/dflash_lora/results/metrics.csv) and [training CSV](paper/data/experiments/dflash_lora/results/training.csv) include all final rows, accepted/proposed token counts, verification iterations, draft-only means, acceptance rates, finish counts, fitting parameters/steps/losses, and available setup/memory measurements. The JSON also includes paired prompt intervals and available per-position counts. Shared controls are identified by identical source paths; they are not independent timing repetitions. Metrics not recorded are left unavailable.

<a id="study-2"></a>


## 2. 2026-09-07_dflash_lora_pilot

Source: [2026-09-07_dflash_lora_pilot.md](paper/data/history/2026-09-07_dflash_lora_pilot.md).

### GSM8K LoRA: feature-reconstruction mapper

A feature-MSE context mapper gives **0.9961× native-drafter speed**, with acceptance length **4.3442 versus 4.3605**. It provides no measured benefit for this target adapter. Equality is an inference-equivalence result, not a math-accuracy score.

#### Models, data and prompts

The target is BF16 `Qwen/Qwen3-4B` (`1cfa9a7208912126459214e8b04321603b3df60c`) with fixed rank 32/alpha 32 attention/MLP LoRA `witcheer/qwen3-4b-gsm8k-grpo` (`f52e1ed1b3accd404d37a9621db0536dd99bdfc5`). The initial draft is `z-lab/Qwen3-4B-DFlash-b16` (`b74e3a329c4d963783143b1e970d95b002be72bd`): five blocks, width2,560. Target and drafter weights remain frozen; only the five context maps train. Original base embedding and tied vocabulary head stay frozen.

Use `openai/gsm8k`, revision `740312add88f781978c0658806c59bc2815b9866`. Python `Random(42)` selects 4,096 unique training questions and128 test questions, deduplicated by stripped-question SHA256, with no overlap. Render this user message through the base Qwen chat template with thinking disabled:

```text
Solve the math problem. Show brief reasoning, then end with '#### <answer>'.

Problem: {q}
Solution:
```

Prompt provenance is author-code revision `816390766c9251cb0e8a1699b145ec56dd83bf67`. Deployment uses original BF16 Qwen instead of the author's Unsloth4-bit derivative; generation caps and sampling are the declared experiment settings, not an author-score replication.

Generate greedy adapted-target continuations with a4,096-new-token cap; persist all token IDs. The4,096 rollouts contain 544,585 response tokens, with two cap terminations. Replay each identical prompt/response sequence through both the base and LoRA target. Capture five post-layer vectors from zero-based layers `[1,9,17,25,33]` (vLLM boundary indices2/10/18/26/34). Select one random position per four-token stratum separately in prompt and response, including partial final strata, using SHA256-derived `42:<group_id>:<segment>` seeds. Persist matched BF16 vectors, position indices, IDs and rollout hashes. The same25% positions are used for both teachers and all epochs. Human reference answers are not training labels, and generated responses are not correctness-filtered.

#### Exact mapper and loss

For each layer j, map adapted-target feature $h_j$ to base-target feature $y_j$ with a bias-free square matrix $M_j$, initialized to identity. All five $M_j$ have shape2,560×2,560: **32,768,000 trainable parameters**. Let $F_0$ be the frozen2,560×12,800 fusion, and $N_0$ its frozen learned RMSNorm with epsilon1e-6. Define

$$
D(u,v)=\frac{\|u-v\|_2^2}{\|v\|_2^2+10^{-6}},\quad
\hat c=N_0(F_0[M_1h_1;\ldots;M_5h_5]),\quad
c=N_0(F_0[y_1;\ldots;y_5]).
$$

The per-position objective is

$$
\ell=\frac15\sum_{j=1}^5D(M_jh_j,y_j)+D(\hat c,c).
$$

There is **no token CE, KL or reward loss**. If example i has $n_i$ cached positions, P is the total number of positions and N=4,096, the actual batch loss is

$$
\mathcal L_B=\frac{P/N}{2048}\sum_{(i,t)\in B}\frac{\ell_{it}}{n_i}.
$$

The denominator stays2,048 for the final partial batch. This equalizes each example's total contribution rather than treating long sequences as proportionally more important. Positions are shuffled within each example with seed 42+epoch and mixed into a bounded buffer; feature-shard order is fixed.

Fused AdamW uses peak LR 1e-3, betas(0.9,0.999), epsilon1e-8, weight decay 0,5%warmup/cosine decay, gradient norm cap 1, FP32 trainable matrices and BF16 autocast. One GPU,2,048 positions/batch,116 batches/epoch, three epochs: **348 updates**. Use the final checkpoint with no dev selection. Fold each $M_j$ into its frozen fusion block, exporting a single BF16 fusion matrix with no added inference multiplication. The32.775-second fitting timer includes training, serialization and export; it excludes rollout/capture and model/optimizer setup.

#### Evaluation and final results

Use vLLM0.28.0+cu129, PyTorch2.13.0+cu129, Transformers5.16.1 on L40S. Greedy temperature0, top-p1, top-k−1, seed 0, thinking disabled, output cap 2,048,15 proposals/block. FLASH_ATTN, batch invariance, synchronous scheduling, compilation mode0 and FULL_DECODE_ONLY CUDA graphs; context capacity5,120, max sequences 16, token-batch budget8,192, memory fraction 0.8, prefix cache off. The draft shares only frozen embedding/head parameters with the target.

Measure one request at a time per GPU, excluding setup, two short warmups, counter reads and cold compilation. Request time includes prefill and decode. Throughput=$N_{\mathrm{tokens}}/T$; speedups=$T_{\mathrm{reference}}/T$. Acceptance $\tau=1+A/V$, with A pooled accepted draft tokens and V verification iterations; draft-only mean=$A/V$. Terminal clipping can make tau differ from returned tokens per step. One timing observation per prompt does not quantify independent hardware-repeat variance.

| Cohort / target | Draft | Tokens | Seconds | Tok/s | Tau | vs native | vs AR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GSM8K/test128 | AR | 17,128 | 543.265 | 31.53 | — | 0.2658× | 1.0000× |
| GSM8K/test128 | Original | 17,128 | 144.389 | 118.62 | 4.3605 | 1.0000× | 3.7625× |
| GSM8K/test128 | MSE mapper | 17,128 | 144.949 | 118.17 | 4.3442 | 0.9961× | 3.7480× |

AR has no speculative acceptance statistic. Counts, draft-only means, acceptance rates, finish counts, allocation memory and setup times are in the machine-readable table. TTFT/TPOT were not recorded for these LoRA studies.

| Fit | Parameters | Steps | GPUs | Fit s | Fit GPU-h | Epoch-average loss |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| full | 32,768,000 | 348 | 1 | 32.775 | 0.0091 | 0.008805 → 0.003463 → 0.000735 |

The two speculative measurements use 13,227/3,936 and13,213/3,951 accepted-token/verification counts respectively. Their evaluation allocations total 445 GPU-seconds (0.1236 GPU-hours). This is separate from fitting, collection and AR-control cost.

#### Reproduction and metric records

The [canonical code collection](paper/data/experiments/dflash_lora/README.md) groups this study's method sources under `src/pilot/` and the shared vLLM implementation under `src/runtime/`. Source hashes, dataset/prompt contracts and raw-measurement identities are pinned in `provenance/`; `artifacts/pilot/` links the executable backing directory and retained tensors/checkpoints.

[Complete metric CSV](paper/data/experiments/dflash_lora/results/metrics.csv) and [training CSV](paper/data/experiments/dflash_lora/results/training.csv) include all final rows, accepted/proposed token counts, verification iterations, draft-only means, acceptance rates, finish counts, fitting parameters/steps/losses, and available setup/memory measurements. The JSON also includes paired prompt intervals and available per-position counts. Shared controls are identified by identical source paths; they are not independent timing repetitions. Metrics not recorded are left unavailable.

<a id="study-3"></a>


## 3. 2026-09-07_dflash_lora_rank

Source: [2026-09-07_dflash_lora_rank.md](paper/data/history/2026-09-07_dflash_lora_rank.md).

### LoRA drafter rank and compression study — 2026-09-07

A two-direction fusion update preserves essentially all of the trained rank 56 fusion update’s latency reduction on a fresh 128-prompt GSM8K set. Removing those directions removes the measured gain. The same trained fusion also improves the base target without LoRA, so these results support workload adaptation and do not isolate a LoRA-specific repair mechanism.

#### Method overview

The experiment asks whether adapting only the target-feature fusion can match adaptation throughout a frozen pretrained drafter, and how much of the learned update is needed at inference.

1. Generate greedy continuations for 4,096 training questions using Qwen3-4B plus the fixed math LoRA. Save token IDs.
2. Replay those sequences through the same target and cache five layers of hidden features at every position. These stages use the shared cached training set.
3. Train fusion-only adapters, parameter-matched drafter-wide adapters, and a full-fusion control with the same sampled-block token CE recipe. Use each final checkpoint.
4. Merge updates into the original draft and benchmark against the unchanged draft and matching autoregressive target. Compress selected learned updates and evaluate them without retraining.
5. Confirm the smallest matched pair on 128 fresh questions, repeat with three training seeds, and test the same adapters against the target without its LoRA.

#### Methods

### Models and data provenance

| Component | Repository | Pinned revision |
| --- | --- | --- |
| Base target | `Qwen/Qwen3-4B` | `1cfa9a7208912126459214e8b04321603b3df60c` |
| Target LoRA | `witcheer/qwen3-4b-gsm8k-grpo` | `f52e1ed1b3accd404d37a9621db0536dd99bdfc5` |
| Initial draft | `z-lab/Qwen3-4B-DFlash-b16` | `b74e3a329c4d963783143b1e970d95b002be72bd` |
| Training implementation | SpecForge | `953d43a0c1c0f5e32989dc43f91ce5fc2d9ddfef` |
| Dataset | `openai/gsm8k` | `740312add88f781978c0658806c59bc2815b9866` |

The target LoRA has rank 32 and alpha 32 and modifies attention/MLP projections, not embedding/head. It was published for an Unsloth four-bit derivative; applying it to the original BF16 target is an explicit deployment deviation from the author's setup.

The target and its LoRA remain frozen. The draft has five blocks, width 2,560, intermediate size 9,728, 32 attention heads, eight KV heads and head dimension 128. Its original Qwen embedding and tied output head are external shared frozen weights. “Training LoRA” below refers to a drafter adapter, not retraining the target adapter.

### Saved training sequences and feature capture

We reused 4,096 GSM8K training questions and their **adapted-target-generated continuations**, not the human reference solutions as CE labels. The generated-response cap was 4,096, not a fixed sequence length: the responses contain 544,585 tokens, 4,094 stopped naturally and two reached the cap. Generation was greedy with thinking disabled. There is no correctness filtering or reward optimization.

The exact user-message text, rendered through the target chat template, is:

```text
Solve the math problem. Show brief reasoning, then end with '#### <answer>'.

Problem: {q}
Solution:
```

In the parent token-objective study, saved prompt-plus-response token IDs were replayed through the adapted target using vLLM pooling capture. This computes hidden states without decoding another response. It persisted BF16 features at all 930,941 prompt-plus-response positions (23.84 GB), along with token IDs and provenance. Each position has five concatenated vectors of total width 12,800.

The draft config selects zero-based target layer IDs `[1,9,17,25,33]`; vLLM capture uses boundary indices `(2,10,18,26,34)` for their post-layer states. These are different indexing conventions for the same five representations.

**Fitting uses cached adapted-target features; no target forward pass or second teacher is needed during optimization.**

The original 128 evaluation questions and fresh 128 confirmation questions come from GSM8K test. The fresh subset was reserved before the rank results using Python `Random(20260907)`, excluding all original train/eval stripped-question SHA256 IDs. It is saved in `setup/confirmation_eval.json`. There is no dev set or best-checkpoint selection.

### Exact parameterizations

Use column vectors. Concatenate the five target features at position t:

$$
h_t=[h_t^{(1)};\ldots;h_t^{(5)}]\in\mathbb R^{12800},\qquad F_0\in\mathbb R^{2560\times12800}.
$$

The original conditioning vector is

$$
c_t=\operatorname{RMSNorm}_\gamma(F_0h_t),\qquad
\operatorname{RMSNorm}_\gamma(u)=\gamma\odot\frac{u}{\sqrt{\frac1{2560}\sum_j u_j^2+10^{-6}}}.
$$

**Fusion LoRA:** replace only the projection by

$$
F_\phi=F_0+\frac\alpha r BA=F_0+BA,
\quad B\in\mathbb R^{2560\times r},\quad A\in\mathbb R^{r\times12800}.
$$

Alpha equals rank, dropout is zero and there is no trainable bias. PEFT's default initialization uses random A and zero B, starting with the original effective projection. All transformer blocks, norms, embedding and head stay frozen. The trainable count is

$$
P_{\rm fusion}(r)=r(2560+12800)=15360r.
$$

**Drafter-wide LoRA:** freeze fusion and add independent $B_\ell A_\ell$ updates to q/k/v/o/gate/up/down projections in each of five draft blocks. Norms, embedding and head stay frozen. Its count is

$$
P_{\rm body}(r)=5r[2(4096+2560)+2(1024+2560)+3(9728+2560)]=286720r.
$$

Consequently fusion ranks 56/112/224 exactly match body ranks 3/6/12 in trainable parameters. Numerical ranks are not equal, because the update locations and matrix dimensions differ.

**Direct full fusion:** train all 32,768,000 entries of F, initialized at $F_0$. Everything else is frozen. This is not full-drafter training.

**Full mapper reference:** reuse the parent CE fit with five identity-initialized square matrices $M_i\in\mathbb R^{2560\times2560}$. Partition $F_0=[F_{0,1},\ldots,F_{0,5}]$. Then

$$
c_t=\operatorname{RMSNorm}_\gamma\left(\sum_{i=1}^{5}F_{0,i}M_i h_t^{(i)}\right),
\quad F_{\rm folded}=[F_{0,1}M_1,\ldots,F_{0,5}M_5].
$$

Only the five maps train, totaling 32,768,000 parameters; $F_0$ stays frozen. This differs from optimizing F directly, despite exporting one fusion matrix. The parent fit took 200.402 seconds on two GPUs with the same CE/anchor recipe. No embedding/head map is trained here.

### Anchor sampling, inputs and causal visibility

Let example i have full tokens $x_{i,0},\ldots,x_{i,n_i-1}$, and response mask $m_{i,t}$, equal to one on saved response tokens and zero on prompt/padding. Eligible anchors are

$$
\mathcal C_i=\{a:0\le a<n_i-1,\ m_{i,a}=m_{i,a+1}=1\}.
$$

Sample up to eight eligible anchors uniformly without replacement per example, resampling each forward pass. Sort selected positions for block construction.

For anchor a, the draft receives one block

$$
[x_{i,a},\mathrm{MASK},\ldots,\mathrm{MASK}]
$$

of length 16, with mask ID 151669 and absolute positions a through a+15. The first token is clean; the next 15 are predicted simultaneously. The draft does not receive the intervening ground-truth continuation tokens inside this block.

The SDPA attention mask allows a block to see target context positions **t < a**, and the anchor/noise positions within its own block. It forbids target features at a or later and forbids other sampled blocks. Thus future features can exist in the file without being visible to that block. Prompt features provide context, but prompt tokens do not supply loss labels. Labels past the response end are excluded.

**Dense storage is not dense supervision.** Eight anchors provide at most 120 supervised slots per example per epoch. Blocks may overlap, so tokens may be supervised repeatedly; terminal blocks can supply fewer slots. We did not train loss at every response position.

### Exact loss and distributed reduction

Let $z_{i,a,k,v}$ be the draft logit for vocabulary token v at offset k, and

$$
q_\phi(v\mid i,a,k)=\frac{\exp z_{i,a,k,v}}{\sum_{u=1}^{151936}\exp z_{i,a,k,u}}.
$$

For retained anchors define

$$
w_{i,a,k}=\mathbf1[1\le k\le15]\mathbf1[a+k<n_i]m_{i,a+k}.
$$

Padded/invalid anchors receive weight zero. The exact loss for one local two-example microbatch is

$$
\boxed{\mathcal L_{\mathcal B}(\phi)=
\frac{\sum_{i\in\mathcal B}\sum_{a\in\mathcal A_i}\sum_{k=0}^{15}
w_{i,a,k}[-\log q_\phi(x_{i,a+k}\mid i,a,k)]}
{\sum_{i\in\mathcal B}\sum_{a\in\mathcal A_i}\sum_{k=0}^{15}w_{i,a,k}}.}
$$

Normalize CE by the number of valid supervised slots in each microbatch. Compute the vocabulary head/loss in chunks of two blocks to limit memory, summing numerators and denominators before dividing.

**Actual settings: `loss_type='dflash'`, `loss_decay_gamma=None`.** Eligible block offsets receive uniform weight. Optional upstream exponential position decay was disabled. “Unchanged upstream CE” refers to the canonical implementation with these explicit settings, not an exact match to every DFlash paper hyperparameter. There is no hidden-state MSE, soft-target KL, label smoothing, reward term, direct acceptance objective or additional regularizer in these fits.

Each GPU accumulates two microbatches with `loss/2`; DDP averages across two ranks. Hence the update gradient corresponds to

$$
\mathcal L_{\rm step}=\frac14\sum_{g=1}^{2}\sum_{b=1}^{2}\mathcal L_{g,b}.
$$

This averages four separately normalized microbatch losses. When valid-token counts differ, it is not identical to normalizing all eight examples by a single global token count. Every compared fit uses the same reduction.

All newly trained variants and the parent full mapper use this hard-token CE. Compression has no training loss. The feature-reconstruction pilot uses MSE; it is not CE-trained.

### Optimization, precision and checkpoint selection

| Setting | Actual value |
| --- | --- |
| Examples / epochs | 4,096 / 3 |
| GPUs per fit | 2, DDP; at most 4 assistant GPUs in concurrent jobs |
| Per-GPU microbatch / accumulation | 2 examples / 2 microbatches |
| Effective batch | 8 examples |
| Updates per epoch / total | 512 / 1,536 |
| Optimizer | Fused AdamW; betas (0.9,0.999), epsilon 1e-8 |
| Weight decay / gradient norm cap | 0 / 1 |
| Peak learning rate | 1e-4 |
| Frozen / trainable parameters | BF16 / FP32 |
| Forward | BF16 CUDA autocast, upstream PyTorch SDPA |
| Checkpoint | Final after three epochs, no dev selection |

Use 5% linear warmup followed by cosine decay to zero across 1,536 updates. Shuffle feature shards and rows each epoch, bucket nearby sequence lengths to reduce padding, and partition batches across the two GPUs. Seeds 42/43/44 vary adapter initialization and anchor draws while keeping data order fixed. Check gradients for finiteness and clip before each optimizer step.

Reported fit seconds start after model/DDP/optimizer setup and include data loading, training and final synchronization. They stop before serialization/export. Slurm accounting includes setup/export separately. The seven initial fit loops total 0.766 GPU-hours; the whole study additionally includes checks, repeated fits, evaluations and compression.

### Merging and post-training compression

Every measured model is a full statically merged BF16 checkpoint. Fusion export is

$$
F_{\rm export}=\operatorname{BF16}(\operatorname{FP32}(F_0)+BA).
$$

There is no extra unfused LoRA branch in this measured inference path. Compact factors establish storage potential, not measured dynamic adapter switching or shared GPU-resident memory savings.

For the saved raw FP32 fusion factors, form $\Delta F=BA$, with

$$
\Delta F=U\Sigma V^\top,\quad
\Delta F_k=U_k\Sigma_kV_k^\top,\quad
E_k=\frac{\sum_{j\le k}\sigma_j^2}{\sum_j\sigma_j^2}.
$$

We compute this truncated SVD through the FP32 Gram eigendecomposition with TF32 disabled, storing $B_k=U_k$ and $A_k=U_k^\top\Delta F$. Check source-weight reconstruction before compression and verify the truncation against a small SVD reference.

Retaining k directions exports $\operatorname{BF16}(F_0+\Delta F_k)$. Removing the leading k exports

$$
F_{{\rm tail},k}=\operatorname{BF16}(F_0+\Delta F-\Delta F_k).
$$

We retained ranks 1/2/4 of fusion-r56 and removed its leading two directions. These interventions have **no retraining**. Rank 2 stores 30,720 FP32 parameters (122,880 payload bytes; 123,024 bytes including safetensors metadata), but inherits the rank-56 source's 860,160-parameter training cost.

The direct-full and full-mapper rank-56 compression controls instead subtract original BF16 weights from trained BF16 exports, then cast to FP32. Their spectra include export rounding. This differs from raw-factor intrinsic update spectra.

The body control compresses each of 35 rank-3 updates independently to rank 1 using QR and an SVD of the 3-by-3 core. It stores 286,720 factor parameters, 9.33 times the fusion-r2 count, and inherits rank-3 source training cost. This is an exploratory one-seed control.

For seed comparisons, QR gives orthonormal row/column-space bases Q. Singular values of $Q_s^\top Q_t$ are principal-angle cosines. Update similarity is

$$
\frac{\langle\Delta F_s,\Delta F_t\rangle_F}{\|\Delta F_s\|_F\|\Delta F_t\|_F}.
$$

These are stability diagnostics for this workload, not a theorem of universal rank-two adaptation or a unique optimum.

### Evaluation and exact metric definitions

Evaluation uses vLLM 0.28.0+cu129 on L40S, BF16, greedy temperature 0, seed 0, 2,048 output-token cap, thinking disabled. SamplingParams leaves top-p/top-k at defaults (1/-1); greedy decoding does not draw samples. Runtime settings: FLASH_ATTN, synchronous scheduling, batch-invariant execution, compilation mode 0, resolved FULL_DECODE_ONLY CUDA graphs, model capacity 5,120, max sequences 16, batched-token capacity 8,192, memory utilization 0.8, prefix caching off. Each speculative block proposes 15 tokens.

The timer submits **one prompt at a time per engine/GPU**. Workers can handle disjoint prompt shards concurrently. “Request seconds” sums individual call durations, not multiworker job wall time, and is not saturated throughput. It includes prompt processing and generation. Setup, two short training-prompt warmups, counter reads and monitoring RPCs are excluded. A detected first-use JIT call is retained as cold timing and repeated for the warm measurement.

For $T_v=\sum_i t_{v,i}$ and identical output count $N=\sum_i n_i$,

$$
\mathrm{TPS}_v=N/T_v,\qquad
S_{v,\rm AR}=T_{\rm AR}/T_v,\qquad
S_{v,0}=T_0/T_v.
$$

Here 0 denotes the original draft. Speedups are ratios of total times, not means of per-prompt speed ratios. Acceptance is

$$
\tau_v=1+\frac{\sum_i A_{v,i}}{\sum_i V_{v,i}},
$$

where A counts accepted proposed draft tokens and V counts verification iterations. One is added for the bonus-token convention. EOS/terminal clipping means this need not exactly equal retained tokens per verification iteration.

Paired bootstrap uses 10,000 shared resamples of the 128 question indices (NumPy seed 20260907), recomputes the total-time ratio, and reports percentiles 2.5/97.5. It measures prompt variation, not independent hardware-repeat uncertainty. Three-seed summaries use sample SD, with reused control timings.

Each speculative token list and finish reason is compared by question ID with the corresponding target's AR output. Base and adapted targets have separate references. Across all 28 full speculative comparisons, 3,584 prompt comparisons and 515,142 output tokens agree exactly. This is evidence on tested inputs, not proof for all inputs or a measure of math accuracy.

For correctness, share frozen embedding/head **parameters**, not target LoRA wrapper objects, with the draft. Verify causal visibility, frozen weights and merged-weight reconstruction before running the full benchmark; detailed checks are retained in machine-readable audits.

### Why a frozen backbone can improve

The effective draft distribution is $q_\phi=D_{\theta_0}(c_\phi(h),\text{anchor/masks})$. Training the interface changes the frozen backbone's activations and predictions. Target-generated token labels supply behavioral supervision even though transformer weights remain fixed.

Better use of target features, workload-specific continuation preferences and fewer local draft errors are plausible mechanisms; this does not establish new mathematical reasoning capability. For greedy verification, if L is accepted draft-prefix length and K=15,

$$
\mathbb E[L]=\sum_{k=1}^{K}\Pr(L\ge k)
=\sum_{k=1}^{K}\Pr(\text{first k proposed tokens all match}).
$$

Correcting an early mismatch can preserve later proposals. This identity makes no independence assumption, and CE remains a surrogate for the prefix statistic. The base-target improvement reported below prevents attributing all gains to target-LoRA repair.

#### Complete rank, compression and seed results

The screening cohort is the original128 test prompts; heldout/base and heldout/adapted use a separate128-question subset fixed before examining the rank results. Rows with seed 43/44 share those held-out prompts and control measurements. The table contains every evaluated rank, compressed/deleted update, seed and target condition.

| Cohort / target | Draft | Tokens | Seconds | Tok/s | Tau | vs native | vs AR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| screening/adapted | fusion_full | 17,128 | 114.254 | 149.91 | 5.6094 | 1.2638× | 4.7549× |
| screening/adapted | draft_r12 | 17,128 | 114.539 | 149.54 | 5.5788 | 1.2606× | 4.7430× |
| heldout/adapted | full_mapper | 19,070 | 123.430 | 154.50 | 5.7172 | 1.2599× | 4.9045× |
| screening/adapted | draft_r6 | 17,128 | 114.919 | 149.04 | 5.5421 | 1.2564× | 4.7274× |
| screening/adapted | fusion_svd56 | 17,128 | 115.198 | 148.68 | 5.5653 | 1.2534× | 4.7159× |
| screening/adapted | fusion_r224 | 17,128 | 116.965 | 146.44 | 5.4553 | 1.2345× | 4.6447× |
| screening/adapted | draft_r3 | 17,128 | 117.230 | 146.11 | 5.4622 | 1.2317× | 4.6342× |
| screening/adapted | fusion_r112 | 17,128 | 118.052 | 145.09 | 5.4143 | 1.2231× | 4.6019× |
| screening/adapted | fusion_r56 | 17,128 | 118.511 | 144.53 | 5.3909 | 1.2184× | 4.5841× |
| screening/adapted | fusion_svd2_from_r56 | 17,128 | 119.034 | 143.89 | 5.3760 | 1.2130× | 4.5640× |
| screening/adapted | fusion_svd4_from_r56 | 17,128 | 119.099 | 143.81 | 5.3751 | 1.2123× | 4.5615× |
| heldout/adapted / seed 43 | draft_r3 | 19,070 | 128.402 | 148.52 | 5.4954 | 1.2111× | 4.7146× |
| heldout/adapted | draft_svd1_from_r3 | 19,070 | 128.861 | 147.99 | 5.4561 | 1.2068× | 4.6978× |
| heldout/adapted / seed 44 | draft_r3 | 19,070 | 129.191 | 147.61 | 5.4730 | 1.2037× | 4.6858× |
| heldout/adapted | draft_r3 | 19,070 | 129.247 | 147.55 | 5.4827 | 1.2032× | 4.6838× |
| heldout/adapted / seed 44 | fusion_r56 | 19,070 | 129.372 | 147.40 | 5.4455 | 1.2020× | 4.6792× |
| heldout/adapted | fusion_r56 | 19,070 | 129.393 | 147.38 | 5.4650 | 1.2019× | 4.6785× |
| heldout/adapted | fusion_svd2_from_r56 | 19,070 | 129.542 | 147.21 | 5.4601 | 1.2005× | 4.6731× |
| heldout/adapted / seed 44 | fusion_svd2_from_r56 | 19,070 | 129.870 | 146.84 | 5.4551 | 1.1974× | 4.6613× |
| heldout/adapted / seed 43 | fusion_r56 | 19,070 | 130.045 | 146.64 | 5.4384 | 1.1958× | 4.6550× |
| heldout/adapted / seed 43 | fusion_svd2_from_r56 | 19,070 | 130.097 | 146.58 | 5.4317 | 1.1953× | 4.6532× |
| heldout/adapted | fusion_svd1_from_r56 | 19,070 | 133.520 | 142.83 | 5.2813 | 1.1647× | 4.5339× |
| screening/adapted | fusion_svd56_from_mapper | 17,128 | 135.541 | 126.37 | 4.6253 | 1.0653× | 4.0081× |
| screening/adapted | Original | 17,128 | 144.389 | 118.62 | 4.3605 | 1.0000× | 3.7625× |
| heldout/adapted | original | 19,070 | 155.512 | 122.63 | 4.4771 | 1.0000× | 3.8927× |
| heldout/adapted | fusion_tail_after2 | 19,070 | 155.828 | 122.38 | 4.4918 | 0.9980× | 3.8848× |
| screening/adapted | AR | 17,128 | 543.265 | 31.53 | — | 0.2658× | 1.0000× |
| heldout/adapted | AR | 19,070 | 605.364 | 31.50 | — | 0.2569× | 1.0000× |
| heldout/base | draft_r3 | 19,918 | 117.218 | 169.92 | 5.5985 | 1.2024× | 4.5747× |
| heldout/base | fusion_r56 | 19,918 | 118.176 | 168.55 | 5.5537 | 1.1926× | 4.5376× |
| heldout/base | original | 19,918 | 140.938 | 141.32 | 4.6225 | 1.0000× | 3.8048× |
| heldout/base | AR | 19,918 | 536.241 | 37.14 | — | 0.2628× | 1.0000× |

AR has no speculative acceptance statistic. Counts, draft-only means, acceptance rates, finish counts, allocation memory and setup times are in the machine-readable table. TTFT/TPOT were not recorded for these LoRA studies.

| Fit | Parameters | Steps | GPUs | Fit s | Fit GPU-h | Epoch-average loss |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| full/draft_r12 | 3,440,640 | 1536 | 2 | 218.514 | 0.1214 | 2.000676 → 1.689013 → 1.646198 |
| full/draft_r3 | 860,160 | 1536 | 2 | 223.984 | 0.1244 | 2.139692 → 1.758259 → 1.718130 |
| full/draft_r6 | 1,720,320 | 1536 | 2 | 219.958 | 0.1222 | 2.062833 → 1.719659 → 1.680087 |
| full/fusion_full | 32,768,000 | 1536 | 2 | 188.124 | 0.1045 | 1.868230 → 1.647486 → 1.594870 |
| full/fusion_r112 | 1,720,320 | 1536 | 2 | 175.121 | 0.0973 | 2.000636 → 1.767849 → 1.742048 |
| full/fusion_r224 | 3,440,640 | 1536 | 2 | 172.974 | 0.0961 | 1.959222 → 1.754421 → 1.725869 |
| full/fusion_r56 | 860,160 | 1536 | 2 | 179.799 | 0.0999 | 2.049910 → 1.785690 → 1.757877 |
| seed 43/draft_r3 | 860,160 | 1536 | 2 | 225.234 | 0.1251 | 2.149043 → 1.751851 → 1.730815 |
| seed 43/fusion_r56 | 860,160 | 1536 | 2 | 181.822 | 0.1010 | 2.062332 → 1.776181 → 1.767528 |
| seed 44/draft_r3 | 860,160 | 1536 | 2 | 222.955 | 0.1239 | 2.138006 → 1.744472 → 1.722201 |
| seed 44/fusion_r56 | 860,160 | 1536 | 2 | 180.033 | 0.1000 | 2.064461 → 1.777524 → 1.765326 |

#### Compression and functional interpretation

Rank56→rank 2 retains99.926% of raw FP32 update energy,99.43% of its source's latency reduction and80.95% of the full mapper reference's reduction on the held-out adapted target. It stores30,720 FP32 factor parameters (123,024 bytes including metadata) but requires the860,160-parameter source fit. Removing the leading two directions gives tau 4.492 and0.998× native speed. Per-projection rank 1 compression of body-r3 stores286,720 factors,9.33× more, and gives 1.207× native speed. All timings use merged weights; no dynamic serving-memory saving is measured.

| Compression source | Retained rank | Squared update energy |
| --- | ---: | ---: |
| Direct full fusion, BF16 weight difference | 56 | 63.234% |
| Fusion-r56, raw FP32 factors | 2 | 99.926% |
| Fusion-r56, raw FP32 factors | 4 | 99.995% |
| Five full maps, folded BF16 weight difference | 56 | 88.706% |

The last row retains more weight energy than direct-full fusion yet loses much more acceptance. Weight energy alone is insufficient to select functional compression rank. FP32-factor and BF16-difference spectra are different measurements.

Across seeds42/43/44, mean±sample-SD acceptance is5.450±0.014 for fusion-r56,5.484±0.011 for body-r3, and5.449±0.015 for compressed fusion-r2. Speedup ranges are1.196–1.202×,1.203–1.211× and1.195–1.200×. Rank2 left/right principal-angle cosines span0.965–0.988. Data order and timing controls are shared, so these are training-seed diagnostics, not independent hardware-repeat variance.

The same math-trained fusion improves the base target from tau 4.622 to5.554 (1.193× speedup). That supports workload adaptation and prevents attributing the entire gain to LoRA repair. The conditioning interface changes the frozen draft's activations and predictions under hard-token supervision; it is not feature reconstruction alone. These results do not establish universal rank 2 sufficiency or a production multi-adapter serving advantage.

#### Cost and application calculation

The study uses 3.870 actual GPU-hours; the seven principal fit loops use 0.766 GPU-hours. Total job-span time is95m46s, including stage gaps; conservative rounded charge4.683 GPU-hours. Fit times exclude setup and serialization/export. Four GPUs maximum in aggregate.

A source-fit rank 2 adapter uses an estimated2,077 GPU-seconds for shared generation(1,212), dense capture(448), fitting allocation(412), and compression(5). At0.2029 seconds saved per comparable request, these components amortize after approximately 10,237 requests. This excludes scientific validation/evaluation and deployment costs. For a baseline application fraction f spent in these LLM calls,

$$
R_{v|s}=\frac{T_0-T_v}{T_0-T_s},\qquad
N_{break-even}=\frac{C_{adaptation}}{(T_0-T_v)/128},\qquad
S_{app}=\frac1{(1-f)+f/S_{LLM}}.
$$

An LLM speedup1.200× gives application speedup1.154× at f=0.8 and1.091× at f=0.5. Saturated throughput, dynamic switching overhead and resident-memory savings are unmeasured. Full geometry, spectra, gain-retention and amortization records are under the rank backing directory's `results/`.

#### Reproduction and metric records

The [canonical code collection](paper/data/experiments/dflash_lora/README.md) groups this study's method sources under `src/rank/` and the shared vLLM implementation under `src/runtime/`. Source hashes, dataset/prompt contracts and raw-measurement identities are pinned in `provenance/`; `artifacts/rank/` links the executable backing directory and retained tensors/checkpoints.

[Complete metric CSV](paper/data/experiments/dflash_lora/results/metrics.csv) and [training CSV](paper/data/experiments/dflash_lora/results/training.csv) include all final rows, accepted/proposed token counts, verification iterations, draft-only means, acceptance rates, finish counts, fitting parameters/steps/losses, and available setup/memory measurements. The JSON also includes paired prompt intervals and available per-position counts. Shared controls are identified by identical source paths; they are not independent timing repetitions. Metrics not recorded are left unavailable.

<a id="study-4"></a>


## 4. 2026-09-08_dflash_fusion_mechanism

Source: [2026-09-08_dflash_fusion_mechanism.md](paper/data/history/2026-09-08_dflash_fusion_mechanism.md).

### Constant and context-dependent components of math fusion adaptation

All four conditions match the fixed math-target AR reference on **128/128 prompts**, token IDs and finish reasons (512 comparisons). This is a post-training intervention on one learned fusion adapter, not a new adapter-training comparison.

#### Method and provenance

Reuse Qwen/Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`, target LoRA `witcheer/qwen3-4b-gsm8k-grpo` revision `f52e1ed1b3accd404d37a9621db0536dd99bdfc5`, DFlash revision `b74e3a329c4d963783143b1e970d95b002be72bd`, and the seed-42 fusion-r56 adapter from the rank study. The same math target LoRA is used in every condition.

Concatenate target layers [1,9,17,25,33] into $h\in\mathbb R^{12800}$. Original fusion $F_0\in\mathbb R^{2560\times12800}$ precedes frozen RMSNorm. The learned raw update is $\Delta F=BA$, with $B\in\mathbb R^{2560\times56}$, $A\in\mathbb R^{56\times12800}$, and 860,160 trainable parameters. Let $F_*$ denote its deployed BF16 merged fusion weight. Compute a training-only mean correction:

$$
\mu=\frac1N\sum_{t\in\text{training cache}}BAh_t.
$$

Use all 930,941 prompt-plus-response positions in the existing 4,096-example dense training cache; no evaluation features enter $\mu$. Accumulate z=Ah first/second moments in FP64, with FP32 A,h products. No dense input covariance is materialized. Response-only moments are descriptive and do not choose the primary intervention. Runtime adds the FP32-derived $\mu$ cast to the fusion dtype, BF16.

Before the same frozen RMSNorm, the four inputs are

$$
F_0h+0,\quad F_0h+\mu,\quad F_*h-\mu,\quad F_*h+0.
$$

These are original, constant, centered and full respectively. Every condition executes the same added-vector operation, including zero controls, so the new runtime overhead is shared. The learned matrices are already merged for centered/full. The constant stores 2,560 values (10,240 raw FP32 bytes), but **inherits the source fusion training cost**; it was not trained directly as a bias-only model. Removing the mean is an intervention, not a separately optimized model. RMSNorm and subsequent draft computation are nonlinear, so speed gains need not add.

The raw-update energy identity is

$$
\mathbb E\|BAh\|^2=\|\mu\|^2+\mathbb E\|BAh-\mu\|^2.
$$

It describes the pre-export FP32 update. BF16 merging/forward rounding means it is not an exact energy decomposition of the deployed matrix difference $F_*-F_0$.

#### Inherited training recipe

The source fit uses 4,096 seed-42 GSM8K training questions, adapted-target greedy rollouts capped at4,096 tokens, and dense persistence of the five target layers. Training labels are target-generated tokens, not ground-truth answers. At each forward, sample eight response anchors per example without replacement. Input each anchor block as $[x_a,\mathrm{MASK}\times15]$, with strictly preceding target context and upstream DFlash block attention. For valid response positions:

$$
\mathcal L=-\frac1{|\mathcal V|}\sum_{(a,k)\in\mathcal V}\log p_\theta(x_{a+k}\mid h_{<a},[x_a,\mathrm{MASK}^{15}]),\qquad k=1,\ldots,15.
$$

Use pinned SpecForge `953d43a0c1c0f5e32989dc43f91ce5fc2d9ddfef`, no loss decay, MSE or KL. Train only fusion LoRA A/B (alpha=rank, zero B/random A, no dropout/bias); freeze the body, norm, embedding and head. Two GPUs, microbatch2/GPU, accumulation2 give effective batch8; average four locally token-normalized microbatch losses. Three epochs give1,536 updates; final checkpoint, no dev selection. Fused AdamW LR1e−4, betas(.9,.999), eps1e−8, weight decay0, clip1,5% warmup/cosine decay, FP32 trainables/BF16 forward, seed42. Source training-loop time179.799 seconds on two GPUs. The [rank report](paper/data/history/2026-09-07_dflash_lora_rank.md) supplies the full inherited data/cost record.

#### Evaluation

Use the rank study's fixed128 fresh held-out GSM8K test questions (seed20260907), identical brief-reasoning/####-answer prompts and model inputs to the workload study. vLLM0.28.0+cu129 on node06 NVIDIA L40S; one condition per GPU, four GPUs maximum. Temperature0, top-p1, top-k−1, seed0, thinking disabled, cap2,048, context5,120,15 proposals. FLASH_ATTN, compilation0, FULL_DECODE_ONLY graphs, batch-invariant execution, synchronous scheduler, max sequences16, batched-token budget8,192, memory fraction.8, prefix cache off. Target LoRA wrapper objects remain isolated from frozen draft embedding/head weights.

Time single requests including prefill/decode; exclude startup, warmup, cold JIT and counter queries. Speedup is the ratio of summed fresh original/full-condition request times. Existing AR outputs establish correctness; no new AR timings are claimed. Tau=1+accepted proposals/verification iterations; draft-only length=tau−1. Per-position survival counts sum to accepted proposals and are monotone. Paired prompt bootstrap uses10,000 resamples, seed20260908; intervals describe prompt variation conditional on this trial, not independent hardware repetitions.

#### Results

| Correction | Seconds | Tok/s | Tau | Accepted/proposed | vs original (95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: |
| original | 155.899 | 122.32 | 4.4771 | 14854/64080 | 1.0000× [1.0000, 1.0000] |
| constant | 138.189 | 138.00 | 5.0816 | 15355/56430 | 1.1282× [1.1054, 1.1516] |
| centered | 139.164 | 137.03 | 5.0438 | 15326/56850 | 1.1203× [1.1031, 1.1387] |
| full | 128.352 | 148.58 | 5.4650 | 15583/52350 | 1.2146× [1.1880, 1.2428] |

The average correction accounts for **44.56%** of raw update energy over all cached positions and **92.51%** over response positions. Moment extraction took 17.379 CPU seconds; no new gradient training was performed.

The acceptance and timing results test whether this fitted update needs context dependence. They do not establish that every low-rank adapter is a constant offset, that a directly trained bias would match this result, or that the same mechanism holds outside math. Full per-position survival, counters and confidence intervals are in `experiments/dflash_kicad_mechanism_20260908/results/math.json`. Raw outputs remain under `measurements/math_full_*`; source factors, means and hashes under `mechanism/`. Scripts `math_moments.py`, `mechanism_runtime_r2.py`, `math_eval.py`, `collect_math.py` and `report_math.py` reproduce the intervention and analysis. Remote family: `/tmp/yashas.kotre/spec_decode/experiments/dflash_kicad_mechanism_20260908`. Slurm GPU jobs29374–29377; CPU moment job29355. Allocation costs are reconciled separately in the root ledger and family GPU-usage JSON.

Total allocated math-mechanism GPU time, including validation/setup, was 0.2783 GPU-hours; conservative ledger charge 0.3667 GPU-hours. Final four-way evaluation completed in a maximum of198 job seconds (four GPUs in parallel).

Acceptance survival (L counts draft proposals, excluding the verifier bonus):

| Correction | P(L≥1) | P(L≥2) | P(L≥4) | P(L≥8) | P(L≥15) |
| --- | ---: | ---: | ---: | ---: | ---: |
| original | 0.7633 | 0.5934 | 0.3418 | 0.1533 | 0.0363 |
| constant | 0.8328 | 0.6776 | 0.4208 | 0.1823 | 0.0385 |
| centered | 0.8211 | 0.6612 | 0.4124 | 0.1855 | 0.0430 |
| full | 0.8309 | 0.6788 | 0.4619 | 0.2246 | 0.0461 |

The constant correction retains 64.3% of the full mapper’s measured latency reduction. Its first-proposal acceptance is very close to the full mapper (0.8328 versus0.8309), whereas accepting at least eight proposals is less frequent (0.1823 versus0.2246). This is consistent with a common shift helping early agreement while context dependence helps sustain longer accepted blocks. It is a mechanistic interpretation of one fitted math adapter, not a general causal law about token positions.

<a id="study-5"></a>


## 5. 2026-09-08_dflash_interface_architecture

Source: [2026-09-08_dflash_interface_architecture.md](paper/data/history/2026-09-08_dflash_interface_architecture.md).

### Fusion updates versus five factorized replacement maps

Every target call uses the selected math or KiCad LoRA on the same pinned Qwen3-4B. The drafter alone changes.

#### Models, data and shared controls

Target Qwen/Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`; DFlash z-lab/Qwen3-4B-DFlash-b16 revision `b74e3a329c4d963783143b1e970d95b002be72bd`. Math adapter witcheer/qwen3-4b-gsm8k-grpo revision `f52e1ed1b3accd404d37a9621db0536dd99bdfc5`; KiCad adapter AbijahKaj/qwen3-4b-kicad-netlist revision `ba1878e0777bf02c22bcef1c3e9b79095181d455`. Existing dense BF16 caches contain five selected target layers [1,9,17,25,33], each 2,560 wide, captured with the corresponding target LoRA active.

Reuse 4,096 training examples per domain, target-generated greedy responses capped at 4,096 tokens. Exact math data/prompt/split contract is in the token-objective report; its held-out 128-prompt contract is in the workload-transfer report. Exact KiCad revision, original messages, publisher-split reconstruction and source-overlap limitation are in the KiCad-mapping report. Setup references and probe indices retain the complete current evaluation identity. No additional rollout generation is performed. The math cache was transferred from node12 to node06 and every shard checked against its original SHA. KiCad cache integrity checks are inherited from its completed collection.

AR/original-drafter controls come from completed node06 runs with the same target, prompt order, greedy settings and caps. Additional existing fusion/body rows are explicitly shared references. They are not charged again as new training or evaluation. These repeatedly used prompts support controlled ablations, not fresh confirmatory generalization claims.

#### Architecture and initialization

For concatenated $h=[h_1;\ldots;h_5]$, partition frozen fusion $F_0=[F_{01},\ldots,F_{05}]$. All variants retain the frozen draft transformer, original embedding/head and RMSNorm, except the declared drafter-wide LoRA control.

Fusion update:

$$
c=N((F_0+(\alpha/r)BA)h),\quad B\in\mathbb R^{2560\times r},\ A\in\mathbb R^{r\times12800}.
$$

Five replacement maps:

$$
W_i=(\alpha/r)B_iA_i,\quad c=N\left(\sum_{i=1}^5F_{0i}W_i h_i\right),\quad B_i\in\mathbb R^{2560\times r},\ A_i\in\mathbb R^{r\times2560}.
$$

Alpha equals rank in both cases. Replacement maps have no identity bypass or residual update. Their effective linear fusion rank is at most 5r; a low-rank fusion update retains the original projection. Trainable counts are 15,360r versus 25,600r.

Replacement initialization uses an uncentered second moment of three deterministic response positions per training example (first, midpoint, last; 12,288 positions). Compute each layer's leading PCA basis Q from training inputs only and set $B=Q,A=Q^\top$. This is a rank-r projector, not identity. The full-map control directly trains five identity-initialized square matrices (32,768,000 parameters). Input-energy fractions and the complete basis provenance are saved separately. Fusion LoRA uses random A/zero B initialization. The drafter-wide control learns rank3 LoRA in q/k/v/o/gate/up/down of all five draft blocks (860,160 parameters).

#### Objective and fitting

Use unchanged SpecForge OnlineDFlashModel revision `953d43a0c1c0f5e32989dc43f91ce5fc2d9ddfef`, block16, mask token151669, eight response anchors sampled without replacement per example/forward. The block input at anchor a is [$x_a$,MASK×15]; predict saved target tokens $x_{a+k}$, k=1..15. Attention sees target context strictly before a and its own bidirectional noise block; blocks cannot access each other's future context. Padding, clean anchors and invalid labels are excluded.

$$
L_{micro}=-\frac{\sum_{i,a,k}m_{iak}\log q_\theta(x_{i,a+k}\mid h_{i,<a},x_{i,a},\mathrm{MASK}^{15})_k}{\sum_{i,a,k}m_{iak}},\quad L_{step}=\frac14\sum_{g=1}^2\sum_{b=1}^2L_{micro,g,b}.
$$

This is hard-token CE, no MSE/KL, positional decay or reward loss. Dense feature persistence does not mean every token is supervised: anchors are sampled. Two GPUs, microbatch2/GPU, accumulation2, effective batch8; three epochs/1,536 updates, final checkpoint, no dev selection. Fused AdamW LR1e-4, betas(.9,.999), eps1e-8, weight decay0, clip1, 5%warmup/cosine; FP32 parameters with BF16 autocast. Seed42 primary; fixed shard/example order and deterministic anchor seeds. Resumable parameter/optimizer/stream checkpoints are saved every100 updates.

Each replacement is folded into its frozen fusion block for evaluation. Exported BF16 projections use the original native DFlash layout; no extra mapping kernel runs during decoding. The comparison is limited to math for whole replacement maps. KiCad retains its completed drafter-wide control and previously completed residual-fusion references. No claim about whole-map performance on KiCad is made.

#### Evaluation

vLLM0.28.0+cu129 on node06 L40S, BF16, greedy temperature0/top-p1/top-k−1/seed0, thinking disabled, 15 draft proposals, batch-invariant execution, synchronous scheduling, FLASH_ATTN, compilation0/FULL_DECODE_ONLY graphs, prefix cache off. Math cap2,048/context5,120; KiCad cap8,192/context16,384. Four one-GPU shards per full configuration; one request at a time per GPU. Eight selected probes cover short/long outputs and a capped KiCad output before full evaluation. Model/data/parameterization changes are declared adaptations, not an official DFlash benchmark reproduction.

Times sum measured prefill+generation latency and exclude setup, warmup, cold-JIT repeats and counter queries. Speedups use matching inherited AR/native sums. Tau=1+$A/V$ includes the verifier bonus, acceptance rate=$A/P$, and per-position survival=$C_k/V$. These survival rates condition on the verification positions visited by each policy; they are not paired per-prefix token-accuracy comparisons. Paired bootstrap intervals use10,000 prompt resamples, seed20260908; hardware-trial variation is not estimated.

#### Results

### math

| Configuration | Tok/s | Tau | Acceptance rate | vs original | 95% CI | vs AR | Fit seconds |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| native (shared) | 122.07 | 4.4771 | 0.2318 | 1.0000× | [1.0000,1.0000] | 3.9111× | inherited |
| fusion_r56 (shared) | 147.42 | 5.4650 | 0.2977 | 1.2077× | [1.1814,1.2355] | 4.7233× | inherited |
| fusion_svd2 (shared) | 147.52 | 5.4601 | 0.2973 | 1.2085× | [1.1826,1.2359] | 4.7266× | inherited |
| draft_r3 (shared) | 148.99 | 5.4827 | 0.2988 | 1.2206× | [1.1949,1.2490] | 4.7737× | inherited |
| maps_r2 | 39.18 | 1.3793 | 0.0253 | 0.3210× | [0.3088,0.3343] | 1.2553× | 184.3 |
| maps_r30 | 74.61 | 2.6765 | 0.1118 | 0.6112× | [0.5801,0.6446] | 2.3905× | 176.2 |
| fusion_r50 | 147.90 | 5.4603 | 0.2974 | 1.2116× | [1.1850,1.2401] | 4.7386× | 172.9 |
| maps_r56 | 93.90 | 3.3791 | 0.1586 | 0.7692× | [0.7316,0.8099] | 3.0084× | 172.8 |
| maps_r112 | 116.30 | 4.2294 | 0.2153 | 0.9527× | [0.9105,0.9980] | 3.7262× | 174.1 |
| maps_full | 154.77 | 5.7281 | 0.3152 | 1.2679× | [1.2317,1.3072] | 4.9588× | 190.0 |

### kicad

| Configuration | Tok/s | Tau | Acceptance rate | vs original | 95% CI | vs AR | Fit seconds |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| native (shared) | 159.57 | 5.9526 | 0.3302 | 1.0000× | [1.0000,1.0000] | 5.5663× | inherited |
| fusion_r56 (shared) | 202.01 | 7.5048 | 0.4337 | 1.2660× | [1.2545,1.2769] | 7.0470× | inherited |
| fusion_svd2 (shared) | 197.00 | 7.3211 | 0.4214 | 1.2346× | [1.2251,1.2436] | 6.8720× | inherited |
| draft_r3 | 225.57 | 8.4115 | 0.4941 | 1.4136× | [1.4003,1.4265] | 7.8687× | 612.8 |

At the matched 768,000-parameter budget, residual fusion is 1.982× faster than five rank30 replacement maps. Increasing replacement rank improves acceptance, but rank112 still trails the original drafter. Full-rank maps reach 1.2679× original speed: about 4.6% above fusion-r50, at 42.7× the trainable parameter count. These results reject the tested compact replacement parameterization, not full-map adaptation. KiCad drafter-wide LoRA outperforms its residual-fusion reference; compact interface adaptation is not universally the strongest equal-budget method. Further replacement sweeps are pruned; coding-LoRA testing precedes any MoE deployment.

#### Training trajectories

| Fit | Parameters | Updates | Seconds | Epoch mean CE |
| --- | ---: | ---: | ---: | --- |
| kicad_s42/draft_r3 | 860,160 | 1536 | 612.8 | 1.095217 → 0.706113 → 0.650230 |
| math_s42/fusion_r50 | 768,000 | 1536 | 172.9 | 2.054942 → 1.785131 → 1.758492 |
| math_s42/maps_full | 32,768,000 | 1536 | 190.0 | 1.893610 → 1.641244 → 1.455419 |
| math_s42/maps_r112 | 2,867,200 | 1536 | 174.1 | 2.600475 → 2.131617 → 1.987675 |
| math_s42/maps_r2 | 51,200 | 1536 | 184.3 | 5.098604 → 4.701770 → 4.631856 |
| math_s42/maps_r30 | 768,000 | 1536 | 176.2 | 3.757738 → 3.026264 → 2.845455 |
| math_s42/maps_r56 | 1,433,600 | 1536 | 172.8 | 3.174238 → 2.525214 → 2.365354 |

Recorded family allocation at report generation: 5.7583 GPU-hours; conservative charge 6.4833. This is incremental experiment cost; inherited rollout/cache/control costs remain in their parent reports. MoE/shared-serving experiments have not been launched.

Full token counts, seconds, acceptance/proposal/iteration counts, all15 survival positions, cap counts, memory observations, confidence intervals and exactness are in `results/primary_metrics.json`; complete raw outputs are in `measurements/`. Learned factors/checkpoints and training summaries are in `modules/`, exports in `exports/`, pinned inputs/bases/immutable job recipes in `setup/`, and validation JSON outside experiment_logs. Source entry points are model.py/train.py/compress.py/evaluate.py/study_r2.py. Remote family is `/tmp/yashas.kotre/spec_decode/experiments/dflash_interface_complete_20260908` on node06.

<a id="study-6"></a>


## 6. 2026-09-08_dflash_kicad_mapping

Source: [2026-09-08_dflash_kicad_mapping.md](paper/data/history/2026-09-08_dflash_kicad_mapping.md).

### KiCad-LoRA target: original versus adapted DFlash drafter

The deployment target is **Qwen3-4B with the KiCad LoRA active in every pipeline**. Only the drafter changes. No unadapted-target comparison is part of this experiment. All three speculative pipelines match the same LoRA-target AR reference on 128/128 prompts, token IDs and finish reason (384 speculative comparisons).

#### Models, data and prompts

Target: Qwen/Qwen3-4B `1cfa9a7208912126459214e8b04321603b3df60c` with AbijahKaj/qwen3-4b-kicad-netlist `ba1878e0777bf02c22bcef1c3e9b79095181d455`. Frozen target LoRA is r64/alpha32 on attention/MLP, 132,120,576 parameters, with no embedding/head updates. Drafter: z-lab/Qwen3-4B-DFlash-b16 `b74e3a329c4d963783143b1e970d95b002be72bd`, 537,427,200 backbone parameters. Training implementation: SpecForge `953d43a0c1c0f5e32989dc43f91ce5fc2d9ddfef`.

Use historical AbijahKaj/kicad-netlist-sft-dataset revision `52ce8608c3c0fd2f1b6b68e93b8784b88bc92373`, whose responses are KiCad s-expressions. Follow publisher train_local.py ordering (tool rows then direct rows), original tokenizer/chat template with thinking disabled, full text encode and <=8192-token filter, then Dataset.train_test_split(test_size=.02,seed=42). This reproduces 16,738 retained rows: 16,403 train and 335 validation, with no template failures. From the direct author training rows, deduplicate system/user prompt hashes and shuffle with Python Random(42), selecting 4,096 mapper-training prompts. From author validation retain direct non-tool rows without exact training-prompt overlap, shuffle with Random(20260908), select 128. Exact mapper train/eval prompt overlap is zero. This reconstructs author validation, not an independent target-quality test; matching the filtered count cannot certify an unpublished original data pin. Source repositories can overlap across splits.

Preserve original system and user messages; append the Qwen generation prompt with enable_thinking=False. The system requests a complete KiCad s-expression export with design/components/nets, pin connections and library/footprint metadata. Exact messages and IDs are in setup/train.json and setup/eval.json. Original assistant references are not CE labels and are not passed to generation. No task-quality verifier filters examples or gates this inference comparison.

#### Rollout generation and feature persistence

Generate greedy continuations from the fixed LoRA target, capped at 4,096 new tokens. The first 32 checked AR rollouts/features are reused. Persist complete prompt/output token IDs, decoded text and finish reasons. Reaching the cap does not append an artificial EOS target.

Replay exactly these sequences through the same active LoRA target with vLLM pooling, saving every selected hidden vector in BF16. Five zero-based target layers [1,9,17,25,33] correspond to vLLM boundaries [2,10,18,26,34]. Each position has 12,800 feature values. Dense persistence and sampled CE anchors are separate. Validate every feature shape, finite value, row ID, sequence boundary and source-rollout hash before fitting. Features remain available for retraining.

#### Parameterization and exact training objective

Concatenate $h_t\in\mathbb{R}^{12800}$ and adapt only the fusion projection:

$$
c_t=\operatorname{RMSNorm}((F_0+BA)h_t),\qquad
F_0\in\mathbb R^{2560\times12800},\ B\in\mathbb R^{2560\times56},\ A\in\mathbb R^{56\times12800}.
$$

Train 860,160 parameters, LoRA alpha 56 (scale 1), random $A$/zero $B$, no dropout or bias adaptation. Freeze $F_0$, draft body, normalization (RMSNorm epsilon1e-6), original embedding and head, and the entire target including its LoRA. The target adapter has no embedding/head deltas, so frozen shared weights are appropriate; target LoRA wrapper objects are kept separate from draft modules.

Use upstream OnlineDFlashModel with block_size16, mask token151669, eight eligible response anchors sampled uniformly without replacement per example/forward. At anchor $a$ the input is $[x_a,\mathrm{MASK}\times15]$; predict $x_{a+k}$, $k=1..15$. Draft noise positions attend bidirectionally inside their own block and only to target context strictly before $a$. No cross-block or future-target leakage. Padding, non-response and out-of-range labels are excluded. The local microbatch loss is

$$
\mathcal L_{\rm micro}(\theta)=-\frac1{|\mathcal V|}\sum_{(a,k)\in\mathcal V}\log p_\theta(x_{a+k}\mid h_{<a},[x_a,\mathrm{MASK}^{15}]).
$$

There is no MSE, KL, reward, positional weighting or loss decay (`loss_type=dflash`, `loss_decay_gamma=None`, `objective_chunk_blocks=2`). Two GPUs, microbatch 2/GPU and accumulation 2 give effective batch 8. Each update averages four locally token-normalized microbatch losses, not one globally token-weighted loss when valid counts differ.

Train three epochs/1536 updates, seed42, final checkpoint, no dev selection. Fused AdamW peak LR1e-4, betas(.9,.999), eps1e-8, weight decay0, gradient clipping 1, 5% warmup/cosine schedule. FP32 trainables with BF16 autocast. Deterministic shard/example shuffling and within-shard length bucketing; anchor seeds depend on epoch/rank/microbatch. Save resumable optimizer/parameter/stream state every 100 updates. Model/export checks cover actual gradients, frozen weights, masking, shortest/capped completions and independent FP32 merge algebra.

Compress the raw learned FP32 $\Delta F=BA$ to its leading two left singular directions $U_2$ using the eigendecomposition of $\Delta F\Delta F^\top$ with TF32 disabled:

$$
\Delta F_2=U_2(U_2^\top\Delta F),\qquad F_{\rm export}=\operatorname{BF16}(F_0+\Delta F_2).
$$

Rank2 stores 30,720 factor parameters and inherits the rank56 training cost; it is not directly trained rank2. Both evaluated learned variants use merged BF16 weights, so these measurements do not establish live adapter-switching or shared-serving memory savings.

#### Evaluation and timing

All 128 fixed prompts use the active KiCad LoRA target for AR, original DFlash, trained fusion-r56 and compressed fusion-r2. vLLM0.28.0+cu129 on node06 L40S; greedy temperature0/top-p1/top-k−1/seed0, thinking disabled,8192 new-token cap and16384 context. BF16/vLLM/greedy inference differs from the publisher's NF4/Transformers/sampled setup; this is our fixed inference comparison, not an exact author task-quality reproduction.

FLASH_ATTN, compilation0, FULL_DECODE_ONLY CUDA graphs, synchronous scheduler, batch-invariant execution, max sequences16, batched-token budget8192, memory fraction.8, prefix caching off, target LoRA rank capacity64,15 speculative proposals. Four one-GPU shards per pipeline, deterministic prompt-index modulo4 assignment; each GPU times one request at a time. No cross-request batch throughput claim. Time includes prefill/decode and excludes startup, warmup, cold JIT and counter queries. Fresh matching AR/native controls define speedups; eight-prompt learned-drafter checks precede full speculative evaluation, using the same AR references.

Let $A$ be accepted draft tokens, $P$ proposed tokens and $V$ verification iterations. Report $\tau=1+A/V$, draft-only mean $A/V$, acceptance rate $A/P$, and survival $C_k/V$ where $C_k$ counts iterations accepting at least $k$ proposals. $C$ is monotone and $\sum_k C_k=A$. EOS/cap clipping can make emitted tokens per iteration differ from $\tau$. Speedups are ratios of summed prompt times. Paired prompt bootstrap uses 10,000 resamples, seed20260908; intervals describe prompt variation conditional on this timing trial, not repeated hardware trials. TTFT/TPOT were not recorded.

#### Results

| Drafter | Seconds | Tok/s | Acceptance length | vs original | vs AR | Training loop |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ar | 22344.800 | 28.67 | — | 0.1797× | 1.0000× | — |
| native | 4014.312 | 159.57 | 5.9526 | 1.0000× | 5.5663× | — |
| fusion_r56 | 3170.844 | 202.01 | 7.5048 | 1.2660× | 7.0470× | 636.9s,2 GPUs |
| fusion_svd2 | 3251.552 | 197.00 | 7.3211 | 1.2346× | 6.8720× | inherits rank56 fit |

| Drafter | 95% CI vs original | Accepted / proposed | Iterations | Cap terminations | Max observed allocated GiB |
| --- | --- | ---: | ---: | ---: | ---: |
| ar | [0.1747,0.1846] | —/— | — | 18 | 34.80 |
| native | [1.0000,1.0000] | 533063/1614480 | 107632 | 18 | 34.57 |
| fusion_r56 | [1.2545,1.2769] | 555343/1280610 | 85374 | 18 | 34.57 |
| fusion_svd2 | [1.2251,1.2436] | 553227/1312800 | 87520 | 18 | 34.57 |

| Drafter | $P(L\ge 1)$ | $P(L\ge 2)$ | $P(L\ge 4)$ | $P(L\ge 8)$ | $P(L\ge 15)$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| native | 0.7277 | 0.6490 | 0.5017 | 0.2834 | 0.0626 |
| fusion_r56 | 0.6919 | 0.6167 | 0.5468 | 0.4078 | 0.2363 |
| fusion_svd2 | 0.6976 | 0.6223 | 0.5481 | 0.3957 | 0.2051 |

Training epoch losses: epoch1: 1.120306, epoch2: 0.925039, epoch3: 0.901093.

Collection: 4096 examples, 13,708,854 response tokens, 2200 cap endings, 382.246GB dense features. Rank2 retains 98.4613% of raw update energy; this is not a speedup guarantee. Training peak allocated memory on rank0: 3.45GiB.

Total study allocation including validation, collection, capture, training and evaluation: 13.1144 GPU-hours; conservative ledger charge 13.4667 GPU-hours. Remaining approved budget: 225.133 GPU-hours. Fitting-loop time excludes collection/capture/model startup. Full per-job costs and all measured counters are in results/final.json.

Overall allocation timeline span: 3.900 hours, including gaps between recorded GPU jobs.

| Phase | Allocated GPU-hours | Phase wall span (min) | Longest job seconds | Slurm jobs |
| --- | ---: | ---: | ---: | --- |
| validation | 0.4108 | 174.95 | 319 | 29426, 29427, 29424, 29434, 29436, 29439, 29440, 29504, 29505, 29506 |
| generation | 1.9925 | 30.37 | 1822 | 29443, 29444, 29445, 29446 |
| feature capture | 0.6333 | 9.73 | 584 | 29468, 29469, 29470, 29471 |
| fit and cache validation | 0.5017 | 15.05 | 903 | 29473 |
| compression | 0.0064 | 0.38 | 23 | 29477 |
| eval ar | 6.4336 | 102.70 | 6156 | 29479, 29482, 29480, 29481 |
| eval native | 1.2044 | 19.35 | 1161 | 29509, 29510, 29511, 29512 |
| eval fusion_r56 | 0.9517 | 18.30 | 904 | 29514, 29515, 29516, 29517 |
| eval fusion_svd2 | 0.9800 | 21.45 | 940 | 29521, 29522, 29523, 29524 |

Phase wall span runs from its first job start to its last job end and includes gaps; overlapping phases must not be added. Validation contains sequential stages. Preparation-cost amortization below conservatively includes generation/capture/fit/compression and validation job allocations, including the reused first shard; it excludes the 128-prompt benchmark allocations.

fusion_r56: saving 6.5896 GPU-seconds/request at this single-request setting; preparation cost breaks even after approximately 1,937 matching requests.

fusion_svd2: saving 5.9591 GPU-seconds/request at this single-request setting; preparation cost breaks even after approximately 2,141 matching requests.

Interpretation: this comparison isolates drafter adaptation while holding the deployment target and its LoRA fixed. Higher mean acceptance therefore measures longer accepted blocks from that same target, without changing its emitted tokens. First-proposal survival decreases (0.7277 to 0.6919 for rank56), while all-15 survival increases (0.0626 to 0.2363): the benefit comes from the acceptance-length distribution, not uniformly better acceptance at every position. It does not establish a target task-quality gain, multi-tenant throughput, or generalization to every adapter. Compression retains weight-update energy; decoding measurements, rather than that energy alone, establish its practical effect.

Artifacts: `/tmp/yashas.kotre/spec_decode/experiments/dflash_kicad_mapping_20260908` on node06. Local source/recipes under the same family name: runtime_r2.py/generate_spec.py/capture.py, model.py/train.py/compress.py, evaluate_r2.py, pipeline_r2.py and report.py. Exact inputs in setup/, immutable job scripts and IDs in setup/jobs.json, training resumable/final checkpoints in modules/, exports in exports/, persistent rollouts/features in their named directories, raw per-shard measurements in measurements/, complete merged rows/metrics in results/. Validation JSON is kept outside experiment_logs. Dense features, rollouts and merged exports remain node-local scratch. Learned rank56 factors and rank2 factors, all source and compact results are also stored locally.

<a id="study-7"></a>


## 7. 2026-09-08_dflash_lora_decisive_controls

Source: [2026-09-08_dflash_lora_decisive_controls.md](paper/data/history/2026-09-08_dflash_lora_decisive_controls.md).

### Matched training-target and direct-rank controls

Training the math drafter update from the base target works about as well as training from its LoRA target. Direct rank 2 improves acceptance, but falls short of training rank 56 and compressing to rank 2 under this recipe. Eight fitted-model/target combinations match their target-specific AR token IDs and finish reasons on 128/128 prompts each (1,024 comparisons).

#### Models, data and target matching

Use frozen BF16 `Qwen/Qwen3-4B` (`1cfa9a7208912126459214e8b04321603b3df60c`), optionally with fixed rank 32/alpha 32 `witcheer/qwen3-4b-gsm8k-grpo` (`f52e1ed1b3accd404d37a9621db0536dd99bdfc5`), and original `z-lab/Qwen3-4B-DFlash-b16` (`b74e3a329c4d963783143b1e970d95b002be72bd`). Training uses SpecForge `953d43a0c1c0f5e32989dc43f91ce5fc2d9ddfef`; GSM8K revision `740312add88f781978c0658806c59bc2815b9866`.

The same4,096 unique GSM8K train questions are selected with Python Random(42), using stripped-question SHA256 IDs. Render `Solve the math problem. Show brief reasoning, then end with '#### <answer>'.\n\nProblem: {q}\nSolution:` with the base Qwen chat template and thinking disabled. Generate each teacher's own greedy continuations, capped at4,096 new tokens. Base responses total 558,647 tokens (one capped output); LoRA-target responses total 544,585 (two capped). There is no answer-correctness filtering or human-solution token supervision.

Replay each teacher's own complete prompt/response sequence through that teacher and persist every five-layer BF16 context vector. Layers are zero-based[1,9,17,25,33], width2,560 each, total 12,800. Shape, finiteness, selected IDs and source hashes are verified. Questions match across teachers; both generated labels and hidden representations can differ. This design does not isolate their separate causal contributions.

The128 evaluation questions are the rank study's held-out GSM8K test subset, Python Random(20260907), excluding all training and original screening question hashes. The same evaluation subset is shared across the compared fits; there is no dev selection. BF16 deployment differs from the target adapter author's Unsloth4-bit setup; this is not an author-quality replication.

#### Parameterization and exact training objective

For concatenated target feature h and frozen fusion $F_0\in\mathbb R^{2560\times12800}$, fusion adaptation uses

$$
c=\operatorname{RMSNorm}_{\gamma_0}( (F_0+BA)h),\quad
B\in\mathbb R^{2560\times r},\ A\in\mathbb R^{r\times12800}.
$$

The frozen normalization has epsilon1e-6. Alpha=r, so LoRA scaling is1; A is random, B zero, dropout0, no trainable bias. Fusion-r2 trains30,720 parameters; fusion-r56 trains860,160. Body-r3 instead trains independent BA updates in q/k/v/o/gate/up/down of all five draft blocks, also860,160 parameters, with fusion frozen. All other draft weights, original embedding/tied head and target weights remain frozen.

Fits: base-generated/base-feature fusion-r2, fusion-r56 and body-r3; adapted-generated/adapted-feature fusion-r2. The rank 56/body-r3 adapted-teacher fits are shared comparison checkpoints, not independently refitted controls.

For response mask $m_{i,t}$, sample up to eight anchors without replacement from positions where $m_{i,a}=m_{i,a+1}=1$. A block contains the clean $x_{i,a}$ followed by15 mask tokens(ID151669); position k predicts $x_{i,a+k}$. It sees only target context $t<a$ and its own bidirectional noise block, never another block or future context. Prompt, padding, clean-anchor and out-of-range labels are excluded. For a local microbatch,

$$
\mathcal L_B=-\frac{\sum_{i,a}\sum_{k=1}^{15}w_{iak}\log q_\theta(x_{i,a+k}\mid h_{i,<a},[x_{i,a},\mathrm{MASK}^{15}])}
{\sum_{i,a,k}w_{iak}},\qquad
w_{iak}=\mathbf1[a+k<n_i]m_{i,a+k}.
$$

Use upstream `loss_type='dflash'`, `loss_decay_gamma=None`, two-block vocabulary-loss chunks, and SDPA. No feature MSE, soft-target KL, reward, label smoothing or direct acceptance objective is added. Dense caching does not imply dense supervision: anchors are resampled each forward pass. Two GPUs, two examples per microbatch and two accumulation steps average four locally normalized losses per optimizer update, not a single global-token normalization.

All four fits use seed 42, three epochs,1,536 updates, effective batch 8, fused AdamW(0.9,0.999; epsilon1e-8), peak LR 1e-4,5%warmup/cosine, weight decay 0, gradient clip 1, BF16 autocast/FP32 trainables, deterministic shard/row shuffle and length bucketing. Use the final checkpoint. Merge BA into original BF16 weights for native inference; there is no extra LoRA kernel in timed decoding.

#### Evaluation and complete results

vLLM0.28.0+cu129/PyTorch2.13.0+cu129/Transformers5.16.1/PEFT0.20.0 on node12 L40S. Temperature0, top-p1, top-k−1, seed 0, thinking disabled,2,048-new-token cap,15 proposals/block. FLASH_ATTN, synchronous scheduling, batch invariance, compilation mode0, FULL_DECODE_ONLY graphs; context5,120, max sequences 16, token-batch budget8,192, memory fraction 0.8, prefix cache off. One request at a time per GPU, with setup/warmup/counter reads excluded and prefill+decode included. Original-draft/AR and adapted-trained rank 56/body controls use the same saved protocol from the rank study; timing ratios do not constitute independent hardware repetitions.

Tau=1+accepted draft tokens/verification iterations, pooled across requests. Draft-only mean excludes1; terminal clipping can make returned tokens/iteration differ. Tok/s=output tokens/summed request seconds; speedup=reference summed time/variant summed time. Fit seconds below measure the training loop, excluding setup/export.

| Cohort / target | Draft | Tokens | Seconds | Tok/s | Tau | vs native | vs AR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| heldout/base | AR | 19,918 | 536.241 | 37.14 | — | 0.2628× | 1.0000× |
| heldout/base | Original | 19,918 | 140.938 | 141.32 | 4.6225 | 1.0000× | 3.8048× |
| heldout/base | adapted_fusion_r56 | 19,918 | 118.176 | 168.55 | 5.5537 | 1.1926× | 4.5376× |
| heldout/base | adapted_body_r3 | 19,918 | 117.218 | 169.92 | 5.5985 | 1.2024× | 4.5747× |
| heldout/base | base_fusion_r56 | 19,918 | 117.771 | 169.13 | 5.5715 | 1.1967× | 4.5533× |
| heldout/base | base_draft_r3 | 19,918 | 116.891 | 170.40 | 5.5854 | 1.2057× | 4.5875× |
| heldout/base | base_fusion_r2 | 19,918 | 125.142 | 159.16 | 5.2421 | 1.1262× | 4.2851× |
| heldout/base | adapted_fusion_r2 | 19,918 | 124.693 | 159.74 | 5.2421 | 1.1303× | 4.3005× |
| heldout/adapted | AR | 19,070 | 605.364 | 31.50 | — | 0.2569× | 1.0000× |
| heldout/adapted | Original | 19,070 | 155.512 | 122.63 | 4.4771 | 1.0000× | 3.8927× |
| heldout/adapted | adapted_fusion_r56 | 19,070 | 129.393 | 147.38 | 5.4650 | 1.2019× | 4.6785× |
| heldout/adapted | adapted_body_r3 | 19,070 | 129.247 | 147.55 | 5.4827 | 1.2032× | 4.6838× |
| heldout/adapted | base_fusion_r56 | 19,070 | 129.271 | 147.52 | 5.4569 | 1.2030× | 4.6829× |
| heldout/adapted | base_draft_r3 | 19,070 | 128.400 | 148.52 | 5.4724 | 1.2112× | 4.7147× |
| heldout/adapted | base_fusion_r2 | 19,070 | 136.833 | 139.37 | 5.1567 | 1.1365× | 4.4241× |
| heldout/adapted | adapted_fusion_r2 | 19,070 | 136.842 | 139.36 | 5.1453 | 1.1364× | 4.4238× |

AR has no speculative acceptance statistic. Counts, draft-only means, acceptance rates, finish counts, allocation memory and setup times are in the machine-readable table. TTFT/TPOT were not recorded for these LoRA studies.

| Fit | Parameters | Steps | GPUs | Fit s | Fit GPU-h | Epoch-average loss |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| adapted/fusion_r2 | 30,720 | 1536 | 2 | 173.820 | 0.0966 | 2.454466 → 1.948402 → 1.916502 |
| base/draft_r3 | 860,160 | 1536 | 2 | 225.810 | 0.1254 | 2.117895 → 1.755461 → 1.725585 |
| base/fusion_r2 | 30,720 | 1536 | 2 | 177.681 | 0.0987 | 2.425743 → 1.943397 → 1.922145 |
| base/fusion_r56 | 860,160 | 1536 | 2 | 178.787 | 0.0993 | 2.033630 → 1.781838 → 1.764915 |

#### Interpretation and cost

The base-trained fusion-r56 is1.00094× as fast as adapted-trained fusion-r56 on the LoRA target; the paired prompt95% interval is[0.9951,1.0072]. On the base target the ratio is1.00344×,[0.9965,1.0101]. Differences are small, not a formal equivalence result or independent hardware-repeat estimate. Bootstrap uses 10,000 paired prompt resamples, seed 20260907.

The compressed rank 56→rank 2 reference achieves tau 5.460 and1.200× native speed on the adapted target, versus 5.145 and1.136× for direct rank 2. Both store30,720 factors, but compression inherits the860,160-parameter source fit. Direct rank 2 takes 173.8–177.7 seconds, versus 179.8 seconds for adapted rank 56; the frozen-backbone forward/backward dominates this recipe. This distinguishes low-rank expressibility from optimization, not an intrinsic impossibility for direct rank 2.

The stronger supported interpretation is workload adaptation rather than necessary LoRA-specific repair. Different optimizer budgets, target adapters and dynamic serving memory/switching are outside this comparison. The four new fits total approximately 756 seconds of two-GPU training loops; the recorded family costs1.762 actual GPU-hours including validation. Shared generation/capture/checkpoint/control costs are separate. Detailed allocation accounting remains in the backing `results/accounting.json`.

#### Reproduction and complete metrics

The [canonical collection](paper/data/experiments/dflash_lora/README.md) groups the selected sources in `src/controls/`, shared inference in `src/runtime/`, source/prompt/data pins in `provenance/`, and backing artifacts in `artifacts/controls/`. [Metric CSV](paper/data/experiments/dflash_lora/results/metrics.csv), [training CSV](paper/data/experiments/dflash_lora/results/training.csv) and their JSON equivalents include accepted/proposed counts, verification steps, draft-only means, acceptance rates, finish counts, available allocation/setup metrics, training losses/costs and paired-prompt intervals. Shared controls retain explicit source identities and are not independent timing repetitions.

<a id="study-8"></a>


## 8. 2026-09-08_dflash_workload_transfer

Source: [2026-09-08_dflash_workload_transfer.md](paper/data/history/2026-09-08_dflash_workload_transfer.md).

### DFlash interface adaptation across math and subtitle workloads

All 20 speculative configurations match AR on 128/128 prompts (2,560 exact token-ID and finish-reason comparisons). Math adaptation improves decoding; subtitle adaptation yields no practically substantial gain under this recipe.

#### Experimental design

The target is the original BF16 Qwen3-4B, optionally with a frozen target LoRA. The original 537,427,200-parameter DFlash drafter is the starting point for every adaptation. No target weights are trained here. The math adapters are reused from the preceding rank study; two new subtitle adapters are fitted, then one is compressed.

| Evaluation workload | Target | Draft variants |
| --- | --- | --- |
| GSM8K | Math target LoRA | Original; math/subtitle fusion-r56, drafter-wide-r3, compressed fusion-r2 |
| English–Spanish batches | Subtitle target LoRA | Same seven variants |
| GSM8K | Base target without LoRA | Original; math fusion-r56; subtitle fusion-r56 |
| English–Spanish batches | Base target without LoRA | Same three variants |

This gives 20 speculative pipelines, each evaluated on 128 prompts, plus four matching AR controls on node06. The base-target rows hold target weights fixed while changing workload and draft adapter. The own-LoRA rows reflect deployment with the respective target adapter. Cross-workload comparisons do not establish equal sample efficiency: math used 4,096 training examples, subtitles use 1,536, and sequence lengths differ.

#### Data and model provenance

The subtitle target adapter is `Hookem22/qwen3-4b-subtitle-es-v4adv-lora`, revision `3b0b976a76472013b9bc99bbae52d11e1c221776`: ordinary rank-16, alpha-16 attention/MLP LoRA. Its published training set is `Hookem22/subtitle-es-v4adv`, revision `56c15228d70ccc8c5d4def53a0c9fcee8aeb30c4`. Python `Random(42)` selects 1,536 of its 1,577 scenes. We retain the system/user messages and generate new target completions; author assistant responses are not training labels. Each training scene contains related subtitle lines.

The prompt is unchanged:

> Translate these subtitles from English to Spanish. Output exactly one Spanish line per source line, and keep every line to 42 characters or fewer.

Evaluation uses `Helsinki-NLP/opus-100`, revision `805090dc28bf78897da9641cdf08b61287580df9`, English–Spanish test split. With Python `Random(20260908)`, select unique, nonempty English single lines of at most 80 characters, excluding all English lines in the author's training set. Group the first 1,024 qualifying lines into 128 batches of eight. These are unrelated sentence batches, **not coherent subtitle scenes**, and do not replicate the author's unavailable 36-scene evaluation. Spanish reference translations are not constrained to 42 characters. We report only explicitly defined formatting diagnostics, not translation fidelity or the author's quality score.

The target adapter was trained on an Unsloth 4-bit derivative; this experiment deploys it on original BF16 Qwen3-4B. This is a declared deviation. The base tokenizer/chat template is used with thinking disabled.

Shared pins are Qwen3-4B `1cfa9a7208912126459214e8b04321603b3df60c`, DFlash drafter `b74e3a329c4d963783143b1e970d95b002be72bd`, and SpecForge `953d43a0c1c0f5e32989dc43f91ce5fc2d9ddfef`. The frozen math target adapter is `witcheer/qwen3-4b-gsm8k-grpo`, revision `f52e1ed1b3accd404d37a9621db0536dd99bdfc5`. Math evaluation uses the rank study's fixed held-out128 prompt IDs. Full resolved contracts and hashes reside in the experiment's `setup/` and `validation/` directories.

#### Training and compression

Generate subtitle target rollouts greedily with a 4,096-new-token cap, stopping normally at EOS. Persist token IDs and text. Replay these exact sequences through the same fixed target plus subtitle LoRA and save **all** selected hidden vectors in BF16. Dense persistence and sampled training anchors are separate: caching every position allows anchors to vary without another target forward pass.

Concatenate five width-2,560 target features from zero-based layers 1, 9, 17, 25, and 33:

$$
h_t=[h_t^{(1)};h_t^{(9)};h_t^{(17)};h_t^{(25)};h_t^{(33)}]\in\mathbb R^{12800},\qquad
c_t=\operatorname{RMSNorm}((F_0+BA)h_t).
$$

For fusion-r56, $F_0\in\mathbb R^{2560\times12800}$ and the draft body, normalization, embedding, and output head stay frozen. Only $B\in\mathbb R^{2560\times56}$, $A\in\mathbb R^{56\times12800}$ train: **860,160 parameters**. LoRA alpha equals rank, so its scaling is one; dropout and bias adaptation are disabled. PEFT initializes B to zero and A randomly. For drafter-wide-r3, rank-3 updates instead train q/k/v/o and gate/up/down projections in all five draft blocks, also exactly **860,160 parameters**; fusion stays frozen. Both variants use original frozen base embedding/head weights, never target LoRA wrapper objects.

Use the pinned upstream `OnlineDFlashModel` DFlash token-CE objective. Each forward samples eight eligible response anchors uniformly without replacement. At anchor a, the draft input is $[x_a,\mathrm{MASK},\ldots,\mathrm{MASK}]$, with 15 mask tokens (ID 151669), and predicts the following 15 response tokens. Target context is strictly before the anchor; each block's noise positions attend bidirectionally within that block, not across blocks. Padding, out-of-range positions, and non-response labels are excluded. For valid supervised pairs $\mathcal V$ in a local microbatch:

$$
\mathcal L_{\mathrm{micro}}(\theta)
=-\frac{1}{|\mathcal V|}\sum_{(a,k)\in\mathcal V}
\log p_\theta(x_{a+k}\mid h_{<a},[x_a,\mathrm{MASK}^{15}]),\quad k=1,\ldots,15.
$$

There is no feature MSE, KL, reward term, or positional loss decay. `loss_decay_gamma=None`, `objective_chunk_blocks=2`. With two GPUs, microbatch two per GPU, and two accumulation steps, each optimizer update averages four locally token-normalized microbatch losses. This is not one global token-normalized loss when valid-token counts differ.

Three epochs over 1,536 examples give **576 optimizer updates**, effective example batch eight. Both fits use seed 42, fused AdamW, peak LR $10^{-4}$, betas (0.9, 0.999), epsilon $10^{-8}$, weight decay zero, gradient clipping one, 5% warmup followed by cosine decay, FP32 trainable parameters and BF16 forward computation. Shards and examples are shuffled reproducibly, with length bucketing within shards. Use the final checkpoint; no dev-set selection. Two two-GPU fits may execute simultaneously, with four assistant GPUs maximum overall.

Compression takes the raw FP32 learned update $\Delta F=BA$, computes its leading two left singular directions $U_2$ through the eigendecomposition of $\Delta F\Delta F^\top$, and exports

$$
\Delta F_2=U_2(U_2^\top\Delta F),\qquad
F_{\mathrm{export}}=\operatorname{BF16}(F_0+\Delta F_2).
$$

TF32 is disabled for this computation. The compact factors contain **30,720 parameters**, but require the rank-56 fit first; they are not a directly trained rank-2 model. All benchmark exports are merged BF16 weights. Thus these timings do not measure dynamic adapter switching or shared-base serving memory savings.

#### Evaluation and interpretation rules

Use vLLM0.28.0+cu129, PyTorch2.13.0+cu129, Transformers5.16.1 and PEFT0.20.0 on NVIDIA L40S. Temperature0, top-p1, top-k−1, seed0, thinking disabled, maximum2,048 new tokens and15 speculative proposals per iteration. FLASH_ATTN, compilation mode0, FULL_DECODE_ONLY CUDA graphs, context capacity5,120, max sequences16, batched-token budget8,192, memory fraction0.8, prefix caching off. The validated runtime keeps target LoRA state isolated from the frozen drafter embedding/head, enables batch-invariant execution, and uses synchronous scheduling. Timings measure one request at a time per GPU, including prefill and decode but excluding model startup, warmup and counter collection. Four pipelines may run on separate GPUs. All AR/native and adapted-draft timing measurements use node06 NVIDIA L40S. Training uses node07 L40S; the target and draft checkpoints are identical across variants.

For summed measured prompt times, report $S_{\mathrm{original}}=T_{\mathrm{original\ draft}}/T_{\mathrm{adapted\ draft}}$ and $S_{\mathrm{AR}}=T_{\mathrm{AR}}/T_{\mathrm{adapted\ draft}}$.

If N is the number of verification iterations and $C_k$ counts iterations accepting at least k draft tokens, report:

$$
P(L\ge k)=C_k/N,\qquad
\mathbb E[L]=\sum_{k=1}^{15}C_k/N,\qquad
\tau=1+\mathbb E[L].
$$

Here L counts accepted draft tokens; the usual acceptance length tau includes one verifier bonus token. End-of-sequence clipping can make emitted tokens per iteration differ. Native vLLM counters are sampled outside timed generation. Check monotonicity of C, $C_1\le N$, and that their sum equals the scalar accepted-token count. The plotted survival curves pool verification steps; they are not an unweighted mean over prompts.

Paired prompt bootstraps use 10,000 resamples, seed 20260908, and the ratio of summed latencies. Their intervals describe prompt variation conditional on this timing trial, not repeated-run hardware noise. Formatting success requires exactly the source number of nonempty output lines and at most 42 Unicode characters per line. A format pass does not establish a correct translation.

#### Complete results and training records

Collection produced 1,536 subtitle examples and 319,221 response tokens (207.8/example), with zero cap terminations. All cached features are finite and agree with source IDs/hashes. Math and subtitle fitting datasets contain4,096 and1,536 examples respectively; comparisons do not establish equal data efficiency. The final table uses24 configurations including four AR references. `math_*` and `subtitle_*` label the draft update's training workload.

| Cohort / target | Draft | Tokens | Seconds | Tok/s | Tau | vs native | vs AR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| math/math | AR | 19,070 | 610.995 | 31.21 | — | 0.2557× | 1.0000× |
| math/math | original | 19,070 | 156.222 | 122.07 | 4.4771 | 1.0000× | 3.9111× |
| math/math | math_fusion | 19,070 | 129.357 | 147.42 | 5.4650 | 1.2077× | 4.7233× |
| math/math | math_body | 19,070 | 127.991 | 148.99 | 5.4827 | 1.2206× | 4.7737× |
| math/math | math_rank2 | 19,070 | 129.267 | 147.52 | 5.4601 | 1.2085× | 4.7266× |
| math/math | subtitle_fusion | 19,070 | 151.635 | 125.76 | 4.6076 | 1.0302× | 4.0294× |
| math/math | subtitle_body | 19,070 | 154.769 | 123.22 | 4.5002 | 1.0094× | 3.9478× |
| math/math | subtitle_rank2 | 19,070 | 151.074 | 126.23 | 4.6043 | 1.0341× | 4.0443× |
| subtitle/subtitle | AR | 12,310 | 397.316 | 30.98 | — | 0.5443× | 1.0000× |
| subtitle/subtitle | original | 12,310 | 216.273 | 56.92 | 2.0573 | 1.0000× | 1.8371× |
| subtitle/subtitle | math_fusion | 12,310 | 218.622 | 56.31 | 2.0281 | 0.9893× | 1.8174× |
| subtitle/subtitle | math_body | 12,310 | 216.268 | 56.92 | 2.0541 | 1.0000× | 1.8371× |
| subtitle/subtitle | math_rank2 | 12,310 | 217.931 | 56.49 | 2.0291 | 0.9924× | 1.8231× |
| subtitle/subtitle | subtitle_fusion | 12,310 | 214.472 | 57.40 | 2.0652 | 1.0084× | 1.8525× |
| subtitle/subtitle | subtitle_body | 12,310 | 215.045 | 57.24 | 2.0511 | 1.0057× | 1.8476× |
| subtitle/subtitle | subtitle_rank2 | 12,310 | 214.892 | 57.28 | 2.0616 | 1.0064× | 1.8489× |
| math/base | AR | 19,918 | 542.645 | 36.71 | — | 0.2593× | 1.0000× |
| math/base | original | 19,918 | 140.708 | 141.56 | 4.6225 | 1.0000× | 3.8565× |
| math/base | math_fusion | 19,918 | 118.148 | 168.59 | 5.5537 | 1.1909× | 4.5929× |
| math/base | subtitle_fusion | 19,918 | 137.255 | 145.12 | 4.7367 | 1.0252× | 3.9536× |
| subtitle/base | AR | 11,517 | 314.039 | 36.67 | — | 0.6237× | 1.0000× |
| subtitle/base | original | 11,517 | 195.870 | 58.80 | 1.9063 | 1.0000× | 1.6033× |
| subtitle/base | math_fusion | 11,517 | 194.613 | 59.18 | 1.9199 | 1.0065× | 1.6137× |
| subtitle/base | subtitle_fusion | 11,517 | 195.991 | 58.76 | 1.9123 | 0.9994× | 1.6023× |

AR has no speculative acceptance statistic. Counts, draft-only means, acceptance rates, finish counts, allocation memory and setup times are in the machine-readable table. TTFT/TPOT were not recorded for these LoRA studies.

| Fit | Parameters | Steps | GPUs | Fit s | Fit GPU-h | Epoch-average loss |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| full/draft_r3 | 860,160 | 576 | 2 | 85.266 | 0.0474 | 5.188511 → 4.920046 → 4.854699 |
| full/fusion_r56 | 860,160 | 576 | 2 | 69.681 | 0.0387 | 5.097506 → 4.939624 → 4.899751 |

#### Interpretation

Math fusion-r56 gives1.2077× native speed, compressed rank2 gives1.2085×, and body-r3 gives1.2206×. Subtitle fusion gives1.0084× (paired prompt95% interval[1.0002,1.0163]), compressed rank2 gives1.0064×, and body-r3 gives1.0057×. This sub-1% subtitle effect is not a material deployment improvement or an independent hardware-repeat result. Shared-base-target subtitle fusion gives0.9994× and tau1.9123 versus1.9063.

Cross-workload transfer is asymmetric: subtitle fusion gives1.0302× on math with its LoRA target and1.0252× on base-target math; math fusion gives0.9893× on subtitle-LoRA decoding and1.0065× on base-target subtitles. This does not support a universal domain-independent gain. Different training sample counts and coherent-scene versus unrelated-batch distribution shift remain confounds.

The subtitle fusion update retains99.9846% squared norm after rank2 compression despite negligible decoding improvement. Spectral concentration alone is therefore not evidence of a useful adaptation. Its30,720 stored factors inherit the860,160-parameter source fit; no direct rank2 subtitle fit is included.

#### Acceptance positions and task diagnostic

| Workload/target | Draft | P(L≥1) | P(L≥2) | P(L≥4) | P(L≥8) |
| --- | --- | ---: | ---: | ---: | ---: |
| Math/math LoRA | Original | 0.7633 | 0.5934 | 0.3418 | 0.1533 |
| Math/math LoRA | Math fusion | 0.8309 | 0.6788 | 0.4619 | 0.2246 |
| Math/math LoRA | Math rank2 | 0.8302 | 0.6779 | 0.4632 | 0.2247 |
| Subtitles/subtitle LoRA | Original | 0.4614 | 0.2108 | 0.0492 | 0.0249 |
| Subtitles/subtitle LoRA | Subtitle fusion | 0.4667 | 0.2100 | 0.0503 | 0.0246 |
| Subtitles/subtitle LoRA | Subtitle rank2 | 0.4659 | 0.2088 | 0.0504 | 0.0247 |

Math gains extend across the draft block; subtitle curves nearly overlap. Complete15-position counts, survival probabilities and accepted-length histograms0–15 are retained in the workload `results/analysis.json` and `acceptance_positions.csv`.

![Acceptance survival curves](paper/data/experiments/dflash_workload_transfer_20260908/results/acceptance_positions.png)

The base target satisfies the declared line-count/42-character formatting contract on6/128 requests(4.69%); subtitle LoRA passes56/128(43.75%), a39.06-percentage-point increase. Manual inspection confirms representative formatting counts but also finds omissions and awkward translations. This is **not translation accuracy or the author's score**. The target changes formatting substantially while its tested small draft adaptations provide almost no latency gain.

#### Cost

The two subtitle training loops take69.681 and85.266seconds on two GPUs each,0.0861 GPU-hours combined, excluding setup/export, generation and capture. Total experimental family cost is4.5267 actual GPU-hours,4.9833 after per-job minute rounding; these broader costs include reference measurements, validation and orchestration. The final four-GPU evaluation allocation lasts41m05s and uses2.7389 GPU-hours, including model startup and allocation time after shorter workers finish. Summed prompt times are not job elapsed time or four-GPU saturated throughput.

All original rollouts, dense feature shards, trained factors and merged exports are retained on node07; final evaluation records and copied exports are on node06. Both use `/tmp/yashas.kotre/spec_decode`; local source and compact records are in the repository. Node-local scratch is temporary storage.

#### Reproduction and complete metrics

The [canonical collection](paper/data/experiments/dflash_lora/README.md) groups the selected sources in `src/workload/`, shared inference in `src/runtime/`, source/prompt/data pins in `provenance/`, and backing artifacts in `artifacts/workload/`. [Metric CSV](paper/data/experiments/dflash_lora/results/metrics.csv), [training CSV](paper/data/experiments/dflash_lora/results/training.csv) and their JSON equivalents include accepted/proposed counts, verification steps, draft-only means, acceptance rates, finish counts, available allocation/setup metrics, training losses/costs and paired-prompt intervals. Shared controls retain explicit source identities and are not independent timing repetitions.

<a id="study-9"></a>


## 9. 2026-09-09_dflash_fusion_interpretability

Source: [2026-09-09_dflash_fusion_interpretability.md](paper/data/history/2026-09-09_dflash_fusion_interpretability.md).

### Fusion steering on Math and KiCad

Paired fixed-prefix interventions use independently fitted Math/KiCad fusion-r56 experts. Reused uniform-CE training takes 179.8/636.9 seconds on two GPUs, respectively. These are offline prediction and geometry results on 32 prompts per domain, not live speed measurements. Methods define training-only mean calibration, target-shift projection and the two-domain cross-expert control.

#### Direction and low-rank geometry

| Domain | Top1 / top2 weight energy | Response correction energy in mean component | Mean $\cos(c,t)$ | Correction energy parallel to t | CE gain retained by $c_\perp$ |
| --- | --- | --- | --- | --- | --- |
| Math | 83.28% /99.93% | 92.60% | −0.0294 | 0.23% | 100.3% |
| KiCad | 95.56% /98.46% | 90.34% | −0.0513 | 0.49% | 96.1% |

#### Causal fixed-prefix results

| Domain | Native | Own correction | Training mean $\mu$ | Centered $c-\mu$ | Parallel only | Perpendicular only | Undo target shift | Reversed correction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Math | 3.0737 | 1.8724 | 1.9914 | 2.7516 | 3.0712 | 1.8692 | 3.0829 | 5.9635 |
| KiCad | 2.1279 | 1.3407 | 1.4449 | 1.6503 | 1.9945 | 1.3717 | 1.7293 | 6.4284 |

#### Causal fixed-prefix results

| Domain | Top1 native / own / mean | All15 fixed-prefix agreement native / own / mean |
| --- | --- | --- |
| Math | 47.16% /53.85% /51.89% | 6.98% /9.77% /7.91% |
| KiCad | 61.47% /71.76% /69.31% | 13.39% /43.75% /32.14% |

#### Causal fixed-prefix results

| Target | Math expert | KiCad expert |
| --- | --- | --- |
| Math | 1.8724 | 2.9346 |
| KiCad | 2.0168 | 1.3407 |

<a id="study-10"></a>


## 10. 2026-09-09_dflash_moe_deployment

Source: [2026-09-09_dflash_moe_deployment.md](paper/data/history/2026-09-09_dflash_moe_deployment.md).

### Two-target fusion-expert routing

The independently trained Math/KiCad pair compares direct target-ID selection, a frozen-expert router, and jointly fitted experts plus router. It uses uniform block CE, 4,096 examples per domain, three epochs and 3,072 updates. This is serial alternating-target evaluation, not concurrent serving. Router-only fitting takes 801.6 seconds and inherits 816.7 seconds of two-GPU expert fitting; scratch takes 802.0 seconds without those expert fits.

#### Exact loss and fitting recipe

| Set | Fit | Trainable params | Steps | Fit seconds | Fit GPU-h | Epoch mean CE |
| --- | --- | --- | --- | --- | --- | --- |
| pair | router | 25,604 | 3072 | 801.6 | 0.4453 | 1.32490 → 1.33357 → 1.32453 |
| pair | scratch | 1,745,924 | 3072 | 802.0 | 0.4455 | 1.53988 → 1.33379 → 1.29622 |

#### Measurement contract and results

| Set | Policy | Domain/traffic | Tok/s | Tau | vs Native | vs Select | vs inherited AR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pair | native | traffic | 157.14 | 5.8963 | 1.0000× | 0.7919× | 5.4687× |
| pair | native | math | 119.85 | 4.4771 | 1.0000× | 0.8310× | 3.8400× |
| pair | native | kicad | 158.61 | 5.9526 | 1.0000× | 0.7904× | 5.5328× |
| pair | select | traffic | 198.42 | 7.4247 | 1.2627× | 1.0000× | 6.9053× |
| pair | select | math | 144.23 | 5.4650 | 1.2034× | 1.0000× | 4.6211× |
| pair | select | kicad | 200.66 | 7.5048 | 1.2652× | 1.0000× | 7.0000× |
| pair | router | traffic | 194.56 | 7.4190 | 1.2382× | 0.9806× | 6.7712× |
| pair | router | math | 141.55 | 5.4517 | 1.1811× | 0.9814× | 4.5353× |
| pair | router | kicad | 196.76 | 7.4996 | 1.2405× | 0.9805× | 6.8637× |
| pair | scratch | traffic | 200.16 | 7.6272 | 1.2738× | 1.0088× | 6.9660× |
| pair | scratch | math | 142.35 | 5.4849 | 1.1878× | 0.9870× | 4.5610× |
| pair | scratch | kicad | 202.61 | 7.7170 | 1.2774× | 1.0097× | 7.0679× |

#### Measurement contract and results

| Set/policy | Traffic95% CI vs Native | vs Select | Routing mean/p95 ms | Routing time fraction |
| --- | --- | --- | --- | --- |
| pair/native | [1.0000, 1.0000] | [0.7854, 0.7989] | 0.750/1.372 | 0.0046% |
| pair/select | [1.2518, 1.2733] | [1.0000, 1.0000] | 2.090/2.662 | 0.0161% |
| pair/router | [1.2275, 1.2485] | [0.9796, 0.9815] | 1.023/1.615 | 0.0077% |
| pair/scratch | [1.2622, 1.2849] | [1.0068, 1.0108] | 1.004/1.625 | 0.0078% |

#### Routing and execution cost

| Set/fit | Context domain | Mean expert probabilities (math, KiCad) | Mean entropy |
| --- | --- | --- | --- |
| pair/router | math | 0.9978, 0.0022 | 0.0139 |
| pair/router | kicad | 0.0300, 0.9700 | 0.0654 |
| pair/scratch | math | 0.9990, 0.0010 | 0.0064 |
| pair/scratch | kicad | 0.0569, 0.9431 | 0.0415 |

<a id="study-11"></a>


## 11. 2026-09-09_dflash_nanocoder

Source: [2026-09-09_dflash_nanocoder.md](paper/data/history/2026-09-09_dflash_nanocoder.md).

### NanoCoder: fusion-only adaptation

The deployed target in every condition is **Qwen3-4B + usernamebetter/nanocoder-v1**. This study fits one fusion-LoRA checkpoint and compares it with the unchanged native DFlash under identical target, prompts and decoding. No drafter-body update, architecture search, learned router or MoE fit is included.

#### Models, data and prompts

- Base Qwen/Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`; native z-lab/Qwen3-4B-DFlash-b16 revision `b74e3a329c4d963783143b1e970d95b002be72bd`.
- NanoCoder revision `f078bfc2722bdb7bf644906cc7d81662ab3de9d2`: standard rank32/alpha32 target LoRA on q/k/v/o and gate/up/down projections, no bias or saved extra modules. The target adapter is frozen throughout.
- Runtime uses the BF16 base plus the adapter. The publisher trained with an Unsloth 4-bit base; this is a documented numerical difference, not a claim of reproducing its original quantized outputs or task-quality scores.
- Dataset `HuggingFaceH4/CodeAlpaca_20K`, revision `798c567f69c8f4b12fc191015e59ee34e9afe00d`, one of the publisher-listed training sources. Use released prompts verbatim, following its instruction-strip + newline + input-strip preprocessing and published 90/10 split (seed42).
- Select 4,096 source-train prompts and 128 source-test prompts with shuffle seed20260909. Deduplicate prompt hashes within splits and exclude test prompts appearing anywhere in source train. Prompts exceeding3,072 tokens are excluded rather than truncated. Shortest/longest selected training prompts are placed first for bounded validation. Exact IDs and prompts are in `setup/train.json` and `setup/eval.json`.
- System: `You are NanoCoder, an expert Senior Full-Stack Engineer and debugging agent.` Use the adapter's released chat template with `enable_thinking=False`, including the template's closed empty thinking block. Training continuation cap4,096; evaluation cap2,048. Greedy temperature0, top_p1, top_k−1, seed0.

The publisher's exact private mixture is unavailable, so the test is held out from **our fusion training**, not demonstrably from target training. Source reference completions are retained for provenance but are neither training labels nor correctness labels. This study measures drafting fidelity and latency, not coding accuracy or improvement over base Qwen.

#### Training and inference

Generate the target's own greedy continuations with vLLM, four independent one-GPU workers using continuous batching, up to32 active sequences/worker. Persist all prompt/output token IDs and finish reasons. Replay those sequences through the same adapted target and persist dense BF16 features at every prompt/response position after zero-based layers [1,9,17,25,33]. No base-only teacher is used. Feature shards include source-token hashes and remain available for reuse on node06.

For concatenated target features $h_t\in\mathbb R^{12800}$, use the frozen native fusion $F_0\in\mathbb R^{2560\times12800}$ and native normalization $N_0$:

$$
c_t=N_0[(F_0+\tfrac\alpha r BA)h_t],\qquad
A\in\mathbb R^{56\times12800},\quad B\in\mathbb R^{2560\times56},\quad r=\alpha=56.
$$

Only A and B are trained: **860,160 parameters**. A has PEFT's random initialization and B starts at zero. The target, native draft transformer, fusion base, normalization, embedding and output head remain frozen. This is an additive learned rank56 update, not the leading56 singular directions of a fitted dense matrix. Export folds BA into $F_0$ and casts the resulting fusion matrix to BF16; standalone inference adds no extra layer.

Training uses unchanged SpecForge DFlash source revision `953d43a0c1c0f5e32989dc43f91ce5fc2d9ddfef`. Sample up to8 random distinct response anchors/example, requiring the anchor and first future label to be supervised. Each length16 block has one clean anchor and15 mask tokens (ID151669). Blocks attend strictly prior target context and bidirectionally within their own block. For anchor a, position k predicts actual generated token $x_{a+k}$. The clean anchor, padding and out-of-range labels receive zero loss weight:

$$
L_{\mathrm{micro}}=-\frac{\sum_{i,a}\sum_{k=1}^{15}m_{iak}\log q_{A,B}(x_{i,a+k}\mid h_{i,<a},x_{i,a},\mathrm{MASK}^{15})_k}{\sum_{i,a,k}m_{iak}},\qquad
L_{\mathrm{step}}=\tfrac14\sum_{g=1}^{2}\sum_{b=1}^{2}L_{\mathrm{micro},g,b}.
$$

This is hard-token CE with no hidden MSE, teacher-logit KL or position decay. Two-GPU DDP; microbatch2/GPU; two accumulation steps, globalbatch8; three epochs/1,536 optimizer updates; seed42; FP32 trainable factors and BF16 forward. AdamW LR1e−4, betas(.9,.999), eps1e−8, weight decay0, gradient clipping1; 5% linear warmup followed by cosine decay. Shards are shuffled per epoch and microbatches bucketed by sequence length. Anchor RNG is determined by seed/epoch/rank/microbatch. Use the final checkpoint, with no dev selection. Dense persistence does not mean every token is scored each update: the objective samples block anchors.

#### Benchmark and results

Use vLLM0.28.0 on node06 NVIDIA L40S, BF16, batch-invariant execution, FLASH_ATTN, synchronous scheduling, prefix caching disabled, context limit8,192 and15 speculative proposals. Each standalone request is measured at concurrency1 on one GPU; conditions are independent GPU workers. AR is partitioned into two64-request workers to shorten wall time. Timing sums per-request prefill+decode, excluding model initialization/warmup and cold compilation. These are request-latency speedups, not batched-server throughput. Runtime integration retains the project's verified target-only LoRA and draft-sharing implementation; it is not an unchanged official DFlash benchmark.

Let $T_v=\sum_i t_{vi}$ and $N=\sum_i|y_i|$. Report throughput $N/T_v$, speedup $T_{AR}/T_v$ and $T_{native}/T_v$. With A accepted draft tokens, V verifier iterations and P proposals, $\tau=1+A/V$ includes the verifier token, draft-only mean is $A/V$, and proposal acceptance is $A/P$.

| Pipeline | Sum request seconds | Tokens/s | vs AR | vs native | Mean length $\tau$ | Proposal acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Adapted target AR | 909.63 | 31.15 | 1.0000× | — | — | — |
| Adapted target + native DFlash | 203.05 | 139.53 | 4.4798× | 1.0000× | 5.0415 | 26.94% |
| Adapted target + fusion-r56 DFlash | 199.18 | 142.25 | 4.5669× | 1.0194× | 5.1320 | 27.55% |

All **256 speculative requests** match their AR reference token IDs and finish reasons exactly. Each condition emits 28,332 tokens; 0/128 requests reach the2,048-token cap. Fusion/native paired-prompt bootstrap95% interval: **[1.0098, 1.0301]×** (10,000 resamples, seed20260909). This measures prompt uncertainty for one timing trial, not repeated-run hardware variation.

Fit time: **197.60s** on2GPUs (**0.1098GPU-hours**), 1536 updates. Epoch mean microbatch CE: 2.0478 → 1.9828 → 1.9820. These are training statistics, not held-out CE.

Total standalone allocation including collection, feature capture, bounded validation and benchmarks: **1.2633GPU-hours**, peak4GPUs, allocation span38.88minutes. Exact per-stage records are in `results/resource_audit.json`.

Persisted training collection: **996,580 response tokens**, 21 capped continuations, **31.68GB** dense feature shards. All4,096 generated texts were checked for thinking tags; none contained them.

#### Prespecified deployment decision

Standalone improvement is **1.0194×**, which does not exceed1.10×. **Stop here: no mixture benchmark or further fit was launched.**

#### Reproduction artifacts

Source family: `experiments/dflash_nanocoder_20260909`; canonical links: `experiments/dflash_lora/src/nanocoder`. `prepare_data.py`, `generate.py`, `capture.py`, `model.py`, `train.py`, `runtime.py` and `evaluate_variants.py` specify the full recipe; `pipeline.py` orchestrates the finite stages. Raw request timings, IDs, finish reasons and acceptance counters are in `measurements/`; final factor checkpoint and training history in `modules/full/fusion_r56/`; merged export and dense feature shards persist remotely. `results/summary.json` includes setup/warmup timings, stop/cap strata, bootstrap intervals and checkpoint checksum; acceptance survival and full0–15 acceptance distributions are in the corresponding CSVs. No failed or nonmatching measurement contributes to the reported table.

<a id="study-12"></a>


## 12. 2026-09-09_dflash_nanocoder_dense

Source: [2026-09-09_dflash_nanocoder_dense.md](paper/data/history/2026-09-09_dflash_nanocoder_dense.md).

### NanoCoder: dense token supervision for fusion LoRA

Dense coverage gives **1.0011× the sampled fit** (paired-prompt 95% interval **0.9952–1.0074×**): no convincing live speed improvement in this trial, despite modestly better fixed-prefix predictions.

Compare dense response-token coverage with the completed 18,016-example sampled-block fit. Both independently continue the same original fusion-r56 checkpoint for three epochs with the same optimizer and 6,756 updates. Every deployment target is BF16 Qwen3-4B plus the NanoCoder target LoRA.

#### Models, data and initialization

Reuse the exact pinned model, adapter, prompt/tokenizer contract, target-generated rollouts and dense features from [the scaling study](paper/data/history/2026-09-09_dflash_nanocoder_scaling.md): Qwen3-4B `1cfa9a7208912126459214e8b04321603b3df60c`, NanoCoder `f078bfc2722bdb7bf644906cc7d81662ab3de9d2`, native DFlash `b74e3a329c4d963783143b1e970d95b002be72bd`, CodeAlpaca `798c567f69c8f4b12fc191015e59ee34e9afe00d`. The adapter publisher's quantized training base differs from this BF16 runtime, consistently across conditions.

Training uses the same 18,016 deterministic source-train prompts, response cap 16,384, 5,260,572 generated response tokens, and cached BF16 target features after layers [1,9,17,25,33]. No new target rollouts or feature collection. Source-test evaluation remains the same 128 prompts, disjoint by mapper-training prompt hash; publisher target-training overlap is unknown. This is a drafting benchmark, not a coding-accuracy evaluation.

Start from the initial 4,096-example fusion checkpoint, SHA256 `796caf147d18fe6478c84db04ab60252692120be3cac46ceba699878aff3526b`, as did the sampled 18K comparator. Do not continue from that comparator. Reset optimizer/schedule. Fit only $B\in\mathbb R^{2560\times56}$ and $A\in\mathbb R^{56\times12800}$, with $c=N_0[(F_0+BA)h]$, alpha/r=1 and 860,160 parameters. Freeze the target, draft body, embeddings/head and normalization. Fold BA into BF16 F0 for inference.

#### Exact dense objective

Let a response occupy positions $p,\ldots,p+T-1$. Its first token is the clean initial anchor under the existing DFlash contract. Choose deterministic anchors

$$\mathcal A=\{p+15j:p+15j<p+T-1,\quad j\ge0\}.$$

Each block contains the clean anchor plus 15 mask tokens (ID 151669). Labels at offset $k=1,\ldots,15$ predict $x_{a+k}$ when that position is in the response. Thus every response token after the first clean anchor contributes **exactly once per epoch**. There are no random token/anchor omissions, no duplicate CE labels within an epoch, and no prompt or padding loss. The first response token is not a draft prediction. This is dense token coverage through adjacent blocks, **not enumeration of every overlapping anchor window**; block alignment is deterministic across epochs.

There are **5,242,556 supervised labels per epoch** and **15,727,668 across three epochs**. The previous recipe sampled up to eight anchors/example/epoch, at most 120 label occurrences, with possible overlap and shorter boundary blocks. Dense tiling changes both coverage and anchor placement; this is not a pure comparison of density with an otherwise identical anchor distribution.

For microbatch $b$, the loss is

$$L_b=-\frac{\sum_i\sum_{a\in\mathcal A_i}\sum_{k=1}^{15}m_{iak}\log q_{A,B}(x_{i,a+k}\mid h_{i,<a},x_{i,a},\mathrm{MASK}^{15})_k}{\sum_i\sum_{a\in\mathcal A_i}\sum_{k=1}^{15}m_{iak}}.$$

Use unchanged SpecForge DFlash attention masks, label alignment and hard-token CE at revision `953d43a0c1c0f5e32989dc43f91ce5fc2d9ddfef`. No hidden MSE, teacher-logit KL, position decay or draft-body adaptation. The **custom components** are deterministic tiled anchor selection, bounded block iteration and weighted gradient accumulation. They are not an upstream official dense-training recipe.

Process up to 64 blocks per example per forward, retaining all needed preceding context. Omit masked future context beyond the current chunk's last label. If chunk $c$ has $D_c$ labels, accumulate $L_b=\sum_c(D_c/\sum_cD_c)L_{bc}$; never average unequal chunks uniformly. Backward each chunk before advancing. DDP synchronizes on the last chunk of each optimizer update. Two GPUs, microbatch two/GPU, accumulation two: $L_{step}=\frac14\sum_{g=1}^2\sum_{b=1}^2L_{gb}$. This retains the previous mean-of-microbatch normalization rather than globally weighting every dataset token equally.

Same shuffled shards/length buckets and seed 42; FP32 factors, BF16 forward; fused AdamW LR 1e−4, betas (0.9,0.999), eps 1e−8, weight decay zero, global gradient clipping 1, 5% warmup/cosine schedule. Three additional epochs, final checkpoint only, no dev selection. Bounded exact-coverage, direct CE, FP32 gradient-equivalence, 16K forward/backward and two-GPU optimizer checks precede the fit. Validation and custom-component provenance reside in `validation/` and `setup/`.

#### Matched vLLM benchmark

Fresh native, sampled-18K and dense measurements use the same actual NanoCoder target, vLLM 0.28.0, L40S, 128 prompts, 2,048 output cap, thinking disabled, greedy temperature 0/top_p 1/top_k −1/seed 0 and 15 proposals. Per-request time includes prefill+decode, excludes setup/warmup/compilation, one active request per GPU. No training overlaps timed evaluation. These are standalone latency measurements, not concurrent serving throughput. Reuse actual-target AR token IDs/finish reasons and its prior timing; AR-relative multipliers inherit that earlier timing trial.

Speedup is the ratio of summed request times. Paired-prompt 95% intervals use 10,000 bootstrap samples, seed 20260909; they do not cover hardware-repeat or training-seed variability. Mean acceptance length $\tau=1+A/V$ includes the verifier token, draft-only length is $A/V$, and acceptance fraction is $A/P$, with $A$ accepted draft tokens, $V$ verification iterations, $P$ proposals.

| Condition | Tok/s | vs native | vs sampled | vs AR (inherited) | Mean τ | Acceptance fraction | Fit minutes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native | 139.90 | 1.0000× | 0.9604× | 4.4917× | 5.0415 | 26.94% | — |
| Sampled 18K | 145.67 | 1.0413× | 1.0000× | 4.6770× | 5.2693 | 28.46% | 15.6 |
| Dense 18K | 145.84 | 1.0424× | 1.0011× | 4.6822× | 5.2810 | 28.54% | 44.7 |

All **384 speculative requests** match AR token IDs and finish reasons exactly. Dense fit: **2683.4 seconds**, **1.4908 GPU-hours**, **15,727,668 supervised labels**. Epoch mean microbatch CE: 1.9595 → 1.9219 → 1.9081.

Dense vs native: 95% paired-prompt interval **[1.0308, 1.0557]×**.

Dense vs sparse: 95% paired-prompt interval **[0.9952, 1.0074]×**.

#### Fixed-prefix diagnostic

Same saved 128-prompt features and 1,024 evenly spaced diagnostic anchors as the prior study; 13,370 valid label positions. This diagnostic sampling is separate from the dense training coverage. Its prefix lengths do not estimate the live verifier visitation distribution.

| Model | CE | Top1 | All15 agreement | Fixed-prefix accepted draft tokens | Mean correction energy fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native | 2.2607 | 50.39% | 8.89% | 4.7256 | — |
| Sampled18K | 1.8739 | 55.30% | 11.93% | 5.3965 | 74.40% |
| Dense18K | 1.8514 | 55.75% | 12.51% | 5.5020 | 73.58% |

#### Interpretation

Dense training used 44.7 minutes versus 15.6 minutes for the sampled fit (2.87× fitting time). Its live throughput is only 0.11% higher and the paired-prompt interval includes no improvement. Mean live acceptance length changes from 5.2693 to 5.2810. The standalone speedup remains below the 1.10× mixture-deployment threshold; no mixture deployment is launched. The sampled comparator's fitting time is inherited from its completed study and is not charged to this run.

At identical diagnostic anchors, dense coverage repairs 254 sampled-fit wrong predictions but changes 194 correct ones to wrong. Prefix acceptance extends at 74 anchors, shortens at 42, and remains unchanged at 908/1,024. CE improves by 0.0225 and top-1 agreement by 0.45 percentage points. These are small prediction improvements; the uniformly placed diagnostic anchors differ from live verifier visits, so their roughly 2% prefix-length gain should not be substituted for the measured live acceptance or speed. Raw paired transitions are in `results/paired_probe.json`.

Correction geometry changes modestly: BA's top singular-direction energy is 77.56% (sampled: 83.99%), top-two energy 90.69% (94.38%), response mean-correction energy 73.58% (74.40%), and correction/native-fusion energy 1.91% (1.52%). This descriptive result is consistent with a somewhat richer/stronger correction, not evidence that it solves the remaining first-failure errors. No causal intervention is implied.

For this data, architecture, optimizer budget and dense-tiling recipe, eliminating supervision omissions does not materially improve deployment speed. This does not rule out other anchor distributions, objectives, architectures or workloads.


Total allocation for this study, including bounded validation and evaluation: **1.7817 GPU-hours**; peak **2 GPUs**; allocation span **65.6 minutes**. The pre-existing cached-data cost is not charged again.

Interpret results as one predeclared recipe and seed on an already examined 128-prompt set. Dense supervision, increased label count and deterministic tiled anchor placement are coupled in this experiment. Raw timings, acceptance survival, factors, fit summaries, diagnostics, pinned source hashes and resource records are retained under `experiments/dflash_nanocoder_dense_20260909`.

<a id="study-13"></a>


## 13. 2026-09-09_dflash_nanocoder_diagnosis

Source: [2026-09-09_dflash_nanocoder_diagnosis.md](paper/data/history/2026-09-09_dflash_nanocoder_diagnosis.md).

### NanoCoder: where fusion adaptation helps and fails

The completed fusion-r56 model improves prediction probabilities more than uninterrupted exact drafting. On all128 held-out prompts, fixed-prefix CE falls2.2607→1.9465, while top1 agreement rises50.39%→53.10%. Approximately99.0% of the **net summed CE reduction** occurs on positions wrong in both conditions. This is an accounting decomposition, not a claim that99% of predictions are wrong.

#### Fixed experiment and method

Target is BF16 Qwen3-4B plus `usernamebetter/nanocoder-v1`, revision `f078bfc2722bdb7bf644906cc7d81662ab3de9d2`. Native DFlash and the completed fusion-r56 checkpoint are exactly those in [the standalone study](paper/data/history/2026-09-09_dflash_nanocoder.md). That report pins base/draft/data revisions and the full training recipe. The fusion checkpoint SHA256 is `796caf147d18fe6478c84db04ab60252692120be3cac46ceba699878aff3526b`. No new fit or target continuation is used in this diagnostic.

Reuse all128 saved CodeAlpaca evaluation continuations (28,332 generated tokens; every request stopped naturally). Capture dense BF16 adapted-target features after layers[1,9,17,25,33] on exactly these saved tokens using canonical vLLM Qwen forward. This is always the NanoCoder-adapted target. Use unchanged SpecForge DFlash blocks, revision `953d43a0c1c0f5e32989dc43f91ce5fc2d9ddfef`, with a custom diagnostic selection of up to8 evenly spaced response anchors from first response token to penultimate token. There are1,024 anchors and13,370 valid prediction positions. The clean anchor is excluded; padding/out-of-bounds labels are excluded; blocks see strictly prior target features and their own bidirectional masked block.

For each identical anchor, compare native fusion and the exported-style BF16 merged fusion:

$$
F'=F_0+BA,\quad F_0\in\mathbb R^{2560\times12800},\quad B\in\mathbb R^{2560\times56},\quad A\in\mathbb R^{56\times12800},\quad\alpha/r=1.
$$

The inherited training optimized only A/B using actual target hard-token CE through the frozen draft body. It used4,096 target-generated continuations, cap4,096,3epochs/1,536updates,8 random anchors/example, globalbatch8, AdamW1e−4, seed42 and the final checkpoint. This diagnostic changes neither model nor loss. Manual per-token CE agrees with the upstream masked objective on bounded shortest/longest cases. Store labels, predictions, valid masks, margins and CE for each anchor under both conditions.

With actual target label y, record $-\log q(y)$, whether $\arg\max q=y$, and prefix survival $s_k=\prod_{j=1}^k\mathbf1[\arg\max q_j=y_j]$. Fixed-prefix accepted draft length is $\sum_{k=1}^{15}s_k$ over available labels, without a verifier bonus. All15 agreement is computed only where15 labels are available. These anchors differ from live verifier visits, so the offline lengths are not replacements for live acceptance metrics.

#### Results

| Metric | Native | Fusion-r56 |
| --- | ---: | ---: |
| Valid-token-weighted CE | 2.2607 | 1.9465 |
| Top1 agreement | 50.39% | 53.10% |
| All15 exact agreement | 8.89% | 10.29% |
| Mean fixed-prefix accepted draft tokens | 4.7256 | 4.9844 |
| Live acceptance length including verifier token (inherited) | 5.0415 | 5.1320 |
| Live speedup vs native (inherited) | 1.0000× | 1.0194× |

| Prediction transition | Token positions | Summed CE reduction (native minus adapted) |
| --- | ---: | ---: |
| Wrong → wrong | 5,717 | +4,159.66 |
| Wrong → correct | 916 | +1,062.35 |
| Correct → wrong | 554 | −586.39 |
| Correct → correct | 6,183 | −435.63 |

The net correction count is362. Of827 anchors where native drafting fails before exhausting available labels,180 extend beyond that first failure after adaptation. Across all1,024 anchors,102 instead fail earlier after adaptation. An isolated corrected token later in a block does not improve acceptance if an earlier mismatch remains.

| Proposal position | Native → adapted top1 | Native → adapted prefix survival |
| --- | ---: | ---: |
| 1 | 82.91% →85.94% | 82.91% →85.94% |
| 5 | 59.53% →61.88% | 43.83% →46.75% |
| 10 | 40.27% →42.76% | 22.85% →23.64% |
| 15 | 23.98% →27.60% | 8.89% →10.29% |

Each position uses anchors with that position available. All15 positions are retained in `results/analysis_original.json`. The update improves early agreement too; this does not support claiming it *only* learns late tokens. The modest benefit is the balance of corrections and regressions across contiguous predictions.

Illustrative first-failure cases are selected deterministically as the first three examples of each transition in evaluation order, not as a prevalence estimate. Both drafters predict ` CSS` where the target writes ` simple` in an introductory sentence; another correction changes `'s` to the target's ` is` and extends the prefix. One regression changes a single space to three spaces. These show that exact drafting failures can involve prose and formatting as well as code structure; they do not by themselves establish a deficit in coding semantics. Token IDs and continuation snippets are retained in `results/first_failure_examples.json`.

#### Geometry and interpretation

For response features h, calculate $c=BAh$ before normalization, with FP32 matrix arithmetic. Across all28,332 response positions:

$$
\text{mean-energy fraction}=\frac{N\|\bar c\|^2}{\sum_t\|c_t\|^2}=90.97\%,\qquad
\frac{\sum_t\|c_t\|^2}{\sum_t\|F_0h_t\|^2}=1.275\%.
$$

Thin QR of B and A-transpose followed by a56×56 SVD measures the effective BA update, invariant to factor reparameterization. Its top1/top2 singular directions contain99.640%/99.992% of squared weight energy. Thus the fitted rank56 parameterization learned an almost rank-one matrix with a large mean response correction. These are descriptive geometric measurements, not a NanoCoder constant-vector intervention or a causal proof that rank-one structure limits speed. No base-target counterfactual or semantic direction labels are used.

The evidence supports probability steering that only modestly improves contiguous exact predictions. It does not establish target coding quality, a universal coding-domain limitation, or that more data cannot help. The ongoing separately reported scale-up tests further optimization, longer natural continuations and more training examples. Evaluation remains fixed; these analyses do not select a checkpoint.

Source/artifacts: `experiments/dflash_nanocoder_scale_20260909/{probe_model,probe,analyze_probe,capture}.py`, `setup/probe_rows.json`, `results/probe_original.json`, `results/analysis_original.json`. Dense evaluation features persist under `features/probe_all` on node06. Probe timing and checkpoint hash are recorded in the raw result; allocation accounting is shared with the scaling family and is not double-counted.

<a id="study-14"></a>


## 14. 2026-09-09_dflash_nanocoder_objectives

Source: [2026-09-09_dflash_nanocoder_objectives.md](paper/data/history/2026-09-09_dflash_nanocoder_objectives.md).

### NanoCoder acceptance-aware objective comparison

The target is the actual BF16 Qwen3-4B with NanoCoder active in every evaluation. This study changes the fusion-adaptation objective while retaining the completed dense experiment’s data, initialization, supervision positions and optimizer budget.

**Models, data and training.** Reuse 18,016 cached adapted-target CodeAlpaca rollouts with a 16,384-token response cap: 5,260,572 actual response tokens, rather than 18,016 full-length responses. The first response token is a clean anchor. Adjacent 15-label DFlash blocks cover the remaining 5,242,556 tokens once per epoch. All frozen target/drafter layers, embeddings, output head and RMSNorm retain their original weights. Only fusion-r56 factors are trained: 860,160 FP32 parameters, alpha/rank=1, BF16 forward computation.

For concatenated target features $h\in\mathbb R^{12800}$, the pre-normalization context is $u=(F_0+BA)h$, with $A\in\mathbb R^{56\times12800}$ and $B\in\mathbb R^{2560\times56}$.

Each fit independently starts from the original 4,096-example NanoCoder fusion checkpoint, resets AdamW, and runs 3 epochs / 6,756 updates. Global batch 8 = 2 examples/GPU × 2 GPUs × 2 accumulation steps; LR 1e-4, betas (0.9,0.999), epsilon1e-8, weight decay 0, gradient clipping 1, 5% linear warmup followed by cosine decay, seed 42, final checkpoint only. The pre-existing uniform dense fit is reused as a training reference; all four inference conditions receive fresh timing.

**Exact objectives.** Let $m_{bj}$ mark eligible response labels and $\ell_{bj}=-\log p_\theta(y_{bj}\mid h_b,\text{masked block})$, with draft positions $j=1,\ldots,15$. Each microbatch minimizes

$$L=\frac{\sum_{b,j}m_{bj}w_{bj}\ell_{bj}}{\sum_{b,j}m_{bj}w_{bj}}.$$

Uniform CE uses $w_{bj}=1$. DFlash decay uses $w_{bj}=\exp(-(j-1)/7)$. Accept-until-fail uses $w_{bj}=\operatorname{stopgrad}\left[\prod_{i<j}\mathbf1\{\arg\max p_\theta(\cdot)_i=y_{bi}\}\right]$: retain the first failure itself and mask later positions. Padding does not break the prefix.

Decay calls the pinned SpecForge implementation. AUF is our implementation of the published formula, not author-released code or a full paper replication. References: [DFlash](https://arxiv.org/html/2602.06036v1), [Spec-AUF](https://arxiv.org/html/2607.01893v1). The frozen-fusion adaptation, dense tiling and optimization recipe are declared deviations from original full-drafter training.

Memory-bounded chunks contain at most 64 blocks/example. Gradients of chunk numerators accumulate before division by the complete microbatch’s detached weight sum; normalized microbatch gradients are averaged over accumulation steps and ranks. This preserves weighting for dynamic AUF support. There are 15,727,668 eligible labels across 3 epochs; AUF activates a subset, so that count must not be described as its number of nonzero-loss labels.

**Evaluation.** Same 128 held-out prompts, 2,048 generated-token cap, thinking disabled, temperature 0 / top_p 1, top-k disabled (resolved top_k=0), 15 proposals plus verifier token, vLLM 0.28.0 on L40S with the audited batch-invariant verifier and CUDA graphs. Disjoint workers (count recorded in setup/eval_execution.json) process prompt indices; variant order rotates across workers. Speed is total output tokens divided by summed timed single-request intervals; model loading and warmup are excluded. This is a latency benchmark, not 12-client serving throughput.

| Objective | tok/s | × native | × uniform | Mean accepted length τ | Accepted/proposed | Fit seconds (2 GPUs) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| native | 139.64 | 1.0000 | 0.9557 | 5.0415 | 26.94% | — |
| uniform | 146.12 | 1.0464 | 1.0000 | 5.2810 | 28.54% | 2683.4 (reused) |
| decay7 | 146.19 | 1.0469 | 1.0005 | 5.2874 | 28.58% | 2714.0 |
| auf | 149.66 | 1.0718 | 1.0242 | 5.4251 | 29.50% | 2792.1 |

τ=1+accepted draft tokens/verifier iterations, including the verifier token. Per-position survival, exact counts, cap counts, startup times and inherited AR ratios are in results/summary.json and acceptance_survival.csv. Paired 10,000-resample prompt intervals quantify prompt variability, not independent hardware-run uncertainty.

decay7 versus uniform: 1.0005×, paired 95% interval [0.9958, 1.0050].

auf versus uniform: 1.0242×, paired 95% interval [1.0156, 1.0333].

| Objective | Timed seconds | × AR (inherited timing) | Native ratio 95% interval | Capped / 128 | Fit GPU-hours |
| --- | ---: | ---: | --- | ---: | ---: |
| native | 202.9 | 4.483 | [1.0000, 1.0000] | 0 | — |
| uniform | 193.9 | 4.691 | [1.0345, 1.0601] | 0 | 1.491 (reused) |
| decay7 | 193.8 | 4.694 | [1.0357, 1.0594] | 0 | 1.508 |
| auf | 189.3 | 4.805 | [1.0586, 1.0868] | 0 | 1.551 |

AR timing is inherited; the matched native/uniform comparisons above use fresh measurements. Training GPU-hours count the measured fit interval on two GPUs; allocation-level costs include setup and validation separately.

**Finding.** AUF improves throughput by 2.42% over the matched uniform fit, whereas decay is indistinguishable from uniform in this trial. AUF reaches 1.0718× native and stays below the predeclared 1.10× NanoCoder mixture threshold, so no NanoCoder serving mixture is launched. Its ordinary CE becomes worse while accepted prefixes and live acceptance improve: unweighted CE alone is insufficient to rank these objectives for inference speed.

**Reproduction and scope.** setup/protocol.json pins the objective source hash and deviations; train.py, objectives.py and dense.py implement the fit. Immutable setup/*.sh recipes record the environment. Initial/final checkpoint hashes, optimizer histories and measured training GPU-hours are in modules/*/fusion_r56/summary.json. Data/model revisions and prompt construction inherit the completed [dense study](paper/data/history/2026-09-09_dflash_nanocoder_dense.md). Validation includes manual full-vocabulary CE, edge-case support, full/chunk gradient agreement and distributed training checks; machine-readable audits live in validation/. No code-execution quality or target capability gain is inferred from acceptance.

**Fixed-prefix diagnosis.** Same 128 saved prefixes, up to 8 fixed anchors each. CE below is ordinary unweighted CE for comparability, regardless of training objective. These prefix lengths exclude the verifier token and are not live τ.

| Fit | CE | Top1 % | Fixed-prefix draft length | Extended / shortened vs uniform |
| --- | ---: | ---: | ---: | --- |
| native | 2.2607 | 50.39 | 4.7256 | — |
| uniform | 1.8514 | 55.75 | 5.5020 | — |
| decay7 | 1.8555 | 55.50 | 5.4893 | 34 / 50 |
| auf | 2.1542 | 56.16 | 5.7578 | 130 / 81 |

Full first-failure transitions, position-wise survival and correction spectra are retained in analysis_*.json and paired_uniform.json.

**Measured allocation cost.** 3.6681 GPU-hours, including startup and validation. All study jobs are terminal; cross-study peak was 4 GPUs. Allocation details are in `experiments/dflash_nanocoder_objectives_20260909/results/resources.json`. Existing feature caches remain available.

<a id="study-15"></a>


## 15. 2026-09-09_dflash_nanocoder_scaling

Source: [2026-09-09_dflash_nanocoder_scaling.md](paper/data/history/2026-09-09_dflash_nanocoder_scaling.md).

### NanoCoder: continued fusion training with more examples and longer rollouts

Test whether NanoCoder's modest initial gain improves through additional fusion-only training. Every target call uses the same BF16 Qwen3-4B plus NanoCoder LoRA. Preserve the original completed experiment and its checkpoint; run three predeclared continuations with identical final-checkpoint selection. No target fine-tuning, draft-body adaptation, MoE training or mixture deployment is included.

#### Fixed contract and data

Use the exact models, adapter, system prompt, tokenizer and nonthinking contract from [the initial study](paper/data/history/2026-09-09_dflash_nanocoder.md). Target adapter revision `f078bfc2722bdb7bf644906cc7d81662ab3de9d2`; base revision `1cfa9a7208912126459214e8b04321603b3df60c`; native DFlash revision `b74e3a329c4d963783143b1e970d95b002be72bd`. BF16 base differs from the publisher's quantized training base, consistently across all conditions.

CodeAlpaca source revision `798c567f69c8f4b12fc191015e59ee34e9afe00d` contains 18,019 source-train prompts. Select 18,016 for full 32-example shards, using the same deterministic ordering as the initial experiment; the original 4,096 are its exact prefix. The 128 source-test prompts are unchanged and disjoint by prompt hash. Publisher prompt strings remain verbatim; exact messages/IDs reside in setup. Target-training overlap with this source is unknown; this is not an independently held-out coding-accuracy benchmark.

All generation is greedy temperature 0/top_p1/top_k−1/seed0. Long conditions raise the response cap from 4,096 to 16,384, with 20,480 maximum context and EOS enabled. Generate new prompts with disjoint one-GPU vLLM workers, continuous batching up to 16 sequences. Three large partitions and three subdivisions of the fourth fill available GPU slots under the four-GPU cap; later jobs persist finished RequestOutput values through a custom observer of the unchanged upstream engine loop, checked against canonical return values and saved reference prefixes. Capture dense BF16 features after target layers [1,9,17,25,33] on actual adapted-target tokens and persist them with source hashes. Hard links reuse unchanged shards; summed feature bytes are logical size, not incremental physical disk use.

A bounded extended example repeats an HTML paragraph to 16K. It is retained as actual target behavior, not as useful complex code. Longer caps need not create longer natural answers or richer supervision. No quality-based filtering of target outputs is introduced. All 4,096 overlapping cached prefix feature sequences were compared and are bitwise identical between the original and extended collections; the added future continuation does not alter those features.

#### Three continued fits

All fits start independently from the same initial final fusion factors (SHA256 `796caf147d18fe6478c84db04ab60252692120be3cac46ceba699878aff3526b`). Reset AdamW and its schedule for each continuation. The control distinguishes more optimization from more data:

| Fit | Training prompts | Response cap | Additional epochs | Additional updates |
| --- | ---: | ---: | ---: | ---: |
| Control | 4,096 original | 4,096 | 3 | 1,536 |
| Longer rollouts | 4,096 original | 16,384 | 3 | 1,536 |
| Expanded data | 18,016 nested | 16,384 | 3 | 6,756 |

Use $c=N_0[(F_0+BA)h]$, $F_0\in\mathbb R^{2560\times12800}$, $B\in\mathbb R^{2560\times56}$, $A\in\mathbb R^{56\times12800}$, alpha/r=1; 860,160 trainable parameters. Only the existing fusion factors are updated. Target, draft body, embeddings/head and normalization remain frozen. Export folds BA into BF16 F0 without an extra standalone layer.

Unchanged SpecForge DFlash source `953d43a0c1c0f5e32989dc43f91ce5fc2d9ddfef` supplies masks and loss. Each example samples up to 8 response anchors, clean anchor + 15 mask tokens (ID 151669), strictly prior target context and isolated bidirectional draft blocks. Predict same-position target-generated labels; exclude anchor/padding/out-of-range slots:

$$L_{micro}=-\frac{\sum_{i,a}\sum_{k=1}^{15}m_{iak}\log q_{A,B}(x_{i,a+k}\mid h_{i,<a},x_{i,a},\mathrm{MASK}^{15})_k}{\sum_{i,a,k}m_{iak}},\qquad L_{step}=\tfrac14\sum_{g=1}^2\sum_{b=1}^2L_{micro,g,b}.$$

No hidden MSE, teacher-logit KL or position decay. Dense feature storage is distinct from the 8 sampled training anchors: a longer response increases their available range, not their count. Only 21/4,096 (0.51%) original examples change when the cap rises; the other 99.49% already stopped naturally. The cap intervention therefore adds many stored tokens but changes only a small fraction of examples sampled by this recipe. Two-GPU DDP, microbatch 2/GPU, accumulation 2/global batch 8, seed 42; FP32 factors/BF16 forward; AdamW LR 1e−4, betas(.9,.999), eps 1e−8, weight decay 0, clip 1, 5% warmup/cosine; same shard and length-bucket ordering. Three additional epochs, final checkpoint only, no dev selection. This is continuation from trained factors with a reset optimizer, not fresh fitting or uninterrupted optimizer-state resumption. Bounded long-context CE, causal feature checks and actual continued updates precede scaling.

#### Evaluation

Fresh vLLM 0.28.0/L40S standalone measurements use native DFlash, the initial fitted checkpoint, and all three continued checkpoints. Exactly the same 128 NanoCoder prompts, response cap 2,048, nonthinking greedy settings and 15 proposals are used. Reuse saved actual-target AR token references and AR timing; AR-relative speeds therefore inherit the earlier timing trial. No training overlaps timed evaluation. Per-request timing includes prefill+decode, excludes setup/warmup/cold compilation; one active request per GPU. These are latency measurements, not concurrent-client serving throughput.

For $T_v=\sum_i t_{vi}$, speedups are $T_{native}/T_v$ and $T_{initial}/T_v$. Throughput is generated tokens/$T_v$. Mean acceptance length $\tau=1+A/V$ includes the verifier token; draft-only length=A/V; acceptance fraction=A/P. Confidence intervals are 10,000 paired-prompt bootstrap resamples, seed 20260909; they do not measure repeated hardware or training-seed variability.

| Condition | Tok/s | vs native | vs initial fit | vs AR (inherited timing) | Mean τ | Acceptance fraction | Incremental fit seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native | 139.59 | 1.0000× | 0.9847× | 4.4815× | 5.0415 | 26.94% | — |
| Initial fit | 141.76 | 1.0156× | 1.0000× | 4.5514× | 5.1320 | 27.55% | — |
| More optimization | 143.60 | 1.0288× | 1.0130× | 4.6106× | 5.1863 | 27.91% | 199.8 |
| Longer rollouts | 142.88 | 1.0236× | 1.0079× | 4.5872× | 5.1684 | 27.79% | 220.8 |
| Expanded data | 145.61 | 1.0431× | 1.0271× | 4.6749× | 5.2693 | 28.46% | 935.6 |

All **640 new speculative requests** match AR exactly. The expanded collection contains **5,260,572 response tokens**, 72 capped responses and **161.62GB** logical dense feature shards.

Median/p99 response lengths are **195/853 tokens**. Capped cases contribute **22.42%** of all response tokens. Of 21 originally capped continuations, 18 remain capped at 16K; three stop at 4,538/5,056/6,816 tokens. The original 4,096-example subset gains 225,306 response tokens.

A descriptive string audit finds 20 outputs with at least 20 consecutive identical nonempty lines (19 capped). Manual examples also show repeated alternating code fragments and very long numeric literals, which that simple statistic does not capture. The three newly natural-stopping extensions contain lengthy advice/explanations. These observations do not constitute a coding-quality score or an exclusion rule; every output is retained.

- control: vs-native 95% interval [1.0186,1.0405]×; epoch mean microbatch CE 1.9657 → 1.9601 → 1.9642; 0.1110 fitting GPU-hours.
- long: vs-native 95% interval [1.0131,1.0355]×; epoch mean microbatch CE 1.9690 → 1.9622 → 1.9661; 0.1227 fitting GPU-hours.
- scale: vs-native 95% interval [1.0312,1.0567]×; epoch mean microbatch CE 1.9633 → 1.9288 → 1.9228; 0.5198 fitting GPU-hours.

#### Fixed-prefix comparison

Repeat the original diagnostic at exactly the same saved features, anchors and labels; no new target outputs or checkpoint selection. Full method and initial results: [NanoCoder diagnosis](paper/data/history/2026-09-09_dflash_nanocoder_diagnosis.md).

| Checkpoint | CE | Top1 | All15 agreement | Fixed-prefix draft length | Correction mean energy |
| --- | ---: | ---: | ---: | ---: | ---: |
| original | 1.9465 | 53.10% | 10.29% | 4.9844 | 90.97% |
| control | 1.9252 | 53.74% | 10.41% | 5.0811 | 87.71% |
| long | 1.9242 | 53.77% | 10.53% | 5.0723 | 87.35% |
| scale | 1.8739 | 55.30% | 11.93% | 5.3965 | 74.40% |

#### Interpretation

The expanded-data fit is the strongest predeclared row: 1.0431× native and 1.0271× the initial fit. It remains below the strict 1.10× deployment gate, so no mixture benchmark is launched. Extra optimization alone improves the initial checkpoint; extra rollout length does not improve on that control.

- long vs control: 0.9949×, paired-prompt 95% interval [0.9925, 0.9974]×.
- scale vs control: 1.0139×, paired-prompt 95% interval [1.0068, 1.0213]×.
- scale vs long: 1.0191×, paired-prompt 95% interval [1.0120, 1.0264]×.

These intervals describe prompt resampling within one timing trial. In particular, the roughly 0.5% long-versus-control difference is too small to call a robust hardware-independent regression without repeated timings. The expanded-data improvement also includes a larger optimizer-update budget.

The expanded fit improves fixed-prefix CE from 1.9465 to 1.8739 and top-1 agreement from 53.10% to 55.30% relative to the initial fit. Against native, it converts 1,194 wrong predictions to correct and 537 correct predictions to wrong; the initial fit gives 916 and 554. It extends beyond the native first failure on 275/827 native-failed anchors versus 180/827 initially, while producing an earlier failure on 101/1,024 anchors versus 102 initially. Thus the gain is not merely lower confidence loss on unchanged predictions. Nevertheless, still-wrong positions contribute 82.9% of the expanded fit’s net CE reduction relative to native (99.0% initially). First-failure errors remain the main acceptance bottleneck. These diagnostic anchors are uniformly distributed across saved responses; their accepted lengths are not estimates of the live verifier’s visitation distribution.

The correction becomes less dominated by a single direction: top singular-direction energy of BA falls from 99.64% initially to 83.99%, and top-two energy from 99.99% to 94.38%. On actual response features, mean-correction energy falls from 90.97% to 74.40%, while correction/native-fusion energy rises from 1.28% to 1.52%. This is consistent with a more input-dependent adjustment after scaling, but is descriptive geometry, not a causal intervention or evidence of new coding capability in the target. All target outputs are unchanged.

![NanoCoder scaling comparison](paper/data/experiments/dflash_nanocoder_scale_20260909/results/scaling_summary.png)


Total incremental allocation, including the associated diagnosis and bounded checks: **5.6536 GPU-hours**, peak 3 GPUs, allocation span 158.78 minutes. Raw records are in `results/resource_audit.json`; inherited initial-run costs are not charged again.

This is one fixed recipe and seed on an already examined 128-prompt set. Report every predeclared final checkpoint; a best observed row is exploratory, not independent confirmation or an upper bound on achievable coding adaptation. A higher cap changes only capped continuations and can add repetition. Expanded data also increases optimizer updates; the original-data control measures extra optimization at the small-data budget, not a fully compute-matched 18K comparison.

Artifacts: `experiments/dflash_nanocoder_scale_20260909`. `prepare_data.py`, `prepare_remote.py`, `generate.py`, `capture.py`, `import_gate.py`, `train.py`, `model.py`, `runtime.py`, `evaluate_variants.py` and `pipeline.py` specify the experiment. `setup/source_manifest.json` pins local source hashes; setup recipes and manifests pin inputs. Raw rollouts are backed up locally; dense features remain persisted remotely. Probe records retain labels, predictions, margins, losses and first-failure transitions. No mixture benchmark is part of this study.

<a id="study-16"></a>


## 16. 2026-09-09_dflash_steering

Source: [2026-09-09_dflash_steering.md](paper/data/history/2026-09-09_dflash_steering.md).

### Learned context steering versus fusion adaptation

This study tests whether the benefit of fusion adaptation can be obtained with a single learned context vector, and whether adding that vector to fusion LoRA helps further. Math and KiCad always use their own active target LoRAs.

**Architecture and loss.** The frozen DFlash drafter receives five target hidden states concatenated as $h∈ℝ^{12800}$. Its fusion matrix $F_0∈ℝ^{2560×12800}$ feeds the original frozen RMSNorm. The compared pre-normalization contexts are:

$$u_{native}=F_0h,\quad u_{mean}=F_0h+\mu,\quad u_{bias}=F_0h+b,$$
$$u_{fusion}=F_0h+BAh,\qquad u_{combined}=F_0h+BAh+b.$$

The direct bias has 2,560 trainable parameters. Fusion uses rank 56, alpha 56: A[56,12800], B[2560,56],860,160 parameters. Combined training has 862,720 parameters. Bias starts at zero; PEFT initializes A randomly and B to zero. All drafter body layers, normalization weights, embeddings and output head remain frozen.

$$L=-\frac{\sum_{a,j}m_{aj}\log p_\theta(y_{aj}\mid h,\text{masked block at anchor }a)}{\sum_{a,j}m_{aj}}.$$

This is uniform token CE from pinned SpecForge: eight random anchors/example,16-position blocks with the clean anchor excluded from loss and up to 15 response labels, no position decay. Each domain uses its existing 4,096 cached adapted-target rollouts/dense five-layer features, 4,096 response cap, 3 epochs / 1,536 updates, global batch 8 (2/GPU × 2 GPUs × 2 accumulation), seed 42, final checkpoint only. AdamW betas 0.9/0.999, epsilon1e-8, weight decay 0, clip 1, 5% warmup/cosine decay. Fusion factors use LR 1e-4; direct bias coordinates use LR 1e-2 in both bias-containing variants. This declared optimizer difference makes the experiment a practical architecture comparison, not a loss-only or identical-LR comparison.

Factors and bias are FP32; forward computation is BF16. Inference folds BA into the fusion weight and adds the FP32 bias before conversion to BF16 and original RMSNorm. The training-derived mean μ is calculated from the newly fitted fusion correction on the first example of 32 uniformly spaced training shards, averaging all prompt+response positions. It is never estimated from evaluation features. Mean calibration depends on the fusion fit and therefore does not have zero acquisition cost.

**Benchmark.** Same 128 held-out prompts/domain, nonthinking greedy vLLM 0.28.0 on L40S, temperature 0 / top_p 1, top-k disabled (resolved top_k=0), 15 draft proposals, audited batch-invariant target verification. Math response cap 2,048; KiCad 8,192. Disjoint prompt workers (count recorded in setup/eval_execution.json) rotate variant order; every prompt’s variants use the same worker. Timed intervals exclude startup/warmup.

| Domain / variant | tok/s | × native | τ | Acceptance % | Fit seconds | Train GPU-hours |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| math/native | 122.49 | 1.0000 | 4.4771 | 23.18 | — | — |
| math/training_mean | 138.17 | 1.1281 | 5.0789 | 27.19 | 193.0 (parent fit) | 0.107 (reused) |
| math/bias | 129.70 | 1.0589 | 4.7534 | 25.02 | 191.8 | 0.107 |
| math/fusion_r56 | 147.84 | 1.2069 | 5.4482 | 29.65 | 193.0 | 0.107 |
| math/bias_fusion | 147.83 | 1.2069 | 5.4588 | 29.73 | 194.7 | 0.108 |
| kicad/native | 159.61 | 1.0000 | 5.9526 | 33.02 | — | — |
| kicad/training_mean | 189.62 | 1.1880 | 7.0554 | 40.37 | 599.8 (parent fit) | 0.333 (reused) |
| kicad/bias | 175.01 | 1.0965 | 6.5208 | 36.81 | 600.5 | 0.334 |
| kicad/fusion_r56 | 202.08 | 1.2661 | 7.5045 | 43.36 | 599.8 | 0.333 |
| kicad/bias_fusion | 202.15 | 1.2665 | 7.5129 | 43.42 | 601.4 | 0.334 |

The mean row reuses the fusion fit; those costs must not be counted twice. Its extraction time was not isolated from the training-stage allocation. τ includes the verifier token: 1+A/V. Acceptance%=100A/P, where A is accepted draft tokens, V verification iterations and P proposed draft tokens. Speedup is the ratio of summed native/request times, not an average of per-request multipliers.

| Domain / variant | Timed seconds | × AR (inherited timing) | Native ratio 95% interval | Capped /128 |
| --- | ---: | ---: | --- | ---: |
| math/native | 155.7 | 3.924 | [1.0000, 1.0000] | 0 |
| math/training_mean | 138.0 | 4.427 | [1.1050, 1.1516] | 0 |
| math/bias | 147.0 | 4.155 | [1.0482, 1.0703] | 0 |
| math/fusion_r56 | 129.0 | 4.737 | [1.1802, 1.2361] | 0 |
| math/bias_fusion | 129.0 | 4.736 | [1.1811, 1.2351] | 0 |
| kicad/native | 4013.2 | 5.568 | [1.0000, 1.0000] | 18 |
| kicad/training_mean | 3378.0 | 6.615 | [1.1800, 1.1960] | 18 |
| kicad/bias | 3660.2 | 6.105 | [1.0911, 1.1015] | 18 |
| kicad/fusion_r56 | 3169.8 | 7.049 | [1.2548, 1.2772] | 18 |
| kicad/bias_fusion | 3168.7 | 7.052 | [1.2554, 1.2777] | 18 |

| Domain / alternative | × fusion | Paired 95% interval |
| --- | ---: | --- |
| math/training_mean | 0.9346 | [0.9261, 0.9426] |
| math/bias | 0.8773 | [0.8612, 0.8926] |
| math/bias_fusion | 1.0000 | [0.9971, 1.0028] |
| kicad/training_mean | 0.9384 | [0.9349, 0.9419] |
| kicad/bias | 0.8660 | [0.8611, 0.8711] |
| kicad/bias_fusion | 1.0004 | [0.9996, 1.0011] |

| Domain / variant | Natural-stop prompts | × native, natural stops | Capped prompts | × native, capped |
| --- | ---: | ---: | ---: | ---: |
| math/training_mean | 128 | 1.1281 | 0 | — |
| math/bias | 128 | 1.0589 | 0 | — |
| math/fusion_r56 | 128 | 1.2069 | 0 | — |
| math/bias_fusion | 128 | 1.2069 | 0 | — |
| kicad/training_mean | 110 | 1.1885 | 18 | 1.1862 |
| kicad/bias | 110 | 1.0978 | 18 | 1.0914 |
| kicad/fusion_r56 | 110 | 1.2618 | 18 | 1.2824 |
| kicad/bias_fusion | 110 | 1.2618 | 18 | 1.2848 |

Finish-reason subsets are descriptive splits of the same fixed prompts. Natural stopping is not a circuit-validity or code-correctness test.

**Finding.** The training-derived mean retains about 62% of the incremental fusion speed gain on Math and 71% on KiCad. The directly trained bias retains about 28% and 36%, respectively. Bias+fusion is indistinguishable from fusion alone in the paired timing intervals. KiCad gains persist on the 110 naturally stopped outputs; they are not confined to capped continuations.

Because the extracted mean is itself a constant vector, the weaker bias-only fit does not establish a capacity limit of constant-vector adaptation. The obtained fit, optimizer choice, generalization and CE-to-acceptance alignment can matter; this experiment does not separate those explanations.

Intervals use 10,000 paired prompt bootstrap samples, seed 20260909; they do not capture repeated-run hardware uncertainty. Full proposal counts and per-position acceptance survival are preserved in results/summary.json and acceptance_survival.csv.

**Interpretation and reproduction.** Compare retained *incremental* speed benefit as $(s_{bias}-1)/(s_{fusion}-1)$, rather than $s_{bias}/s_{fusion}$. Raw bias storage is 10,240 FP32 bytes versus 3,440,640 factor bytes, excluding serialization and shared model weights. Standalone benchmarks load exported drafts; shared-serving memory benefits must be measured separately.

setup/protocol.json, model.py, train.py and immutable setup/*.sh give the full recipe; modules/{domain}/{variant}/summary.json records training histories, optimizer settings, source hashes and allocated GPU-time. Runtime correction hooks are explicitly custom, layered over upstream vLLM; no custom token verifier is substituted. The mathematical mask, frozen gradients, training decrease, merged output and representative exact-target token checks passed before full evaluation. Machine-readable audit artifacts are in validation/. Model and dataset revisions inherit the [Math objective study](paper/data/history/2026-09-07_dflash_lora_objective.md) and [KiCad study](paper/data/history/2026-09-08_dflash_kicad_mapping.md). This study measures drafting efficiency, not target task accuracy.

**Measured allocation cost.** 7.6819 GPU-hours, including startup and validation. All study jobs are terminal; cross-study peak was 4 GPUs. Allocation details are in `experiments/dflash_steering_20260909/results/resources.json`. Existing feature caches remain available.

<a id="study-17"></a>


## 17. 2026-09-09_dflash_steering_causality

Source: [2026-09-09_dflash_steering_causality.md](paper/data/history/2026-09-09_dflash_steering_causality.md).

### Causal analysis of learned context steering

This separate analysis uses the completed Math/KiCad steering checkpoints. It introduces no further gradient training or target generation; two rank-one reader controls are calibrated analytically on cached training features. All labels and cached hidden states come from the actual adapted targets.

**Design.** Reuse 32 length-stratified held-out prefixes/domain and up to 8 evenly spaced response anchors/prefix, scoring 15 draft positions per anchor. These inherited diagnostic caches retain at most 4,096 response tokens, even though the live KiCad benchmark allows 8,192. Selection uses 32 uniformly spaced ranks in original response-length order, with group ID breaking ties. This intentionally fixed-prefix sample is not the live verifier’s visitation distribution. Every intervention is added before the original RMSNorm; the drafter body, output head, target features and labels are held fixed. Analytic geometry uses FP32 F₀h. Scoring adds each FP32 correction to the model’s BF16 fusion output and converts back to BF16 before the original RMSNorm. Small rounding effects can therefore remain in nominally radial controls. These counterfactual corrections are not claims of live speedup.

For the independently fitted fusion let $c(h)=BAh$ and training-only mean $μ=E_{train}[c(h)]$. For combined training let $c_c(h)=B_cA_ch$ and $b_c$ be its learned bias. We compare native, fusion, μ, c−μ, learned bias b, 0.5b, 2b, −b, three signed coordinate permutations, the other domain’s bias rescaled to||b||, c_c+b_c, b_c alone and c_c alone. Cross-domain vectors use only training data.

For $u=F_0h$, decompose $b=b_\parallel+b_\perp$, with $b_\parallel=u(u^Tb)/(u^Tu)$ at each token. The original normalization is $N(u)=g\odot u/r$, $r=\sqrt{\|u\|^2/d+\epsilon}$, $d=2560$. Its exact Jacobian action is

$$J_N(u)b=\frac{g}{r}\odot\left(b-\frac{u(u^Tb)}{d r^2}\right).$$

We compare the actual normalized shift N(u+b)−N(u) against this linearization and separately apply radial-only and tangent-only bias. A constant pre-normalization vector therefore can induce an input-dependent direction change after normalization; it need not correspond to restoration of base-model features. Centered FP64 finite differences (step 1e-4, relative tolerance 1e-5) validate the FP32 analytic Jacobian on bounded real examples. Radial suppression is a local statement: a finite radial perturbation that makes 1+(uᵀb)/(uᵀu) negative flips the direction. We record such sign crossings and the full-bias linearization error instead of assuming the small-perturbation approximation holds.

**Results.** Token CE and top1 pool eligible labels; fixed-prefix accepted draft length counts consecutive correct proposals, excluding the verifier token. This is distinct from live τ.

| Domain / intervention | CE | Top1 % | Fixed-prefix draft length | All15 % |
| --- | ---: | ---: | ---: | ---: |
| math/random_rowspace_reader | 1.9975 | 51.77 | 5.1484 | 7.91 |
| math/constant_reader | 1.9814 | 51.65 | 5.1719 | 7.91 |
| math/reader_residual | 2.7740 | 49.24 | 4.7930 | 7.44 |
| math/svd1 | 1.9860 | 52.15 | 5.1445 | 8.37 |
| math/native | 3.0737 | 47.16 | 4.6875 | 6.98 |
| math/fusion | 1.8730 | 53.79 | 5.3125 | 9.77 |
| math/fusion_mean | 1.9904 | 51.89 | 5.1641 | 7.91 |
| math/fusion_centered | 2.7506 | 49.42 | 4.8359 | 7.44 |
| math/bias | 2.4139 | 50.04 | 4.8711 | 7.91 |
| math/bias_half | 2.6951 | 48.92 | 4.7500 | 7.91 |
| math/bias_double | 2.1260 | 50.70 | 4.9062 | 6.98 |
| math/negative_bias | 4.0568 | 44.40 | 4.4375 | 6.51 |
| math/cross_bias_norm_matched | 2.8256 | 48.14 | 4.7188 | 8.37 |
| math/combo | 1.8720 | 53.79 | 5.3398 | 10.23 |
| math/combo_bias_only | 3.0268 | 47.37 | 4.6875 | 6.98 |
| math/combo_residual_only | 1.8728 | 53.70 | 5.2773 | 8.84 |
| math/radial_bias | 3.0736 | 47.22 | 4.6953 | 6.98 |
| math/tangent_bias | 2.4137 | 49.99 | 4.8750 | 7.91 |
| math/permuted_bias_10 | 3.1098 | 47.16 | 4.7266 | 7.91 |
| math/permuted_bias_11 | 3.0548 | 47.61 | 4.7344 | 6.98 |
| math/permuted_bias_12 | 3.0768 | 47.13 | 4.7070 | 8.37 |
| kicad/random_rowspace_reader | 1.4527 | 68.87 | 7.8438 | 29.46 |
| kicad/constant_reader | 1.4326 | 69.49 | 7.8398 | 31.25 |
| kicad/reader_residual | 1.6665 | 67.75 | 7.4336 | 26.79 |
| kicad/svd1 | 1.4409 | 69.10 | 7.8867 | 33.93 |
| kicad/native | 2.1279 | 61.47 | 6.3633 | 13.39 |
| kicad/fusion | 1.3409 | 71.85 | 8.3359 | 43.75 |
| kicad/fusion_mean | 1.4441 | 69.28 | 7.8906 | 31.70 |
| kicad/fusion_centered | 1.6509 | 67.87 | 7.4805 | 27.23 |
| kicad/bias | 1.6889 | 67.28 | 7.3477 | 23.66 |
| kicad/bias_half | 1.8324 | 64.98 | 6.9688 | 18.75 |
| kicad/bias_double | 1.6219 | 66.51 | 7.3203 | 21.43 |
| kicad/negative_bias | 3.1794 | 51.59 | 5.1094 | 4.91 |
| kicad/cross_bias_norm_matched | 1.9988 | 62.21 | 6.5156 | 15.62 |
| kicad/combo | 1.3399 | 71.79 | 8.3320 | 43.75 |
| kicad/combo_bias_only | 2.0882 | 62.18 | 6.5195 | 14.29 |
| kicad/combo_residual_only | 1.3421 | 71.88 | 8.3594 | 44.20 |
| kicad/radial_bias | 2.1272 | 61.65 | 6.4375 | 13.84 |
| kicad/tangent_bias | 1.6883 | 67.07 | 7.3047 | 23.21 |
| kicad/permuted_bias_10 | 2.1647 | 61.26 | 6.4336 | 12.50 |
| kicad/permuted_bias_11 | 2.1686 | 61.11 | 6.4414 | 13.39 |
| kicad/permuted_bias_12 | 2.1339 | 61.32 | 6.3242 | 12.50 |

| Domain | Cos(b, μ) | Fusion mean energy % | Combined mean energy % | Radial bias energy % | RMS linearization relative error, median |
| --- | ---: | ---: | ---: | ---: | ---: |
| math | 0.5781 | 92.61 | 92.60 | 0.17 | 0.0138 |
| kicad | 0.4271 | 90.33 | 90.13 | 0.23 | 0.0173 |

**Does the matrix read an almost constant feature?** Let Q be an orthonormal basis of the fitted A row space and z=Qᵀh. On the same 32 training examples used for μ, estimate m=E[z] and C=Cov(z), then solve

$$q=\frac{(C+\lambda I)^{-1}m}{m^T(C+\lambda I)^{-1}m},\quad\lambda=\max(10^{-8},0.01\,\mathrm{tr}(C)/56).$$

This minimizes $q^TCq+\lambda\|q\|^2$ subject to $q^Tm=1$. Using an orthonormal basis avoids dependence on arbitrary rescaling of the LoRA factors. The extracted rank-one correction is $c_{reader}(h)=\mu\,q^TQ^Th$. We compare it, its residual $c-c_{reader}$, the fixed μ correction, and the original leading-SVD rank-one correction. This is closed-form calibration on cached training features, with no new CE fitting or model forward. Its held-out variability tests whether the learned input subspace supports an approximately constant scalar; low variability alone does not establish predictive importance.

| Domain | Reader held-out mean | Reader held-out CV | SVD1 input scalar CV | CPU calibration seconds |
| --- | ---: | ---: | ---: | ---: |
| math | 0.9730 | 0.1313 | 0.2637 | 0.65 |
| kicad | 1.0142 | 0.1008 | 0.3157 | 9.34 |

A matched random-rowspace control draws a Gaussian 12,800×56 matrix (seed 20260909), orthonormalizes it, and fits the same minimum-variance reader on the same training rows. Its output direction remains μ. This separates the ability to read a constant from generic feature anisotropy versus the learned input subspace.

| Domain | Learned-rowspace reader CV | Random-rowspace reader CV |
| --- | ---: | ---: |
| math | 0.1313 | 0.1912 |
| kicad | 0.1008 | 0.2042 |

Mean-energy fraction is $\|\sum_t c_t\|^2/(n\sum_t\|c_t\|^2)$ on response positions. Weight spectra use singular values of BA via thin QR, avoiding conclusions that depend on arbitrary rescaling of A and B. Geometry pools all response positions, while CE scores the fixed anchor sample. Mean-energy dominance and low matrix rank are different statements.

The combined-fit loss interaction is $I=L(c_c+b_c)-L(b_c)-L(c_c)+L(0)$. Negative I indicates superadditive loss reduction on these fixed prefixes; it is not a causal attribution of uniquely identifiable parameters, because constant and matrix components can compensate for one another.

math: mean per-prefix interaction I=0.04618.

kicad: mean per-prefix interaction I=0.03742.

![Fixed-prefix causal effects](paper/data/experiments/dflash_steering_causality_20260909/results/causal_effects.png)


| Domain / intervention | Retained fusion CE gain % | CE reduction vs native, paired 95% interval |
| --- | ---: | --- |
| math/constant_reader | 91.0 | 1.0923 [0.8995, 1.2978] |
| math/random_rowspace_reader | 89.6 | 1.0762 [0.8829, 1.2794] |
| math/reader_residual | 25.0 | 0.2997 [0.1913, 0.4216] |
| math/svd1 | 90.6 | 1.0877 [0.8960, 1.2909] |
| math/fusion_mean | 90.2 | 1.0833 [0.8896, 1.2877] |
| math/fusion_centered | 26.9 | 0.3231 [0.1855, 0.4717] |
| math/bias | 55.0 | 0.6598 [0.5514, 0.7756] |
| math/combo | 100.1 | 1.2017 [0.9807, 1.4335] |
| math/radial_bias | 0.0 | 0.0001 [-0.0024, 0.0027] |
| math/tangent_bias | 55.0 | 0.6600 [0.5514, 0.7755] |
| math/cross_bias_norm_matched | 20.7 | 0.2481 [0.1777, 0.3280] |
| kicad/constant_reader | 88.3 | 0.6952 [0.5834, 0.8130] |
| kicad/random_rowspace_reader | 85.8 | 0.6752 [0.5638, 0.7919] |
| kicad/reader_residual | 58.6 | 0.4614 [0.3672, 0.5639] |
| kicad/svd1 | 87.3 | 0.6869 [0.5675, 0.8135] |
| kicad/fusion_mean | 86.9 | 0.6838 [0.5736, 0.7999] |
| kicad/fusion_centered | 60.6 | 0.4770 [0.3819, 0.5806] |
| kicad/bias | 55.8 | 0.4390 [0.3523, 0.5315] |
| kicad/combo | 100.1 | 0.7880 [0.6523, 0.9311] |
| kicad/radial_bias | 0.1 | 0.0007 [-0.0012, 0.0028] |
| kicad/tangent_bias | 55.9 | 0.4395 [0.3530, 0.5322] |
| kicad/cross_bias_norm_matched | 16.4 | 0.1291 [0.0971, 0.1644] |

Retained CE gain is (L_native−L_intervention)/(L_native−L_fusion). Intervals resample the 32 complete prefixes 10,000 times, seed 20260909, preserving within-prefix anchor dependence. They describe this exploratory, length-stratified diagnostic sample and do not replace a new independent evaluation.

| Domain / rank-one control | CE excess over full fusion | Paired 95% interval |
| --- | ---: | --- |
| math/constant_reader | 0.1084 | [0.0656, 0.1564] |
| math/random_rowspace_reader | 0.1245 | [0.0807, 0.1727] |
| math/svd1 | 0.1131 | [0.0721, 0.1530] |
| kicad/constant_reader | 0.0917 | [0.0527, 0.1372] |
| kicad/random_rowspace_reader | 0.1118 | [0.0705, 0.1587] |
| kicad/svd1 | 0.1000 | [0.0615, 0.1517] |

Positive CE excess means worse loss than full fusion. A confidence interval crossing zero is not an equivalence test.

| Domain | Minimum finite radial scale | Radial sign crossings % |
| --- | ---: | ---: |
| math | 0.99871 | 0.00000 |
| kicad | 0.99769 | 0.00000 |

**Prior-art boundary.** Bias and representation adaptation are established: [BitFit](https://aclanthology.org/2022.acl-short.1/), [RED](https://arxiv.org/abs/2402.15179), and [ReFT](https://arxiv.org/html/2404.03592v3). This study tests a specific DFlash interface mechanism and its separately measured inference benefit. It does not establish novelty of bias tuning or semantic meaning for a learned direction. The random-rowspace control uses one declared seed.

**Interpretation.** The training-mean correction retains 90.2% of fusion’s fixed-prefix CE reduction on Math and 86.9% on KiCad. A calibrated rank-one reader retains 91.0% and 88.3%; a reader calibrated in an independent random subspace still retains 89.6% and 85.8%. Thus reading an approximately constant scalar is partly a generic property of the feature distribution, not evidence by itself that the learned input subspace encodes a special skill. The learned subspace gives a steadier scalar, but every rank-one control has significantly higher CE than full fusion in the paired intervals above.

The residual after subtracting the mean still reduces CE, especially on KiCad. Mean and residual effects are nonlinear and cannot be added as independent percentages. In the jointly fitted model, the matrix component alone retains essentially all the CE gain, whereas its explicit bias alone retains only 3.9%/5.0%. The positive interaction values indicate overlapping rather than superadditive benefit in this decomposition.

Reversing the bias is harmful, signed permutations mostly remove its benefit, and a norm-matched bias from the other domain transfers only a small part of the gain. Direction therefore matters in these controls. Doubling the separately fitted bias improves CE but does not reliably improve fixed-prefix acceptance, further separating loss reduction from accepted length.

Radial-only CE changes are indistinguishable from zero here, while tangent-only bias tracks full bias. However, only 0.17%/0.23% of the learned bias energy is radial to begin with. This control and the analytic Jacobian are consistent with directional steering; they do not isolate RMSNorm as the cause of the method’s benefit. BF16 rounding can change individual argmax decisions even when aggregate CE barely moves. A no-normalization counterfactual was not tested.

| Domain | Top-two singular-direction weight energy, fusion % | Joint-fit matrix % |
| --- | ---: | ---: |
| math | 99.924 | 99.926 |
| kicad | 98.460 | 98.399 |

Weight-energy concentration, correction mean energy, retained CE reduction and retained live speed are distinct quantities. For example, the constant mean retains about 90%/87% of the CE improvement here but only about 62%/71% of the incremental live speed improvement in the standalone study. This supports a largely constant steering mechanism with useful residual adaptation, not equivalence to full fusion or an identified semantic capability.

**Limits and reproduction.** This is exploratory analysis on the previously defined diagnostic prefixes, not an independent test set or proof of semantic meaning for an individual direction. probe.py contains all interventions and the analytic Jacobian; collect.py defines every aggregate. setup/protocol.json fixes conditions and data scope. results/{domain}.json retains every anchor, label/prediction ID, validity mask, CE, margin and prefix outcome, together with checkpoint hashes. Representative canonical-CE, finite-difference and decomposition checks precede all 32-prefix probes. Live speed and fitting costs belong to the separate [steering comparison](paper/data/history/2026-09-09_dflash_steering.md), not this report.

**Measured allocation cost.** 0.0311 GPU-hours, including startup and validation. All study jobs are terminal; cross-study peak was 4 GPUs. Allocation details are in `experiments/dflash_steering_causality_20260909/results/resources.json`. Existing feature caches remain available.

<a id="study-18"></a>


## 18. 2026-09-10_dflash_auf_domains

Source: [2026-09-10_dflash_auf_domains.md](paper/data/history/2026-09-10_dflash_auf_domains.md).

### GSM8K and KiCad: accept-until-fail fusion adaptation

Standalone inference always uses Qwen3-4B with the corresponding target LoRA active.


4096 cached active-target rollouts/domain, original 4096 response cap; no new decoding. Reuse dense five-layer features. Match the steering study uniform fusion-r56 baseline: native draft initialization, PEFT A random/B zero, seed42; 8 sampled block anchors/example, block16 (clean anchor plus15 labels), 3epochs,1536 updates, global batch8 over2 GPUs with2 accumulation steps, AdamW1e-4, betas .9/.999 eps1e-8 wd0, clip1,5% warmup cosine, final checkpoint. Only fusion factors train (860160 parameters, alpha/r=1). All other weights frozen. AUF weights CE through the first incorrect proposal inclusive with detached strict-prefix correctness; subsequent labels receive zero weight. Upstream sums chunk numerators/denominators within microbatch; DDP/accumulation averages microbatch losses as in the uniform reference. No dense anchor sweep: this matches the prior loss baseline, and differs from the larger dense NanoCoder recipe.

Canonical pinned SpecForge and the previously validated AUF formula implementation are reused. AUF is a custom formula adaptation, not author-released code. No new target generation is authorized.

Fresh standalone vLLM comparisons: native, reused matched uniform fusion-r56, new AUF.128 fixed prompts/domain, greedy nonthinking, response cap2048 math and8192 KiCad; existing prompt contracts unchanged. Max4 assistant GPUs. Cross-domain effect sizes are not a matched comparison against NanoCoder's larger dense training.

Target/data provenance: Qwen/Qwen3-4B revision1cfa9a7208912126459214e8b04321603b3df60c; z-lab/Qwen3-4B-DFlash-b16 revisionb74e3a329c4d963783143b1e970d95b002be72bd. GSM8K target adapter witcheer/qwen3-4b-gsm8k-grpo revisionf52e1ed1b3accd404d37a9621db0536dd99bdfc5; openai/gsm8k revision740312add88f781978c0658806c59bc2815b9866, publisher prompt source816390766c9251cb0e8a1699b145ec56dd83bf67. Preserve the parent selected training/test messages and IDs.

KiCad target adapter AbijahKaj/qwen3-4b-kicad-netlist revisionba1878e0777bf02c22bcef1c3e9b79095181d455 (r64,alpha32). Dataset AbijahKaj/kicad-netlist-sft-dataset revision52ce8608c3c0fd2f1b6b68e93b8784b88bc92373: publisher tool-then-direct ordering, original template with thinking off, full-text <=8192 filter, 2% split seed42. Existing direct training prompts deduplicated and shuffled seed42; evaluation direct validation prompts shuffled20260908 after removing prompt overlap. Reuse the exact selected messages; generated active-target responses are CE labels, not the publisher assistant references. Source-repository and unknown target-training overlaps are not certified by mapper prompt disjointness.

Five target hidden layers use vLLM boundaries[2,10,18,26,34], concatenated width12800; original fusion maps to2560 before frozen RMSNorm. BF16 forward computation; FP32 trainable LoRA factors; folded BF16 exports. vLLM0.28.0 on L40S, temperature0/top_p1/top_k disabled/seed0, batch-invariant target computation, frozen-drafter embedding/head sharing and canonical target verification. The exact upstream source hashes are in setup/upstream.json.

Hardware policy: all measurements must use NVIDIA L40S; identical physical GPU UUIDs are not required. Preserve per-request GPU names/UUIDs for audit. Use complete128-prompt measurements for every condition.


For block labels y_j and logits p_j, w_j=stopgrad[prod_{i<j} 1(argmax p_i=y_i)], with clean anchors and padding excluded from loss and ignored in prefix correctness. The exact microbatch objective is L=sum_{b,j} m_{bj} w_{bj} (-log p_{bj}(y_{bj}))/sum_{b,j}m_{bj}w_{bj}. The first failure remains supervised. Uniform CE sets w=1.

AUF uses the same newly initialized fusion factors and anchor RNG schedule as the matched uniform fit; it does not continue that fitted uniform checkpoint. Dense feature storage does not mean dense loss supervision: eight sampled anchors are retained for a loss-only comparison.

| Domain / variant | tokens/s | × native [paired 95% CI] | × uniform [paired 95% CI] | τ | draft-only | accepted/proposed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| math/native | 121.85 | 1.0000 [1.0000, 1.0000] | 0.8282 [0.8085, 0.8468] | 4.4771 | 3.4771 | 23.18% |
| math/uniform | 147.12 | 1.2074 [1.1809, 1.2368] | 1.0000 [1.0000, 1.0000] | 5.4482 | 4.4482 | 29.65% |
| math/auf | 152.18 | 1.2490 [1.2216, 1.2795] | 1.0344 [1.0229, 1.0460] | 5.6604 | 4.6604 | 31.07% |
| kicad/native | 159.37 | 1.0000 [1.0000, 1.0000] | 0.7896 [0.7826, 0.7970] | 5.9526 | 4.9526 | 33.02% |
| kicad/uniform | 201.83 | 1.2665 [1.2547, 1.2778] | 1.0000 [1.0000, 1.0000] | 7.5045 | 6.5045 | 43.36% |
| kicad/auf | 210.80 | 1.3227 [1.3067, 1.3382] | 1.0444 [1.0400, 1.0488] | 7.8209 | 6.8209 | 45.47% |

Speedups divide summed matched prompt request times. All retained measurements use NVIDIA L40S GPUs; physical cards may differ between conditions. Four workers rotate condition order. Timings exclude loading, warmup and detected first-use JIT; these costs persist in raw measurements. The 10,000 paired-prompt bootstrap samples quantify prompt uncertainty, not training-seed or hardware-repeat uncertainty. No AR timing is reused to imply a fresh speedup baseline.

Training and cached data:
- math: 544,585 cached response tokens across4,096 examples; endings {'stop': 4094, 'length': 2}. AUF training 3.41 minutes on2GPUs, 0.114 GPU-hours, 1536 updates. Eligible anchor labels and nonzero AUF labels differ; no full-token supervision count is claimed.
- kicad: 13,708,854 cached response tokens across4,096 examples; endings {'length': 2200, 'stop': 1896}. AUF training 8.93 minutes on2GPUs, 0.298 GPU-hours, 1536 updates. Eligible anchor labels and nonzero AUF labels differ; no full-token supervision count is claimed.

Pinned upstream SpecForge revision: `953d43a0c1c0f5e32989dc43f91ce5fc2d9ddfef`. Exact cache paths, prior uniform training summaries and source SHA256s are in validation/cache.json; target prompt/token contracts and exact AR references are in setup/ar_math.json and setup/ar_kicad.json. Inputs retain the previous study's dataset split and adapter revisions; source metadata is copied into setup/inherited/. Data and fit initialization follow the [steering study](paper/data/history/2026-09-09_dflash_steering.md).

Interpret this as two within-target loss comparisons. NanoCoder used more examples and dense block tiling, so these numbers do not isolate domain differences against NanoCoder. Quality of the target model itself is not measured by acceptance. No expanded generation or deployment is authorized by this experiment.

Raw per-request outputs, timing and counters: measurements/. Position survival and compact tables: results/. Frozen-target verification is unchanged; AUF affects only training.

Total allocated GPU-hours including checks and evaluation: 1.5619444444444444.

<a id="study-19"></a>


## 19. 2026-09-10_dflash_auf_scale

Source: [2026-09-10_dflash_auf_scale.md](paper/data/history/2026-09-10_dflash_auf_scale.md).

### GSM8K / KiCad AUF: training-duration checkpoints on 4,096 unique examples


Reuse the same4,096 cached active-target examples/domain and4096 response cap. Counts denote cumulative example presentations, not unique prompts. No new decoding/capture. Initialize fusion-r56 afresh from native DFlash, PEFT random A/zero B, seed42. AUF training keeps8 sampled masked-block anchors/example,block16 with clean anchor excluded,alpha/r=1 and860,160 trainables; drafter body, norms, embedding/head and target frozen.

Global batch8 on2GPUs,microbatch2 and2 accumulation steps. Process exactly16,000 examples=2000 optimizer updates, cycling the cached4096 examples with the inherited epoch shuffle. Save fixed snapshots at4000/8000/12000/16000 presentations (steps500/1000/1500/2000). Evaluate only8000 and16000, after training finishes. AdamW1e-4,betas(.9,.999),eps1e-8,wd0,clip1;100-step warmup and cosine to step2000. BF16 forwards,FP32 trainable factors. Loss L=sum m*w*CE/sum m*w per microbatch; w is detached strict-prefix correctness, retaining the first failure and ignoring padding. DDP/accumulation averages microbatch losses as in the matched prior run.

Exactly128 prompts/domain/checkpoint with active target LoRA. Greedy nonthinking vLLM,L40S,math evalcap2048,KiCad8192. Reuse complete native/uniform/prior-AUF timing controls from the preceding study; do not launch additional controls. Report baseline source and paired-prompt intervals; unchanged eval dataset and target contracts. Fixed checkpoints do not use a dev set. Same-type L40S cards allowed; UUID matching is not required.

Target/data provenance: Qwen/Qwen3-4B revision1cfa9a7208912126459214e8b04321603b3df60c; z-lab/Qwen3-4B-DFlash-b16 revisionb74e3a329c4d963783143b1e970d95b002be72bd. GSM8K target adapter witcheer/qwen3-4b-gsm8k-grpo revisionf52e1ed1b3accd404d37a9621db0536dd99bdfc5; openai/gsm8k revision740312add88f781978c0658806c59bc2815b9866, publisher prompt source816390766c9251cb0e8a1699b145ec56dd83bf67. Preserve the parent selected training/test messages and IDs.

KiCad target adapter AbijahKaj/qwen3-4b-kicad-netlist revisionba1878e0777bf02c22bcef1c3e9b79095181d455 (r64,alpha32). Dataset AbijahKaj/kicad-netlist-sft-dataset revision52ce8608c3c0fd2f1b6b68e93b8784b88bc92373: publisher tool-then-direct ordering, original template with thinking off, full-text <=8192 filter, 2% split seed42. Existing direct training prompts deduplicated and shuffled seed42; evaluation direct validation prompts shuffled20260908 after removing prompt overlap. Reuse the exact selected messages; generated active-target responses are CE labels, not the publisher assistant references. Source-repository and unknown target-training overlaps are not certified by mapper prompt disjointness.

Five target hidden layers use vLLM boundaries[2,10,18,26,34], concatenated width12800; original fusion maps to2560 before frozen RMSNorm. BF16 forward computation; FP32 trainable LoRA factors; folded BF16 exports. vLLM0.28.0 on L40S, temperature0/top_p1/top_k disabled/seed0, batch-invariant target computation, frozen-drafter embedding/head sharing and canonical target verification. The exact upstream source hashes are in setup/upstream.json.



| Domain / checkpoint | tokens/s | × native | × uniform | × prior AUF | τ | draft-only | accepted/proposed | Training min / GPU-hours |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| math/native | 121.85 | 1.0000 | 0.8282 | 0.8007 | 4.4771 | 3.4771 | 23.18% | reused |
| math/uniform | 147.12 | 1.2074 | 1.0000 | 0.9667 | 5.4482 | 4.4482 | 29.65% | reused |
| math/auf_12288 | 152.18 | 1.2490 | 1.0344 | 1.0000 | 5.6604 | 4.6604 | 31.07% | reused |
| math/auf_8000 | 152.77 | 1.2538 | 1.0384 | 1.0038 | 5.6587 | 4.6587 | 31.06% | 1.94 / 0.065 |
| math/auf_16000 | 153.44 | 1.2593 | 1.0430 | 1.0083 | 5.6914 | 4.6914 | 31.28% | 3.83 / 0.128 |
| kicad/native | 159.37 | 1.0000 | 0.7896 | 0.7560 | 5.9526 | 4.9526 | 33.02% | reused |
| kicad/uniform | 201.83 | 1.2665 | 1.0000 | 0.9575 | 7.5045 | 6.5045 | 43.36% | reused |
| kicad/auf_12288 | 210.80 | 1.3227 | 1.0444 | 1.0000 | 7.8209 | 6.8209 | 45.47% | reused |
| kicad/auf_8000 | 210.11 | 1.3184 | 1.0410 | 0.9967 | 7.7954 | 6.7954 | 45.30% | 5.37 / 0.179 |
| kicad/auf_16000 | 213.37 | 1.3389 | 1.0572 | 1.0122 | 7.9123 | 6.9123 | 46.08% | 10.51 / 0.350 |

The768 reused baseline outputs are checked again during aggregation. GPU UUIDs may differ; all measured cards are NVIDIA L40S.

Paired-prompt95% bootstrap intervals (10,000 samples) for all ratios are in results/summary.json. They cover prompt resampling, not run/seed/hardware uncertainty. Acceptance survival is in results/acceptance_survival.csv. Speed ratios divide sums of matched-prompt request times; no average of per-prompt ratios. Model load/warmup and first-use JIT are excluded from steady timing and retained in raw measurements.

The prior AUF baseline used a1536-update cosine schedule terminating after12,288 presentations. The new8k and16k checkpoints share a2000-update schedule. Thus comparison to the prior run includes a schedule change, not just extra optimizer updates. The8k checkpoint is midway through the16k schedule, not a separately converged8k fit.

All four milestones retain trainable factors, optimizer state, step/epoch/batch and deterministic anchor-seed metadata, plus folded exports. The4k/12k checkpoints are not evaluated. No new rollout generation or feature extraction. Files live under experiments/dflash_auf_scale_20260910; evaluation baselines are explicitly reused from dflash_auf_domains_20260910.

Total allocated compute, including validation and evaluation: 2.562 GPU-hours.

<a id="study-20"></a>


## 20. 2026-09-10_dflash_kicad_auf_data_scale

Source: [2026-09-10_dflash_kicad_auf_data_scale.md](paper/data/history/2026-09-10_dflash_kicad_auf_data_scale.md).

### KiCad AUF: expanded unique training data

Target: Qwen3-4B with the pinned KiCad LoRA active. Reuse the canonical KiCad prompt/split contract and 4096-response-token cap. The eligible deduplicated training pool contains 15,366 prompts: 4,096 cached plus 11,270 newly generated. Evaluation uses the same 128 held-out prompts and 8192 response-token cap as the preceding study. Thinking disabled; greedy vLLM on L40S.

Only fusion-r56 is fitted: F = F0 + BA, A in R^(56 x 12800), B in R^(2560 x 56), alpha/r=1; 860,160 trainable parameters. Initialize native drafter plus random A/zero B; freeze target, drafter body, normalization, embeddings and head. Fold BA into the fusion matrix for inference.

Dense BF16 features are captured from target layers with vLLM boundaries [2,10,18,26,34]; training samples eight DFlash blocks per example, each a clean anchor plus 15 prediction slots. This is sampled-anchor AUF, not the dense NanoCoder recipe. L = sum(m*w*CE)/sum(m*w), where w_j is the detached product of correctness indicators at eligible positions before j. Include the first failure; ignore padding/anchor and positions after failure. The AUF formula adaptation and runtime inherit the validated preceding study, not an official full-paper replication.

One pass through all 15,366 unique examples, shuffled shards/length buckets with seed42; the six-example tail stays last. Two-GPU DDP, microbatch2, two accumulation steps, global batch8 (final batch6 split3 per rank). Weight the last microbatches by their example fractions; ordinary full batches retain the prior mean-of-microbatches normalization. AdamW LR1e-4, betas .9/.999, eps1e-8, weight decay0, clipping1, 5% warmup/cosine over 1921 updates, FP32 factors/BF16 forwards. Save4000/8000/12000/15366; evaluate only8000 and15366. The8k checkpoint shares this full-run schedule; no checkpoint selection.

Native/uniform/priorAUF controls reuse matching earlier L40S measurements. Ratios use summed request times, including prefill/decode and excluding load/warmup. Bootstrap intervals reflect prompt variability only. All new and reused rows are checked against the active target AR token IDs and finish reasons.

| Checkpoint | Tokens/s | x native | x uniform | Mean acceptance length | Fit seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8000 | 210.31 | 1.3196 | 1.0420 | 7.7954 | 469.2 |
| 15366 | 212.88 | 1.3358 | 1.0547 | 7.8950 | 944.9 |

Acceptance length = 1 + accepted draft tokens / verifier iterations. Full counters, acceptance fraction, confidence intervals, and fitting GPU-hours are in results/summary.json; survival in results/acceptance_survival.csv. Source pins, capture checks and immutable job recipes reside in setup/ and validation/. This run changes data coverage and schedule from the prior repeated-4096 duration study; it is not a matched compute scaling law.

<a id="study-21"></a>


## 21. 2026-09-10_dflash_nanocoder_direct

Source: [2026-09-10_dflash_nanocoder_direct.md](paper/data/history/2026-09-10_dflash_nanocoder_direct.md).

### NanoCoder: dense causal next-token and hidden-state objectives

**Result.** The best new variant is **full_ce**, at **0.6690× native DFlash** (paired-prompt 95% interval 0.6536–0.6845). None of the new variants exceeds the requested 10% threshold. This is exploratory selection among nine variants on one fixed evaluation set, not independent confirmation of the winning configuration.

#### Question and controlled design

Can training the fusion interface with clean causal next-token prediction, direct final-hidden regression, or both improve NanoCoder drafting beyond the previously fitted block-objective interface? The target is always BF16 Qwen3-4B with the `usernamebetter/nanocoder-v1` LoRA active. The native control uses that same adapted target with its unmodified DFlash drafter. `auf` is the previous best interface, included as a reference; it used the earlier 18,016-example training set. All nine new variants share the full 18,019-example data and optimization schedule below.

This is a custom causal training contract. It is **not** the upstream masked-block DFlash loss. Deployment uses ordinary DFlash block proposals and target verification, so a causal-to-block training mismatch remains part of what this experiment tests.

#### Data and supervision

Use all **18,019 training examples** from `HuggingFaceH4/CodeAlpaca_20K`, revision `798c567f69c8f4b12fc191015e59ee34e9afe00d`. Preserve the released parquet prompt verbatim as the user message (publisher preprocessing: stripped instruction + newline + stripped input). The system message is `You are NanoCoder, an expert Senior Full-Stack Engineer and debugging agent.`; render using the pinned adapter tokenizer/chat template with `enable_thinking=False`. Generate greedy responses from the actual NanoCoder target, capped at **16,384 response tokens** and stopping at EOS. The responses contain **5,260,941 tokens in total**, mean 291.97, median 195, p99 853; 72 reach the cap. The cap is not a fixed rollout length. Reuse 18,016 saved rollouts and generate the three remaining source rows. Source IDs are preserved in `setup/train_ids.json`.

Cache the concatenated five target hidden layers (2,10,18,26,34 in the existing capture convention), width 12,800, at every prefix position. Also persist the adapted target's final normalized pre-head state, width 2,560, for direct MSE: 30.15 GiB of additional teacher-state shards. Paired capture checks match the existing five-layer states exactly. No feature-position sampling is used.

For a sequence x₀,…,xₙ₋₁ with p prompt tokens, supervise every query t=p−1,…,n−2, predicting xₜ₊₁. This includes the first response token, predicted from the final prompt position. There are **15,782,823 supervised token presentations over three epochs**. Full prefixes and full-sequence backpropagation are retained.

Evaluation uses the unchanged **128 mapper-held-out NanoCoder prompts**, a **2,048-response-token cap**, thinking disabled, greedy temperature 0, top-p 1 and no top-k restriction. No new dev set or checkpoint selection is used. These prompts are held out from mapper training; overlap with the target adapter’s private training mixture is unknown, so this is not an independent target-accuracy benchmark.

#### Forward computation and exact losses

Write hₜ∈ℝ¹²⁸⁰⁰ for the five-layer concatenation and F(hₜ)∈ℝ²⁵⁶⁰ for the trainable fusion output, followed by the existing frozen hidden RMSNorm. The frozen drafter processes clean token embeddings with two allowed attention regions: query t can access target-context keys k<t and clean draft-stream keys k≤t. Other keys are masked. Absolute positions are shared by the context and clean-token streams. Let zₜ be the drafter's final normalized output and gₜ the adapted target's final normalized pre-head state at the same prefix. U is the unchanged vocabulary head shared with the drafter.

For T response labels and d=2,560:

$$
\mathcal L_{CE}=-\frac1T\sum_{t=p-1}^{n-2}\log\operatorname{softmax}(Uz_t)_{x_{t+1}},\qquad
\mathcal L_{MSE}=\frac1{Td}\sum_{t=p-1}^{n-2}\lVert z_t-g_t\rVert_2^2.
$$

The three objectives are CE, MSE, and CE+MSE with coefficient **1** on each. Neither objective uses masked diffusion inputs or sampled anchors. CE is full-vocabulary cross entropy, projected in 128-token chunks with checkpointed recomputation. Every global optimizer update averages over its actual total response-token count, so long examples contribute proportionally to their token count.

#### Architectures and fitting

All variants start from the same BF16 reconstruction F₀ of the initial fitted NanoCoder fusion. This is an already adapted fusion, not the native matrix. New residuals use Kaiming-uniform A and zero B; all other target/drafter weights and norms stay frozen.

- `linear56`: F(h)=F₀h+BAh, A∈ℝ⁵⁶ˣ¹²⁸⁰⁰, B∈ℝ²⁵⁶⁰ˣ⁵⁶, α/r=1. **860,160** trainable parameters. Rank 56 refers to the new residual, not the total change from native.
- `full`: F(h)=Wh, W initialized to F₀. **32,768,000** trainable parameters.
- `mlp56`: F(h)=F₀h+B SiLU(Ah), same shapes and **860,160** trainable parameters. The nonlinear residual remains explicit in vLLM; its runtime cost is measured.

Each architecture is crossed with all three objectives. Use seed 42, **three epochs**, global batch **8 examples over two GPUs**, and retain the last three-example batch: **2,253 updates/epoch, 6,759 total**. AdamW: lr 10⁻⁴, betas (0.9,0.999), epsilon 10⁻⁸, weight decay 0, gradient-norm clipping at 1, 5% linear warmup followed by cosine decay. BF16 forward computation and FP32 trainable parameters; activate draft-layer checkpointing. Export the **final checkpoint**, folding linear updates into BF16 fusion weights.

#### Results

| Variant | Output tokens/s | Speedup vs native [95% CI] | Vs prior AUF | Acceptance length τ | Accepted draft-only length | Accepted/proposed | Training wall / GPU-hours |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| native | 139.72 | 1.0000× [1.0000, 1.0000] | 0.9340× | 5.0415 | 4.0415 | 26.94% | reused checkpoint |
| auf | 149.60 | 1.0707× [1.0570, 1.0860] | 1.0000× | 5.4251 | 4.4251 | 29.50% | reused checkpoint |
| linear56_ce | 82.65 | 0.5915× [0.5763, 0.6070] | 0.5525× | 2.9287 | 1.9287 | 12.86% | 41.64 min / 1.388 |
| linear56_mse | 37.15 | 0.2659× [0.2513, 0.2812] | 0.2483× | 1.2900 | 0.2900 | 1.93% | 37.40 min / 1.247 |
| linear56_ce_mse | 69.09 | 0.4945× [0.4772, 0.5126] | 0.4619× | 2.4348 | 1.4348 | 9.57% | 41.06 min / 1.369 |
| full_ce | 93.47 | 0.6690× [0.6536, 0.6845] | 0.6248× | 3.3212 | 2.3212 | 15.47% | 41.87 min / 1.396 |
| full_mse | 41.61 | 0.2978× [0.2823, 0.3145] | 0.2781× | 1.4532 | 0.4532 | 3.02% | 37.28 min / 1.243 |
| full_ce_mse | 64.45 | 0.4613× [0.4429, 0.4809] | 0.4308× | 2.2712 | 1.2712 | 8.47% | 41.63 min / 1.388 |
| mlp56_ce | 84.74 | 0.6065× [0.5919, 0.6215] | 0.5664× | 3.0192 | 2.0192 | 13.46% | 40.91 min / 1.364 |
| mlp56_mse | 37.33 | 0.2672× [0.2531, 0.2821] | 0.2495× | 1.3034 | 0.3034 | 2.02% | 36.81 min / 1.227 |
| mlp56_ce_mse | 69.10 | 0.4946× [0.4777, 0.5122] | 0.4619× | 2.4474 | 1.4474 | 9.65% | 41.07 min / 1.369 |

τ=1+Σ accepted draft tokens / Σ verification iterations; the draft-only column omits the bonus token. Accepted/proposed is a different metric. Full per-position acceptance survival is in `results/acceptance_survival.csv`.

Native and AUF are freshly timed alongside new variants using vLLM 0.28.0 on L40S, one request at a time per worker, four independent workers with fixed prompt strata and rotated condition order. Each worker keeps its allocated GPU throughout its conditions. Batch-invariant target computation, audited frozen embedding/head sharing, CUDA graphs, and the canonical DFlash verifier are enabled. Model load, warmup, and detected JIT first-use time are excluded from steady request latency and recorded separately; training time is reported separately. Speedup is Σ native request seconds / Σ variant request seconds, **not** an average of per-request ratios. The intervals use 10,000 paired prompt bootstrap samples; they do not cover training-seed or hardware-repeat uncertainty. AR reference outputs establish token agreement; their old timings are not reused as a fresh speed baseline.

#### Compute cost

The complete study consumed **14.771 allocated GPU-hours**, including bounded validation, teacher-state capture, model initialization, fitting, export, and evaluation. Training allocations used 12.279 GPU-hours, new teacher capture 0.405, and final evaluation 1.892. The table's fit wall timers begin after model/data initialization and include optimization/checkpoint export; they are not whole-allocation wall time. Full job timing and parallelism are preserved in `results/resources.json`. At most four GPUs are allocated to this study concurrently.

![Dense direct-objective comparison](paper/data/experiments/dflash_nanocoder_direct_20260910/results/direct_objectives.png)

#### Reproduction artifacts and limits

Family: `experiments/dflash_nanocoder_direct_20260910`. `README.md`, `model.py`, `train.py`, `capture_runtime.py`, `capture.py`, `runtime.py`, `interface_runtime.py`, and `evaluate.py` define the computation. `setup/protocol.json` pins models and SpecForge; `results/data_inventory.json` inventories cached shards; module summaries contain code hashes, token coverage, final checkpoints and timing. Raw request outputs, finish reasons, timings and acceptance counters persist in `measurements/`; aggregate tables are generated by `collect.py`.

Interpret differences within the matched nine-variant grid. The previous AUF reference differs in objective and initial training recipe, and its earlier source count lacks three examples. A gain in exact-target inference speed does not establish improved task accuracy. This fixed-set sweep does not establish generalization across coding adapters, training seeds, or production concurrency.

<a id="study-22"></a>


## 22. 2026-09-11_dflash_auf_mapper_probes

Source: [2026-09-11_dflash_auf_mapper_probes.md](paper/data/history/2026-09-11_dflash_auf_mapper_probes.md).

### Final AUF fusion mapper mechanism probes

This separate study investigates the final individual fusion-r56 AUF mappers and the one pooled fusion-r56 AUF mapper. It does not train new mappers or measure live throughput. The body-LoRA comparison is a separate experiment.

For concatenated adapted-target hidden vector h and frozen native fusion F0, the correction is c(h)=BAh and the draft receives RMSNorm(F0h+c(h)). All draft blocks, norms, embeddings and head are frozen. Both A and B belong to existing fitted AUF checkpoints; the pooled checkpoint is identical across all three target domains. Provenance hashes are retained in the raw probe results.

Use32 fixed heldout prefixes/domain and up to8 evenly spaced response anchors/prefix, with15 continuation slots and truncated-tail masks. Math and KiCad reuse the existing32 diagnostic prefixes. NanoCoder selects32 length-stratified prefixes from its128 previously captured prefixes. Every feature vector comes from the actual domain-adapted target on that exact prefix; no base-target substitution and no new decoding. Canonical pinned SpecForge draft blocks produce the predictions. This is an explicitly custom diagnostic readout, not an official benchmark implementation.

The calibration set is the first training example in each of128 immutable cache shards, using response positions only. Define the token-weighted training mean mu=mean_train c(h). The heldout data never fit the mean. Compare c(h), mu, c(h)-mu, the rank2 truncated SVD correction, the complement after removing those two directions, -c(h), and three random signed coordinate permutations of c(h). Random controls preserve each token's correction norm exactly. The top directions are singular directions of the full learned BA product, computed using reduced QR and a small FP64 SVD, not independent singular directions of B or A. Exact reconstruction and zero-correction checks precede the full probes.

Each intervention adds its FP32 correction to the native BF16 fusion output immediately before RMSNorm and casts back to BF16. This hook isolates correction content; it does not exactly emulate the rounding of a merged live weight. Identical anchors, labels and clean prefixes are retained across all interventions. Record token CE, top1 predictions, target-vs-largest-other-logit margins, fixed-native-competitor margins and prefix survival. For K consecutive correct draft slots, E[K]=sum_j P(K>=j) on this finite fixed-prefix sample. This standard identity defines the diagnostic statistic; it is not claimed as a novel theorem or as a live speedup estimate.

A first-failure repair counts only when the old first failed slot becomes correct and all earlier slots remain correct. A regression counts any newly incorrect slot in the native accepted prefix. Separately record finite differences from correction scales+0.01 and-0.01 against the same native competitor. Those are BF16-quantized finite differences, not exact Jacobians; predicting one competitor margin does not guarantee full argmax or prefix correctness. Prefix bootstrap intervals resample whole prompts to retain correlated anchors.

| Domain | Checkpoint | Intervention | Mean CE | Fixed-prefix draft length | First-failure repair fraction | Prefix regression fraction | New training min |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | native | none | 3.0737 | 4.6875 | 0.0000 | 0.0000 | 0 |
| math | individual | full | 2.1050 | 5.4297 | 0.3206 | 0.0938 | 0 |
| math | individual | mean | 2.2545 | 5.3203 | 0.3254 | 0.0781 | 0 |
| math | individual | centered | 2.8278 | 4.9219 | 0.1005 | 0.0469 | 0 |
| math | individual | svd2 | 2.1086 | 5.4648 | 0.3301 | 0.0898 | 0 |
| math | individual | without_svd2 | 3.0611 | 4.6836 | 0.0096 | 0.0078 | 0 |
| math | individual | negative | 4.9029 | 2.5117 | 0.0766 | 0.5469 | 0 |
| math | individual | random_10 | 3.1415 | 4.5820 | 0.1531 | 0.1445 | 0 |
| math | individual | random_11 | 3.1729 | 4.3789 | 0.1340 | 0.1719 | 0 |
| math | individual | random_12 | 3.3019 | 4.4922 | 0.1100 | 0.1445 | 0 |
| math | pooled | full | 2.1220 | 5.5156 | 0.3206 | 0.0898 | 0 |
| math | pooled | mean | 2.2907 | 5.2500 | 0.3014 | 0.0898 | 0 |
| math | pooled | centered | 2.8257 | 5.0352 | 0.1340 | 0.0664 | 0 |
| math | pooled | svd2 | 2.4344 | 5.2383 | 0.2727 | 0.0508 | 0 |
| math | pooled | without_svd2 | 2.5335 | 5.1172 | 0.2632 | 0.0820 | 0 |
| math | pooled | negative | 4.8673 | 2.5625 | 0.0526 | 0.5312 | 0 |
| math | pooled | random_10 | 3.1609 | 4.5391 | 0.1675 | 0.1875 | 0 |
| math | pooled | random_11 | 3.1680 | 4.2891 | 0.1388 | 0.1797 | 0 |
| math | pooled | random_12 | 3.3186 | 4.4648 | 0.1292 | 0.1641 | 0 |
| kicad | native | none | 2.1279 | 6.3633 | 0.0000 | 0.0000 | 0 |
| kicad | individual | full | 1.6347 | 8.4688 | 0.5306 | 0.0195 | 0 |
| kicad | individual | mean | 1.7552 | 8.0156 | 0.4745 | 0.0430 | 0 |
| kicad | individual | centered | 1.6873 | 7.7422 | 0.3980 | 0.0234 | 0 |
| kicad | individual | svd2 | 1.7035 | 8.1875 | 0.5102 | 0.0312 | 0 |
| kicad | individual | without_svd2 | 1.8947 | 7.0430 | 0.2449 | 0.0352 | 0 |
| kicad | individual | negative | 6.1063 | 1.0664 | 0.0051 | 0.8203 | 0 |
| kicad | individual | random_10 | 2.9539 | 4.6328 | 0.0765 | 0.4180 | 0 |
| kicad | individual | random_11 | 2.7715 | 4.5117 | 0.0459 | 0.4023 | 0 |
| kicad | individual | random_12 | 2.7874 | 4.7930 | 0.1020 | 0.4258 | 0 |
| kicad | pooled | full | 1.6746 | 8.4141 | 0.5306 | 0.0195 | 0 |
| kicad | pooled | mean | 1.7743 | 7.9727 | 0.4745 | 0.0430 | 0 |
| kicad | pooled | centered | 1.7543 | 7.6797 | 0.3673 | 0.0234 | 0 |
| kicad | pooled | svd2 | 1.7457 | 8.0508 | 0.4694 | 0.0312 | 0 |
| kicad | pooled | without_svd2 | 1.8533 | 7.2344 | 0.2602 | 0.0273 | 0 |
| kicad | pooled | negative | 5.7360 | 1.2969 | 0.0000 | 0.7930 | 0 |
| kicad | pooled | random_10 | 2.7080 | 5.1680 | 0.0918 | 0.3320 | 0 |
| kicad | pooled | random_11 | 2.6625 | 4.6836 | 0.0561 | 0.4102 | 0 |
| kicad | pooled | random_12 | 2.6760 | 4.9336 | 0.1224 | 0.3984 | 0 |
| nanocoder | native | none | 2.3808 | 4.4297 | 0.0000 | 0.0000 | 0 |
| nanocoder | individual | full | 2.2544 | 5.1836 | 0.3119 | 0.0742 | 0 |
| nanocoder | individual | mean | 2.3029 | 4.8711 | 0.2294 | 0.0703 | 0 |
| nanocoder | individual | centered | 2.3054 | 4.9453 | 0.2064 | 0.0391 | 0 |
| nanocoder | individual | svd2 | 2.2595 | 5.1992 | 0.3073 | 0.0664 | 0 |
| nanocoder | individual | without_svd2 | 2.3725 | 4.5000 | 0.0367 | 0.0039 | 0 |
| nanocoder | individual | negative | 2.8634 | 3.7344 | 0.0642 | 0.3086 | 0 |
| nanocoder | individual | random_10 | 2.4292 | 4.4023 | 0.1284 | 0.1133 | 0 |
| nanocoder | individual | random_11 | 2.4378 | 4.4219 | 0.0917 | 0.1172 | 0 |
| nanocoder | individual | random_12 | 2.4002 | 4.3906 | 0.0872 | 0.1250 | 0 |
| nanocoder | pooled | full | 2.2846 | 5.2383 | 0.2936 | 0.0625 | 0 |
| nanocoder | pooled | mean | 2.3235 | 4.8320 | 0.1972 | 0.0703 | 0 |
| nanocoder | pooled | centered | 2.3102 | 4.7227 | 0.1651 | 0.0469 | 0 |
| nanocoder | pooled | svd2 | 2.2915 | 4.7109 | 0.1606 | 0.0352 | 0 |
| nanocoder | pooled | without_svd2 | 2.3806 | 4.9336 | 0.2248 | 0.0742 | 0 |
| nanocoder | pooled | negative | 2.7937 | 3.7344 | 0.0550 | 0.2969 | 0 |
| nanocoder | pooled | random_10 | 2.4250 | 4.2812 | 0.0963 | 0.1250 | 0 |
| nanocoder | pooled | random_11 | 2.4666 | 4.4375 | 0.1193 | 0.1055 | 0 |
| nanocoder | pooled | random_12 | 2.3977 | 4.3789 | 0.0872 | 0.1250 | 0 |

No fitting occurs in this probe study. Existing checkpoint training costs are reported in the corresponding training and body-baseline reports. All interventions, per-position survival, correction energy fractions, weight spectra, pooled cross-domain mean cosines and finite-difference diagnostics are in `results/probe_summary.json` and `results/probe_metrics.csv`; per-token records remain in `results/probe_{domain}.json`. These fixed-prefix results require separate live validation before assigning speedup implications. The theoretical interpretation is deferred until the body baseline and probe results have both been reviewed.

<a id="study-23"></a>


## 23. 2026-09-11_dflash_auf_mixture

Source: [2026-09-11_dflash_auf_mixture.md](paper/data/history/2026-09-11_dflash_auf_mixture.md).

### Multi-LoRA mixture with matched sampled-AUF fusion mappers

One BF16 Qwen3-4B target and one frozen DFlash drafter; active GSM8K,KiCad,NanoCoder LoRAs. All mapper checkpoints are reused: fusion-r56,alpha56,4096 unique examples/domain,4096 response-token cap,8 sampled clean-anchor+15-mask blocks,2000 optimizer updates/16000 presentations. AUF loss L=sum(m*w*CE)/sum(m*w),detached strict-prefix correctness weights,first failure included. Native random-A/zero-B initialization,AdamW1e-4,betas(.9,.999),eps1e-8,wd0,clip1,100-step warmup/cosine to2000,seed42,globalbatch8 across2GPUs. Frozen draft body/norms/embed/head. Target feature layers[1,9,17,25,33],dense cache; sampled loss. Checkpoint paths/hashes and measured fitting costs in setup/selection.json. No mapper retraining in this deployment study.

Canonical vLLM AsyncLLM and LoRARequest/Punica kernels with the existing audited request-indexed fusion hook; rank padded to64,alpha/r=1,no bias. Target-ID selects the matching mapper for the entire request. Request batches preserve target/drafter token ordering and original verification. Native and mapped use identical active-target requests. Adapter-specific chat templates,thinking disabled,temperature0,top_p1,top_k disabled,seed0. Actual installed package/source hashes in setup/runtime_*.json. This is custom fusion integration,not an official upstream experiment.

Fixed128 prompts/domain,384 total; existing held-out evaluation data,math/NanoCoder cap2048,KiCad8192. Twelve closed-loop streaming clients; same order in each native/mapped pair. Two repetitions reverse condition order. L40S GPUs,one engine/GPU,max4 assistant GPUs. Exclude model loading/short warmup,retain bounded full-workload JIT warmup separately. Report throughput including workload drain,latency excluding client semaphore wait; include route/render overhead.

Clients supply their target adapter ID.

| Domain | Mapper training wall min (2 GPUs) |
| --- | ---: |
| math | 3.83 |
| kicad | 10.51 |
| nanocoder | 4.54 |

Router CPU fitting: 0.944s; held-out source-domain accuracy: 99.740%.

| Repeat | Speedup vs native | Native τ | Mapped τ |
| --- | ---: | ---: | ---: |
| 0 | 1.3176x | 5.855 | 7.678 |
| 1 | 1.3288x | 5.855 | 7.678 |

τ=1+accepted draft tokens/verification steps. Full throughput,latency,TTFT,TPOT,memory,acceptance fraction/survival,router probabilities/confusion and per-source latency ratios in results/oracle.json. Per-domain ratios are latency under contention; aggregate counters do not establish per-domain acceptance. Existing test sets have been used in prior model selection; these results are exploratory and two repeats do not quantify seed or workload-population uncertainty.

<a id="study-24"></a>


## 24. 2026-09-11_dflash_auf_target_router

Source: [2026-09-11_dflash_auf_target_router.md](paper/data/history/2026-09-11_dflash_auf_target_router.md).

### Learned target-LoRA router with matched sampled-AUF fusion mappers

One BF16 Qwen3-4B target and one frozen DFlash drafter; active GSM8K,KiCad,NanoCoder LoRAs. All mapper checkpoints are reused: fusion-r56,alpha56,4096 unique examples/domain,4096 response-token cap,8 sampled clean-anchor+15-mask blocks,2000 optimizer updates/16000 presentations. AUF loss L=sum(m*w*CE)/sum(m*w),detached strict-prefix correctness weights,first failure included. Native random-A/zero-B initialization,AdamW1e-4,betas(.9,.999),eps1e-8,wd0,clip1,100-step warmup/cosine to2000,seed42,globalbatch8 across2GPUs. Frozen draft body/norms/embed/head. Target feature layers[1,9,17,25,33],dense cache; sampled loss. Checkpoint paths/hashes and measured fitting costs in setup/selection.json. No mapper retraining in this deployment study.

Canonical vLLM AsyncLLM and LoRARequest/Punica kernels with the existing audited request-indexed fusion hook; rank padded to64,alpha/r=1,no bias. Target-ID selects the matching mapper for the entire request. Request batches preserve target/drafter token ordering and original verification. Native and mapped use identical active-target requests. Adapter-specific chat templates,thinking disabled,temperature0,top_p1,top_k disabled,seed0. Actual installed package/source hashes in setup/runtime_*.json. This is custom fusion integration,not an official upstream experiment.

Fixed128 prompts/domain,384 total; existing held-out evaluation data,math/NanoCoder cap2048,KiCad8192. Twelve closed-loop streaming clients; same order in each native/mapped pair. Two repetitions reverse condition order. L40S GPUs,one engine/GPU,max4 assistant GPUs. Exclude model loading/short warmup,retain bounded full-workload JIT warmup separately. Report throughput including workload drain,latency excluding client semaphore wait; include route/render overhead.

Prompt-only hard top1 router:4096 user-only examples/domain,word1/2gram TF-IDF(max32768,min_df2,sublinearTF,L2),multinomial logistic regression C1,lbfgs,max300,tol1e-6,seed42,float64. Fit locally,freeze before timing,verify canonical/export parity and malformed inputs. Router labels are source domain,not quality-optimal experts. Misroutes use AR references for the actually selected target. Router is fixed for each response; no token-level soft target mixing.

| Domain | Mapper training wall min (2 GPUs) |
| --- | ---: |
| math | 3.83 |
| kicad | 10.51 |
| nanocoder | 4.54 |

Router CPU fitting: 0.944s; held-out source-domain accuracy: 99.740%.

| Repeat | Speedup vs native | Native τ | Mapped τ |
| --- | ---: | ---: | ---: |
| 0 | 1.3212x | 5.856 | 7.680 |
| 1 | 1.3243x | 5.856 | 7.680 |

τ=1+accepted draft tokens/verification steps. Full throughput,latency,TTFT,TPOT,memory,acceptance fraction/survival,router probabilities/confusion and per-source latency ratios in results/learned.json. Per-domain ratios are latency under contention; aggregate counters do not establish per-domain acceptance. Existing test sets have been used in prior model selection; these results are exploratory and two repeats do not quantify seed or workload-population uncertainty.

<a id="study-25"></a>


## 25. 2026-09-11_dflash_body_decay

Source: [2026-09-11_dflash_body_decay.md](paper/data/history/2026-09-11_dflash_body_decay.md).

### Body LoRA: DFlash decay CE versus AUF

### Body-LoRA DFlash decay CE versus AUF

Frozen fusion/norms/embedding/head. Same4096 cached target-LoRA trajectories/domain,4096 response cap,8 sampled anchors,2000 optimizer steps/16000 presentations,global batch8 on2L40S,AdamW1e-4,weight decay0,clip1,100-step warmup/cosine. Only replace detached first-failure AUF with pinned upstream SpecForge loss_decay_gamma=7. No AUF truncation; all eligible masked positions receive exponentially decaying CE weights. Upstream normalization is preserved and audited. Fixed final checkpoint, no dev selection. Same128heldout prompts/domain,greedy/nonthinking,target LoRA active,vLLM,2048/8192/2048 caps,15 proposals. Reuse completed AUF/native timings and label inherited comparisons. No new training rollouts or feature extraction. Validation gates block scale. Eight simultaneous assistant GPUs authorized.

For eligible label mask m and block position k (first predicted position k=1), w_k=exp(-(k-1)/7). Each microbatch uses sum(m*w*CE)/sum(m*w); four microbatch/rank means are averaged per optimizer step. No first-failure masking. This is the unchanged upstream DFlash implementation, with normalization preserved. AUF uses detached strict-prefix correctness support in place of w. Cache byte transfer is lossless; pinned upstream Python source hashes match across nodes. Final snapshots every500updates are retained, with only the final2000update checkpoint evaluated. Startup/warmup and feature-cache transfer are excluded from fit time, separately recorded when available.


| Target | Decay/native | Decay/AUF body | Decay/AUF mapper | Acceptance length | Fit minutes (2 GPUs) |
| --- | ---: | ---: | ---: | ---: | ---: |
| math | 1.2628x | 0.9623x | 1.0103x | 5.7161 | 5.18 |
| kicad | 1.7973x | 1.0341x | 1.3447x | 10.7627 | 13.47 |
| nanocoder | 1.0583x | 0.9759x | 1.0048x | 5.3200 | 5.49 |

Control timings and their fitting costs are inherited, not fresh paired timings; body architecture, seed and initialization match AUF; only loss is changed. No mixed-serving or router inference in this experiment. Full per-request counters, fitted parameters, hashes, exact export audits and retained checkpoints are preserved.

<a id="study-26"></a>


## 26. 2026-09-11_dflash_joint_auf

Source: [2026-09-11_dflash_joint_auf.md](paper/data/history/2026-09-11_dflash_joint_auf.md).

### Joint fusion mapper and drafter-body AUF

Protocol: joint fusion-r56/alpha56 and body-r16/alpha32,5,447,680 parameters; exact existing4096 unique/domain,4096 response cap,8anchors,AUF,2000updates/16000presentations,seed42,AdamW1e-4,global batch8 on2L40S. Fixed final checkpoint. Frozen target LoRA, embedding, output head and norms. Reuse controls from completed body-AUF study.

| Target | Joint/native | Joint/body | Joint/mapper | Acceptance length | Fit minutes (2 GPUs) |
| --- | ---: | ---: | ---: | ---: | ---: |
| math | 1.3081x | 0.9968x | 1.0466x | 5.9439 | 5.37 |
| kicad | 1.7661x | 1.0162x | 1.3213x | 10.5273 | 16.67 |
| nanocoder | 1.0919x | 1.0069x | 1.0368x | 5.5085 | 5.53 |

Control timings and their fitting costs are inherited, not fresh paired timings; architecture/seed-specific random initialization differs because the joint model has an additional trainable fusion adapter. No mixed-serving or router inference in this experiment. Full per-request counters, fitted parameters, hashes, exact export audits and retained checkpoints are preserved.

<a id="study-27"></a>


## 27. 2026-09-11_dflash_nanocoder_sampled_auf

Source: [2026-09-11_dflash_nanocoder_sampled_auf.md](paper/data/history/2026-09-11_dflash_nanocoder_sampled_auf.md).

### NanoCoder sampled-anchor AUF matched to GSM8K/KiCad

Fusion mapper only: r56,alpha56,860160 trainable parameters. Native DFlash body,norms,embedding/head frozen. Reuse4096 unique CodeAlpaca trajectories generated by Qwen3-4B with NanoCoder adapter active,greedy nonthinking,4096 response cap,EOS enabled. Exact prompts,IDs,token references and source locations/hashes are retained in setup and validation/cache.json. Target adapter revision f078bfc2722bdb7bf644906cc7d81662ab3de9d2; source contract inherits dflash_nanocoder_20260909. No new generation or feature extraction.

Training exactly matches the GSM8K/KiCad duration study: randomA/zeroB from native draft,seed42;8 sampled clean-anchor+15-mask blocks/example; no dense anchor tiling. AUF loss L=sum(m_j*w_j*CE_j)/sum(m_j*w_j),with detached w_j equal to the product of correctness at earlier eligible positions of the same block. First failure included,later labels masked. Prompt/padding/anchor labels excluded. Mean within each microbatch,then equal microbatch/rank averaging. AdamW1e-4,betas(.9,.999),eps1e-8,wd0,clip1;100-step warmup/cosine to2000 updates. Two GPUs,microbatch2,accumulation2,globalbatch8. Cycle4096 unique examples to16000 presentations. Save4000/8000/12000/16000 snapshots,evaluate final only. SpecForge953d43a0c1c0f5e32989dc43f91ce5fc2d9ddfef plus inherited custom AUF formula; no claim of an official full-paper reproduction. Five target feature layers[1,9,17,25,33],dense BF16 storage. Export folds BA into native BF16 fusion.

Fresh native and mapped vLLM/L40S timings on the same128 prompts,2048 output cap,greedy/nonthinking,15 proposals,batch invariance and safe frozen-drafter head sharing. Four disjoint32-prompt workers alternate condition order. Sum request wall times excluding setup/warmup. Prompt bootstrap interval is not repeat/seed uncertainty.

| Target / checkpoint | Speedup vs native | Acceptance length | Training wall min (2 GPUs) |
| --- | ---: | ---: | ---: |
| GSM8K sampled AUF seen16000 (existing) | 1.2593x | 5.691 | 3.83 |
| KiCad sampled AUF seen16000 (existing) | 1.3389x | 7.912 | 10.51 |
| NanoCoder sampled AUF seen16000 | 1.0590x | 5.356 | 4.54 |

NanoCoder native acceptance length 5.042; mapped throughput 147.55 tokens/s versus native 139.34. Acceptance length=1+accepted draft tokens/verification steps. Accepted/proposed=0.2904. Training GPU-hours=0.1512; excludes historical collection and current evaluation. Paired-prompt95% speedup interval [1.0476821796416116, 1.071981646337932]. Full counters,position survival,memory and training history in results/summary.json. GSM8K/KiCad timings are inherited standalone results from dflash_auf_scale_20260910; this is not a serving-mixture benchmark.

<a id="study-28"></a>


## 28. 2026-09-11_dflash_pooled_mixture

Source: [2026-09-11_dflash_pooled_mixture.md](paper/data/history/2026-09-11_dflash_pooled_mixture.md).

### Pooled AUF mapper: Explicit-ID mixture

One fusion-r56 alpha56 shared across GSM8K,KiCad,NanoCoder; target LoRAs active. Native and three-separate controls inherited from dflash_auf_serving_20260911, unchanged exact request manifests. New pooled and AR measurements:12clients,128requests/domain,2repeats,greedy/nonthinking,caps2048/8192/2048. Router reused without training.

Train on the exact union:4096 unique examples/domain,4096 response cap,dense feature cache with8 sampled anchors and15 mask labels. AUF L=sum(m*w*CE)/sum(m*w), detached strict-prefix correctness including first failure. Same native random-A/zero-B initialization,860160 parameters; target/body/norm/embed/head frozen. Round-robin domain-homogeneous global batches8 on2GPUs,microbatch2,accum2. Preserve per-domain shuffle/anchor seeds. Each domain gets2000 updates/16000presentations; pooled optimizer takes6000steps/48000presentations. Each triplet shares the original domain learning rate:100 warmup then cosine over2000 domain steps,peak1e-4;AdamW(.9,.999),eps1e-8,wd0,clip1. Shared optimizer history differs intentionally. Final checkpoint folded into BF16 fusion, no routing overhead for the pooled draft correction.

Pooled training: 27.12 wall minutes on2GPUs, 0.9041GPU-hours. Separate mapper fitting inherited: math3.83,KiCad10.51,NanoCoder4.54 minutes on2GPUs. Router fitting inherited0.944CPU seconds; no refit. Data generation/capture excluded from fit times.

| Repeat | Mode | Tokens/s | Acceptance length | Speed vs fresh AR | Fit wall min (2 GPUs) |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | native (inherited) | 1273.98 | 5.8553 | 6.5200 | — |
| 0 | three separate (inherited) | 1678.61 | 7.6782 | 8.5909 | 3.83 /10.51 /4.54 |
| 0 | one pooled | 1671.89 | 7.6170 | 8.5565 | 27.12 |
| 0 | AR | 195.39 | — | 1.0000 | — |
| 1 | native (inherited) | 1273.32 | 5.8553 | 6.5500 | — |
| 1 | three separate (inherited) | 1691.96 | 7.6782 | 8.7036 | 3.83 /10.51 /4.54 |
| 1 | one pooled | 1673.98 | 7.6170 | 8.6111 | 27.12 |
| 1 | AR | 194.40 | — | 1.0000 | — |

Throughput includes workload drain; load/warmup/JIT excluded and retained separately. Counters,full15-position survival,TTFT/TPOT/latency/memory,source-domain router accuracy and per-domain latency are in results/oracle.json. Cross-trial ratios using inherited controls are not paired hardware-repeat estimates. Equal requests are not equal tokens/compute.

[
{
"repeat": 0,
"pooled_vs_fresh_AR": 8.556494192997638,
"pooled_vs_inherited_native": 1.312339422880064,
"pooled_vs_inherited_separate": 0.9959970830501954,
"inherited_separate_vs_fresh_AR": 8.590882783304712,
"inherited_native_vs_fresh_AR": 6.5200313606517515
},
{
"repeat": 1,
"pooled_vs_fresh_AR": 8.611064751732115,
"pooled_vs_inherited_native": 1.3146584983417102,
"pooled_vs_inherited_separate": 0.9893727618051832,
"inherited_separate_vs_fresh_AR": 8.703559552236507,
"inherited_native_vs_fresh_AR": 6.550039240300031
}
]

<a id="study-29"></a>


## 29. 2026-09-11_dflash_pooled_target_router

Source: [2026-09-11_dflash_pooled_target_router.md](paper/data/history/2026-09-11_dflash_pooled_target_router.md).

### Pooled AUF mapper: Learned target router

One fusion-r56 alpha56 shared across GSM8K,KiCad,NanoCoder; target LoRAs active. Native and three-separate controls inherited from dflash_auf_serving_20260911, unchanged exact request manifests. New pooled and AR measurements:12clients,128requests/domain,2repeats,greedy/nonthinking,caps2048/8192/2048. Router reused without training.

Train on the exact union:4096 unique examples/domain,4096 response cap,dense feature cache with8 sampled anchors and15 mask labels. AUF L=sum(m*w*CE)/sum(m*w), detached strict-prefix correctness including first failure. Same native random-A/zero-B initialization,860160 parameters; target/body/norm/embed/head frozen. Round-robin domain-homogeneous global batches8 on2GPUs,microbatch2,accum2. Preserve per-domain shuffle/anchor seeds. Each domain gets2000 updates/16000presentations; pooled optimizer takes6000steps/48000presentations. Each triplet shares the original domain learning rate:100 warmup then cosine over2000 domain steps,peak1e-4;AdamW(.9,.999),eps1e-8,wd0,clip1. Shared optimizer history differs intentionally. Final checkpoint folded into BF16 fusion, no routing overhead for the pooled draft correction.

Pooled training: 27.12 wall minutes on2GPUs, 0.9041GPU-hours. Separate mapper fitting inherited: math3.83,KiCad10.51,NanoCoder4.54 minutes on2GPUs. Router fitting inherited0.944CPU seconds; no refit. Data generation/capture excluded from fit times.

| Repeat | Mode | Tokens/s | Acceptance length | Speed vs fresh AR | Fit wall min (2 GPUs) |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | native (inherited) | 1280.52 | 5.8560 | 6.5518 | — |
| 0 | three separate (inherited) | 1691.87 | 7.6798 | 8.6565 | 3.83 /10.51 /4.54 |
| 0 | one pooled | 1682.03 | 7.6188 | 8.6061 | 27.12 |
| 0 | AR | 195.45 | — | 1.0000 | — |
| 1 | native (inherited) | 1285.61 | 5.8560 | 6.5729 | — |
| 1 | three separate (inherited) | 1702.59 | 7.6798 | 8.7048 | 3.83 /10.51 /4.54 |
| 1 | one pooled | 1695.52 | 7.6188 | 8.6687 | 27.12 |
| 1 | AR | 195.59 | — | 1.0000 | — |

Throughput includes workload drain; load/warmup/JIT excluded and retained separately. Counters,full15-position survival,TTFT/TPOT/latency/memory,source-domain router accuracy and per-domain latency are in results/learned.json. Cross-trial ratios using inherited controls are not paired hardware-repeat estimates. Equal requests are not equal tokens/compute.

[
{
"repeat": 0,
"pooled_vs_fresh_AR": 8.606100936633515,
"pooled_vs_inherited_native": 1.3135511588563147,
"pooled_vs_inherited_separate": 0.9941832060660092,
"inherited_separate_vs_fresh_AR": 8.656453744263017,
"inherited_native_vs_fresh_AR": 6.551782074576138
},
{
"repeat": 1,
"pooled_vs_fresh_AR": 8.668657536086116,
"pooled_vs_inherited_native": 1.3188400453437916,
"pooled_vs_inherited_separate": 0.9958465788296484,
"inherited_separate_vs_fresh_AR": 8.704812287725893,
"inherited_native_vs_fresh_AR": 6.572940794974415
}
]

<a id="study-30"></a>


## 30. 2026-09-12_decay32

Source: [2026-09-12_decay32.md](experiment_logs/2026-09-12_decay32.md).

### Matched decaying-CE architecture comparison

18 additional configurations: GSM8K, KiCad and NanoCoder × five full W matrices, full fusion residual, body LoRA rank128, and five BA matrices at ranks28/56/128. These are fresh training runs; the ongoing AUF Transformers checkpoint evaluations continue separately.

Use precisely the original AUF cohort and cached target-LoRA features: 4,096 examples per domain, 4,096 response-token cap, 32 sampled anchors per example (bounded by valid positions), one epoch, global batch8, two GPUs, per-GPU batch2, gradient accumulation2, 512 optimizer updates, seed42. AdamW, learning rate1e-4, zero weight decay, gradient clipping1, 5% linear warmup followed by cosine decay. Frozen embeddings/head and the same native draft initialization. Full-W matrices start at identity; full fusion residual starts at zero; five-BA uses the original cached PCA bases; body LoRA uses rank128, alpha256, zero dropout, all attention/MLP projections and frozen fusion. No parameter matching changes.

Only the loss changes. Use the **unchanged positional CE implementation** in SpecForge revision `953d43a0c1c0f5e32989dc43f91ce5fc2d9ddfef`, `specforge/algorithms/common/dflash_family_model.py`, with `loss_type=dflash`, `loss_decay_gamma=7`. This implements the DFlash paper's decaying objective; this experiment is not a full reproduction of its training recipe (anchors/data/epochs/LR remain matched to our AUF experiment).

For block position j (zero-based, including the known anchor slot), mask m and token CE c:

$$
w_{bj}=m_{bj}\exp[-\max(j-1,0)/7],\qquad
L=\frac{\sum_{b,j}w_{bj}[-\log q_\theta(y_{bj})]}{\sum_{b,j}w_{bj}}.
$$

The known anchor slot is masked. Labels after a greedy prediction failure still receive decaying CE; there is no AUF prefix-support mask. The upstream wrapper performs chunk aggregation and normalization. `check_loss.py` checks its additive terms against the formula on full, padded and fully masked blocks, including masked gradients. Separate two-update execution checks cover all six architectures on each domain before full fits are released.

Nodes08/09 have eight L40S GPUs total. Cached features are copied once: GSM8K/NanoCoder to node08, KiCad to node09. `stage_assets.py` releases only readiness-held preflight jobs; full fits remain held for audit. Evaluation shards depend only on their own fit. The earlier AUF jobs are neither cancelled nor retrained.

`train_job.sh` trains and exports; `pack_checkpoint.py` preserves exported BF16 bytes for changed tensors. Each checkpoint receives four one-GPU Transformers evaluation shards, covering the same128 held-out prompts. Pinned DFlash0.1.0, Transformers5.16.1, Torch2.13.0+cu129, PEFT0.20.0, BF16/defaultSDPA, active target LoRA merged, greedy/nonthinking, caps2048/8192/2048. Existing fresh Transformers AR/native baselines are reused. No additional AR/native evaluations.

Primary metric: arithmetic mean of generated tokens/s per rollout; speedup percentage is100×(meanTPS_method/meanTPS_baseline−1). Record mean acceptance length, parameters, original/new fit seconds, GPU-hours and raw per-prompt timing.

Entry points: `sync.py`, `launch_jobs.py`, `stage_assets.py`, `preflight_job.sh`, `train_job.sh`. Job IDs and dependencies: `setup/jobs.json`. No new rollout generation or feature extraction.

#### Recorded results

| Domain | Architecture | Prompts | Mean TPS | vs native (%) | vs AR (%) | Acceptance | Params | Fit seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| kicad | body_r128 | 128 | 302.10 | +84.9% | +686.3% | 10.875 | 36,700,160 | 337.7 |
| kicad | fusion_delta | 128 | 244.52 | +49.7% | +536.4% | 8.901 | 32,768,000 | 359.8 |
| kicad | maps_full | 128 | 316.11 | +93.5% | +722.7% | 11.375 | 32,768,000 | 386.3 |
| kicad | maps_r128 | 128 | 266.09 | +62.9% | +592.6% | 9.354 | 3,276,800 | 278.6 |
| kicad | maps_r28 | 128 | 172.99 | +5.9% | +350.3% | 6.412 | 716,800 | 339.1 |
| kicad | maps_r56 | 128 | 227.60 | +39.3% | +492.4% | 8.069 | 1,433,600 | 328.0 |
| math | body_r128 | 128 | 171.40 | +34.8% | +341.6% | 5.754 | 36,700,160 | 162.4 |
| math | fusion_delta | 128 | 168.44 | +32.5% | +334.0% | 5.621 | 32,768,000 | 160.4 |
| math | maps_full | 128 | 173.97 | +36.8% | +348.2% | 5.780 | 32,768,000 | 160.2 |
| math | maps_r128 | 128 | 127.70 | +0.4% | +229.0% | 4.223 | 3,276,800 | 147.3 |
| math | maps_r28 | 128 | 70.44 | -44.6% | +81.5% | 2.278 | 716,800 | 146.4 |
| math | maps_r56 | 128 | 97.19 | -23.6% | +150.4% | 3.104 | 1,433,600 | 146.8 |
| nanocoder | body_r128 | 128 | 163.20 | +8.7% | +318.4% | 5.457 | 36,700,160 | 167.4 |
| nanocoder | fusion_delta | 128 | 159.43 | +6.1% | +308.7% | 5.384 | 32,768,000 | 162.1 |
| nanocoder | maps_full | 128 | 154.88 | +3.1% | +297.1% | 5.362 | 32,768,000 | 167.2 |
| nanocoder | maps_r128 | 128 | 108.00 | -28.1% | +176.9% | 3.302 | 3,276,800 | 151.6 |
| nanocoder | maps_r28 | 128 | 56.77 | -62.2% | +45.5% | 1.818 | 716,800 | 151.1 |
| nanocoder | maps_r56 | 128 | 73.03 | -51.4% | +87.2% | 2.346 | 1,433,600 | 151.2 |

<a id="study-31"></a>


## 31. 2026-09-12_dflash_auf_scaling

Source: [2026-09-12_dflash_auf_scaling.md](experiment_logs/2026-09-12_dflash_auf_scaling.md).

### Five-W AUF data-size and anchor study

Completed: 24 configurations, 128 exact-reference evaluation prompts each (3,072/3,072 agreement checks).

Five trainable identity-initialized W matrices precede the frozen fusion blocks: z = sum_i F_i W_i h_i, followed by the original RMSNorm. All draft body/head/embedding weights remain frozen; F_i W_i folds into one native-shaped fusion at inference. 32,768,000 trainable parameters.

AUF: at each sampled block, let j* be the first greedy draft mismatch (or last eligible position if none). With eligibility mask m_j, support s_j = 1[j <= j*] is detached. L = sum_j m_j s_j CE(logits_j, target_j) / sum_j m_j s_j. Each microbatch is normalized separately, then averaged across accumulation steps and DDP ranks. First failure is included.

One epoch per manifest; global batch8, 512/1024/1536/2048 updates for4096/8192/12288/16384 presentations. AdamW lr1e-4, wd0, clipping1, seed42,5%warmup/cosine. Cached target-LoRA rollouts capped at4096 generated tokens; no new generation/extraction in this sweep. Requested anchors8/24/32/64/128 at4096 examples, and32anchors across all four sizes. Larger cells initialized afresh. Three existing4096×32cells reused. GSM8K uses4096cached unique prompts then repeats; KiCad uses up to15366unique then repeats to16384; NanoCoder supports16384unique. Exact manifest unique counts are in table.csv. This is a reduced two-axis study, not a full interaction grid.

vLLM, L40S, active domain target LoRA, greedy/nonthinking,128fixed prompts/domain; generation caps2048/8192/2048 for GSM8K/KiCad/NanoCoder. Two workers process disjoint64prompt shards with serial requests per worker. Speedups use inherited same-domain native summed latencies. Acceptance is1+accepted draft tokens/verification iterations. Reported frontier is descriptive; no repeated-timing confidence claim.

| Domain | Presentations | Anchors | Speedup/native | Acceptance incl. bonus | Fit seconds | Fit GPU-hours |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| math | 4096 | 8 | 1.2637× | 5.7336 | 70.20 | 0.0390 |
| math | 4096 | 24 | 1.2977× | 5.8984 | 121.09 | 0.0673 |
| math | 4096 | 32 | 1.3172× | 5.9840 | 159.76 | 0.0888 |
| math | 4096 | 64 | 1.3208× | 6.0202 | 293.66 | 0.1631 |
| math | 4096 | 128 | 1.3282× | 6.0526 | 548.78 | 0.3049 |
| math | 8192 | 32 | 1.3368× | 6.0844 | 309.52 | 0.1720 |
| math | 12288 | 32 | 1.3411× | 6.1023 | 466.97 | 0.2594 |
| math | 16384 | 32 | 1.3505× | 6.1997 | 618.88 | 0.3438 |
| kicad | 4096 | 8 | 1.6671× | 9.9357 | 178.18 | 0.0990 |
| kicad | 4096 | 24 | 1.8505× | 11.0734 | 260.22 | 0.1446 |
| kicad | 4096 | 32 | 1.8815× | 11.2774 | 319.25 | 0.1774 |
| kicad | 4096 | 64 | 1.9501× | 11.6779 | 448.98 | 0.2494 |
| kicad | 4096 | 128 | 1.9850× | 11.9066 | 788.16 | 0.4379 |
| kicad | 8192 | 32 | 2.0121× | 12.0614 | 662.57 | 0.3681 |
| kicad | 12288 | 32 | 2.0705× | 12.4166 | 1034.44 | 0.5747 |
| kicad | 16384 | 32 | 2.0993× | 12.6120 | 1426.56 | 0.7925 |
| nanocoder | 4096 | 8 | 1.0407× | 5.2446 | 69.94 | 0.0389 |
| nanocoder | 4096 | 24 | 1.0783× | 5.4408 | 125.28 | 0.0696 |
| nanocoder | 4096 | 32 | 1.0872× | 5.4929 | 165.37 | 0.0919 |
| nanocoder | 4096 | 64 | 1.0894× | 5.4964 | 298.66 | 0.1659 |
| nanocoder | 4096 | 128 | 1.0969× | 5.5348 | 586.43 | 0.3258 |
| nanocoder | 8192 | 32 | 1.1013× | 5.5495 | 327.32 | 0.1818 |
| nanocoder | 12288 | 32 | 1.0974× | 5.5311 | 490.16 | 0.2723 |
| nanocoder | 16384 | 32 | 1.1128× | 5.6300 | 656.52 | 0.3647 |

Raw outputs, checkpoint paths, unique counts, optimizer steps and Pareto flags: `experiments/dflash_auf_scaling_20260912/results/{summary.json,table.csv}`. Plot PDFs in the same directory. Source cohort manifests and jobs recorded under setup. Negative/dominated cells retained. No claim of a validated scaling law or independent generalization from the repeatedly examined evaluation set.

<a id="study-32"></a>


## 32. 2026-09-12_dflash_optimized_baselines

Source: [2026-09-12_dflash_optimized_baselines.md](experiment_logs/2026-09-12_dflash_optimized_baselines.md).

### Optimized standalone AR / native DFlash

4/6 complete rows. Node09 L40S; same128prompts/domain, active target LoRA, one request/GPU, independent shards. No fitting.

Compilation mode3; batch invariance disabled; backend-selected async scheduling and CUDA graphs. Pinned environment/configuration and custom frozen draft-weight sharing correction documented in the study README. BF16 greedy non-thinking; response caps2048/8192/2048. Startup/compilation/warmup excluded; detected JIT replays logged. Throughput = total generated tokens / sum(request latency).

IMPORTANT: production settings are not bitwise invariant. Throughput ratios are not fixed-output latency-reduction estimates. No proof of global peak performance or statistical uncertainty from one timing trial.

| Domain | Method | Old tok/s | New tok/s | Throughput ratio | Acceptance | Training s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | ar | 31.17 | 66.80 | 2.143× | None | 0 |
| math | native | 122.23 | 249.98 | 2.045× | 4.496919431279621 | 0 |
| nanocoder | ar | 31.15 | 67.32 | 2.161× | None | 0 |
| nanocoder | native | 139.21 | 286.69 | 2.059× | 5.1037438768369485 | 0 |

Old AR source paths:
```json
{
  "math": "/home/yashas/spec_decode/experiments/dflash_workload_transfer_20260908/measurements/full_math_math_ar/ar.json",
  "kicad": "/home/yashas/spec_decode/experiments/dflash_kicad_auf_data_scale_20260910/setup/ar_kicad.json",
  "nanocoder": "/home/yashas/spec_decode/experiments/dflash_nanocoder_20260909/measurements/full_ar/ar.json"
}
```
Native denominators are the inherited controls recorded by the completed scaling study. Raw measurements and new resolved engine configuration remain in measurements/.

<a id="study-33"></a>


## 33. 2026-09-12_dflash_transformers_ablation

Source: [2026-09-12_dflash_transformers_ablation.md](experiment_logs/2026-09-12_dflash_transformers_ablation.md).

### Transformers standalone architecture ablation

All24 evaluations are complete: AR/native controls and six existing AUF checkpoints for each of GSM8K,KiCad,NanoCoder. Each configuration covers the same128 held-out domain prompts. No new training was performed for this table.

#### Models and execution

The target is always Qwen3-4B with the corresponding task LoRA applied and merged using PEFT `merge_and_unload()`. The native DFlash draft is the common initialization for all variants. Adapter locations and exact checkpoint provenance/hashes are recorded in `experiments/dflash_transformers_20260912/assets/manifest.json`; per-shard records retain the active target adapter path and DFlash package provenance.

Use pinned DFlash0.1.0 upstream `dflash_generate`, Torch2.13.0+cu129,Transformers5.16.1,PEFT0.20.0,BF16 and default SDPA on eight L40S GPUs across nodes08/09. Each GPU executes one request at a time; there is no tensor parallel inference. AR uses the same upstream generator's block-size1 path; native/adapted DFlash uses block size16. Temperature0,top-p1,top-k0,thinking disabled. Response caps are2048 for GSM8K/NanoCoder and8192 for KiCad. Preserve the stored chat messages; apply the target tokenizer's nonthinking chat template. NanoCoder uses its adapter tokenizer, the other domains use the base tokenizer. Stops come from upstream `stop_token_ids`.

Eight prompt shards/domain,16 requests/shard. Load target and draft once per worker; reset native draft tensors before overlaying each architecture's exported BF16 weights. Each method gets two64-token warmup requests from training prompts. CUDA synchronization brackets each timed `dflash_generate` call. Tokenization,model loading,warmup and JSON writes are outside request timing; per-request target prefill and generation are inside it. Raw records include generated IDs,text,request seconds,TTFT,upstream TPOT and acceptance-step lengths.

Generated token counts are used only to compute rates here.

#### Existing training recipe and architectures

All18 checkpoints use4096 cached training examples/domain,4096 maximum response tokens,32 sampled anchors/example (bounded by valid positions),one epoch,512 optimizer updates,global batch8,two GPUs,per-GPU batch2 and gradient accumulation2. Seed42,AdamW1e-4,zero weight decay,gradient clip1,5% warmup then cosine schedule. Exact original per-checkpoint fit times and source hashes are preserved in the asset manifest. No target decoding or feature extraction occurs in this reevaluation.

Let h concatenate five2560-dimensional context vectors and partition frozen fusion F=[F1,…,F5]. Five-W learns W_i∈R^(2560×2560), initialized to identity, giving sum_i F_i W_i h_i. Five-BA replaces each W_i with B_i A_i; the original cached PCA initialization and ranks28/56/128 are retained. These are pure low-rank replacements, not identity-plus-low-rank residuals. Full fusion residual learns a zero-initialized ΔF, giving(F+ΔF)h. Body LoRA trains rank128,alpha256 adapters on all attention and MLP projections in the five draft blocks; fusion stays frozen. All updates are folded into exported weights before evaluation, so five-W/BA/residual variants use the normal fusion layer at inference.

The original AUF objective is token CE through the first greedy failure inclusive. For supervised position j, define c_j=1[argmax q_j=y_j], validity m_j, and detached strict-prefix support s_j=prod_(k<j)(c_k OR NOT m_k). The loss is sum_j m_j s_j[-log q_j(y_j)] / sum_j m_j s_j, aggregated through the existing training wrapper. Slots after the first failure have zero support. This is our modified objective, not the upstream decaying CE objective.

#### Metrics and complete results

Primary rate is the arithmetic mean across prompts: meanTPS=(1/128)sum_i(N_i/t_i). Speedup%=100(meanTPS_method/meanTPS_baseline−1). Each prompt has equal weight. Aggregate N/sum(t) is retained only as a secondary CSV field. Acceptance length is the mean produced-token count per upstream verification step,including bonus tokens,excluding the initial prefill token. Original training times are shown separately from this evaluation-only run.

| Domain | Method | Prompts | Tokens/s | vs AR (%) | vs native (%) | Acceptance length | Trainable params | Original fit (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| math | ar | 128 | 38.81 | +0.0% | -69.5% | — | 0 | 0.0 |
| math | native | 128 | 127.13 | +227.6% | +0.0% | 4.472 | 0 | 0.0 |
| math | maps_full | 128 | 174.90 | +350.6% | +37.6% | 5.992 | 32,768,000 | 159.8 |
| math | fusion_delta | 128 | 168.27 | +333.5% | +32.4% | 5.809 | 32,768,000 | 156.3 |
| math | body_r128 | 128 | 173.42 | +346.8% | +36.4% | 5.937 | 36,700,160 | 163.5 |
| math | maps_r28 | 128 | 72.39 | +86.5% | -43.1% | 2.382 | 716,800 | 151.1 |
| math | maps_r56 | 128 | 99.26 | +155.7% | -21.9% | 3.213 | 1,433,600 | 152.6 |
| math | maps_r128 | 128 | 131.24 | +238.2% | +3.2% | 4.339 | 3,276,800 | 148.0 |
| kicad | ar | 128 | 38.42 | +0.0% | -76.5% | — | 0 | 0.0 |
| kicad | native | 128 | 163.34 | +325.1% | +0.0% | 5.991 | 0 | 0.0 |
| kicad | maps_full | 128 | 309.56 | +705.7% | +89.5% | 11.259 | 32,768,000 | 319.3 |
| kicad | fusion_delta | 128 | 235.96 | +514.2% | +44.5% | 8.777 | 32,768,000 | 304.7 |
| kicad | body_r128 | 128 | 286.59 | +645.9% | +75.5% | 10.495 | 36,700,160 | 310.0 |
| kicad | maps_r28 | 128 | 184.28 | +379.6% | +12.8% | 6.618 | 716,800 | 310.6 |
| kicad | maps_r56 | 128 | 220.72 | +474.5% | +35.1% | 8.119 | 1,433,600 | 299.2 |
| kicad | maps_r128 | 128 | 258.69 | +573.3% | +58.4% | 9.456 | 3,276,800 | 289.2 |
| nanocoder | ar | 128 | 39.01 | +0.0% | -74.0% | — | 0 | 0.0 |
| nanocoder | native | 128 | 150.21 | +285.1% | +0.0% | 5.111 | 0 | 0.0 |
| nanocoder | maps_full | 128 | 162.32 | +316.1% | +8.1% | 5.549 | 32,768,000 | 165.4 |
| nanocoder | fusion_delta | 128 | 162.45 | +316.5% | +8.2% | 5.542 | 32,768,000 | 165.3 |
| nanocoder | body_r128 | 128 | 164.96 | +322.9% | +9.8% | 5.593 | 36,700,160 | 168.8 |
| nanocoder | maps_r28 | 128 | 56.40 | +44.6% | -62.5% | 1.833 | 716,800 | 156.9 |
| nanocoder | maps_r56 | 128 | 76.12 | +95.1% | -49.3% | 2.400 | 1,433,600 | 152.6 |
| nanocoder | maps_r128 | 128 | 106.96 | +174.2% | -28.8% | 3.387 | 3,276,800 | 155.1 |

Acceptance length is the upstream mean number of produced tokens per verification step, including the bonus token. The first prefill token is excluded. Fit times belong to the original checkpoint training and are not costs of this evaluation rerun.

Reproduce evaluation with `launch_rerun.py`, collect with `collect_rerun.py`, aggregate with `report_rerun.py`. All48 shard jobs completed successfully; job IDs and statuses are in `setup/rerun_jobs.json` and `setup/rerun_status.json`.

<a id="study-34"></a>


## 34. 2026-09-12_dflash_transformers_gsm8k

Source: [2026-09-12_dflash_transformers_gsm8k.md](experiment_logs/2026-09-12_dflash_transformers_gsm8k.md).

### GSM8K Transformers BF16 execution comparison

All three profiles completed128prompts with AR and native DFlash. None achieved128/128exact AR/native token agreement. The small forced-SDPA-math successes did not generalize. No architecture sweep is released.

Identical GSM8K held-out cohort and active merged math target LoRA; native frozen drafter; pinned DFlash0.1.0 generation. BF16, greedy temperature0/top-p1/top-k0, thinking disabled,2048output-token cap, block16(native)/1(AR). L40S onnode08/node09; matching Torch2.13.0+cu129, Transformers5.16.1, PEFT0.20.0. Each AR/native shard shares its GPU and model object. Two training prompts warm each method for64tokens. No fitting.

Tokens/sec is total generated tokens divided by synchronized summed request wall time; startup/warmup excluded. Outputs differ, so throughput ratios are not fixed-output speedups or validated lossless comparisons. Eight one-GPU jobs32668–32675; SDPA/math split43/43/42, eager64/64. All128requests remain sequential within each worker.

| Profile | AR tok/s | Native tok/s | 128/128? | Mean acceptance incl. bonus |
| --- | ---: | ---: | --- | ---: |
| sdpa | 39.12 | 127.86 | No | 4.4717 |
| sdpa_math | 31.24 | 104.54 | No | 4.4590 |
| eager | 31.96 | 114.06 | No | 4.4815 |

Raw token IDs, finish reasons, timings and acceptance lists are preserved in `experiments/dflash_transformers_20260912/measurements/*comparison*.json`; aggregate results in `results/gsm8k_comparison.json` and CSV. Unforced SDPA is the fastest tested Transformers profile; no global-optimum claim. No precision or tie-breaking modification was applied.

<a id="study-35"></a>


## 35. 2026-09-12_jointw32

Source: [2026-09-12_jointw32.md](experiment_logs/2026-09-12_jointw32.md).

### One joint full-W mapper: AUF and decaying CE

User-confirmed architecture: concatenate five2560-dimensional target context vectors into h∈R^12800. Learn one full W∈R^(12800×12800), initialized to identity, before the frozen native fusion F∈R^(2560×12800). Output FWh. All off-diagonal blocks may mix the five context sources. 163,840,000 trainable parameters. This is neither a shared2560×2560 map nor a residual added directly to fusion.

For efficient training, compute the effective matrix FW and apply it to h, retaining autograd through W. This avoids an additional12800-dimensional per-token activation. During export fold FW into the normal BF16 fc.weight; no additional inference layer. W itself is stored in FP32 trainable checkpoints. This reparameterizes the fusion map but has a different optimizer geometry and parameter count from direct fusion training.

Six fits: three active target LoRA domains × AUF / upstream DFlash positional decaying CE gamma7. Same existing4096 examples,4096 response-token cap,32 sampled anchors,one epoch,512 updates,two GPUs,batch2/GPU,accumulation2,AdamW1e-4,warmup5%/cosine,seed42 as the original18. Reuse staged features via symlinks; no duplication/generation/extraction. Loss implementations and upstream pin follow ../dflash_decay32_20260912/README.md. AUF uses the unchanged existing detached strict-prefix CE implementation from ../dflash_auf32_20260912/objectives.py. Known anchor/padding are masked; first greedy failure remains supervised.

Two-update checks are separate from full fits. Reuse native/AR Transformers baselines. Same128 held-out prompts,greedy/nonthinking,BF16/defaultSDPA,domain caps2048/8192/2048. Each fit gets four single-GPU evaluation shards. Mean per-rollout TPS is the primary metric; report speedup percentages,acceptance,parameter count and training time.

#### Recorded results

| Domain | Architecture | Prompts | Mean TPS | vs native (%) | vs AR (%) | Acceptance | Params | Fit seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| kicad | joint_auf | 128 | 307.78 | +88.4% | +701.1% | 10.869 | 163,840,000 | 340.5 |
| kicad | joint_decay7 | 128 | 308.23 | +88.7% | +702.2% | 10.933 | 163,840,000 | 343.1 |
| math | joint_auf | 128 | 158.06 | +24.3% | +307.2% | 5.064 | 163,840,000 | 204.8 |
| math | joint_decay7 | 128 | 152.08 | +19.6% | +291.9% | 4.987 | 163,840,000 | 203.1 |
| nanocoder | joint_auf | 128 | 139.62 | -7.0% | +257.9% | 4.486 | 163,840,000 | 207.6 |
| nanocoder | joint_decay7 | 128 | 137.17 | -8.7% | +251.6% | 4.465 | 163,840,000 | 208.0 |

<a id="study-36"></a>


## 36. 2026-09-13_dflash_eagle3

Source: [2026-09-13_dflash_eagle3.md](experiment_logs/2026-09-13_dflash_eagle3.md).

### EAGLE-3 target-LoRA baseline

Published EAGLE-3 checkpoints are evaluated without new fitting; target is always Qwen3 plus its selected target LoRA. Use the pinned upstream EAGLE revision `cb7e084…` under `dflash_ready_20260913/upstream/EAGLE` with the documented Qwen3 head-dimension compatibility patch in `compat/EAGLE`. A separate environment uses Transformers4.53.3,PEFT0.15.2 and the shared Torch build. Patch/config provenance and complete commands are in ready setup/validation records.

Standalone greedy/nonthinking generation:BF16, one L40S,128 prompts/domain,tree60/depth7/top-k10. Persistent KV buffers are warmed before measurement. AR uses the same EAGLE model wrapper/runtime without speculative proposals; this companion avoids crediting runtime differences as speculative acceleration. The DFlash standalone backend has a different upstream Transformers environment and should not be described as an identical-runtime algorithm-only comparison. Caps are2,048 for math/code and8,192 for4B KiCad.

Primary TPS is the mean of generated tokens divided by per-request elapsed seconds; acceptance is total committed new tokens divided by verification steps, as returned by upstream. Model loading/warmup is excluded. Token-agreement/length-validity checks were waived; no bitwise-equivalence claim is inferred. No EAGLE-specific fine-tuning:additional trainable parameters and training time are zero.

| Target | Domain | AR TPS | EAGLE TPS | EAGLE × companion AR | Acceptance | Additional fit min | Additional trained params |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen4b | math | 31.49 | 71.19 | 2.260 | 3.347 | 0 | 0 |
| qwen4b | kicad | 30.37 | 73.82 | 2.431 | 3.794 | 0 | 0 |
| qwen4b | nanocoder | 31.80 | 81.77 | 2.572 | 3.852 | 0 | 0 |
| qwen8b | math | 31.19 | 76.25 | 2.445 | 3.728 | 0 | 0 |
| qwen8b | codealpaca | 30.32 | 76.21 | 2.514 | 4.051 | 0 | 0 |

The live multi-LoRA vLLM baseline uses3 speculative tokens, not the Transformers tree60; DFlash uses15 speculative tokens. It shares one target backbone and one draft backbone and uses the same frozen route assignments and workload as the other serving methods. Client counts1/4/8/16/32 and12-client learned routing are recorded in `mixture.md`, including mean and aggregate TPS,latency,TTFT/TPOT,acceptance and memory. This is a disclosed configuration difference, not a proposal-budget-matched comparison. A100 and Spark evaluations are deferred.

<a id="study-37"></a>


## 37. 2026-09-13_dflash_fivew_mechanism

Source: [2026-09-13_dflash_fivew_mechanism.md](experiment_logs/2026-09-13_dflash_fivew_mechanism.md).

### Five-W AUF: mechanistic interventions

Partial experiment; completed live interventions are shown below. Remaining KiCad probes were cancelled by the user; completed outputs are retained. GSM8K/NanoCoder rank, subspace and timing-repeat results are reported as available. No conclusion is based on an incomplete intervention.

#### Model and training

The active target is Qwen3-4B with the corresponding GSM8K, KiCad or NanoCoder target adapter. The frozen native DFlash drafter receives five target-layer vectors, in the pinned checkpoint's order. Each vector has width2560. For frozen fusion blocks $F_i$ and trained full matrices $W_i$,

$$z_0=\sum_{i=1}^5F_i h_i,\quad z=\sum_{i=1}^5F_iW_i h_i,\quad u=N(z).$$

Only the five identity-initialized $W_i\in\mathbb R^{2560\times2560}$ were fitted:32,768,000 trainable parameters. All other drafter parameters were frozen. Inference folds each $F_iW_i$ into the existing fusion matrix.

The fixed checkpoints use4096 cached training examples/domain, one epoch,32 requested sampled anchors/example, block size16, response cap4096, global batch8 (two examples/GPU, two GPUs, accumulation2),512 optimizer updates, seed42, AdamW learning rate$10^{-4}$, zero weight decay,5% warmup followed by cosine decay, and gradient norm clipping at1. Each domain's target generated its cached training sequences. Requested anchors are blocks, not dense supervision of every token.

For block$a$, valid-position mask$m_{aj}$, target-generated label$y_{aj}$, and draft distribution$q_{aj}$, the implemented accept-until-fail objective is

$$c_{aj}=\mathbf1[\arg\max_vq_{aj}(v)=y_{aj}\ \lor\ m_{aj}=0],\qquad s_{aj}=\operatorname{stopgrad}\!\left(\prod_{k<j}c_{ak}\right),$$
$$L_{\rm AUF}=\frac{\sum_{a,j}m_{aj}s_{aj}[-\log q_{aj}(y_{aj})]}{\sum_{a,j}m_{aj}s_{aj}}.$$

The first wrong token remains supervised; later positions in that block are excluded. The mask is recomputed from the current model and receives no gradient. Reduction is within each microbatch across its objective chunks, followed by the implemented gradient-accumulation/DDP average; it is not a global dataset-level ratio. This is our modification to pinned SpecForge's DFlash block objective, not an upstream-released AUF objective.

#### Interventions and measurement

Write $D_i=W_i-I$ and $\delta=z-z_0$. Whole-map truncation uses $[W_i]_r$; identity-preserving truncation uses $I+[D_i]_r$. Both use FP32 SVD followed by BF16 folded exports. Whole-map truncation removes most of the identity path, so its failure cannot establish that the learned correction requires high rank.

The exact algebraic reparameterization exports $F_i+F_i(W_i-I)$ without fitting new parameters. Algebraic equivalence does not imply bitwise equality after FP32 evaluation/BF16 rounding. Do not attribute observed differences between these exports to learned capacity or optimization.

RMSNorm uses the actual gain vector and epsilon, $N(z)=g\odot z/\sqrt{\|z\|^2/d+\epsilon}$. Radial and perpendicular interventions use $\delta_\parallel=(\delta^Tz_0)/(\|z_0\|^2+\epsilon_p)z_0$ and $\delta_\perp=\delta-\delta_\parallel$. Dynamic hooks also execute native/full controls to expose hook overhead. Static strength, layer-removal and bank-swap interventions remain folded into dense fusion.

Each completed row uses128 held-out prompts, Transformers, one request/GPU, L40S, BF16 SDPA, greedy decoding with thinking disabled, and response caps2048/8192/2048. The active target adapter stays fixed when the mapper bank changes. Timing synchronizes GPU work and excludes model loading/warmup. Primary gain is the ratio of arithmetic mean per-rollout TPS to the corresponding fresh native-control mean, minus one. Acceptance includes the bonus token per verification step.

Retention is $R=(T_0-T_r)/(T_0-T_*)$, where each $T$ sums per-request seconds. Intervals use2000 paired-prompt bootstrap draws with seed42; they do not measure run-to-run GPU noise. Repeated timing controls are included below; a single run crossing98% does not establish stable retention. These dense-folded timing tests do not demonstrate a faster low-rank kernel. Calibration uses training examples only; repeated inspection of the evaluation cohort makes the mechanism findings exploratory.

#### Completed live results

| Domain | Intervention | Mean TPS | Gain vs native | Acceptance | Retained reduction | Original fit min | Original params |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| math | full | 179.83 | +39.51% | 5.988 | 1.000 | 2.66 | 32,768,000 |
| math | leave_out layer=3 | 176.18 | +36.68% | 5.908 | 0.949 | 2.66 | 32,768,000 |
| math | leave_out layer=4 | 171.68 | +33.18% | 5.833 | 0.864 | 2.66 | 32,768,000 |
| math | full hook=native | 131.47 | +1.99% | 4.472 | 0.082 | 2.66 | 32,768,000 |
| math | full hook=full | 180.17 | +39.77% | 5.988 | 1.006 | 2.66 | 32,768,000 |
| math | full hook=radial | 124.61 | -3.33% | 4.466 | -0.136 | 2.66 | 32,768,000 |
| math | full hook=perpendicular | 177.64 | +37.81% | 5.924 | 0.949 | 2.66 | 32,768,000 |
| math | strength alpha=0.5 | 177.02 | +37.33% | 5.757 | 0.989 | 2.66 | 32,768,000 |
| math | strength alpha=1.5 | 175.61 | +36.24% | 5.802 | 0.898 | 2.66 | 32,768,000 |
| math | strength alpha=2.0 | 165.93 | +28.72% | 5.400 | 0.728 | 2.66 | 32,768,000 |
| math | full source_bank=kicad | 127.63 | -0.99% | 4.430 | -0.025 | 5.32 | 32,768,000 |
| math | native | 128.90 | +0.00% | 4.472 | 0.000 | 0.00 | 0 |
| math | full source_bank=nanocoder | 129.72 | +0.63% | 4.565 | 0.042 | 2.76 | 32,768,000 |
| math | reparameterized | 181.94 | +41.14% | 5.995 | 1.032 | 2.66 | 32,768,000 |
| math | whole_rank rank=1 | 34.16 | -73.50% | 1.145 | -11.425 | 2.66 | 32,768,000 |
| math | whole_rank rank=2 | 34.50 | -73.24% | 1.165 | -11.250 | 2.66 | 32,768,000 |
| math | delta_rank rank=1 | 143.26 | +11.14% | 4.913 | 0.369 | 2.66 | 32,768,000 |
| math | delta_rank rank=2 | 148.99 | +15.58% | 5.072 | 0.496 | 2.66 | 32,768,000 |
| math | leave_out layer=0 | 174.69 | +35.52% | 5.982 | 0.886 | 2.66 | 32,768,000 |
| math | leave_out layer=1 | 178.72 | +38.65% | 5.979 | 0.979 | 2.66 | 32,768,000 |
| math | leave_out layer=2 | 184.68 | +43.27% | 6.006 | 1.096 | 2.66 | 32,768,000 |
| math | delta_rank rank=128 | 179.10 | +38.94% | 5.928 | 0.992 | 2.66 | 32,768,000 |
| math | delta_rank rank=256 | 172.59 | +33.89% | 5.964 | 0.870 | 2.66 | 32,768,000 |
| math | delta_rank rank=64 | 172.81 | +34.06% | 5.931 | 0.876 | 2.66 | 32,768,000 |
| math | fusion_complement rank=404 | 148.42 | +15.14% | 5.273 | 0.461 | 2.66 | 32,768,000 |
| math | fusion_top rank=404 | 174.08 | +35.05% | 5.846 | 0.905 | 2.66 | 32,768,000 |
| math | output_scale alpha=2 | 181.07 | +40.47% | 5.988 | 1.021 | 2.66 | 32,768,000 |
| math | shift_complement rank=413 | 163.41 | +26.77% | 5.713 | 0.736 | 2.66 | 32,768,000 |
| math | shift_top rank=413 | 156.94 | +21.75% | 5.417 | 0.622 | 2.66 | 32,768,000 |
| math | full (repeat 0) | 176.03 | +40.92% | 5.988 | 1.000 | 2.66 | 32,768,000 |
| math | native (repeat 0) | 124.91 | +0.00% | 4.472 | 0.000 | 0.00 | 0 |
| math | delta_rank rank=128 (repeat 0) | 172.81 | +38.35% | 5.928 | 0.951 | 2.66 | 32,768,000 |
| math | delta_rank rank=256 (repeat 0) | 180.10 | +44.18% | 5.964 | 1.067 | 2.66 | 32,768,000 |
| math | full (repeat 1) | 184.00 | +44.50% | 5.988 | 1.000 | 2.66 | 32,768,000 |
| math | native (repeat 1) | 127.34 | +0.00% | 4.472 | 0.000 | 0.00 | 0 |
| math | delta_rank rank=128 (repeat 1) | 183.07 | +43.76% | 5.928 | 0.996 | 2.66 | 32,768,000 |
| math | delta_rank rank=256 (repeat 1) | 176.10 | +38.29% | 5.964 | 0.886 | 2.66 | 32,768,000 |
| nanocoder | full | 171.73 | +13.89% | 5.552 | 1.000 | 2.76 | 32,768,000 |
| nanocoder | leave_out layer=3 | 170.51 | +13.08% | 5.539 | 0.928 | 2.76 | 32,768,000 |
| nanocoder | leave_out layer=4 | 168.31 | +11.62% | 5.447 | 0.849 | 2.76 | 32,768,000 |
| nanocoder | full hook=native | 150.58 | -0.13% | 5.111 | 0.006 | 2.76 | 32,768,000 |
| nanocoder | full hook=full | 161.63 | +7.19% | 5.552 | 0.458 | 2.76 | 32,768,000 |
| nanocoder | full hook=radial | 150.78 | -0.01% | 5.114 | 0.013 | 2.76 | 32,768,000 |
| nanocoder | full hook=perpendicular | 170.39 | +13.00% | 5.527 | 0.895 | 2.76 | 32,768,000 |
| nanocoder | strength alpha=0.5 | 169.23 | +12.23% | 5.532 | 0.908 | 2.76 | 32,768,000 |
| nanocoder | strength alpha=1.5 | 163.65 | +8.53% | 5.321 | 0.540 | 2.76 | 32,768,000 |
| nanocoder | strength alpha=2.0 | 152.46 | +1.11% | 5.032 | -0.044 | 2.76 | 32,768,000 |
| nanocoder | full source_bank=math | 148.97 | -1.20% | 5.016 | -0.020 | 2.66 | 32,768,000 |
| nanocoder | native | 150.79 | +0.00% | 5.111 | 0.000 | 0.00 | 0 |
| nanocoder | full source_bank=kicad | 148.12 | -1.77% | 4.990 | -0.111 | 5.32 | 32,768,000 |
| nanocoder | reparameterized | 165.03 | +9.44% | 5.539 | 0.657 | 2.76 | 32,768,000 |
| nanocoder | whole_rank rank=1 | 31.06 | -79.40% | 1.057 | -34.844 | 2.76 | 32,768,000 |
| nanocoder | whole_rank rank=2 | 33.06 | -78.08% | 1.076 | -32.481 | 2.76 | 32,768,000 |
| nanocoder | delta_rank rank=1 | 157.11 | +4.19% | 5.239 | 0.388 | 2.76 | 32,768,000 |
| nanocoder | delta_rank rank=2 | 151.77 | +0.65% | 5.259 | 0.021 | 2.76 | 32,768,000 |
| nanocoder | leave_out layer=0 | 167.18 | +10.87% | 5.577 | 0.749 | 2.76 | 32,768,000 |
| nanocoder | leave_out layer=1 | 170.44 | +13.03% | 5.569 | 0.906 | 2.76 | 32,768,000 |
| nanocoder | leave_out layer=2 | 170.78 | +13.26% | 5.587 | 0.929 | 2.76 | 32,768,000 |
| nanocoder | delta_rank rank=128 | 163.60 | +8.50% | 5.492 | 0.601 | 2.76 | 32,768,000 |
| nanocoder | delta_rank rank=256 | 168.04 | +11.44% | 5.521 | 0.831 | 2.76 | 32,768,000 |
| nanocoder | delta_rank rank=512 | 167.20 | +10.89% | 5.528 | 0.762 | 2.76 | 32,768,000 |
| nanocoder | fusion_complement rank=404 | 159.58 | +5.83% | 5.337 | 0.447 | 2.76 | 32,768,000 |
| nanocoder | fusion_top rank=404 | 157.99 | +4.78% | 5.484 | 0.284 | 2.76 | 32,768,000 |
| nanocoder | output_scale alpha=2 | 166.34 | +10.32% | 5.552 | 0.724 | 2.76 | 32,768,000 |
| nanocoder | shift_complement rank=244 | 162.92 | +8.05% | 5.479 | 0.612 | 2.76 | 32,768,000 |
| nanocoder | shift_top rank=244 | 152.37 | +1.05% | 5.299 | 0.077 | 2.76 | 32,768,000 |
| nanocoder | delta_rank rank=1024 | 168.84 | +11.97% | 5.546 | 0.845 | 2.76 | 32,768,000 |
| nanocoder | delta_rank rank=2048 | 166.49 | +10.41% | 5.553 | 0.723 | 2.76 | 32,768,000 |
| nanocoder | full (repeat 0) | 171.03 | +12.20% | 5.552 | 1.000 | 2.76 | 32,768,000 |
| nanocoder | native (repeat 0) | 152.44 | +0.00% | 5.111 | 0.000 | 0.00 | 0 |
| nanocoder | delta_rank rank=256 (repeat 0) | 163.49 | +7.25% | 5.521 | 0.578 | 2.76 | 32,768,000 |
| nanocoder | delta_rank rank=512 (repeat 0) | 166.52 | +9.24% | 5.528 | 0.724 | 2.76 | 32,768,000 |
| nanocoder | full (repeat 1) | 172.93 | +10.12% | 5.552 | 1.000 | 2.76 | 32,768,000 |
| nanocoder | native (repeat 1) | 157.04 | +0.00% | 5.111 | 0.000 | 0.00 | 0 |
| nanocoder | delta_rank rank=256 (repeat 1) | 171.45 | +9.18% | 5.521 | 0.893 | 2.76 | 32,768,000 |
| nanocoder | delta_rank rank=512 (repeat 1) | 166.80 | +6.22% | 5.528 | 0.521 | 2.76 | 32,768,000 |

Original fit time is inherited checkpoint cost, not probe execution time; native controls require no fit. Raw JSON supplies request times, confidence intervals and preparation provenance.

#### Current evidence

Whole-W rank1/2 truncation sharply reduces acceptance in the completed GSM8K/NanoCoder tests. Preserving the identity path performs better, but rank1/2 do not preserve98% of full-mapper latency reduction. Higher-rank and subspace interventions show substantial timing variability: GSM8K delta-rank128 retained about95.1% and99.6% in the two matched repeat rounds, so a robust98% threshold is not established by its initial point estimate. Reparameterized export differences, especially on NanoCoder, require numerical/timing interpretation; they are not evidence for a new architecture benefit.

#### Reproduction artifacts

Drivers and exports: `experiments/dflash_ready_20260913/` and `experiments/dflash_fivew_probes_20260912/probes.py`. Training: `experiments/dflash_auf32_20260912/{model,train,objectives}.py`. Tasks and audited releases: ready-family `setup/probe_tasks.json`, `setup/launch_plan.json`, and `validation/`. Raw outputs: `measurements/`; merged metrics and intervals: `results/probes.json`. Run `report_probes.py`, then `scripts/report_fivew_mechanism.py` to refresh this report.

#### Paired-feature geometry and optimization

Geometry uses64 distinct cached training examples per domain and16 evenly spaced response positions each (1,024 tokens). The same target-generated token IDs are teacher-forced through base and adapted targets; there is no new decoding. Layer IDs are1/9/17/25/33. These base forwards are offline diagnostics, never a deployed target. With $t=\sum_iF_i(h_i^{LoRA}-h_i^{base})$, compare its tokenwise cosine with $\delta$ against a shuffled-token control. The RMSNorm linearization is

$$J_N(z_0)=\operatorname{diag}(g)\left(I/s-z_0z_0^T/(d s^3)\right),\quad s=\sqrt{\|z_0\|^2/d+\epsilon}.$$

| Domain | Radial energy fraction | Jacobian relative error | cos(correction, shift) | Shuffled cosine | Geometry prep seconds | Original fit min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| math | 0.2522 | 0.1755 | -0.0429 | -0.0413 | 29.19 | 2.66 |
| nanocoder | 0.3482 | 0.1488 | 0.1248 | 0.1052 | 30.11 | 2.76 |

The shuffled cosines are close to the aligned cosines: these measurements do not support a strong token-specific reversal/recovery explanation. Radial energy is nonzero, and the finite-correction Jacobian approximation has noticeable error; neither an energy fraction nor a cosine predicts speedup by itself. Live radial/perpendicular, fixed fusion/shift-subspace and strength interventions in the table supply the causal tests. Exact tokenwise shift projection with a second live base model was not run: the fixed training-derived shift projector tests alignment at ordinary deployment cost, as permitted by the probe plan.

For $G_i=F_iW_i$ and $H_i=\partial L/\partial G_i$, plain SGD induces

$$\Delta G_i=-\eta F_iF_i^T H_i.$$

This algebraic identity is distinct from an explanation for AdamW training outcomes. We capture one actual AUF objective gradient at the fitted checkpoint, on the first canonical microbatch with32 anchors and seed42000. At identical effective weights, compare one zero-moment AdamW update in W coordinates against one directly in G coordinates (learning rate$10^{-4}$, epsilon$10^{-8}$, clipping1, weight decay0). Both coordinate systems apply their own clipping. No optimizer history or further training trajectory is simulated.

| Domain | Max SGD identity relative error | Effective Adam update cosine range (5 layers) | W/direct update norm ratio range | Analysis seconds | Original fit min |
| --- | ---: | --- | --- | ---: | ---: |
| math | 4.85e-07 | 0.238–0.515 | 5.85–79.11 | 14.68 | 2.66 |
| nanocoder | 5.26e-07 | 0.269–0.495 | 5.72–93.55 | 14.95 | 2.76 |

The coordinate systems produce different effective update directions and magnitudes. That supports parameterization-dependent optimization, but does not prove why one trained checkpoint wins. Geometry/gradient diagnostics have no standalone speed multiplier. A matched training trajectory and inherited Adam-state transformation remain a separate optional study, not a result here. Full per-layer spectra, principal-angle overlaps/random controls, conditioning and gradient energies are retained in `results/*_geometry.json`, `results/*_rank_calibration.json`, and `results/*_gradient_analysis.json`. The KiCad gradient extension was cancelled by the user.

<a id="study-38"></a>


## 38. 2026-09-13_dflash_matched_scaling

Source: [2026-09-13_dflash_matched_scaling.md](experiment_logs/2026-09-13_dflash_matched_scaling.md).

### AUF and decaying-CE data/anchor scaling

All96 cells have completed training and128-prompt Transformers evaluation. This is an empirical trade-off study: two intersecting sweeps, not a fitted or validated scaling law. Full results, including weaker settings, are in [scaling.md](scaling.md).

#### Intervention and training

For each active Qwen3-4B target LoRA (GSM8K, KiCad, NanoCoder), reuse its frozen target-generated response IDs and hidden-state cache. The target is always the adapted model. Each example includes its full prompt and up-to4096 generated response tokens. No target decoding or feature extraction was repeated for this sweep.

The five-W mapper has five independent2560×2560 matrices, initialized to identity, with frozen native fusion blocks Fᵢ and a frozen drafter:

$$
z=\sum_{i=1}^{5}F_iW_ih_i,\qquad G=[F_1W_1|\cdots|F_5W_5].
$$

It trains32,768,000 parameters. Export folds the matrices into the existing fusion weight G, so inference adds no separate mapper GEMM. The comparator is ordinary body LoRA r128, alpha256, dropout0 on q/k/v/o and gate/up/down projections in all five draft transformer blocks, excluding fusion:36,700,160 trainable parameters. Its adapters are merged for standalone inference. The backbone, embedding, and head remain frozen during both fits.

At4096 example presentations, vary requested anchors in{8,24,32,64,128}; at32 anchors, vary presentations in{4096,8192,12288,16384}. Each fit starts independently from the native drafter and runs one epoch over the specified cohort. GSM8K repeats after exhausting its compatible4096 unique cached questions; KiCad repeats only beyond15366 unique cached examples; NanoCoder has enough unique cached examples. These are not all unique-data scaling points.

Training uses canonical SpecForge DFlash block construction (block16, response masks, masked draft inputs), with cached target context. Anchors sample draft blocks; they do not imply all-token dense supervision. Two GPUs each process microbatch2 with two accumulation steps: global batch8, giving512/1024/1536/2048 optimizer updates. AdamW uses peak LR1e-4, zero weight decay, gradient clipping1,5% warmup and cosine decay, seed42. Final checkpoints are evaluated. Trainable weights are FP32 over a BF16 frozen draft. Exact source hashes and realized history are retained in each fit summary.

#### Losses

For block b and position j∈{0,…,15}, let mᵦⱼ be the canonical valid-response mask, yᵦⱼ the target token, and ℓᵦⱼ=−log qᵦⱼ(yᵦⱼ). Position0 is the anchor and excluded by the canonical mask. AUF includes the first wrong prediction and excludes later positions:

$$
s_{bj}=\operatorname{stopgrad}\left[\prod_{k<j}
\mathbf1\{m_{bk}=0\;\lor\;\arg\max_vq_{bk}(v)=y_{bk}\}\right],
\qquad
L_{\rm AUF}=\frac{\sum_{b,j}m_{bj}s_{bj}\ell_{bj}}{\sum_{b,j}m_{bj}s_{bj}}.
$$

This is our objective override on SpecForge, not upstream author-released AUF code. The comparison leaves the upstream DFlash objective intact with gamma7:

$$
w_j=\exp[-\max(j-1,0)/7],\qquad
L_{\rm decay}=\frac{\sum_{b,j}m_{bj}w_j\ell_{bj}}{\sum_{b,j}m_{bj}w_j}.
$$

The implementation aggregates additive numerator/denominator terms over its objective chunks; gradient accumulation follows the training script. There is no MSE or hidden-state alignment term in either fit.

#### Measurement and results

Single-request BF16 greedy/nonthinking Transformers on L40S,128 held-out prompts/domain, generation caps2048/8192/2048. Primary TPS is the arithmetic mean of each request's output-token count divided by timed generation seconds. Native-relative gain is100( meanTPS / nativeMeanTPS −1). Acceptance includes the verifier bonus token. Initialization and warmup are outside request timing.

The table below shows the highest observed mean TPS within each architecture/loss/domain group, not an independently selected optimum. Every cell remains in the full report. Intervals use2000 paired-prompt bootstrap draws, seed42; they do not measure run-to-run hardware noise.

| Domain | Loss | Architecture | Examples | Anchors | Gain vs native | 95% prompt CI | Fit min | GPU-hours |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| GSM8K | AUF | Five W | 12288 | 32 | +46.98% | [42.45, 51.61]% | 7.78 | 0.2594 |
| GSM8K | AUF | Body r128 | 16384 | 32 | +49.16% | [44.97, 53.67]% | 10.81 | 0.3604 |
| GSM8K | Decaying CE | Five W | 16384 | 32 | +44.44% | [40.51, 48.66]% | 10.26 | 0.3421 |
| GSM8K | Decaying CE | Body r128 | 16384 | 32 | +41.05% | [36.96, 45.25]% | 10.85 | 0.3617 |
| KiCad | AUF | Five W | 16384 | 32 | +112.96% | [105.73, 119.92]% | 23.78 | 0.7925 |
| KiCad | AUF | Body r128 | 16384 | 32 | +113.11% | [106.27, 119.85]% | 23.35 | 0.7782 |
| KiCad | Decaying CE | Five W | 8192 | 32 | +112.42% | [105.57, 119.04]% | 10.64 | 0.3548 |
| KiCad | Decaying CE | Body r128 | 16384 | 32 | +115.97% | [108.66, 123.17]% | 23.46 | 0.7822 |
| NanoCoder | AUF | Five W | 8192 | 32 | +14.67% | [12.19, 17.24]% | 5.46 | 0.1818 |
| NanoCoder | AUF | Body r128 | 12288 | 32 | +15.55% | [13.06, 18.17]% | 8.49 | 0.2829 |
| NanoCoder | Decaying CE | Five W | 4096 | 64 | +11.31% | [8.84, 13.94]% | 4.94 | 0.1648 |
| NanoCoder | Decaying CE | Body r128 | 16384 | 32 | +14.37% | [11.99, 17.02]% | 11.16 | 0.3719 |

The five-W AUF results improve substantially with exposure on GSM8K and KiCad, while NanoCoder gains remain smaller and nonmonotonic. AUF does not universally dominate CE: KiCad's strongest observed CE body result exceeds its AUF counterpart, and mapper/body differences near the top are small. These observations support reporting domain-dependent trade-offs, not a universal superiority claim.

Fit time is per domain, not shared across the three targets. This sweep's marginal preparation cost equals fit cost because its caches already existed; historical cold-start generation/extraction is excluded, not assumed free. The paired-analysis artifact records summed request latency reduction, evaluation seconds/GPU-hours and prompt intervals for all96 cells. Cost plots retain dominated cells. Repeated measurements of close frontier candidates remain required before treating their ordering as stable.

#### Reproduction artifacts

- AUF: `experiments/dflash_scaling_transformers_20260912/` (mapper fits referenced by `setup/mapper_manifest.json`).
- CE: `experiments/dflash_scaling_decay_20260913/`.
- Reused4096×32 evaluations: original Transformers AUF and `dflash_decay32_20260912` families.
- Raw128-request rows: each family's `measurements/`; optimizer/source provenance: `modules/**/summary.json`.
- Regenerate: `scripts/report_scaling.py`, `scripts/analyze_scaling.py`, `scripts/plot_scaling.py`, `scripts/report_scaling_experiment.py`.

#### Realized sampled blocks

Counts below are exact from response-mask eligibility. Label presentations are expectations under uniform anchor sampling; actual sampled offsets and AUF retained-prefix counts were not logged. No unique-token coverage claim is made.

| Domain | Presentations | Requested anchors | Actual sampled blocks | Expected valid label presentations |
| --- | ---: | ---: | ---: | ---: |
| math | 4096 | 8 | 32,768 | 460,842 |
| math | 4096 | 24 | 98,304 | 1,382,527 |
| math | 4096 | 32 | 131,072 | 1,843,370 |
| math | 4096 | 64 | 261,155 | 3,673,876 |
| math | 4096 | 128 | 445,763 | 6,299,483 |
| math | 8192 | 32 | 262,144 | 3,686,739 |
| math | 12288 | 32 | 393,216 | 5,530,109 |
| math | 16384 | 32 | 524,288 | 7,373,479 |
| kicad | 4096 | 8 | 32,768 | 490,285 |
| kicad | 4096 | 24 | 98,304 | 1,470,855 |
| kicad | 4096 | 32 | 131,072 | 1,961,141 |
| kicad | 4096 | 64 | 262,144 | 3,922,281 |
| kicad | 4096 | 128 | 524,288 | 7,844,562 |
| kicad | 8192 | 32 | 262,144 | 3,922,261 |
| kicad | 12288 | 32 | 393,216 | 5,883,365 |
| kicad | 16384 | 32 | 524,288 | 7,844,410 |
| nanocoder | 4096 | 8 | 32,721 | 466,040 |
| nanocoder | 4096 | 24 | 97,646 | 1,394,857 |
| nanocoder | 4096 | 32 | 129,821 | 1,856,461 |
| nanocoder | 4096 | 64 | 256,370 | 3,676,206 |
| nanocoder | 4096 | 128 | 481,903 | 6,940,656 |
| nanocoder | 8192 | 32 | 260,005 | 3,721,160 |
| nanocoder | 12288 | 32 | 390,068 | 5,581,518 |
| nanocoder | 16384 | 32 | 520,117 | 7,444,998 |

<a id="study-39"></a>


## 39. 2026-09-13_dflash_qwen8b

Source: [2026-09-13_dflash_qwen8b.md](experiment_logs/2026-09-13_dflash_qwen8b.md).

### Qwen3-8B: AUF five-W adaptation and multi-LoRA serving

Two domains: GSM8K math and CodeAlpaca coding. Every inference target is Qwen3-8B plus the corresponding published target LoRA. The native 8B DFlash drafter is frozen except for our external five-W mapper during fitting. No target LoRA is trained in this study.

#### Assets, data and fitting

Target: Qwen/Qwen3-8B revision `b968826…`; native draft: z-lab/Qwen3-8B-DFlash-b16 revision `9b41424…`. Exact pinned asset records and active paths are in `experiments/dflash_ready_20260913/setup/qwen8b.json` and the download manifests. Math adapter: `sumitdotml/lora-and-friends`, revision `6c91e19043bb9c784b7f982f488a134f043dd944`, subfolder `checkpoints/best-checkpoints/attention_only/seed-0/step-3169`; rank8,alpha32,q/k/v/o. It was published as an OpenMathSFT adaptation, not a GSM8K-trained target. Saved tensors are attention-only despite broader adapter metadata. Coding uses the pinned published CodeAlpaca adapter in `model_staging/qwen3_8b_loras/codealpaca` (rank8,alpha16,q/v); full dataset/prompt records are in `setup/qwen8b_data`.

Each domain uses4,096 unique training prompts and128 held-out evaluation prompts. Apply each target's tokenizer/chat template with thinking disabled; coding system message is “You are a helpful coding assistant.” Training rollouts use the active target with native DFlash, greedy decoding, block16, cap4,096, then one teacher-forced target prefill captures all five context layers. Dense features and exact token IDs are persisted. Sampling32 anchor blocks per example is **not dense loss supervision**. No rollout regeneration occurs during mapper fitting or serving.

For five frozen fusion blocks $F_i$, context vectors $h_i\in\mathbb R^{4096}$ and identity-initialized $W_i\in\mathbb R^{4096\times4096}$,

$$z=\sum_{i=1}^{5}F_iW_i h_i,\qquad u=\operatorname{RMSNorm}(z).$$

Only W is trained:83,886,080 parameters/domain. The target and all native drafter weights remain frozen. Inference exports the folded weights $F_iW_i$, adding no new matrix multiplication to fusion. The target's untied output head is loaded separately from input embeddings.

For valid mask $m_{aj}$ and draft probabilities $q_{aj}$ at the target-generated label $y_{aj}$, AUF uses the detached preceding-correctness mask

$$s_{aj}=\operatorname{stopgrad}\prod_{k<j}\mathbf1[\arg\max_v q_{ak}(v)=y_{ak}\ \lor\ m_{ak}=0],$$
$$L=\frac{\sum_{a,j}m_{aj}s_{aj}[-\log q_{aj}(y_{aj})]}{\sum_{a,j}m_{aj}s_{aj}}.$$

The first wrong position is included; later block positions are omitted. The mask changes with the current drafter predictions. This is our AUF modification, not the upstream DFlash decaying CE. The implementation reduces valid labels within microbatches and then averages through gradient accumulation/DDP; it does not compute one global dataset ratio.

Training: one epoch,seed42,AdamW learning rate1e−4,weight decay0,clipping1,5% warmup/cosine schedule; two GPUs,2 examples/GPU,accumulation2,global batch8,512 optimizer updates. Use the final checkpoint; no evaluation-based checkpoint selection. Exact source hashes and training history are preserved in each `modules/<domain>/maps_full/summary.json`.

#### Standalone results

Transformers, one L40S per evaluation, BF16 SDPA, greedy/nonthinking,128 prompts/domain,2,048-token response cap. Primary TPS is the arithmetic mean of per-request generated tokens/seconds. Model load/warmup is excluded. EAGLE uses its separate pinned upstream environment and tree configuration; its AR multiplier uses its own companion AR. Consequently the EAGLE comparison is not an identical-backend algorithm-only comparison.

| Domain | Method | Mean TPS | × matching AR | Gain vs native | Acceptance | Fit min | Trainable params |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| math | ar | 37.89 | 1.000 | -70.56% | 1.000 | 0.00 | 0 |
| math | native | 128.69 | 3.397 | +0.00% | 4.764 | 0.00 | 0 |
| math | mapped | 173.88 | 4.589 | +35.12% | 6.350 | 4.06 | 83,886,080 |
| math | eagle_ar | 31.19 | 1.000 | -75.77% | — | 0.00 | 0 |
| math | eagle3 | 76.25 | 2.445 | -40.75% | 3.728 | 0.00 | 0 |
| codealpaca | ar | 37.56 | 1.000 | -75.40% | 1.000 | 0.00 | 0 |
| codealpaca | native | 152.70 | 4.065 | +0.00% | 5.618 | 0.00 | 0 |
| codealpaca | mapped | 162.13 | 4.316 | +6.18% | 5.631 | 4.02 | 83,886,080 |
| codealpaca | eagle_ar | 30.32 | 1.000 | -80.15% | — | 0.00 | 0 |
| codealpaca | eagle3 | 76.21 | 2.514 | -50.09% | 4.051 | 0.00 | 0 |

#### vLLM serving

One shared target and drafter on one L40S. Explicit-ID workload:256 requests (128/domain),fixed order,clients1/4/8/16/32,engine capacity32. Learned-router workload:the same two domains,12 clients/capacity12. Frozen prompt router:8,192 training prompts,255/256 held-out correct,confusion `[[128,0],[1,127]]`,3.23 CPU seconds. The router selects the target LoRA and the matching folded mapper bank. It does not mix weights token by token.

Optimized vLLM compilation mode3,CUDA graphs,supported asynchronous scheduling,BF16,batch invariance disabled. Prefix caches reset before timing; only complete rows with zero timed JIT events are reported. DFlash uses15 speculative tokens; vLLM EAGLE uses3. Timing includes loopback HTTP and request routing,excludes semaphore wait; aggregate throughput includes workload drain. Full operational metrics and exact sources are in `mixture.md` and portable `results/comparison.json`.

| Routing | Clients | Method | Mean TPS | Aggregate TPS | Gain vs native | × AR | Acceptance | Fit min (math / code) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| learned | 12 | ar | 34.20 | 394.54 | -68.33% | 1.000 | — | 0 / 0 |
| learned | 12 | eagle3 | 63.43 | 737.91 | -41.27% | 1.855 | 2.351908003890383 | 0 / 0 |
| learned | 12 | mapped | 124.28 | 1384.27 | +15.07% | 3.634 | 6.005316855870096 | 4.06 / 4.02 |
| learned | 12 | native | 108.00 | 1248.05 | +0.00% | 3.158 | 5.183989111606038 | 0 / 0 |
| oracle | 1 | ar | 39.22 | 39.69 | -75.01% | 1.000 | — | 0 / 0 |
| oracle | 1 | eagle3 | 77.87 | 80.67 | -50.39% | 1.985 | 2.3761778901401978 | 0 / 0 |
| oracle | 1 | mapped | 183.69 | 178.59 | +17.02% | 4.683 | 5.930070921985815 | 4.06 / 4.02 |
| oracle | 1 | native | 156.97 | 159.17 | +0.00% | 4.002 | 5.186187299234001 | 0 / 0 |
| oracle | 4 | ar | 35.38 | 143.12 | -72.85% | 1.000 | — | 0 / 0 |
| oracle | 4 | eagle3 | 72.74 | 299.96 | -44.18% | 2.056 | 2.352728938777848 | 0 / 0 |
| oracle | 4 | mapped | 150.72 | 593.43 | +15.66% | 4.260 | 5.896268549200403 | 4.06 / 4.02 |
| oracle | 4 | native | 130.31 | 525.87 | +0.00% | 3.683 | 5.059563448020718 | 0 / 0 |
| oracle | 8 | ar | 35.20 | 272.58 | -70.70% | 1.000 | — | 0 / 0 |
| oracle | 8 | eagle3 | 68.61 | 555.52 | -42.90% | 1.949 | 2.3642804090727303 | 0 / 0 |
| oracle | 8 | mapped | 139.09 | 1043.65 | +15.76% | 3.951 | 5.911773009599097 | 4.06 / 4.02 |
| oracle | 8 | native | 120.15 | 931.62 | +0.00% | 3.413 | 5.107587216394243 | 0 / 0 |
| oracle | 16 | ar | 33.51 | 495.40 | -65.49% | 1.000 | — | 0 / 0 |
| oracle | 16 | eagle3 | 62.56 | 945.01 | -35.58% | 1.867 | 2.3743208464398053 | 0 / 0 |
| oracle | 16 | mapped | 110.93 | 1431.21 | +14.22% | 3.310 | 5.898839841539332 | 4.06 / 4.02 |
| oracle | 16 | native | 97.12 | 1482.33 | +0.00% | 2.898 | 5.173816840811309 | 0 / 0 |
| oracle | 32 | ar | 30.59 | 512.06 | -50.64% | 1.000 | — | 0 / 0 |
| oracle | 32 | eagle3 | 55.80 | 1270.93 | -9.96% | 1.824 | 2.3705426356589148 | 0 / 0 |
| oracle | 32 | mapped | 70.05 | 1521.07 | +13.03% | 2.290 | 5.985349579344358 | 4.06 / 4.02 |
| oracle | 32 | native | 61.98 | 1569.56 | +0.00% | 2.026 | 5.1993769470404985 | 0 / 0 |

Fitting costs exclude generation/feature extraction. Raw per-example generation/capture durations and token hashes remain in the feature metadata and worker manifests; they must be included when estimating a cold-start adaptation cost. The results support a measured benefit on these two specific targets/workloads, not universal adapter performance. A100 and DGX Spark were deferred by the user. Reproduce with the qwen8b family training/export scripts and ready/portable benchmark scripts; regenerate this report with `python3 scripts/report_qwen8b.py`.

<a id="study-40"></a>


## 40. 2026-09-13_dflash_reconstruction

Source: [2026-09-13_dflash_reconstruction.md](experiment_logs/2026-09-13_dflash_reconstruction.md).

### Five-W feature/context reconstruction versus token objectives

Three-domain experiment: GSM8K, KiCad and NanoCoder. Compare reconstruction-trained five-W mappers against the existing native/AUF/decaying-CE original ablation. No new target training, rollouts, or baseline measurements. Every evaluation target remains Qwen3-4B plus its corresponding active LoRA.

#### Architecture and exact objective

Five identity-initialized matrices W_i ∈ R^(2560×2560), 32,768,000 trainable parameters. Freeze source fusion blocks F_i, learned RMSNorm N, embedding/head and draft body. For the same saved LoRA-target token IDs, extract h_i from the adapted target and y_i from the unadapted base target, using Transformers BF16 SDPA and the pinned DFlash `extract_context_feature` helper (layers1/9/17/25/33).

D(u,v) = ||u−v||² / (||v||² + 1e−6).

Per-position loss:

L_it = (1/5) Σ_i D(W_i h_i, y_i) + D(N(Σ_i F_i W_i h_i), N(Σ_i F_i y_i)).

Average sampled positions within each example, then average examples. No token CE, AUF support, KL, or draft-body backpropagation. Base-model forwards supply offline labels only; the base model is never the evaluation target. Export folds F_i W_i into the ordinary BF16 fusion projection without extra inference multiplication.

#### Data and fitting

Use the existing 4,096 examples per domain and target-generated responses capped at 4,096 tokens. MSE50 samples half the prompt positions and half the response positions separately; MSE100 uses all valid positions. Sampling is fixed across epochs. Both teachers process the same saved token sequence.

One epoch, global batch eight, 512 updates, seed 42, AdamW LR $10^{-4}$, zero weight decay, clipping at one, 5% warmup followed by cosine decay, and the final checkpoint. Trainable maps and reconstruction arithmetic are FP32; cached features and exports are BF16. Earlier AUF uses vLLM features and block-token supervision, so comparisons with it do not isolate loss choice from those differences.

#### Evaluation and costs

Reuse pinned upstream DFlash Transformers runner: one L40S/request, BF16 SDPA, nonthinking,greedy,128 identical held-out prompts/domain,caps2048/8192/2048. Eight16-prompt shards/domain, so even slow negative results fit bounded allocations. Primary TPS is mean_i(output_tokens_i/request_seconds_i). Report gain against native and existing five-W AUF, multiplier against AR, acceptance including bonus tokens, parameters, separate per-domain fitting wall time/GPU-hours. Paired prefill and export costs are separate; generation cost for this experiment is zero because saved IDs are reused.

Code: `experiments/dflash_reconstruction_20260913/`. Source hashes, paired token identities and positions, extraction timing, final checkpoint, optimizer state and fit histories are persisted. Small validation records live in that family's `validation/`; completed result tables are generated by `report.py` after `collect.py`.

#### LoRA MSE coverage results — completed 14 September

All fits use the same active LoRA target, 4,096 examples, one epoch, 512 updates, LR $10^{-4}$, and 32,768,000 trainable parameters. The 50% and 100% variants cover prompt **and** response positions. Evaluation is 128 prompts/domain with caps 2,048 / 8,192 / 2,048 for GSM8K / KiCad / NanoCoder. Acceptance includes the correction/bonus token. Fitting excludes feature extraction, setup, export and evaluation.

| Domain | Coverage | Mean tok/s | × native | Gain/native | Change/AUF | Observed × AR | Acceptance | Fit seconds | Fit GPU h |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| math | 50% | 128.54 | 1.0111 | +1.11% | -26.51% | 3.312 | 4.4665 | 24.00 | 0.0133 |
| math | 100% | 124.80 | 0.9817 | -1.83% | -28.65% | 3.215 | 4.4696 | 34.12 | 0.0190 |
| kicad | 50% | 194.45 | 1.1905 | +19.05% | -37.19% | 5.061 | 7.2083 | 181.75 | 0.1010 |
| kicad | 100% | 196.83 | 1.2050 | +20.50% | -36.42% | 5.123 | 7.2064 | 381.61 | 0.2120 |
| nanocoder | 50% | 146.78 | 0.9772 | -2.28% | -9.57% | 3.763 | 4.9573 | 29.13 | 0.0162 |
| nanocoder | 100% | 148.15 | 0.9863 | -1.37% | -8.73% | 3.798 | 4.9742 | 37.45 | 0.0208 |

Full coverage leaves acceptance almost unchanged relative to 50%. KiCad retains approximately 20% mean-throughput gain over native DFlash; GSM8K and NanoCoder remain near native performance. These are separate timing passes, so small differences do not establish a reliable coverage benefit. AR ratios are descriptive timings. Comparisons with AUF also differ in feature backend and supervision structure, and do not isolate the loss alone.

Sources: [50% measurements](experiments/dflash_mse50_20260914/results/table.json), [100% measurements](experiments/dflash_mse100_20260914/results/table.json).

<a id="study-41"></a>


## 41. 2026-09-13_eagle3_transformers

Source: [2026-09-13_eagle3_transformers.md](experiment_logs/2026-09-13_eagle3_transformers.md).

### EAGLE-3 standalone comparison

Status: GSM8K and NanoCoder complete; KiCad and companion EAGLE-AR measurements pending. These are Transformers results on one L40S per request, not vLLM serving results.

The target is Qwen3-4B with the same active domain LoRA used in the DFlash studies, merged through PEFT before inference. The published AngelSlim/Qwen3-4B_eagle3 drafter is reused without adaptation training. No local training was performed; the publisher's drafter training time is unknown, not zero. Model/config hashes and revisions are in `experiments/dflash_ready_20260913/setup/artifacts.json` and `eagle_stage_audit.json`.

Implementation: SafeAILab/EAGLE revision `cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b`; publisher checkpoint revision `fd331e59626c8e95c392381a16ee59d518727fbb`. The documented three-line head-width compatibility patch reads the checkpoint's explicit 128-dimensional attention heads. Upstream tree construction and verification are unchanged. This is a modified upstream integration, not an unmodified official reproduction. The isolated runtime uses Transformers4.53.3, PEFT0.15.2 and the installed PyTorch runtime; runtime install logs are retained on each node.

Evaluation: identical 128 held-out prompt messages/domain; tokenizer chat template with thinking disabled, greedy temperature0/top-p1/top-k0, BF16. Tree configuration: total tokens60, depth7, branching top-k10. Generation caps are 2048 for GSM8K/NanoCoder and8192 for KiCad. Persistent KV capacity covers the largest workload prompt plus the generation cap and tree buffer, including during warmup. Two training prompts with64 generated tokens warm each process. Time is CUDA-synchronized generation including prefill, excluding model load, prompt formatting and warmup. Raw generated token IDs, times and verification counts are saved. Upstream stopping can overshoot the requested cap by a final block; actual output counts are used.

Primary throughput is the arithmetic mean of each rollout's generated tokens divided by its generation seconds. Acceptance below is total upstream accepted-new-token count divided by total verification steps. The companion EAGLE-AR path is being measured because EAGLE uses its own target KV implementation; an apparent difference from the DFlash driver must not be attributed solely to the drafting algorithm.

| Domain | Prompts | Mean tokens/s | Acceptance | Local fitting time |
| --- | ---: | ---: | ---: | --- |
| math | 128 | 71.19 | 3.347 | No adaptation fit; pretrained drafter |
| nanocoder | 128 | 81.77 | 3.852 | No adaptation fit; pretrained drafter |
| KiCad | Pending | — | — | No adaptation fit; pretrained drafter |

Reproduction entry point: `experiments/dflash_ready_20260913/eagle_benchmark.py`, configuration `setup/qwen4b.json`. Run with the isolated `.venv-eagle/bin/python`, `--domain math/kicad/nanocoder --method eagle3 --phase full`. Full runs require the preceding bounded domain check. Raw outputs and derived summaries are under `measurements/` and `results/standalone.json`; the two completed rows are not the complete three-domain baseline.

## Complete matched numerical exports

These tables preserve the paper’s numerical evidence, including fitting costs and metrics omitted from compact log tables. Field names retain their source meaning: `gain` is percent versus native; `ar` is an AR throughput multiplier; `accept` is live acceptance; `fit` is seconds; `gpuh` is GPU-hours. Domain `math` in Phase 1 denotes GSM8K. Missing values are not zero.

### ablation

| domain | loss | arch | tps | gain | ar | accept | params | fit | gpuh |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| math | AUF | maps_full | 174.9020839344481 | 37.57749920237758 | 4.5063973895733085 | 5.9916718075262185 | 32768000 | 159.75759311811998 | 0.08875421839895554 |
| math | AUF | fusion_delta | 168.26613433227308 | 32.35767945773451 | 4.335420433256485 | 5.808911483253588 | 32768000 | 156.27670038910583 | 0.08682038910505879 |
| math | AUF | body_r128 | 173.4227376717027 | 36.41384949218216 | 4.468281650831114 | 5.936735941320293 | 36700160 | 163.48096746997908 | 0.09082275970554393 |
| math | AUF | maps_r28 | 72.38991447564052 | -43.05829194828089 | 1.8651448529722798 | 2.3816821971554685 | 716800 | 151.06898103328422 | 0.0839272116851579 |
| math | AUF | maps_r56 | 99.25532523000294 | -21.9260335812788 | 2.5573391034344306 | 3.2128680119087 | 1433600 | 152.62913565104827 | 0.08479396425058237 |
| math | AUF | maps_r128 | 131.24350750835424 | 3.2357828069969896 | 3.3815228860037925 | 4.338842975206612 | 3276800 | 147.95132832415402 | 0.08219518240230779 |
| kicad | AUF | maps_full | 309.56172196053893 | 89.51965975646755 | 8.057053602323458 | 11.258683888985862 | 32768000 | 319.25154255796224 | 0.1773619680877568 |
| kicad | AUF | fusion_delta | 235.96400680566177 | 44.46171833312913 | 6.141504314588948 | 8.777437572293666 | 32768000 | 304.66711899312213 | 0.16925951055173452 |
| kicad | AUF | body_r128 | 286.5912664121881 | 75.45670361187852 | 7.45919482814839 | 10.49496412242308 | 36700160 | 310.0278574200347 | 0.17223769856668594 |
| kicad | AUF | maps_r28 | 184.28204151311095 | 12.821021880877304 | 4.796362667235716 | 6.618270246975651 | 716800 | 310.5813084333204 | 0.17254517135184466 |
| kicad | AUF | maps_r56 | 220.72135001122965 | 35.12986970801511 | 5.744779221910291 | 8.118848022556202 | 1433600 | 299.1572087239474 | 0.16619844929108188 |
| kicad | AUF | maps_r128 | 258.6865475522248 | 58.372896252084 | 6.732910537608261 | 9.455544968115516 | 3276800 | 289.2031414047815 | 0.16066841189154527 |
| nanocoder | AUF | maps_full | 162.3202694529798 | 8.065706723183897 | 4.161262532116293 | 5.549419767907163 | 32768000 | 165.36662476230413 | 0.09187034709016896 |
| nanocoder | AUF | fusion_delta | 162.45046641161446 | 8.152386140335445 | 4.16460027747359 | 5.541658341658342 | 32768000 | 165.31304154079407 | 0.09184057863377448 |
| nanocoder | AUF | body_r128 | 164.95618201121135 | 9.82058154211909 | 4.228837112934008 | 5.593063117564025 | 36700160 | 168.7616890487261 | 0.09375649391595896 |
| nanocoder | AUF | maps_r28 | 56.395364632299874 | -62.45444296365874 | 1.4457585526458583 | 1.832815700786361 | 716800 | 156.86177701689303 | 0.08714543167605168 |
| nanocoder | AUF | maps_r56 | 76.1209308562587 | -49.32202726664953 | 1.9514456115027685 | 2.4001384562132224 | 1433600 | 152.63151792017743 | 0.08479528773343191 |
| nanocoder | AUF | maps_r128 | 106.95846048510685 | -28.79175434282385 | 2.742000340496266 | 3.387396189545677 | 3276800 | 155.12268558098003 | 0.08617926976721113 |
| kicad | Decay | body_r128 | 302.0962426045634 | 84.94914923435661 | 7.8627473846257745 | 10.875048473301748 | 36700160 | 337.6645624511875 | 0.18759142358399306 |
| kicad | Decay | fusion_delta | 244.5180038016083 | 49.6986403594134 | 6.364141699708677 | 8.901109516449548 | 32768000 | 359.80683541018516 | 0.19989268633899177 |
| kicad | Decay | maps_full | 316.1053573604197 | 93.52579961995451 | 8.227366717384957 | 11.37463407752266 | 32768000 | 386.34409778984264 | 0.2146356098832459 |
| kicad | Decay | maps_r128 | 266.0940600119174 | 62.90791832171085 | 6.925708033918966 | 9.354071495903126 | 3276800 | 278.6388954790309 | 0.1547993863772394 |
| kicad | Decay | maps_r28 | 172.9915948678106 | 5.908792574346178 | 4.502502905637174 | 6.412012644889357 | 716800 | 339.07934205187485 | 0.18837741225104157 |
| kicad | Decay | maps_r56 | 227.59890785951956 | 39.340443338076135 | 5.923783434336192 | 8.068878380745078 | 1433600 | 328.0282713091001 | 0.1822379285050556 |
| math | Decay | body_r128 | 171.39799792443924 | 34.82119476390253 | 4.416113708023451 | 5.7538507109004735 | 36700160 | 162.43941126205027 | 0.0902441173678057 |
| math | Decay | fusion_delta | 168.43912910630866 | 32.493756672205066 | 4.33987768831364 | 5.620659722222222 | 32768000 | 160.43551453901455 | 0.08913084141056364 |
| math | Decay | maps_full | 173.96888700987296 | 36.84344906267476 | 4.482353329661253 | 5.779529901814936 | 32768000 | 160.21201536990702 | 0.0890066752055039 |
| math | Decay | maps_r128 | 127.6962484598045 | 0.445518574894499 | 3.2901268399594468 | 4.2228260869565215 | 3276800 | 147.29083394724876 | 0.08182824108180486 |
| math | Decay | maps_r28 | 70.44031656794826 | -44.591840311781326 | 1.8149129590788096 | 2.277790806754221 | 716800 | 146.43453157739714 | 0.08135251754299841 |
| math | Decay | maps_r56 | 97.19058746934788 | -23.55015063711734 | 2.5041406014757195 | 3.1040268456375837 | 1433600 | 146.7968154461123 | 0.08155378635895127 |
| nanocoder | Decay | body_r128 | 163.20451258490286 | 8.654396966789424 | 4.183931098565528 | 5.456620106236475 | 36700160 | 167.39610937517136 | 0.09299783854176187 |
| nanocoder | Decay | fusion_delta | 159.42794830892637 | 6.140126328598106 | 4.087114629034579 | 5.383540372670807 | 32768000 | 162.07367620337754 | 0.09004093122409863 |
| nanocoder | Decay | maps_full | 154.88198348642 | 3.113622596572263 | 3.970573711797535 | 5.36168567562343 | 32768000 | 167.249285064172 | 0.09291626948009556 |
| nanocoder | Decay | maps_r128 | 107.99924954941837 | -28.098842692519042 | 2.7686821378574415 | 3.3022978926062625 | 3276800 | 151.59123237105086 | 0.08421735131725049 |
| nanocoder | Decay | maps_r28 | 56.774733248790845 | -62.201875999669255 | 1.4554840934854882 | 1.8176813683727635 | 716800 | 151.06899277912453 | 0.08392721821062474 |
| nanocoder | Decay | maps_r56 | 73.02688390307627 | -51.3819078456873 | 1.8721262405931713 | 2.345935887676563 | 1433600 | 151.2326415013522 | 0.08401813416741788 |
| kicad | AUF | joint | 307.7791226388711 | 88.4283180531556 | 8.010657367688305 | 10.868817929058894 | 163840000 | 340.4729793178849 | 0.1891516551766027 |
| kicad | Decay | joint | 308.22516376493365 | 88.7013930377286 | 8.022266610713464 | 10.932743482829926 | 163840000 | 343.06878998083994 | 0.19059377221157775 |
| math | AUF | joint | 158.05733204838148 | 24.327463599381893 | 4.072388004323041 | 5.063868613138686 | 163840000 | 204.8279697992839 | 0.11379331655515772 |
| math | Decay | joint | 152.08482479125456 | 19.629505782529733 | 3.918504811468621 | 4.987163029525032 | 163840000 | 203.11626756889746 | 0.1128423708716097 |
| nanocoder | AUF | joint | 139.61936316122936 | -7.047559721863561 | 3.5792992867662834 | 4.485848293708556 | 163840000 | 207.59103596489877 | 0.11532835331383265 |
| nanocoder | Decay | joint | 137.16774935098786 | -8.679736528302461 | 3.516449411478638 | 4.464906632324533 | 163840000 | 208.0204990208149 | 0.11556694390045272 |

### scaling

| domain | architecture | examples | anchors | prompts | mean_rollout_tps | speedup_pct_vs_native | acceptance | training_seconds | training_gpu_hours | parameters | reused | pareto | loss | source_method | native_gain_ci95 | summed_latency_reduction_pct | summed_latency_reduction_ci95 | evaluation_seconds | evaluation_gpu_hours | incremental_generation_gpu_hours | incremental_feature_extraction_gpu_hours | preparation_cost_scope | speedup_pct_vs_ar | speedup_pct_vs_auf | acceptance_length |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| math | maps | 4096 | 32 | 128 | 174.9020839344481 | 37.57749920237758 | 5.9916718075262185 | 159.75759311811998 | 0.08875421839895554 | 32768000 | True | False | AUF | maps_full | [34.01700255897581, 41.239954259842165] | 23.408981915389038 | [21.334404528585562, 25.73268753125892] | 119.00416081771255 | 0.03305671133825348 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| math | body | 4096 | 32 | 128 | 173.4227376717027 | 36.41384949218216 | 5.936735941320293 | 163.48096746997908 | 0.09082275970554393 | 36700160 | True | False | AUF | body_r128 | [32.80066546305231, 40.05129704338645] | 23.012531815460257 | [21.08656179367182, 25.1261471540063] | 119.62014964548871 | 0.033227819345969085 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| kicad | maps | 4096 | 32 | 128 | 309.56172196053893 | 89.51965975646755 | 11.258683888985862 | 319.25154255796224 | 0.1773619680877568 | 32768000 | True | True | AUF | maps_full | [83.50669695614398, 95.04209602742323] | 46.65508702784834 | [45.03009574944089, 47.94232051328996] | 2093.8311791196465 | 0.5816197719776796 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| kicad | body | 4096 | 32 | 128 | 286.5912664121881 | 75.45670361187852 | 10.49496412242308 | 310.0278574200347 | 0.17223769856668594 | 36700160 | True | False | AUF | body_r128 | [71.0150444859504, 79.52156970209522] | 42.790946271763985 | [41.79500159363925, 43.67925415939976] | 2245.5018435711972 | 0.6237505121031104 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| nanocoder | maps | 4096 | 32 | 128 | 162.3202694529798 | 8.065706723183897 | 5.549419767907163 | 165.36662476230413 | 0.09187034709016896 | 32768000 | True | False | AUF | maps_full | [5.678252831058904, 10.444067084306996] | 5.0129304554398235 | [3.3304394108190953, 6.806496819384601] | 181.79720189748332 | 0.05049922274930092 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| nanocoder | body | 4096 | 32 | 128 | 164.95618201121135 | 9.82058154211909 | 5.593063117564025 | 168.7616890487261 | 0.09375649391595896 | 36700160 | True | False | AUF | body_r128 | [7.735850601268114, 11.86831186887907] | 6.621082375579768 | [5.113810395519249, 8.170636828563227] | 178.71933539723977 | 0.0496442598325666 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| kicad | body | 4096 | 8 | 128 | 260.2552876436226 | 59.33330920781093 | 9.527896361729471 | 187.4690807298757 | 0.1041494892943754 | 36700160 | False | False | AUF | scale_body_n4096_a8 | [55.74945901002061, 62.46415763667094] | 37.583505985411556 | [36.58373240343556, 38.415305899843915] | 2449.8981060725637 | 0.6805272516868233 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| kicad | body | 4096 | 24 | 128 | 283.82939048637536 | 73.76582987453601 | 10.302028429963265 | 248.67300831526518 | 0.13815167128625844 | 36700160 | False | True | AUF | scale_body_n4096_a24 | [69.26872707057406, 77.74471405266496] | 42.34004354220342 | [41.3492226662906, 43.19213479282826] | 2263.2001420836896 | 0.6286667061343583 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| kicad | body | 4096 | 64 | 128 | 304.65123272882073 | 86.51336349173859 | 10.955583864118896 | 450.06162220891565 | 0.2500342345605087 | 36700160 | False | False | AUF | scale_body_n4096_a64 | [81.39158857394122, 91.39420801005662] | 45.905773421592976 | [44.834137983545254, 46.865800639157335] | 2123.242347013671 | 0.5897895408371309 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| kicad | body | 4096 | 128 | 128 | 312.8016662180806 | 91.50321615171686 | 11.283500104961165 | 810.2730175098404 | 0.45015167639435577 | 36700160 | False | False | AUF | scale_body_n4096_a128 | [85.80202166640467, 96.73510630469171] | 47.167896324999724 | [46.06483237577879, 48.154336888196966] | 2073.702997527551 | 0.5760286104243197 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| kicad | body | 8192 | 32 | 128 | 316.9875546950589 | 94.06589785186767 | 11.56221991180583 | 652.410364327021 | 0.3624502024039005 | 36700160 | False | False | AUF | scale_body_n8192_a32 | [88.30388208724216, 99.64466913674443] | 47.994076873912796 | [46.777018059041716, 49.08870627244159] | 2041.2747396766208 | 0.5670207610212835 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| kicad | body | 12288 | 32 | 128 | 331.91132193317947 | 103.20251613703752 | 12.009346664432405 | 1062.4692848571576 | 0.590260713809532 | 36700160 | False | False | AUF | scale_body_n12288_a32 | [96.61915759071935, 109.34241022117891] | 50.11078027763698 | [48.861386134742034, 51.20693677342029] | 1958.1924111708067 | 0.5439423364363352 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| kicad | body | 16384 | 32 | 128 | 348.09306040905886 | 113.10928868887045 | 12.292929292929292 | 1400.8110776161775 | 0.7782283764534319 | 36700160 | False | True | AUF | scale_body_n16384_a32 | [106.26871058508924, 119.84627105493114] | 52.369061399111125 | [51.20046530013085, 53.44160677477399] | 1869.5530422055162 | 0.5193202895015323 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| kicad | maps | 4096 | 8 | 128 | 260.77333555079997 | 59.650468133287916 | 10.006050076013775 | 178.18257793411613 | 0.09899032107450896 | 32768000 | False | True | AUF | scale_maps_n4096_a8 | [55.663988397558875, 63.18854070235877] | 37.467055898393376 | [36.30101596756666, 38.44573957578916] | 2454.4688666085713 | 0.6817969073912697 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| kicad | maps | 4096 | 24 | 128 | 298.46930297344954 | 82.72866680360711 | 11.137183803850471 | 260.2220773231238 | 0.1445678207350688 | 32768000 | False | True | AUF | scale_maps_n4096_a24 | [77.30322576731147, 87.710349430165] | 44.95223151122493 | [43.66732275484472, 46.05818368534921] | 2160.6696417881176 | 0.6001860116078105 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| kicad | maps | 4096 | 64 | 128 | 317.9056716952202 | 94.62798679615057 | 11.717015749604897 | 448.97664255229756 | 0.24943146808460975 | 32768000 | False | True | AUF | scale_maps_n4096_a64 | [88.62296875377109, 100.63355242101693] | 48.14674109580916 | [46.83473220325679, 49.382545052650784] | 2035.2825449211523 | 0.5653562624780979 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| kicad | maps | 4096 | 128 | 128 | 330.3455632642582 | 102.24392846568287 | 11.930489789878662 | 788.1631591860205 | 0.4378684217700114 | 32768000 | False | False | AUF | scale_maps_n4096_a128 | [95.57261440228548, 108.41438093545794] | 49.9720229369254 | [48.53926087082823, 51.118316249762444] | 1963.6387495398521 | 0.5454552082055145 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| kicad | maps | 8192 | 32 | 128 | 332.5596188646514 | 103.59941602858606 | 12.095827473042663 | 662.5696266382001 | 0.3680942370212223 | 32768000 | False | True | AUF | scale_maps_n8192_a32 | [97.10031122932456, 109.73374230356211] | 50.29669815765765 | [49.01439432103824, 51.418742270039644] | 1950.8949833139777 | 0.5419152731427715 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| kicad | maps | 12288 | 32 | 128 | 347.75157736616256 | 112.90022618045552 | 12.41884554661327 | 1034.436440674588 | 0.5746869114858822 | 32768000 | False | True | AUF | scale_maps_n12288_a32 | [105.8200371623393, 119.66371187522951] | 52.29834115492584 | [51.0466997929751, 53.36969713967413] | 1872.3288692529313 | 0.5200913525702587 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| kicad | maps | 16384 | 32 | 128 | 347.84506569476264 | 112.95746153925661 | 12.593915963761324 | 1426.5628143800423 | 0.7925348968778012 | 32768000 | False | False | AUF | scale_maps_n16384_a32 | [105.73292216014353, 119.92205302587992] | 52.20935023128015 | [50.960842055619935, 53.27981369140547] | 1875.821835314855 | 0.521061620920793 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| math | body | 4096 | 8 | 128 | 175.28792593766372 | 37.88100146315989 | 5.877458396369137 | 79.17222662596032 | 0.043984570347755735 | 36700160 | False | True | AUF | scale_body_n4096_a8 | [34.505150246192855, 41.41153910126027] | 23.742440720919745 | [21.839421076160594, 25.764137791898538] | 118.48604542622343 | 0.032912790396173175 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| math | body | 4096 | 24 | 128 | 178.18548007958412 | 40.160209598829134 | 5.922256097560975 | 130.05507783219218 | 0.07225282101788455 | 36700160 | False | True | AUF | scale_body_n4096_a24 | [36.6846434971217, 43.73606606871945] | 25.34158046059327 | [23.46221411171096, 27.312674397005715] | 116.00136396475136 | 0.032222601101319824 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| math | body | 4096 | 64 | 128 | 178.16595277050686 | 40.14484947110184 | 5.989824236817761 | 308.3480454967357 | 0.17130446972040875 | 36700160 | False | False | AUF | scale_body_n4096_a64 | [36.27660807927319, 44.14510625074649] | 24.96816141722592 | [23.014804416230174, 27.03691415788107] | 116.58156802784652 | 0.03238376889662403 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| math | body | 4096 | 128 | 128 | 178.70467477462648 | 40.56860672101459 | 6.013931888544891 | 582.1691037244163 | 0.32342727984689795 | 36700160 | False | False | AUF | scale_body_n4096_a128 | [36.712804864798095, 44.50112832600852] | 25.034155450265683 | [22.818528301135732, 27.263514665665177] | 116.47902905242518 | 0.03235528584789588 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| math | body | 8192 | 32 | 128 | 179.49602320460218 | 41.19107922417007 | 6.0514018691588785 | 323.65649117995054 | 0.1798091617666392 | 36700160 | False | True | AUF | scale_body_n8192_a32 | [37.23400992506342, 45.10545330292124] | 24.999189456983718 | [22.875019229831555, 27.218871967095392] | 116.53335786005482 | 0.032370377183348564 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| math | body | 12288 | 32 | 128 | 184.28976648177596 | 44.96182452952022 | 6.158845909955612 | 486.09291714290157 | 0.2700516206349453 | 36700160 | False | False | AUF | scale_body_n12288_a32 | [41.12168466183966, 49.03048724108997] | 27.09096987805564 | [24.894457792394384, 29.3754780022646] | 113.28323036665097 | 0.03146756399073638 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| math | body | 16384 | 32 | 128 | 189.6234947965378 | 49.157320583442996 | 6.174507310870947 | 648.6744858892635 | 0.3603747143829241 | 36700160 | False | True | AUF | scale_body_n16384_a32 | [44.972836928785895, 53.67017359192247] | 28.695053541560778 | [26.310325537851817, 31.15082379973122] | 110.7908672275953 | 0.03077524089655425 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| math | maps | 4096 | 8 | 128 | 171.43113302420485 | 34.84725874243855 | 5.7967770814682185 | 70.19876116607338 | 0.03899931175892966 | 32768000 | False | True | AUF | scale_maps_n4096_a8 | [31.19541912372899, 38.5430361237629] | 22.2070091951996 | [20.140365211879764, 24.42589627537497] | 120.8717395295389 | 0.03357548320264969 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| math | maps | 4096 | 24 | 128 | 174.3323903299455 | 37.12937977660544 | 5.924062214089662 | 121.09142233198509 | 0.06727301240665838 | 32768000 | False | False | AUF | scale_maps_n4096_a24 | [33.582647923130345, 40.63958553685031] | 22.992751601566454 | [20.757998882085055, 25.343863650873548] | 119.65088337659836 | 0.03323635649349954 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| math | maps | 4096 | 64 | 128 | 175.44469901428468 | 38.004318734964706 | 6.043870566272558 | 293.6629083612934 | 0.16314606020071853 | 32768000 | False | False | AUF | scale_maps_n4096_a64 | [34.32823809075074, 41.81229317560664] | 23.386327994821098 | [21.242274016203453, 25.729279535572672] | 119.03935960307717 | 0.033066488778632545 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| math | maps | 4096 | 128 | 128 | 182.38818712024622 | 43.46604742264386 | 6.112334801762114 | 548.7806235551834 | 0.30487812419732413 | 32768000 | False | False | AUF | scale_maps_n4096_a128 | [39.56300748833706, 47.65172088434459] | 25.84927327104497 | [23.579383833404872, 28.36548639918567] | 115.21253051701933 | 0.032003480699172034 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| math | maps | 8192 | 32 | 128 | 179.1058692885582 | 40.88418522459107 | 6.064626912269747 | 309.52218452095985 | 0.17195676917831104 | 32768000 | False | True | AUF | scale_maps_n8192_a32 | [36.78325576332362, 45.29319778050188] | 24.25473166296851 | [21.786501274463664, 26.816918549828724] | 117.6900675795041 | 0.03269168543875114 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| math | maps | 12288 | 32 | 128 | 186.85303777012888 | 46.97808669001582 | 6.1335648879065365 | 466.97136477194726 | 0.2594285359844151 | 32768000 | False | True | AUF | scale_maps_n12288_a32 | [42.44970979634454, 51.607674179679215] | 27.28265821876742 | [24.749884741978597, 29.97036931281905] | 112.98539243871346 | 0.03138483123297596 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| math | maps | 16384 | 32 | 128 | 184.2767631187572 | 44.95159611985753 | 6.1981493299298025 | 618.8797111520544 | 0.34382206175114133 | 32768000 | False | False | AUF | scale_maps_n16384_a32 | [40.73834516862477, 49.50800057348492] | 26.413447006470548 | [23.719382102660667, 29.076492698377372] | 114.33593919314444 | 0.03175998310920679 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| nanocoder | body | 4096 | 8 | 128 | 163.9416953607647 | 9.145180883832783 | 5.493365022776787 | 81.90371846873313 | 0.045502065815962854 | 36700160 | False | True | AUF | scale_body_n4096_a8 | [7.120998142418315, 11.12159770118522] | 5.941532952399752 | [4.548071121224723, 7.427103715169141] | 180.01993540814146 | 0.050005537613372626 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| nanocoder | body | 4096 | 24 | 128 | 167.21728641376242 | 11.325925551589 | 5.567242071457246 | 134.1444064481184 | 0.07452467024895466 | 36700160 | False | True | AUF | scale_body_n4096_a24 | [9.14554427731055, 13.58827885739109] | 7.861742104943037 | [6.376029595754737, 9.448471888539702] | 176.34481780882925 | 0.04898467161356368 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| nanocoder | body | 4096 | 64 | 128 | 168.6673093462147 | 12.291287138828544 | 5.602100585740255 | 312.85022426908836 | 0.17380568014949352 | 36700160 | False | False | AUF | scale_body_n4096_a64 | [10.167185245717278, 14.438527411936137] | 8.607203287044829 | [7.212082359903249, 10.0828122288491] | 174.91806827671826 | 0.04858835229908841 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| nanocoder | body | 4096 | 128 | 128 | 165.81388702442635 | 10.391603871773025 | 5.628246753246753 | 620.335596498102 | 0.34463088694338995 | 36700160 | False | False | AUF | scale_body_n4096_a128 | [8.168300317356838, 12.680836013711918] | 7.100583562209617 | [5.5767320110302165, 8.773708850205972] | 177.80161075899377 | 0.049389336321942715 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| nanocoder | body | 8192 | 32 | 128 | 172.41798282540762 | 14.788320821576928 | 5.6999588984792435 | 338.90066461404786 | 0.18827814700780438 | 36700160 | False | True | AUF | scale_body_n8192_a32 | [12.5634926912672, 16.980674318926066] | 10.628947482754224 | [9.14177834336749, 12.205142177811263] | 171.04862120887265 | 0.04751350589135351 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| nanocoder | body | 12288 | 32 | 128 | 173.56754817130889 | 15.553650942996477 | 5.698787754263407 | 509.2841326277703 | 0.28293562923765014 | 36700160 | False | True | AUF | scale_body_n12288_a32 | [13.059963478856998, 18.166047316314334] | 10.321432495835781 | [8.615575446230466, 12.14235638754378] | 171.6371788349934 | 0.0476769941208315 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| nanocoder | body | 16384 | 32 | 128 | 167.55898733530668 | 11.553415018548874 | 5.742443064182194 | 678.3400903251022 | 0.3768556057361679 | 36700160 | False | False | AUF | scale_body_n16384_a32 | [9.155945768251101, 14.18945654453462] | 7.610242593903327 | [5.755597676113589, 9.548767612836238] | 176.8261665608734 | 0.04911837960024261 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| nanocoder | maps | 4096 | 8 | 128 | 157.25951488594825 | 4.6964785874625825 | 5.330770709206227 | 69.94306988967583 | 0.0388572610498199 | 32768000 | False | True | AUF | scale_maps_n4096_a8 | [2.5120019619605194, 6.980261789409793] | 1.9099234068768345 | [0.07361555240136268, 3.733504671317831] | 187.73609443940222 | 0.052148915122056175 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| nanocoder | maps | 4096 | 24 | 128 | 165.7599644949264 | 10.355704619767403 | 5.51740600755918 | 125.2754918821156 | 0.06959749549006422 | 32768000 | False | True | AUF | scale_maps_n4096_a24 | [8.17124871820052, 12.59467012198072] | 7.073519252501481 | [5.316676065964185, 8.875549856303973] | 177.85340955434367 | 0.049403724876206576 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| nanocoder | maps | 4096 | 64 | 128 | 169.2051819632019 | 12.649379105270286 | 5.587429492344883 | 298.6559931198135 | 0.16591999617767417 | 32768000 | False | True | AUF | scale_maps_n4096_a64 | [10.336294917959284, 15.028713637944351] | 8.442758657367143 | [6.9512435901313125, 10.175433144445718] | 175.23280136287212 | 0.04867577815635337 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| nanocoder | maps | 4096 | 128 | 128 | 169.27988362060853 | 12.699112188071604 | 5.624822551206652 | 586.4291566489264 | 0.32579397591607023 | 32768000 | False | False | AUF | scale_maps_n4096_a128 | [10.252600215559493, 15.30013866545164] | 8.415611087432628 | [6.547670718958139, 10.355723132506384] | 175.28475951123983 | 0.04869021097534439 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| nanocoder | maps | 8192 | 32 | 128 | 172.2452348992387 | 14.67331283896749 | 5.605497170573969 | 327.3232179051265 | 0.18184623216951473 | 32768000 | False | True | AUF | scale_maps_n8192_a32 | [12.19215671225607, 17.241388705012582] | 9.710803839341697 | [7.884535653834153, 11.560526270687879] | 172.80587033880875 | 0.04800163064966909 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| nanocoder | maps | 12288 | 32 | 128 | 167.33147762372337 | 11.401949044203885 | 5.60663028097837 | 490.1637491490692 | 0.2723131939717051 | 32768000 | False | False | AUF | scale_maps_n12288_a32 | [8.764895265706071, 14.229961947006208] | 7.597567259205107 | [5.896536288854149, 9.523178278829782] | 176.85042607737705 | 0.04912511835482696 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| nanocoder | maps | 16384 | 32 | 128 | 169.88020497719737 | 13.098779782755997 | 5.6719836400818 | 656.5232654050924 | 0.36473514744727353 | 32768000 | False | False | AUF | scale_maps_n16384_a32 | [10.206022666232425, 15.974100475069108] | 8.497981595440606 | [6.709193460145576, 10.41800234161903] | 175.12710933899507 | 0.04864641926083196 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | Not measured | Not measured | Not measured |
| kicad | body_r128 | 4096 | 32 | 128 | 302.0962426045634 | 84.94914923435661 | Not measured | 337.6645624511875 | 0.18759142358399306 | 36700160 | True | Not measured | Decaying CE | decay32_body_r128 | [79.57893930966206, 89.91003512864455] | 45.29714149633824 | [44.16396597210785, 46.311130218995466] | 2147.1316446186975 | 0.5964254568385271 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 686.2747384625775 | 5.410135621535428 | 10.875048473301748 |
| kicad | maps_full | 4096 | 32 | 128 | 316.1053573604197 | 93.52579961995451 | Not measured | 386.34409778984264 | 0.2146356098832459 | 32768000 | True | Not measured | Decaying CE | decay32_maps_full | [87.5752246189031, 98.9621291928729] | 47.67673865871086 | [46.352702858396746, 48.81767166125716] | 2053.7305224738084 | 0.5704807006871689 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 722.7366717384957 | 2.113838674380708 | 11.37463407752266 |
| math | body_r128 | 4096 | 32 | 128 | 171.39799792443924 | 34.82119476390253 | Not measured | 162.43941126205027 | 0.0902441173678057 | 36700160 | True | Not measured | Decaying CE | decay32_body_r128 | [31.580530366008055, 38.05344108503992] | 22.64399555755262 | [20.71879119566562, 24.59455144577105] | 120.1927670254372 | 0.03338687972928811 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 341.61137080234505 | -1.167516886451403 | 5.7538507109004735 |
| math | maps_full | 4096 | 32 | 128 | 173.96888700987296 | 36.84344906267476 | Not measured | 160.21201536990702 | 0.0890066752055039 | 32768000 | True | Not measured | Decaying CE | decay32_maps_full | [33.305930443564435, 40.586366914944556] | 23.435877345682776 | [21.407108117953285, 25.631670899852786] | 118.96237173862755 | 0.033045103260729874 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 348.23533296612527 | -0.5335539197605432 | 5.779529901814936 |
| nanocoder | body_r128 | 4096 | 32 | 128 | 163.20451258490286 | 8.654396966789424 | Not measured | 167.39610937517136 | 0.09299783854176187 | 36700160 | True | Not measured | Decaying CE | decay32_body_r128 | [6.462049769495034, 11.051056894301135] | 5.009545067940168 | [3.476845299132261, 6.727105508830752] | 181.8036812422797 | 0.05050102256729992 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 318.39310985655277 | -1.0618998360361176 | 5.456620106236475 |
| nanocoder | maps_full | 4096 | 32 | 128 | 154.88198348642 | 3.113622596572263 | Not measured | 167.249285064172 | 0.09291626948009556 | 32768000 | True | Not measured | Decaying CE | decay32_maps_full | [0.8295111571369196, 5.5320084368573905] | 0.3121722482914424 | [-1.655288430573445, 2.389053393953149] | 190.7940547633916 | 0.052998348545386556 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 297.0573711797535 | -4.582475122562856 | 5.36168567562343 |
| kicad | body_r128 | 4096 | 8 | 128 | 273.8571050735264 | 67.66060431089723 | Not measured | 197.84503088193014 | 0.10991390604551675 | 36700160 | False | Not measured | Decaying CE | scale_ce_body_r128_n4096_a8 | [63.38268458075898, 71.38999384149096] | 40.006423984706 | [39.01512772594015, 40.83546268710899] | 2354.7966058785096 | 0.654110168299586 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 612.7759081388617 | Not measured | 9.896282430918882 |
| kicad | body_r128 | 4096 | 24 | 128 | 295.56570389615797 | 80.95102741811604 | Not measured | 260.4992013229057 | 0.1447217785127254 | 36700160 | False | Not measured | Decaying CE | scale_ce_body_r128_n4096_a24 | [76.17384959724743, 85.49022744221044] | 44.20815403425639 | [43.17258055918538, 45.14003791100104] | 2189.875287353061 | 0.6082986909314059 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 669.2775140989082 | Not measured | 10.684280271658109 |
| kicad | body_r128 | 4096 | 64 | 128 | 313.17051140405835 | 91.72903029212995 | Not measured | 456.0211913106032 | 0.2533451062836684 | 36700160 | False | Not measured | Decaying CE | scale_ce_body_r128_n4096_a64 | [86.08954943959101, 97.11375616262684] | 47.23096299111176 | [46.06833153598017, 48.29193587588539] | 2071.227579638362 | 0.5753409943439894 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 715.0980622117079 | Not measured | 11.270684443201873 |
| kicad | body_r128 | 4096 | 128 | 128 | 324.9692485848487 | 98.9524448729043 | Not measured | 821.7555369650945 | 0.45653085386949693 | 36700160 | False | Not measured | Decaying CE | scale_ce_body_r128_n4096_a128 | [92.82367594440719, 104.99006697439184] | 48.991177681686494 | [47.79501352541245, 50.08171309040112] | 2002.1377227855846 | 0.5561493674404402 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 745.806981035164 | Not measured | 11.496070009089776 |
| kicad | body_r128 | 8192 | 32 | 128 | 328.8651801359894 | 101.33760934779632 | Not measured | 659.1787082739174 | 0.3662103934855097 | 36700160 | False | Not measured | Decaying CE | scale_ce_body_r128_n8192_a32 | [95.07470655560806, 107.45178496862786] | 49.600194968096254 | [48.381001339407405, 50.711450224598316] | 1978.2332994421013 | 0.5495092498450281 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 755.947036187889 | Not measured | 11.716377243333575 |
| kicad | body_r128 | 12288 | 32 | 128 | 341.96876191181127 | 109.35987496906515 | Not measured | 1025.812018983066 | 0.5698955661017033 | 36700160 | False | Not measured | Decaying CE | scale_ce_body_r128_n12288_a32 | [102.41982318303957, 115.9802080423252] | 51.38651926852855 | [50.11687932462603, 52.52522831074258] | 1908.1186191872694 | 0.5300329497742415 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 790.0521122552988 | Not measured | 12.096054309504163 |
| kicad | body_r128 | 16384 | 32 | 128 | 352.76047061097904 | 115.9667672808927 | Not measured | 1407.87277075788 | 0.7821515393099334 | 36700160 | False | Not measured | Decaying CE | scale_ce_body_r128_n16384_a32 | [108.65706728350906, 123.17226068207206] | 52.82122198348975 | [51.56647738916922, 53.96310914091508] | 1851.8053718693554 | 0.5143903810748209 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 818.1400085556492 | Not measured | 12.36551512595376 |
| kicad | maps_full | 4096 | 8 | 128 | 275.67158691329064 | 68.77146510702062 | Not measured | 210.71909942803904 | 0.11706616634891058 | 32768000 | False | Not measured | Decaying CE | scale_ce_maps_full_n4096_a8 | [64.56287757651877, 72.57054017991747] | 40.52629419323656 | [39.43475466632617, 41.464424600416784] | 2334.3912777774967 | 0.6484420216048602 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 617.4985131660061 | Not measured | 10.16916819070442 |
| kicad | maps_full | 4096 | 24 | 128 | 319.589155920792 | 95.65864832502145 | Not measured | 257.08109400002286 | 0.1428228300000127 | 32768000 | False | Not measured | Decaying CE | scale_ce_maps_full_n4096_a24 | [89.945387880653, 101.03000788706908] | 48.08580210956127 | [46.96786429619754, 49.11504285744199] | 2037.6744496468455 | 0.5660206804574571 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 731.8040562855408 | Not measured | 11.188572221547641 |
| kicad | maps_full | 4096 | 64 | 128 | 333.14846356393235 | 103.95991811634535 | Not measured | 443.06827441602945 | 0.24614904134223858 | 32768000 | False | Not measured | Decaying CE | scale_ce_maps_full_n4096_a64 | [97.00937646837941, 110.4844380984326] | 49.94011182919997 | [48.530154917706184, 51.137429514862774] | 1964.8912864471786 | 0.5458031351242163 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 767.0952634151813 | Not measured | 11.737271172253157 |
| kicad | maps_full | 4096 | 128 | 128 | 339.60766785177765 | 107.91436762378801 | Not measured | 777.6901539959945 | 0.43205008555333024 | 32768000 | False | Not measured | Decaying CE | scale_ce_maps_full_n4096_a128 | [101.33569471083702, 114.16727042999634] | 51.05806686818881 | [49.783886478783074, 52.189495482044926] | 1921.0106427818537 | 0.5336140674394038 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 783.9068235932066 | Not measured | 12.033095162584184 |
| kicad | maps_full | 8192 | 32 | 128 | 346.97325926465766 | 112.42372481961121 | Not measured | 638.65421316633 | 0.35480789620351666 | 32768000 | False | Not measured | Decaying CE | scale_ce_maps_full_n8192_a32 | [105.56754340633124, 119.03726479712924] | 52.04890822550181 | [50.849860234005895, 53.11767453085649] | 1882.1193144074641 | 0.52281092066874 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 803.0774640879486 | Not measured | 12.117870293830315 |
| kicad | maps_full | 12288 | 32 | 128 | 335.6568772469928 | 105.49561737764704 | Not measured | 1038.6455385531299 | 0.5770252991961833 | 32768000 | False | Not measured | Decaying CE | scale_ce_maps_full_n12288_a32 | [98.95302811545473, 111.76897424868646] | 50.51467028966239 | [49.29687433070239, 51.644821547331695] | 1942.3394000213593 | 0.5395387222281554 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 773.6239851748426 | Not measured | 12.366226346364002 |
| kicad | maps_full | 16384 | 32 | 128 | 344.06618555542417 | 110.64395819739579 | Not measured | 1403.0634060851298 | 0.7794796700472943 | 32768000 | False | Not measured | Decaying CE | scale_ce_maps_full_n16384_a32 | [103.41340399596572, 117.53214493414097] | 51.76497080887561 | [50.442311685758256, 52.91347147316379] | 1893.2640887210146 | 0.5259066913113929 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 795.5111382021579 | Not measured | 12.593915963761324 |
| math | body_r128 | 4096 | 8 | 128 | 166.22121486601029 | 30.749151417873442 | Not measured | 79.80302410619333 | 0.04433501339232963 | 36700160 | False | Not measured | Decaying CE | scale_ce_body_r128_n4096_a8 | [27.882368717721548, 33.84076283082713] | 20.281777115466415 | [18.419902090858915, 22.23671568191943] | 123.86309065343812 | 0.03440641407039948 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 328.27325547740986 | Not measured | 5.684811237928007 |
| math | body_r128 | 4096 | 24 | 128 | 170.0236997318101 | 33.74017557735512 | Not measured | 130.44502144819126 | 0.07246945636010625 | 36700160 | False | Not measured | Decaying CE | scale_ce_body_r128_n4096_a24 | [30.428073597903683, 37.09515739459519] | 21.669913368323147 | [19.63627068551081, 23.64426983866809] | 121.70625824667513 | 0.03380729395740976 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 338.0704559953606 | Not measured | 5.733471074380165 |
| math | body_r128 | 4096 | 64 | 128 | 171.52676220497696 | 34.92248039357868 | Not measured | 308.12566369585693 | 0.17118092427547607 | 36700160 | False | Not measured | Decaying CE | scale_ce_body_r128_n4096_a64 | [31.553118537125417, 38.37677465054964] | 22.36308775271747 | [20.442541188755822, 24.476567356233083] | 120.62923070508987 | 0.033508119640302744 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 341.94313529858914 | Not measured | 5.758968277497776 |
| math | body_r128 | 4096 | 128 | 128 | 169.21591168459977 | 33.10477171640158 | Not measured | 579.5990715869702 | 0.3219994842149835 | 36700160 | False | Not measured | Decaying CE | scale_ce_body_r128_n4096_a128 | [29.507789493911506, 36.82773396692095] | 21.30210110916967 | [19.234376360732362, 23.490265644686755] | 122.27775070536882 | 0.03396604186260245 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 335.9891692174161 | Not measured | 5.758968277497776 |
| math | body_r128 | 8192 | 32 | 128 | 171.43472776889726 | 34.8500863587478 | Not measured | 324.5429281820543 | 0.18030162676780795 | 36700160 | False | Not measured | Decaying CE | scale_ce_body_r128_n8192_a32 | [31.439741329999396, 38.561734195853916] | 22.482882203457944 | [20.48079751539824, 24.65258346441689] | 120.44309872202575 | 0.03345641631167382 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 341.7060061957387 | Not measured | 5.852666465802953 |
| math | body_r128 | 12288 | 32 | 128 | 176.90234093305557 | 39.150895867757555 | Not measured | 487.4008721662685 | 0.2707782623145936 | 36700160 | False | Not measured | Decaying CE | scale_ce_body_r128_n12288_a32 | [35.544038105005605, 43.15752674801055] | 24.503366087551548 | [22.20698764302529, 26.798372861699843] | 117.30374902952462 | 0.032584374730423506 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 355.79345280351856 | Not measured | 5.902461257976299 |
| math | body_r128 | 16384 | 32 | 128 | 179.31404004851672 | 41.04793177304777 | Not measured | 651.0671083629131 | 0.3617039490905073 | 36700160 | False | Not measured | Decaying CE | scale_ce_body_r128_n16384_a32 | [36.9624670088255, 45.25257960047318] | 24.964376784804788 | [22.731763891127144, 27.42874644538262] | 116.58744844328612 | 0.03238540234535726 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 362.007257895986 | Not measured | 5.987977805178792 |
| math | maps_full | 4096 | 8 | 128 | 171.97527548932382 | 35.275279712096875 | Not measured | 66.36353142885491 | 0.036868628571586064 | 32768000 | False | Not measured | Decaying CE | scale_ce_maps_full_n4096_a8 | [31.957382890608592, 38.707581936131554] | 22.485721829218765 | [20.373677268466395, 24.719364985323107] | 120.43868662137538 | 0.03345519072815983 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 343.0987413658828 | Not measured | 5.632067265874166 |
| math | maps_full | 4096 | 24 | 128 | 168.57788377951383 | 32.60290071733471 | Not measured | 121.48632239596918 | 0.06749240133109399 | 32768000 | False | Not measured | Decaying CE | scale_ce_maps_full_n4096_a24 | [29.1771847458072, 36.01372714395799] | 21.16552349398394 | [18.977507754765128, 23.47565135042718] | 122.4899597200565 | 0.034024988811126804 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 334.34527383248064 | Not measured | 5.733471074380165 |
| math | maps_full | 4096 | 64 | 128 | 174.691681197563 | 37.41199698699611 | Not measured | 291.6973713040352 | 0.16205409516890842 | 32768000 | False | Not measured | Decaying CE | scale_ce_maps_full_n4096_a64 | [33.89370369435766, 41.07375289727157] | 23.64974720732832 | [21.49149302527276, 25.813526778627935] | 118.63006902160123 | 0.032952796950444785 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 350.09763086866263 | Not measured | 5.822841726618705 |
| math | maps_full | 4096 | 128 | 128 | 173.90781007597988 | 36.7954061141007 | Not measured | 544.9296472053975 | 0.3027386928918875 | 32768000 | False | Not measured | Decaying CE | scale_ce_maps_full_n4096_a128 | [33.40468778962458, 40.38235958437691] | 23.320899455832144 | [21.188617879895308, 25.557552803625285] | 119.14101993571967 | 0.03309472775992213 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 348.07796666764136 | Not measured | 5.865036231884058 |
| math | maps_full | 8192 | 32 | 128 | 176.15644200142478 | 38.5641738152636 | Not measured | 307.6427364703268 | 0.17091263137240376 | 32768000 | False | Not measured | Decaying CE | scale_ce_maps_full_n8192_a32 | [34.94329461143789, 42.169442046461754] | 23.92164839423545 | [21.759311986000355, 26.10515537106461] | 118.20759947644547 | 0.03283544429901263 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 353.87162493116097 | Not measured | 5.927677754043333 |
| math | maps_full | 12288 | 32 | 128 | 180.12243214662274 | 41.68381077877805 | Not measured | 460.70617061201483 | 0.25594787256223045 | 32768000 | False | Not measured | Decaying CE | scale_ce_maps_full_n12288_a32 | [37.828509866430785, 45.571971226659855] | 25.27853793198842 | [23.087585456915168, 27.470846817946303] | 116.0993170067668 | 0.03224981027965745 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 364.09010102667276 | Not measured | 5.960417305922062 |
| math | maps_full | 16384 | 32 | 128 | 183.63014252856783 | 44.44296613827101 | Not measured | 615.82985429978 | 0.3421276968332111 | 32768000 | False | Not measured | Decaying CE | scale_ce_maps_full_n16384_a32 | [40.51457148514107, 48.659945595177525] | 26.38503239077734 | [23.97374560527531, 28.820288307469582] | 114.3800887237303 | 0.03177224686770286 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 373.1278074696108 | Not measured | 5.980603448275862 |
| nanocoder | body_r128 | 4096 | 8 | 128 | 163.303795264296 | 8.720495014495654 | Not measured | 83.04743313509971 | 0.04613746285283317 | 36700160 | False | Not measured | Decaying CE | scale_ce_body_r128_n4096_a8 | [6.817894228591926, 10.822229520389135] | 5.660161036676659 | [4.244611310790397, 7.176549812278297] | 180.55845741136 | 0.05015512705871111 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 318.6476321631256 | Not measured | 5.395059326979187 |
| nanocoder | body_r128 | 4096 | 24 | 128 | 163.95890062748708 | 9.156635394806557 | Not measured | 133.9897920829244 | 0.07443877337940244 | 36700160 | False | Not measured | Decaying CE | scale_ce_body_r128_n4096_a24 | [6.988009917209886, 11.507668573765283] | 5.709327500563011 | [4.323081670284852, 7.232665168359501] | 180.46435696585104 | 0.05012898804606973 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 320.3270683861076 | Not measured | 5.455546813532651 |
| nanocoder | body_r128 | 4096 | 64 | 128 | 165.643929682665 | 10.27845373770704 | Not measured | 312.2212305967696 | 0.17345623922042755 | 36700160 | False | Not measured | Decaying CE | scale_ce_body_r128_n4096_a64 | [8.07836993963873, 12.511863217299092] | 6.596384698537284 | [5.150313955690633, 8.15023364841594] | 178.76660465821624 | 0.049657390182837845 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 324.6468297421411 | Not measured | 5.465221674876847 |
| nanocoder | body_r128 | 4096 | 128 | 128 | 164.06428555837962 | 9.22679605356629 | Not measured | 618.7006749678403 | 0.3437225972043557 | 36700160 | False | Not measured | Decaying CE | scale_ce_body_r128_n4096_a128 | [7.089163044536767, 11.533312782461763] | 5.689718733090832 | [4.0894771422439655, 7.423317632555909] | 180.5018864851445 | 0.050139412912540135 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 320.59723450020454 | Not measured | 5.477093206951027 |
| nanocoder | body_r128 | 8192 | 32 | 128 | 166.3236596844141 | 10.730988120920637 | Not measured | 332.39625775720924 | 0.18466458764289403 | 36700160 | False | Not measured | Decaying CE | scale_ce_body_r128_n8192_a32 | [8.63512878925777, 12.9991912923699] | 7.261121533834325 | [5.71500644764262, 8.90391546850577] | 177.49435468530282 | 0.04930398741258412 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 326.3893939935227 | Not measured | 5.528403428343632 |
| nanocoder | body_r128 | 12288 | 32 | 128 | 169.44816901373832 | 12.811149212166306 | Not measured | 498.931756334845 | 0.27718430907491387 | 36700160 | False | Not measured | Decaying CE | scale_ce_body_r128_n12288_a32 | [10.284673344115891, 15.558567054757537] | 8.19134055112528 | [6.521897485391431, 9.95834118163362] | 175.71399431303144 | 0.048809442864730955 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 334.3994248092558 | Not measured | 5.548309661932387 |
| nanocoder | body_r128 | 16384 | 32 | 128 | 171.7902401731264 | 14.370397332491613 | Not measured | 669.4678538050503 | 0.37192658544725016 | 36700160 | False | Not measured | Decaying CE | scale_ce_body_r128_n16384_a32 | [11.990775868219103, 17.015189529301065] | 9.693333520401792 | [8.139790227876228, 11.33737107322942] | 172.83930704882368 | 0.04801091862467324 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 340.40358744153554 | Not measured | 5.595319749848699 |
| nanocoder | maps_full | 4096 | 8 | 128 | 157.94198691226293 | 5.150838490209453 | Not measured | 71.6748823611997 | 0.03981937908955539 | 32768000 | False | Not measured | Decaying CE | scale_ce_maps_full_n4096_a8 | [2.9342158659204243, 7.4680512191213] | 2.5883752370281177 | [0.9200300029552297, 4.461967848466223] | 186.4375951285474 | 0.051788220869040946 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 304.90203386237437 | Not measured | 5.220402785620177 |
| nanocoder | maps_full | 4096 | 24 | 128 | 163.24496588632792 | 8.681328998275784 | Not measured | 125.77232767408714 | 0.06987351537449285 | 32768000 | False | Not measured | Decaying CE | scale_ce_maps_full_n4096_a24 | [6.5562746977717, 10.829153005165303] | 5.779744861594405 | [4.209022730801518, 7.430649743202912] | 180.32958410400897 | 0.050091551140002494 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 318.49681644112644 | Not measured | 5.382495633611488 |
| nanocoder | maps_full | 4096 | 64 | 128 | 167.1988738809755 | 11.313667295889385 | Not measured | 296.59989930316806 | 0.16477772183509337 | 32768000 | False | Not measured | Decaying CE | scale_ce_maps_full_n4096_a64 | [8.835685218157003, 13.940765956580263] | 7.578278211556944 | [6.068649616876015, 9.234061616737584] | 176.88734368002042 | 0.049135373244450116 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 328.63310394792376 | Not measured | 5.435234175974917 |
| nanocoder | maps_full | 4096 | 128 | 128 | 163.74359379741733 | 9.01329356305487 | Not measured | 583.7624719031155 | 0.3243124843906197 | 32768000 | False | Not measured | Decaying CE | scale_ce_maps_full_n4096_a128 | [6.652116509814531, 11.390945339337858] | 5.808732612990886 | [4.121510642852358, 7.560560106003474] | 180.27410400426015 | 0.05007614000118338 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 319.7751051298258 | Not measured | 5.473850404578646 |
| nanocoder | maps_full | 8192 | 32 | 128 | 165.96582666648175 | 10.492758612664721 | Not measured | 317.37998558208346 | 0.1763222142122686 | 32768000 | False | Not measured | Decaying CE | scale_ce_maps_full_n8192_a32 | [8.146456260255658, 12.972172282212092] | 7.025863698664669 | [5.277732848072188, 8.807042032749207] | 177.9446182460524 | 0.04942906062390345 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 325.4720488367569 | Not measured | 5.484674708325094 |
| nanocoder | maps_full | 12288 | 32 | 128 | 166.57789249630827 | 10.900245161844424 | Not measured | 479.6087824120186 | 0.26644932356223255 | 32768000 | False | Not measured | Decaying CE | scale_ce_maps_full_n12288_a32 | [8.297889137690982, 13.73346465890101] | 6.670007386074806 | [4.831129661313847, 8.653443033880608] | 178.62569707306102 | 0.049618249186961394 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 327.04114837893303 | Not measured | 5.511923688394277 |
| nanocoder | maps_full | 16384 | 32 | 128 | 165.87198204011244 | 10.430280981836804 | Not measured | 642.036615468096 | 0.35668700859338665 | 32768000 | False | Not measured | Decaying CE | scale_ce_maps_full_n16384_a32 | [8.013772639612343, 13.141086421143115] | 6.618478703038866 | [4.821984596870732, 8.562668244457178] | 178.72431860584766 | 0.0496456440571799 | 0 | 0 | Marginal cost conditional on the existing frozen caches; historical cache construction is excluded, not zero. | 325.2314675902699 | Not measured | 5.527301713830211 |

### serving

| file | trial | size | device | policy | concurrency | capacity | cohort_sha256 | method | requests | mean_request_tps | aggregate_serving_tps | workload_seconds | mean_latency_seconds | mean_ttft_seconds | mean_tpot_seconds | acceptance | allocated_bytes | reserved_bytes | gain_pct_vs_ar | gain_pct_vs_native |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| experiments/dflash_peak_deployment_20260912/measurements/peak_trial0_full_oracle_mapped_c12.json | peak_trial0 | qwen4b | NVIDIA L40S | oracle | 12 | 12 | 9967deee5c658071054c4e0322f423543fd4dcb14b4abd40bcd2a8152093a53c | mapped | 384 | 202.42228744200656 | 3227.256210709981 | 212.33610077993944 | 6.352362916442265 | 0.11239261046527342 | 0.0052694619358926624 | 10.497389093916054 | 36474215936 | 37111201792 | 362.0643523111834 | 45.63666721155097 |
| experiments/dflash_peak_deployment_20260912/measurements/peak_clients_trial0_full_oracle_mapped_c1.json | peak_clients_trial0 | qwen4b | NVIDIA L40S | oracle | 1 | 32 | 9967deee5c658071054c4e0322f423543fd4dcb14b4abd40bcd2a8152093a53c | mapped | 384 | 390.7086962238815 | 516.2385260179855 | 1322.3635307997465 | 3.4435048407855597 | 0.043112069614532324 | 0.002689463688428895 | 10.46316837750881 | 35771598848 | 36094083072 | 506.7425814042135 | 44.59661076968171 |
| experiments/dflash_peak_deployment_20260912/measurements/peak_clients_trial0_full_oracle_mapped_c4.json | peak_clients_trial0 | qwen4b | NVIDIA L40S | oracle | 4 | 32 | 9967deee5c658071054c4e0322f423543fd4dcb14b4abd40bcd2a8152093a53c | mapped | 384 | 283.22374382033064 | 1572.0859876327802 | 434.8483514119871 | 4.485149997271947 | 0.07911553993835696 | 0.003753456404236107 | 10.507159097892215 | 35771598848 | 36161191936 | 410.8611949017856 | 46.934861943826704 |
| experiments/dflash_peak_deployment_20260912/measurements/peak_clients_trial0_full_oracle_mapped_c32.json | peak_clients_trial0 | qwen4b | NVIDIA L40S | oracle | 32 | 32 | 9967deee5c658071054c4e0322f423543fd4dcb14b4abd40bcd2a8152093a53c | mapped | 384 | 108.14430176763909 | 4323.955743094702 | 161.6133562689647 | 12.221465530173495 | 0.21298124577394142 | 0.009939648877214118 | 10.576358629871702 | 35771598848 | 37327208448 | 248.75002118053837 | 43.76697004843035 |
| experiments/dflash_peak_deployment_20260912/measurements/peak_clients_trial0_full_oracle_native_c8.json | peak_clients_trial0 | qwen4b | NVIDIA L40S | oracle | 8 | 32 | 9967deee5c658071054c4e0322f423543fd4dcb14b4abd40bcd2a8152093a53c | native | 384 | 171.82291519333862 | 1534.730226284391 | 453.3317895773798 | 9.169091239897776 | 0.09394173777764081 | 0.00583391165824687 | 5.906091693607922 | 35823765504 | 36406558720 | 247.5570285617698 | 0.0 |
| experiments/dflash_peak_deployment_20260912/measurements/peak_clients_trial0_full_oracle_mapped_c8.json | peak_clients_trial0 | qwen4b | NVIDIA L40S | oracle | 8 | 32 | 9967deee5c658071054c4e0322f423543fd4dcb14b4abd40bcd2a8152093a53c | mapped | 384 | 246.9871614384343 | 2689.5250879408836 | 255.32277169637382 | 5.207266077320431 | 0.09256854809670283 | 0.004289869993914655 | 10.464477211796247 | 35771598848 | 36318478336 | 399.59648179555717 | 43.74518157867322 |
| experiments/dflash_peak_deployment_20260912/measurements/peak_trial0_full_oracle_native_c12.json | peak_trial0 | qwen4b | NVIDIA L40S | oracle | 12 | 12 | 9967deee5c658071054c4e0322f423543fd4dcb14b4abd40bcd2a8152093a53c | native | 384 | 138.99129341374527 | 1842.7744090854965 | 382.25080429110676 | 11.471853704178406 | 0.11530895846954081 | 0.007260762114192443 | 5.868238938716367 | 36787019264 | 37385928704 | 217.27199005453167 | 0.0 |
| experiments/dflash_peak_deployment_20260912/measurements/peak_clients_trial0_full_oracle_ar_c4.json | peak_clients_trial0 | qwen4b | NVIDIA L40S | oracle | 4 | 32 | 9967deee5c658071054c4e0322f423543fd4dcb14b4abd40bcd2a8152093a53c | ar | 384 | 55.440449704695446 | 226.27887318934705 | 3006.9842155817896 | 30.937985589543434 | 0.09167369516095884 | 0.017726335543748646 | Not measured | 37123638272 | 37316722688 | 0.0 | -71.23781109033438 |
| experiments/dflash_peak_deployment_20260912/measurements/peak_trial0_full_learned_ar_c12.json | peak_trial0 | qwen4b | NVIDIA L40S | learned | 12 | 12 | 8f5366aae3c6e629aea7795d28d295cc49d40d2ade3ca0cd6fc9dedf2cc875cd | ar | 384 | 43.79963981757763 | 503.17146704814314 | 1363.5808962401934 | 40.89067679176151 | 0.1055917831727129 | 0.02260267564596314 | Not measured | 37188944896 | 37385928704 | 0.0 | -68.81318208781585 |
| experiments/dflash_peak_deployment_20260912/measurements/peak_clients_trial0_full_oracle_ar_c32.json | peak_clients_trial0 | qwen4b | NVIDIA L40S | oracle | 32 | 32 | 9967deee5c658071054c4e0322f423543fd4dcb14b4abd40bcd2a8152093a53c | ar | 384 | 31.009116903151593 | 827.7307387034713 | 824.2402608711272 | 60.910005646645004 | 0.14079694205550672 | 0.032774522645289704 | Not measured | 37123638272 | 37474009088 | 0.0 | -58.77649854707647 |
| experiments/dflash_peak_deployment_20260912/measurements/peak_clients_trial0_full_oracle_ar_c8.json | peak_clients_trial0 | qwen4b | NVIDIA L40S | oracle | 8 | 32 | 9967deee5c658071054c4e0322f423543fd4dcb14b4abd40bcd2a8152093a53c | ar | 384 | 49.43733001296542 | 390.6974958096318 | 1754.3316948567517 | 35.62090038589546 | 0.09912275063349323 | 0.01993704703017245 | Not measured | 37123638272 | 37346082816 | 0.0 | -71.22774342564405 |
| experiments/dflash_peak_deployment_20260912/measurements/peak_clients_trial0_full_oracle_native_c4.json | peak_clients_trial0 | qwen4b | NVIDIA L40S | oracle | 4 | 32 | 9967deee5c658071054c4e0322f423543fd4dcb14b4abd40bcd2a8152093a53c | native | 384 | 192.75462614761042 | 889.1117587869693 | 770.094415278174 | 7.993233282981237 | 0.0811390800117806 | 0.005173740698893314 | 5.792730470963322 | 35823765504 | 36240883712 | 247.67868437994537 | 0.0 |
| experiments/dflash_peak_deployment_20260912/measurements/peak_clients_trial0_full_oracle_mapped_c16.json | peak_clients_trial0 | qwen4b | NVIDIA L40S | oracle | 16 | 32 | 9967deee5c658071054c4e0322f423543fd4dcb14b4abd40bcd2a8152093a53c | mapped | 384 | 175.93318031073878 | 3693.0893134062853 | 186.73174176877365 | 7.429616930561072 | 0.12922448765675654 | 0.006030413612485899 | 10.496302720467408 | 35771598848 | 36781948928 | 330.0356959954101 | 41.11787001581455 |
| experiments/dflash_peak_deployment_20260912/measurements/peak_clients_trial0_full_oracle_native_c1.json | peak_clients_trial0 | qwen4b | NVIDIA L40S | oracle | 1 | 32 | 9967deee5c658071054c4e0322f423543fd4dcb14b4abd40bcd2a8152093a53c | native | 384 | 270.2059848734735 | 291.61440901826114 | 2316.661931330338 | 6.032792021898786 | 0.04564002121818097 | 0.0036947029633857217 | 5.783518035171794 | 35823765504 | 36106665984 | 319.61051381118006 | 0.0 |
| experiments/dflash_peak_deployment_20260912/measurements/peak_clients_trial0_full_oracle_ar_c16.json | peak_clients_trial0 | qwen4b | NVIDIA L40S | oracle | 16 | 32 | 9967deee5c658071054c4e0322f423543fd4dcb14b4abd40bcd2a8152093a53c | ar | 384 | 40.911296887460374 | 595.4556409682889 | 1147.0258286399767 | 44.555529812234454 | 0.1124835744461355 | 0.024304966772388983 | Not measured | 37123638272 | 37392220160 | 0.0 | -67.18461482850466 |
| experiments/dflash_peak_deployment_20260912/measurements/peak_clients_trial0_full_oracle_native_c32.json | peak_clients_trial0 | qwen4b | NVIDIA L40S | oracle | 32 | 32 | 9967deee5c658071054c4e0322f423543fd4dcb14b4abd40bcd2a8152093a53c | native | 384 | 75.22193848225976 | 2439.743363254623 | 280.0847869049758 | 20.736439062310335 | 0.21031139434732418 | 0.013473988999375717 | 5.790301541589023 | 35823765504 | 37132173312 | 142.58007319974539 | 0.0 |
| experiments/dflash_peak_deployment_20260912/measurements/peak_trial0_full_learned_native_c12.json | peak_trial0 | qwen4b | NVIDIA L40S | learned | 12 | 12 | 8f5366aae3c6e629aea7795d28d295cc49d40d2ade3ca0cd6fc9dedf2cc875cd | native | 384 | 140.4427984314035 | 1824.2088868649746 | 372.901921977289 | 11.035361399293228 | 0.11369806686343509 | 0.007175427569532135 | 5.826731740731227 | 36787019264 | 37478203392 | 220.64829532009335 | 0.0 |
| experiments/dflash_peak_deployment_20260912/measurements/peak_trial0_full_learned_mapped_c12.json | peak_trial0 | qwen4b | NVIDIA L40S | learned | 12 | 12 | 8f5366aae3c6e629aea7795d28d295cc49d40d2ade3ca0cd6fc9dedf2cc875cd | mapped | 384 | 201.98010302103373 | 3195.9004456579255 | 211.2644030940719 | 6.3038009376772 | 0.1139849512874207 | 0.0052168431293041885 | 10.458813328378545 | 36474215936 | 37113298944 | 361.14557987752056 | 43.816632306488046 |
| experiments/dflash_peak_deployment_20260912/measurements/peak_trial0_full_oracle_ar_c12.json | peak_trial0 | qwen4b | NVIDIA L40S | oracle | 12 | 12 | 9967deee5c658071054c4e0322f423543fd4dcb14b4abd40bcd2a8152093a53c | ar | 384 | 43.808245849202095 | 491.383986250438 | 1396.6368852122687 | 40.83377878504931 | 0.10693927677978839 | 0.022581646753473388 | Not measured | 37188944896 | 37385928704 | 0.0 | -68.4813021209933 |
| experiments/dflash_peak_deployment_20260912/measurements/peak_clients_trial0_full_oracle_native_c16.json | peak_clients_trial0 | qwen4b | NVIDIA L40S | oracle | 16 | 32 | 9967deee5c658071054c4e0322f423543fd4dcb14b4abd40bcd2a8152093a53c | native | 384 | 124.67108544865552 | 2097.747832777154 | 332.176245930139 | 12.8800466613975 | 0.12868302327842684 | 0.00805496937599839 | 5.831983602442901 | 35823765504 | 36612079616 | 204.73510969745905 | 0.0 |
| experiments/dflash_peak_deployment_20260912/measurements/peak_clients_trial0_full_oracle_ar_c1.json | peak_clients_trial0 | qwen4b | NVIDIA L40S | oracle | 1 | 32 | 9967deee5c658071054c4e0322f423543fd4dcb14b4abd40bcd2a8152093a53c | ar | 384 | 64.3944743946676 | 63.93236515952127 | 10821.842712586746 | 28.181618368827912 | 0.08601331050158478 | 0.015203884573075645 | Not measured | 37123638272 | 37316722688 | 0.0 | -76.16837597996916 |
| experiments/dflash_portable_deployment_20260913/measurements/l40s_qwen8b_oracle_c8_eagle3_full_oracle_eagle3_c8.json | l40s_qwen8b_oracle_c8_eagle3 | qwen8b | NVIDIA L40S | oracle | 8 | 32 | f61f527209d8d8c6cd3a21cb99e86b9f573987c82c0e7dffa30fbf81acf093cc | eagle3 | 256 | 68.61096841584987 | 555.516444241312 | 74.95367676625028 | 2.2753654406715214 | 0.11520636265049689 | 0.013740932207506313 | 2.3642804090727303 | 34643930112 | 35190210560 | 94.91130117731853 | -42.896071223332186 |
| experiments/dflash_portable_deployment_20260913/measurements/l40s_qwen8b_oracle_c1_eagle3_full_oracle_eagle3_c1.json | l40s_qwen8b_oracle_c1_eagle3 | qwen8b | NVIDIA L40S | oracle | 1 | 32 | f61f527209d8d8c6cd3a21cb99e86b9f573987c82c0e7dffa30fbf81acf093cc | eagle3 | 256 | 77.87417366792107 | 80.67102974994148 | 515.8109438912943 | 2.014752130711713 | 0.06666711677462445 | 0.012421881514816295 | 2.3761778901401978 | 36615973888 | 36960206848 | 98.53237003061525 | -50.388844439512816 |
| experiments/dflash_portable_deployment_20260913/measurements/l40s_qwen8b_learned_c12_eagle3_full_learned_eagle3_c12.json | l40s_qwen8b_learned_c12_eagle3 | qwen8b | NVIDIA L40S | learned | 12 | 12 | 560c3f266e8e6494adc69ea7efa852edce488af21f499a7adef2e8e095e69e19 | eagle3 | 256 | 63.42911124824343 | 737.9116657724354 | 56.05548999784514 | 2.4436228787817527 | 0.1176413897810562 | 0.01487534796392847 | 2.351908003890383 | 35984686080 | 36268146688 | 85.46220642522748 | -41.26951051026667 |
| experiments/dflash_portable_deployment_20260913/measurements/l40s_qwen4b_learned_c12_eagle3_full_learned_eagle3_c12.json | l40s_qwen4b_learned_c12_eagle3 | qwen4b | NVIDIA L40S | learned | 12 | 12 | 8f5366aae3c6e629aea7795d28d295cc49d40d2ade3ca0cd6fc9dedf2cc875cd | eagle3 | 384 | 78.71086439937041 | 960.2511053828507 | 721.391515320167 | 21.282832528160117 | 0.10604340429927106 | 0.012469845667898624 | 2.5611537921192538 | 36882333696 | 37224448000 | 79.70664765097506 | -43.95521502100005 |
| experiments/dflash_portable_deployment_20260913/measurements/l40s_qwen8b_oracle_c8_ar_full_oracle_ar_c8.json | l40s_qwen8b_oracle_c8_ar | qwen8b | NVIDIA L40S | oracle | 8 | 32 | f61f527209d8d8c6cd3a21cb99e86b9f573987c82c0e7dffa30fbf81acf093cc | ar | 256 | 35.201123793961926 | 272.57866336335405 | 152.31933228997514 | 4.5366548587026045 | 0.10753355209635629 | 0.027399849456005442 | Not measured | 36912312320 | 37100716032 | 0.0 | -70.70260758009198 |
| experiments/dflash_portable_deployment_20260913/measurements/l40s_qwen8b_oracle_c4_mapped_full_oracle_mapped_c4.json | l40s_qwen8b_oracle_c4_mapped | qwen8b | NVIDIA L40S | oracle | 4 | 32 | f61f527209d8d8c6cd3a21cb99e86b9f573987c82c0e7dffa30fbf81acf093cc | mapped | 256 | 150.72184303807623 | 593.4251771046193 | 68.98595067998394 | 1.07246606453009 | 0.10930301300504652 | 0.00604154189014739 | 5.896268549200403 | 35195162624 | 35662069760 | 326.0134348173297 | 15.66089980017955 |
| experiments/dflash_portable_deployment_20260913/measurements/l40s_qwen8b_oracle_c16_eagle3_full_oracle_eagle3_c16.json | l40s_qwen8b_oracle_c16_eagle3 | qwen8b | NVIDIA L40S | oracle | 16 | 32 | f61f527209d8d8c6cd3a21cb99e86b9f573987c82c0e7dffa30fbf81acf093cc | eagle3 | 256 | 62.56088020664904 | 945.0090755914613 | 44.20063370699063 | 2.4875414255002397 | 0.1411996025381086 | 0.014969841161457553 | 2.3743208464398053 | 34643930112 | 35190210560 | 86.68531780864272 | -35.58323595988464 |
| experiments/dflash_portable_deployment_20260913/measurements/l40s_qwen4b_oracle_c1_eagle3_full_oracle_eagle3_c1.json | l40s_qwen4b_oracle_c1_eagle3 | qwen4b | NVIDIA L40S | oracle | 1 | 32 | 9967deee5c658071054c4e0322f423543fd4dcb14b4abd40bcd2a8152093a53c | eagle3 | 384 | 135.13976884558554 | 136.27246452198244 | 5025.380603500642 | 13.086730541301222 | 0.04336628021216408 | 0.007262958368867498 | 2.542182181290392 | 36832260096 | 37119590400 | 109.86236803072073 | -49.98638949138524 |
| experiments/dflash_portable_deployment_20260913/measurements/l40s_qwen8b_oracle_c16_native_full_oracle_native_c16.json | l40s_qwen8b_oracle_c16_native | qwen8b | NVIDIA L40S | oracle | 16 | 32 | f61f527209d8d8c6cd3a21cb99e86b9f573987c82c0e7dffa30fbf81acf093cc | native | 256 | 97.11894277658752 | 1482.3255020428762 | 28.410089377779514 | 1.6457391144340363 | 0.15162847120518563 | 0.009685128096955617 | 5.173816840811309 | 33659981824 | 34449915904 | 189.80859344686266 | 0.0 |
| experiments/dflash_portable_deployment_20260913/measurements/l40s_qwen4b_oracle_c8_eagle3_full_oracle_eagle3_c8.json | l40s_qwen4b_oracle_c8_eagle3 | qwen4b | NVIDIA L40S | oracle | 8 | 32 | 9967deee5c658071054c4e0322f423543fd4dcb14b4abd40bcd2a8152093a53c | eagle3 | 384 | 92.51655223890836 | 775.0946500520502 | 888.1237923055887 | 17.985163767096918 | 0.08296107684752012 | 0.010605742614589235 | 2.544621173756991 | 36832260096 | 37155241984 | 87.13905507163311 | -46.155870923964436 |
| experiments/dflash_portable_deployment_20260913/measurements/l40s_qwen8b_oracle_c16_ar_full_oracle_ar_c16.json | l40s_qwen8b_oracle_c16_ar | qwen8b | NVIDIA L40S | oracle | 16 | 32 | f61f527209d8d8c6cd3a21cb99e86b9f573987c82c0e7dffa30fbf81acf093cc | ar | 256 | 33.511408899748375 | 495.39945391859095 | 81.71789387287572 | 4.632558123634226 | 0.1330131996764976 | 0.028644212964278937 | Not measured | 36233785344 | 36482056192 | 0.0 | -65.49446694777347 |
| experiments/dflash_portable_deployment_20260913/measurements/l40s_qwen8b_oracle_c4_eagle3_full_oracle_eagle3_c4.json | l40s_qwen8b_oracle_c4_eagle3 | qwen8b | NVIDIA L40S | oracle | 4 | 32 | f61f527209d8d8c6cd3a21cb99e86b9f573987c82c0e7dffa30fbf81acf093cc | eagle3 | 256 | 72.73767602538308 | 299.96176039812934 | 137.8075656881556 | 2.1384049830940057 | 0.09842995812505251 | 0.01297844473530765 | 2.352728938777848 | 36615973888 | 36960206848 | 105.59214629810097 | -44.18257573758279 |
| experiments/dflash_portable_deployment_20260913/measurements/l40s_qwen4b_oracle_c4_eagle3_full_oracle_eagle3_c4.json | l40s_qwen4b_oracle_c4_eagle3 | qwen4b | NVIDIA L40S | oracle | 4 | 32 | 9967deee5c658071054c4e0322f423543fd4dcb14b4abd40bcd2a8152093a53c | eagle3 | 384 | 111.32223387249856 | 484.3438147939223 | 1411.6666283658706 | 14.628097007490092 | 0.07034738892131524 | 0.00881136460281819 | 2.539153103171685 | 36832260096 | 37121687552 | 100.79605137667254 | -42.2466603798922 |
| experiments/dflash_portable_deployment_20260913/measurements/l40s_qwen8b_oracle_c32_ar_full_oracle_ar_c32.json | l40s_qwen8b_oracle_c32_ar | qwen8b | NVIDIA L40S | oracle | 32 | 32 | f61f527209d8d8c6cd3a21cb99e86b9f573987c82c0e7dffa30fbf81acf093cc | ar | 256 | 30.59281532731879 | 512.0606942353534 | 85.22036643559113 | 5.39748592646356 | 0.13584602094670117 | 0.03135350330819232 | Not measured | 36233785344 | 36576428032 | 0.0 | -50.637100357933676 |
| experiments/dflash_portable_deployment_20260913/measurements/l40s_qwen8b_oracle_c8_native_full_oracle_native_c8.json | l40s_qwen8b_oracle_c8_native | qwen8b | NVIDIA L40S | oracle | 8 | 32 | f61f527209d8d8c6cd3a21cb99e86b9f573987c82c0e7dffa30fbf81acf093cc | native | 256 | 120.15104719709538 | 931.6171412504647 | 44.968043357133865 | 1.3426293751472258 | 0.11985961816935742 | 0.00784955513388115 | 5.107587216394243 | 35444133888 | 35938893824 | 241.32730506093947 | 0.0 |
| experiments/dflash_portable_deployment_20260913/measurements/l40s_qwen8b_oracle_c4_native_full_oracle_native_c4.json | l40s_qwen8b_oracle_c4_native | qwen8b | NVIDIA L40S | oracle | 4 | 32 | f61f527209d8d8c6cd3a21cb99e86b9f573987c82c0e7dffa30fbf81acf093cc | native | 256 | 130.31356603525424 | 525.8722463020534 | 78.06839073309675 | 1.2145413950256625 | 0.10981998555507744 | 0.007213601227808702 | 5.059563448020718 | 35444133888 | 35792093184 | 268.3296909788249 | 0.0 |
| experiments/dflash_portable_deployment_20260913/measurements/l40s_qwen8b_oracle_c32_eagle3_full_oracle_eagle3_c32.json | l40s_qwen8b_oracle_c32_eagle3 | qwen8b | NVIDIA L40S | oracle | 32 | 32 | f61f527209d8d8c6cd3a21cb99e86b9f573987c82c0e7dffa30fbf81acf093cc | eagle3 | 256 | 55.80197816868909 | 1270.9266740983103 | 32.68402563780546 | 2.759868665933027 | 0.20259369359519042 | 0.016490220185922885 | 2.3705426356589148 | 34643930112 | 35190210560 | 82.40223258844377 | -9.960968982478114 |
| experiments/dflash_portable_deployment_20260913/measurements/l40s_qwen8b_oracle_c1_mapped_full_oracle_mapped_c1.json | l40s_qwen8b_oracle_c1_mapped | qwen8b | NVIDIA L40S | oracle | 1 | 32 | f61f527209d8d8c6cd3a21cb99e86b9f573987c82c0e7dffa30fbf81acf093cc | mapped | 256 | 183.69222124117147 | 178.59097680439476 | 234.07677559088916 | 0.9142334175612632 | 0.0671264377906482 | 0.005193534330415408 | 5.930070921985815 | 35195162624 | 35662069760 | 368.30483485721555 | 17.02446310516943 |
| experiments/dflash_portable_deployment_20260913/measurements/l40s_qwen8b_oracle_c1_ar_full_oracle_ar_c1.json | l40s_qwen8b_oracle_c1_ar | qwen8b | NVIDIA L40S | oracle | 1 | 32 | f61f527209d8d8c6cd3a21cb99e86b9f573987c82c0e7dffa30fbf81acf093cc | ar | 256 | 39.22492521290722 | 39.68938443450886 | 1041.8402953119949 | 4.069536431132292 | 0.0884921335382387 | 0.024773729647338323 | Not measured | 36912312320 | 37100716032 | 0.0 | -75.0110495568875 |
| experiments/dflash_portable_deployment_20260913/measurements/l40s_qwen8b_oracle_c32_mapped_full_oracle_mapped_c32.json | l40s_qwen8b_oracle_c32_mapped | qwen8b | NVIDIA L40S | oracle | 32 | 32 | f61f527209d8d8c6cd3a21cb99e86b9f573987c82c0e7dffa30fbf81acf093cc | mapped | 256 | 70.04777537115326 | 1521.0670100660204 | 27.136214070022106 | 2.2210420216324565 | 0.24789931342638738 | 0.013019825139675884 | 5.985349579344358 | 33660899328 | 34915483648 | 128.968058747447 | 13.02527305188974 |
| experiments/dflash_portable_deployment_20260913/measurements/l40s_qwen8b_oracle_c8_mapped_full_oracle_mapped_c8.json | l40s_qwen8b_oracle_c8_mapped | qwen8b | NVIDIA L40S | oracle | 8 | 32 | f61f527209d8d8c6cd3a21cb99e86b9f573987c82c0e7dffa30fbf81acf093cc | mapped | 256 | 139.08706038288398 | 1043.647424828183 | 40.13615997461602 | 1.1896018027473474 | 0.11934570864286798 | 0.006580965706359855 | 5.911773009599097 | 33660899328 | 34183577600 | 295.1210796478653 | 15.760173238212417 |
| experiments/dflash_portable_deployment_20260913/measurements/l40s_qwen8b_learned_c12_native_full_learned_native_c12.json | l40s_qwen8b_learned_c12_native | qwen8b | NVIDIA L40S | learned | 12 | 12 | 560c3f266e8e6494adc69ea7efa852edce488af21f499a7adef2e8e095e69e19 | native | 256 | 108.00031091062414 | 1248.0522278062697 | 33.57631921675056 | 1.4679187539404666 | 0.13525299385037215 | 0.008705124837723375 | 5.183989111606038 | 36116833792 | 36637245440 | 215.78522167374086 | 0.0 |
| experiments/dflash_portable_deployment_20260913/measurements/l40s_qwen4b_oracle_c32_eagle3_full_oracle_eagle3_c32.json | l40s_qwen4b_oracle_c32_eagle3 | qwen4b | NVIDIA L40S | oracle | 32 | 32 | 9967deee5c658071054c4e0322f423543fd4dcb14b4abd40bcd2a8152093a53c | eagle3 | 384 | 58.31186991972729 | 1639.7003045455365 | 420.24447887809947 | 30.780397475737118 | 0.15119617395854826 | 0.017378967309343962 | 2.5467033779288935 | 36832260096 | 37465620480 | 88.04750261624123 | -22.480235026807392 |
| experiments/dflash_portable_deployment_20260913/measurements/l40s_qwen8b_oracle_c32_native_full_oracle_native_c32.json | l40s_qwen8b_oracle_c32_native | qwen8b | NVIDIA L40S | oracle | 32 | 32 | f61f527209d8d8c6cd3a21cb99e86b9f573987c82c0e7dffa30fbf81acf093cc | native | 256 | 61.97532063381473 | 1569.5648023446292 | 26.591447474900633 | 2.4989303875318 | 0.24568177799847035 | 0.015124288339712872 | 5.1993769470404985 | 33659981824 | 34892414976 | 102.58129227639921 | 0.0 |
| experiments/dflash_portable_deployment_20260913/measurements/l40s_qwen8b_oracle_c16_mapped_full_oracle_mapped_c16.json | l40s_qwen8b_oracle_c16_mapped | qwen8b | NVIDIA L40S | oracle | 16 | 32 | f61f527209d8d8c6cd3a21cb99e86b9f573987c82c0e7dffa30fbf81acf093cc | mapped | 256 | 110.92838976378248 | 1431.2149462338002 | 29.13887959998101 | 1.4724357129325654 | 0.15424755622916564 | 0.008164801117727351 | 5.898839841539332 | 33660899328 | 34475081728 | 231.0167892243271 | 14.219107614219206 |
| experiments/dflash_portable_deployment_20260913/measurements/l40s_qwen8b_oracle_c1_native_full_oracle_native_c1.json | l40s_qwen8b_oracle_c1_native | qwen8b | NVIDIA L40S | oracle | 1 | 32 | f61f527209d8d8c6cd3a21cb99e86b9f573987c82c0e7dffa30fbf81acf093cc | native | 256 | 156.96907840208416 | 159.1688999890714 | 263.8203820148483 | 1.030429227213972 | 0.06987729729371495 | 0.00622809477886377 | 5.186187299234001 | 33659981824 | 34156314624 | 300.17687108408415 | 0.0 |
| experiments/dflash_portable_deployment_20260913/measurements/l40s_qwen4b_oracle_c16_eagle3_full_oracle_eagle3_c16.json | l40s_qwen4b_oracle_c16_eagle3 | qwen4b | NVIDIA L40S | oracle | 16 | 32 | 9967deee5c658071054c4e0322f423543fd4dcb14b4abd40bcd2a8152093a53c | eagle3 | 384 | 73.86921712788268 | 1150.1739401757925 | 587.1790138948709 | 22.824037238242454 | 0.10352829283389535 | 0.013413291621142102 | 2.533536402376808 | 36832260096 | 37285265408 | 80.55946095056244 | -40.74871742549724 |
| experiments/dflash_portable_deployment_20260913/measurements/l40s_qwen4b_oracle_c12_eagle3_full_oracle_eagle3_c12.json | l40s_qwen4b_oracle_c12_eagle3 | qwen4b | NVIDIA L40S | oracle | 12 | 12 | 9967deee5c658071054c4e0322f423543fd4dcb14b4abd40bcd2a8152093a53c | eagle3 | 384 | 78.53255496176318 | 967.814958482137 | 712.1061665350571 | 21.23990123932769 | 0.10565491002004516 | 0.01251518145957261 | 2.562976928912491 | 36882333696 | 37224448000 | 79.264322137184 | -43.49821990073167 |
| experiments/dflash_portable_deployment_20260913/measurements/l40s_qwen8b_oracle_c4_ar_full_oracle_ar_c4.json | l40s_qwen8b_oracle_c4_ar | qwen8b | NVIDIA L40S | oracle | 4 | 32 | f61f527209d8d8c6cd3a21cb99e86b9f573987c82c0e7dffa30fbf81acf093cc | ar | 256 | 35.37959855719204 | 143.12209874581487 | 292.0583219942637 | 4.538971348725681 | 0.10651774296275107 | 0.027272813706705472 | Not measured | 36912312320 | 37100716032 | 0.0 | -72.85041025765449 |
| experiments/dflash_portable_deployment_20260913/measurements/l40s_qwen8b_learned_c12_mapped_full_learned_mapped_c12.json | l40s_qwen8b_learned_c12_mapped | qwen8b | NVIDIA L40S | learned | 12 | 12 | 560c3f266e8e6494adc69ea7efa852edce488af21f499a7adef2e8e095e69e19 | mapped | 256 | 124.28085279053425 | 1384.2743355641767 | 30.19199224188924 | 1.3061606098981429 | 0.13622604933152616 | 0.007269009956844589 | 6.005316855870096 | 36104316416 | 36616273920 | 263.38836728663273 | 15.074532418136389 |
| experiments/dflash_portable_deployment_20260913/measurements/l40s_qwen8b_learned_c12_ar_full_learned_ar_c12.json | l40s_qwen8b_learned_c12_ar | qwen8b | NVIDIA L40S | learned | 12 | 12 | 560c3f266e8e6494adc69ea7efa852edce488af21f499a7adef2e8e095e69e19 | ar | 256 | 34.20055895529101 | 394.5430027522317 | 103.9024889911525 | 4.594877856427047 | 0.11644332152172865 | 0.028045972619463615 | Not measured | 36324093952 | 36557553664 | 0.0 | -68.33290694543116 |

### eight standalone

| domain | method | source | target_adapter | mean_rollout_tps | evaluation_seconds | acceptance | mapper_training_seconds | mapper_training_gpu_hours | trainable_parameters | speedup_vs_ar | speedup_vs_native |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| math | ar | /home/yashas/spec_decode/experiments/dflash_ready_20260913/measurements/math_bfloat16_full_l40s_qwen8b_ar_part0.json | /tmp/yashas.kotre/spec_decode/model_staging/qwen3_8b_loras/gsm8k_sumit_seed0 | 37.88761694325838 | 537.2122408058494 | 1 | 0 | 0 | 0 | 1.0 | 0.2944192732739126 |
| math | native | /home/yashas/spec_decode/experiments/dflash_ready_20260913/measurements/math_bfloat16_full_l40s_qwen8b_native_part0.json | /tmp/yashas.kotre/spec_decode/model_staging/qwen3_8b_loras/gsm8k_sumit_seed0 | 128.68592644072484 | 158.79259284958243 | 4.763554926921263 | 0 | 0 | 0 | 3.396516773104223 | 1.0 |
| math | mapped | /home/yashas/spec_decode/experiments/dflash_ready_20260913/measurements/math_bfloat16_full_l40s_qwen8b_mapped_part0.json | /tmp/yashas.kotre/spec_decode/model_staging/qwen3_8b_loras/gsm8k_sumit_seed0 | 173.8756774782872 | 121.91319076903164 | 6.350408548082966 | 243.68252908997238 | 0.13537918282776243 | 83886080 | 4.589248189947882 | 1.351163116958074 |
| math | eagle_ar | /home/yashas/spec_decode/experiments/dflash_ready_20260913/measurements/qwen8b_math_eagle_ar_full_part0.json | /tmp/yashas.kotre/spec_decode/model_staging/qwen3_8b_loras/gsm8k_sumit_seed0 | 31.18540549301584 | 663.6102174082771 | Not measured | 0 | 0 | 0 | 1.0 | 0.2423373429835035 |
| math | eagle3 | /home/yashas/spec_decode/experiments/dflash_ready_20260913/measurements/qwen8b_math_eagle3_full_part0.json | /tmp/yashas.kotre/spec_decode/model_staging/qwen3_8b_loras/gsm8k_sumit_seed0 | 76.251954764408 | 269.78373852325603 | 3.7281921618204805 | 0 | 0 | 0 | 2.4451166678427536 | 0.5925430765696906 |
| codealpaca | ar | /home/yashas/spec_decode/experiments/dflash_ready_20260913/measurements/codealpaca_bfloat16_full_l40s_qwen8b_ar_part0.json | /tmp/yashas.kotre/spec_decode/model_staging/qwen3_8b_loras/codealpaca | 37.5621711228766 | 570.2265847781673 | 1 | 0 | 0 | 0 | 1.0 | 0.24599123326097652 |
| codealpaca | native | /home/yashas/spec_decode/experiments/dflash_ready_20260913/measurements/codealpaca_bfloat16_full_l40s_qwen8b_native_part0.json | /tmp/yashas.kotre/spec_decode/model_staging/qwen3_8b_loras/codealpaca | 152.6971942249105 | 144.71886774944142 | 5.617817606747496 | 0 | 0 | 0 | 4.065185522034771 | 1.0 |
| codealpaca | mapped | /home/yashas/spec_decode/experiments/dflash_ready_20260913/measurements/codealpaca_bfloat16_full_l40s_qwen8b_mapped_part0.json | /tmp/yashas.kotre/spec_decode/model_staging/qwen3_8b_loras/codealpaca | 162.1292998346589 | 142.7288188561797 | 5.631175693527081 | 241.38026973698288 | 0.13410014985387939 | 83886080 | 4.316292029666965 | 1.0617699994923002 |
| codealpaca | eagle_ar | /home/yashas/spec_decode/experiments/dflash_ready_20260913/measurements/qwen8b_codealpaca_eagle_ar_full_part0.json | /tmp/yashas.kotre/spec_decode/model_staging/qwen3_8b_loras/codealpaca | 30.316286987843572 | 691.3801501151174 | Not measured | 0 | 0 | 0 | 1.0 | 0.19853859883758018 |
| codealpaca | eagle3 | /home/yashas/spec_decode/experiments/dflash_ready_20260913/measurements/qwen8b_codealpaca_eagle3_full_part0.json | /tmp/yashas.kotre/spec_decode/model_staging/qwen3_8b_loras/codealpaca | 76.20844644900848 | 254.57602682057768 | 4.051479062620054 | 0 | 0 | 0 | 2.5137790284003794 | 0.499082166085905 |
