# Abstract and introduction source notes

Checked on 15 September 2026. Literature claims were checked against official proceedings, arXiv records, and the supplied reference PDFs. Experimental figures are reported results, not independently rerun measurements.

## Chronology

Dates below refer to initial arXiv releases. Bibliography years use the publication venue where available. Release order does not establish that a later method improves every metric or workload.

| Work | Initial release | Source |
| --- | --- | --- |
| Speculative decoding, Leviathan et al. | November 2022 | https://proceedings.mlr.press/v202/leviathan23a.html |
| Speculative sampling, Chen et al. | February 2023 | https://arxiv.org/abs/2302.01318 |
| Medusa | January 2024 | https://proceedings.mlr.press/v235/cai24b.html |
| EAGLE | January 2024 | https://arxiv.org/abs/2401.15077 |
| EAGLE-2 | June 2024 | https://aclanthology.org/2024.emnlp-main.422/ |
| EAGLE-3 | March 2025 | https://arxiv.org/abs/2503.01840 |
| PARD | April 2025 | https://arxiv.org/abs/2504.18583 |
| DFlash | February 2026 | https://arxiv.org/abs/2602.06036 |
| PARD-2 | May 2026 | https://arxiv.org/abs/2605.08632 |

## Scope of the motivation

- PARD can reuse a trained drafter across supported targets within a family. Do not claim that every target change requires full drafter retraining. None of these claims requires retraining the target LLM.
- PARD and PARD-2 train the draft model itself. Their results do not establish that a frozen DFlash transformer can directly use arbitrary new target features.
- [Huang et al., ACL 2025](https://aclanthology.org/2025.acl-long.185/) study linear transformations for transferring concept steering across LLMs.
- [Chen et al., NeurIPS 2025](https://proceedings.nips.cc/paper_files/paper/2025/hash/4569a868e7aa891248832ec08445d071-Abstract-Conference.html) use affine maps, including a bias, to transfer features, predictors, and steering vectors. These results motivate mapping. They do not prove that maps preserve a speculative drafter's acceptance or speed.
- [Spec-AUF](https://arxiv.org/abs/2607.01893) supplies the cited accept-until-fail training rule. Its support includes the first incorrect proposal and excludes later positions.

## RelaySpec claims

The method description is grounded in the existing RelaySpec draft and phase 1 and phase 2 experimental reports. The reports are evidence sources, not canonical papers.

The abstract's Qwen3-4B drafter to Qwen3-8B results come from the T1 summary in `../phase2_results.md`:

| Dataset | Autoregressive tokens/s | Native drafter tokens/s | Mapped drafter tokens/s | Reported comparison |
| --- | ---: | ---: | ---: | --- |
| MATH | 37.8373 | 219.79 | 220.26 | 5.82 times autoregressive, 100.2% of native |
| GSM8K | 37.7622 | 177.94 | 176.87 | 4.68 times autoregressive, 99.4% of native |

Throughput uses the arithmetic mean of per-request token rates. The reported evaluation uses greedy decoding in Transformers, 128 prompts per dataset, and a 2048-token generation cap. These results should not be presented as universal serving speedups. The abstract also preserves the weaker transfer on code and chat.

Contributions summarize the reported interface, draft-body and fusion comparisons, supervision budgets, shared serving, and controlled map interventions. Detailed evidence and limitations still need to be drafted in the remaining manuscript sections.

## SD² positioning update, 16 September 2026

[Berdoz et al., AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/40255), *Steering Pretrained Drafters During Speculative Decoding*, is directly related. Sections 3 and 4.1 of the [extended paper](https://arxiv.org/html/2511.09844v1) describe target-feature projections, steering inside drafter feed-forward layers, joint training, and a frozen-drafter ablation. Do not claim that linear target-feature steering or frozen-drafter adaptation is new in general. RelaySpec studies changing an existing DFlash feature interface across target variants, with maps folded into the fusion projection. No direct experimental comparison against SD² has been established here. This is a different paper from *Self-Distilled Sparse Drafters*.

## Sections 1–3 and diagram review

The DFlash, PARD-2, and RepSpec PDFs in `../relayspec_ref/` were used as structural references: a short motivation, concrete method overview, contribution bullets, task-specific preliminaries, and grouped related work. Manuscript prose is original.

Additional verified sources:

| Key | Source and status | Use |
| --- | --- | --- |
| zhou2024distillspec | https://proceedings.iclr.cc/paper_files/paper/2024/hash/8766fbc68e1ed1cdef712ce273e0a363-Abstract-Conference.html , ICLR 2024 | Distillation for draft–target agreement and training-objective choices. |
| liu2024online | https://arxiv.org/abs/2310.07177 , arXiv version | Updating the drafter using serving feedback. |
| cheng2024redrafter | https://arxiv.org/abs/2403.09919 , arXiv version | Target-conditioned recurrent drafting, beam search, dynamic tree attention. |
| zhang2019rmsnorm | https://proceedings.neurips.cc/paper/2019/hash/1e8a19426224ca89e83cef47f1e7f53b-Abstract.html , NeurIPS 2019 | Definition of RMSNorm. |
| huo2026repspec | https://proceedings.iclr.cc/paper_files/paper/2026/hash/d70ea003729b440b89a2f958a5554c1f-Abstract-Conference.html , ICLR 2026 | Training-time linear structures merged before inference. |

The five-stream native fusion and the 4096-to-2560 map dimensions are checked against `../phase2_results.md`, section “T1 pruned sweep: method and interpretation”, and `../phase1_probe.md`, section “What exactly was adapted?”. The diagram uses column vectors: H_i has shape d_t by n and W_i has shape d_r by d_t. A single position uses W_i h_i. F_i has shape d_d by d_r. All arrow segments in the generator are checked to be horizontal or vertical.

The token vocabulary assumption applies to the displayed speculative-sampling equation. The Llama-to-Qwen transfer uses the separate tokenizer bridge documented in the T4 report. Greedy bridge measurements do not establish stochastic distribution preservation. Reference features used as reconstruction targets are denoted H_i^(r), distinct from new-target inputs H_i.

## Evidence-based rewrite after objective and probe review

Consulted both report PDFs through `tmp/pdfs/phase1.txt` and `tmp/pdfs/phase2 (1).txt`, plus all four results/probe Markdown reports.

- AUF motivation: Phase 1 PDF discussion of expected accepted-prefix length and `phase1_probe.md` section “What AUF taught these maps”. Later correct predictions can lower CE without extending the prefix. This is motivation, not a proof of optimal throughput.
- Same-model versus cross-model motivation: Phase 2 PDF opening and `phase2_probe.md` section 11. Identity preserves the same-model starting function. Random rectangular maps need to learn inputs usable by the frozen drafter.
- Compression and direction: `phase1_probe.md` sections 5 and 9. Whole-map truncation destroys the identity path. Direction-only correction retains most acceptance gains in the tested GSM8K and NanoCoder checkpoints. Very low-rank residual approximations do not fully preserve the learned function.
- LoRA shift: `phase1_probe.md` section 10 and “Paired-feature geometry and optimization”. Aligned versus shuffled shift cosines do not establish token-specific reversal. This is not proof that no useful information is recovered.
- Cross-size intervention and refinement: `phase2_probe.md` sections 6, 7, and 11. Midpoint maps improve the tested Math and Code endpoints. Warm AUF improves GSM8K and weakens other tested workloads. The recipes differ in initialization, coverage, and optimizer settings. No universal loss ranking is claimed.
- Performance scope: `phase1_results.md` architecture/objective and pooled-serving comparisons, and `phase2_results.md` selected T1 standalone table. Keep mean per-request, pooled, and aggregate throughput separate.
- Added `lin2026eda`: https://arxiv.org/abs/2603.09527 , verified preprint metadata and shared/private adaptation description.
- Added `ramakrishnan2025omnidraft`: https://proceedings.neurips.cc/paper_files/paper/2025/file/3c2fe1417eed1c6ff9acf169617981ea-Paper-Conference.pdf , NeurIPS 2025. Cross-vocabulary cache and online adaptation.

Root AGENTS.md now requires checking the supplied report PDFs and Markdown evidence when drafting technical claims, and following the reference papers' structure with original, simple-English prose.


## Method draft and main results figure

- Figure 1 reads the selected `maps_full` AUF rows (4,096 examples, 32 anchors) in `phase1_results.md` and the completed T1 standalone AR-baseline table in `phase2_results.md`. Both panels use ratios of arithmetic mean per-request token rates. The LoRA baseline is unchanged DFlash with the adapted target. The transfer baseline is autoregressive Qwen3-8B. Native 8B drafter results and weaker Code/Chat transfer are displayed. No repeated-run uncertainty is available in these aggregate rows. Plotted values and provenance are saved in `figures/speedups_data.json`.
- Method equations and data preparation were checked against both report PDFs and all four Markdown reports. AUF uses a detached exclusive prefix mask, including the first failure, with local microbatch normalization. MSE averages five relative feature errors and adds normalized fused-context error, with epsilon 1e-6. Positions are averaged within each example before the example mean.
- Initialization, frozen components, 15 proposal positions, causal feature access, block attention, Xavier-uniform cross-model initialization, and BF16 folding follow the selected report recipes. Distinguish the selected MSE recipe from the cold token-loss and warm-start probe recipes.
- The architecture diagram is Figure 2 in Method. The title page uses normal template dimensions, with top floats deferred so Figure 1 can appear on page 2. No margin or style-file edits were made.

## Method presentation order

Following the organization of RepSpec Section 3.1, Method now explains the procedure, defines the map and fusion dimensions, and derives the folded computation before presenting the architecture figure. AUF and MSE training details follow. Parameterization was rechecked against Phase 1 and Phase 2 reports and their Markdown evidence. This is a presentation change, not a new experimental claim.

## Method compression and layout

Removed repeated Method explanations and shortened the architecture caption while retaining the equations, dimensions, frozen components, initialization, loss normalization, and numerical qualifications. Replaced forced figure placement with a normal float after the architecture explanation. This avoids stretching the preceding page to accommodate an unbreakable figure. Method source word count decreased from 726 to 566 by whitespace token count, including LaTeX. The rebuilt draft is eight pages including references. Official margins, fonts, and style files are unchanged.

## Method rewrite for standalone readability

Rewrote Method as connected explanations of the five maps, combining their weights with fusion before inference, AUF training for a LoRA-adapted target, and MSE reconstruction for a different target. Rechecked both report PDFs and all four Markdown reports. Removed deployment qualifications and numerical-export details from the core explanation. Moved initialization into the relevant training subsection and retained the correction definition inline. Preserved the AUF first-failure rule and local normalization, and the MSE relative-error terms and per-example averaging.

## Inherited architecture choices and evaluation scope

DFlash Section 5.5.3 and Table 7 discuss five target features, distinct from the number of draft transformer layers. Its block-size experiments and Appendix A.5.2 document 16-token training blocks. RelaySpec retains one map per inherited feature input and one known token plus 15 masked positions for its main Qwen setup. Phase 2 Appendix A explicitly records the Llama checkpoint exception: pretrained block size 10, evaluated at 16. Phase 2 Appendix D explicitly identifies EAGLE-3 as an external native checkpoint, not a transferred EAGLE mapper. The manuscript therefore includes EAGLE as a baseline without claiming demonstrated RelaySpec adaptation to EAGLE. Model and workload coverage follows both reports and their results tables.

## Current Method and provisional experiment values

The newly supplied `RelaySpec_MSE_Paper-single-layer-old-paper.pdf` documents actual RelaySpec-adapted EAGLE-3 experiments. This corrects the earlier source inventory, which only had native EAGLE baselines in the Phase reports. Its Sections 4, 5.2, 5.3 and Appendix A describe a single learned projection over concatenated features, not one target layer. EAGLE MATH-500 throughput is 84.92/59.18 tokens per second for 8B/14B, retaining 90.2/89.1 percent of native throughput. Its Table 44 reports dense/factored/MLP throughput 198.09/196.25/190.15 on 128 MATH development questions, with 2,048 fitting records and 8,192 updates.

At the user's request, the manuscript describes the current per-input-map design only. These earlier numerical values are retained solely as starred draft placeholders in Experiments, with an explicit source/configuration note. They must not be interpreted as current five-map or new per-input EAGLE measurements. The pending studies replace these cells, and no current linear-over-MLP conclusion is asserted. Older-method exposition and its appendix were removed. Provenance remains here for replacement and audit.

Method now includes the block-matrix identity establishing equivalence of separate maps and combined projection in exact arithmetic, the identity-initialization argument, and the conditional induction argument for greedy verification. None proves speedup or linear superiority. The proposed EAGLE extension uses three input maps and its existing autoregressive proposal procedure. The currently running EAGLE implementation and exact training support must be checked against its eventual experiment records.

## Citation audit for named techniques

Added original-source citations at the relevant Method statements for Xavier-uniform initialization (Glorot and Bengio, AISTATS 2010), GELU (Hendrycks and Gimpel, 2016 preprint), LoRA (Hu et al., ICLR 2022), and RMSNorm (Zhang and Sennrich, NeurIPS 2019). Existing citations cover DFlash's features and block construction, EAGLE-3's features, the borrowed AUF objective, and speculative verification. Block-matrix multiplication and identity initialization are derived directly, not attributed to an unrelated work. RelaySpec's particular relative-MSE objective remains grounded in the experimental reports.

Primary sources verified in this audit:
- Xavier: https://proceedings.mlr.press/v9/glorot10a.html
- GELU: https://arxiv.org/abs/1606.08415
- LoRA: https://arxiv.org/abs/2106.09685
- RMSNorm: https://arxiv.org/abs/1910.07467
- Qwen3: https://arxiv.org/abs/2505.09388
- GSM8K: https://arxiv.org/abs/2110.14168
- MATH dataset: https://arxiv.org/abs/2103.03874

Also added model/dataset citations where Qwen3, GSM8K, and MATH appear in prose. These identify external assets, not evidence for RelaySpec's own measured performance. All 26 used bibliography keys resolve, with no duplicate keys or LaTeX warnings.

## Final DFlash Method clarifications

Clarified one map per corresponding feature input, shared over token positions. The frozen embeddings and vocabulary head belong to the original drafter, as documented in Phase 2's architecture description and Phase 1's parameterization. Defined the original fused context explicitly. Paired MSE supervision uses identical saved prompt/response text, not separately generated responses, and selected valid positions include prompt and response positions. The average feature-error term and context-error term have equal weight. Added prompt/verification feature reuse and removal of rejected speculative cache positions, consistent with the recorded generation loop in the earlier manuscript. The EAGLE paragraph was left unchanged pending its new experiment records.

## Report domain names differ between phases

The two phases use the same domain strings for different datasets. In `../phase1_results.md`, section "Complete matched numerical exports", domain `math` denotes GSM8K. In `../phase2_results.md`, domain `math` denotes the MATH dataset and domain `gsm` denotes GSM8K. A row copied from one phase into a table labelled by the other phase's convention would therefore report the wrong dataset.

Always resolve the raw domain string to a dataset name before writing it into the manuscript. Phase 1 uses `math` for GSM8K and `kicad` for KiCad. Phase 2 uses `math` for MATH, `gsm` for GSM8K, `code` for its pinned LiveCodeBench releases and `chat` for its UltraFeedback-derived instructions. The Phase 2 code and chat workloads are not HumanEval or MT-Bench.

`figures/draw_speedups.py` now holds these two mappings explicitly, records both the raw `domain` and the resolved `dataset` for every plotted row in `figures/speedups_data.json`, and labels both panels from the resolved names instead of a separate hardcoded list. Figure 1 panel (a) therefore reads Phase 1 `math` and prints GSM8K, and panel (b) reads Phase 2 `math` and prints MATH. The plotted values and the figure are unchanged by this correction. An unrecognized Phase 2 domain now fails the render instead of being plotted under a wrong label.

## Adaptation cost against published drafter training

The DFlash paper reports its training data and schedule but no training time, no batch size and no hardware budget, and it runs its experiments on H200 GPUs. A GPU-hour ratio against our L40S fits is therefore not derivable from published numbers and would not be comparable across that hardware. Section~5.1 instead compares how much supervision each recipe consumes, and reports measured time only for our own fits. The manuscript states this limitation in the text rather than estimating the missing figure.

DFlash recipe values are taken from its Section 5.1 and Appendix A.1 in `../relayspec_ref/2602.06036v2.pdf`: a mixture of about 800,000 samples, six epochs, 512 anchor positions randomly sampled per sequence at every epoch, five draft layers for Qwen3 and block size 16. The paper's own wording is "around 800K samples", so the derived ratios are stated as approximate.

Derived scale quantities, computed only from the two published recipes:

| Recipe | Unique examples | Sequence passes | Anchor blocks |
| --- | ---: | ---: | ---: |
| DFlash drafter training | 800,000 | 800,000 times 6 = 4,800,000 | 4,800,000 times 512 = 2,457,600,000 |
| RelaySpec AUF maps, LoRA target | 4,096 | 4,096 times 1 = 4,096 | 4,096 times 32 = 131,072 |
| RelaySpec MSE maps, cross-model | 16,384 | 16,384 times 3 = 49,152 | Not applicable |

The selected LoRA checkpoint makes one pass, not three. Phase 1 studies 31 and 38 both record one epoch over the specified cohort with 512 optimizer updates at 4,096 presentations, and study 33 repeats the same recipe for its reevaluation. Only the cross-model T1 recipe uses three epochs, with 6,144 updates at 16,384 examples. An earlier version of this entry applied three epochs to both settings and understated the LoRA ratios.

Sequence-pass ratios are 1,171.9 for the LoRA setting and 97.66 for the cross-model setting. The LoRA anchor-block ratio is 18,750. Counted as unique examples instead, the ratios are 195.31 and 48.83. The manuscript rounds these to about 1,170, about 18,750, about 98, about 195 and about 49. The abstract and introduction quote the unique-example ratios, which do not depend on the epoch count. Anchor blocks do not apply to the reconstruction objective because it supervises positions directly rather than sampling anchors.

GSM8K holds exactly 4,096 unique cached questions and repeats only beyond that point, so 4,096 presentations are 4,096 unique examples. KiCad has up to 15,366 unique cached examples, so its 4,096-presentation cell is also fully unique.

Measured fitting times are the selected checkpoints' optimizer-loop wall time on two L40S GPUs, from the `ablation` and `scaling` exports in `../phase1_results.md` and the selected T1 recipe table in `../phase2_results.md`. GSM8K is 159.8 seconds and 0.089 GPU-hours, KiCad is 319.3 seconds and 0.177 GPU-hours, and T1 is 24.91 minutes and 0.8303 GPU-hours. Phase 1 fitting ran on node07 L40S. Feature capture for the 16,384 T1 records is 0.9238 GPU-hours from the shared extraction table, which one cache amortizes across several fits.

Two qualifications are stated beside the table and must be retained. The supervision ratio understates the compute difference for the cross-model setting, because DFlash trains draft weights while our reconstruction objective passes no gradient through the draft transformer. The ratio also overstates our generality, because the published recipe yields one drafter that serves its target across workloads while our LoRA maps are fitted per workload and do not transfer, as the bank-swap probe in `../phase1_probe.md` section 8 records. No claim is made that RelaySpec replaces drafter training, and no cost advantage is claimed over the matched draft-body control, which fits in a comparable time.


## Figure 1 caption reduction

Shortened the caption to the two panel results, the warning that the panels use different baselines, the fitting time, and a pointer to Section~5. The evaluation settings it previously carried moved to a lead-in paragraph in Experiments: single requests, greedy decoding, Transformers, L40S, 128 held-out prompts per workload, mean per-request token rates, and response caps of 2,048 tokens with 8,192 for KiCad. Those settings are recorded in Phase 1 study 33 under "Models and execution" and in the Phase 2 T1 evaluation contract. The caption states fitting as under six minutes in panel (a), which covers the 159.8-second GSM8K and 319.3-second KiCad checkpoints, with the exact values in Table 1. The caption no longer restates that bars are not shared-serving speedups, because the lead-in paragraph now defines the throughput statistic for the whole section.

## New MLP and EAGLE-3 measurements, 16 September 2026

Sources are `../new_phase1.md` and `../new_phase2.md`, both complete 128-request cohorts regenerated from saved per-request JSON by `scripts/report_extensions_20260916.py`.

### Figure 1 rebased onto the new snapshots

Panel (a) previously used the `ablation` and `scaling` exports in `../phase1_results.md`, which report the selected checkpoints as plus 37.58 percent on GSM8K and plus 89.52 percent on KiCad. The new snapshot re-evaluates the same checkpoints, confirmed identical by their 32,768,000 trainable parameters and their 2.663 and 5.321 minute fits, and reports 1.3951 and 1.8547 times unchanged DFlash. That is plus 39.51 and plus 85.47 percent. The two runs differ only in evaluation, which is the run-to-run timing variation the Phase 1 probes document. The manuscript now uses the new snapshot throughout so that Figure 1, the EAGLE-3 table and the nonlinear-map table share one cohort. Dividing the rounded tokens-per-second columns instead gives 85.46 percent, so the figure and text take the snapshot's own ratio column, which is computed from full-precision data.

Panel (b) already agreed with `../new_phase2.md` and changed only in the fourth decimal of the native MATH ratio, from 5.8088 to 5.8087. `figures/draw_speedups.py` now reads both panels from the two new documents and parses their ratio columns directly instead of re-deriving them. Its table match requires the multiplication sign in the ratio cell, because the token-count tables have the same column count.

### Nonlinear map comparison

The measured alternative is a parallel branch, $g_i(h)=W_i h+B_i\operatorname{SiLU}(A_i h)$, with $A_i$ reducing to 256 coordinates and $B_i$ starting at zero, so training begins at exactly the linear solution. It is **not** parameter matched: it adds parameters, 32.8 to 39.3 million for a LoRA-adapted target and 52.4 to 60.9 million for cross-model transfer. The manuscript says "more parameters" and never "parameter matched". This replaces the earlier proposed dense versus factored versus GELU comparison, which was never run.

Across the seven matched cells the nonlinear arm changes throughput between minus 2.07 and plus 2.97 percent, favoring the linear map in five of seven, with acceptance deltas between minus 0.035 and plus 0.154. Both source documents warn that small throughput differences without acceptance gains are not a robust architectural improvement, so the manuscript claims only that this nonlinear interface does not pay for its extra computation here.

The SiLU activation is defined inline as $x\sigma(x)$ rather than cited, because no primary source for it has been verified in this workspace. Add a citation after verifying one. The GELU entry `hendrycks2016gelu` is now uncited and harmless, retained in case a GELU arm is measured later.

### EAGLE-3 results and the Method correction

The Method EAGLE paragraph previously stated that maps for a LoRA-adapted target start at identity and receive token supervision. `../new_phase1.md` shows the implementation uses normalized layer plus post-RMSNorm context MSE at 100 percent position coverage, with source features from base Qwen3-4B. Method now states reconstruction for both EAGLE settings and notes that this differs from the DFlash LoRA recipe. Initialization follows width: identity for the square 2,560 to 2,560 maps, Xavier for the rectangular 4,096 to 2,560 maps. Trainable parameters are 19,660,800 and 31,457,280. The three taps are HF hidden-state indices 2, 18 and 33. The tap indices and parameter counts are held for Experiments or the appendix rather than Method, to protect the page budget.

Against native EAGLE, cross-model transfer gains 8.05, 6.63, 2.57 and 2.57 percent on MATH, GSM8K, Code and Chat. LoRA targets are mixed: plus 0.99 on GSM8K, plus 6.60 on KiCad, minus 7.87 on NanoCoder. EAGLE acceptance uses upstream emitted-token counters and is not comparable with the DFlash rows, and EAGLE's absolute throughput is far below DFlash's for reasons the documents attribute to different draft bodies and proposal structures, not to the maps.

### Preliminaries and Method generalized to K streams

Because EAGLE-3 now carries measured results, Preliminaries defines $K$ feature streams with $K=5$ for DFlash and $K=3$ for EAGLE-3, and Method's context, folding and reconstruction equations use $K$. The leftover superscript in the original fused-context equation was corrected from $c^{(r)}_t$ to $c^{(o)}_t$, matching the agreed preference for "original target" over "reference model".

### New cost figures

`../new_phase1.md` and `../new_phase2.md` independently confirm the fitting costs already in Table 1: 0.0888 GPU-hours for GSM8K, 0.1774 for KiCad and 0.830 for cross-model. Not yet used in the manuscript: EAGLE fits are much cheaper because no gradient reaches the draft body, at 0.247 to 3.734 minutes in Phase 1 and 13.132 minutes in Phase 2, but EAGLE needs fresh paired capture costing 8.43, 57.88 and 9.00 minutes of wall time for GSM8K, KiCad and NanoCoder. The nonlinear arm costs about the same to fit as the linear one. The cross-model AUF comparator at 8,192 examples cost 1.0548 GPU-hours, more than the selected MSE recipe.

## Experiments section cleared for redrafting, 16 September 2026

At the user's request, `sections/experiments.tex` is empty again. The drafted adaptation-cost, EAGLE-3 and linear-versus-nonlinear subsections were removed pending a decision on the section's structure. The preceding entries retain every number, source and derivation, so those subsections can be rebuilt from this log plus `../new_phase1.md` and `../new_phase2.md`.

Three cross-references into that section were removed so the build stays free of undefined references: the cost pointer in the introduction's second contribution bullet, the nonlinear-map pointer in Method, and the evaluation-settings pointer in the Figure 1 caption. The claims themselves were kept in the abstract, introduction and Method, so they currently have no supporting section in the main text and must be reconnected when Experiments is redrafted. The shared evaluation settings that had moved out of the Figure 1 caption into the Experiments lead-in paragraph are no longer stated anywhere in the manuscript and need a home in the new section.

## Experiments section drafted, 16 September 2026

Section 5 now has nine subsections in the order: setup, fine-tuned targets, replaced targets, where the trained parameters belong, which objective, how much supervision, shared serving, generality, and cost. The order follows DFlash's pattern of main results before design ablations and PARD-2's separation of the two adaptation settings, with each subsection closing in an explicit statement of what it supports.

Claim-to-source map for every table and figure:

| Manuscript object | Source |
| --- | --- |
| Table 2, fine-tuned targets, Qwen3-4B rows | `../new_phase1.md` results table, 5W AUF and DFlash native, plus its saved AR comparison |
| Table 2, Qwen3-8B rows | `../phase1_results.md`, "eight standalone" export, domains math and codealpaca |
| Table 3, cross-model versus AR | `../phase2_results.md`, completed standalone AR baselines, Mapped/AR column |
| Table 4, native recovery | same table, mapped mean TPS divided by native mean TPS. T3 is absent because its Llama3.2-3B target has no published native drafter, stated in the caption rather than left blank |
| Table 5, placement for a fine-tuned target | `../phase1_results.md` `ablation` export, AUF rows, gain versus native |
| Table 6, placement for T1 | `../phase2_results.md`, "Selected T1 architecture results", 8,192-example three-epoch fits |
| Table 7, objective comparison | `ablation` export for the upper block, T1 architecture table for the lower block |
| Figure 3, supervision scaling | `../phase1_results.md` `scaling` export, AUF rows only, parsed by `figures/draw_scaling.py` |
| Figure 4, shared serving | `../phase1_results.md` `serving` export, trial `peak_clients_trial0`, oracle policy, parsed by `figures/draw_serving.py` |
| Table 8, nonlinear maps | `../new_phase1.md` and `../new_phase2.md` |
| Table 9, cost | `../new_phase1.md` and `../new_phase2.md` fitting-cost tables |

Both new figure scripts parse the reports at render time and assert grid completeness, so a change in report shape fails the render instead of plotting stale values. Plotted values and provenance are saved to `figures/scaling_data.json` and `figures/serving_data.json`.

Distinctions preserved in the prose, each of which a reviewer could otherwise challenge:

- The T1 architecture and objective tables use the 8,192-example three-epoch fits, which are not the selected 16,384-example 25 percent recipe used in Figure 1 and Table 3. The selected recipe reaches 1.0022 on Math where the 8k MSE50 fit reaches 0.9986.
- The direct fusion control has the same 52.4M parameters as the five maps and the same initial function, so its collapse to 0.365 of native under a matched objective is attributed to the parameterization, not to capacity.
- BA cells are low-rank replacements of each whole map, not low-rank corrections added to it, so their degradation is not evidence about correction rank.
- The draft-body control uses a different backbone in the T1 setting and wins on NanoCoder in the LoRA setting, so no general claim of input adaptation over body adaptation is made.
- Reconstruction runs use learning rate 1e-3 and token runs 1e-4, so the objective comparison is between recipes, not losses at one setting.
- The Qwen3-8B serving sweep shows mapped aggregate throughput below native at 16 and 32 clients. This is reported as a boundary in the same paragraph as the 4B gains.
- EAGLE runs do not return identical sequences, unlike the DFlash comparisons, and the fine-tuned EAGLE maps are reconstruction-trained rather than token-trained. Both are stated where the numbers appear.

Per-column completeness: no table contains a pending or blank cell. Autoregressive absolute throughput was moved out of Tables 2 and 3 into their captions to keep both within the text width, and the missing T3 native drafter and the missing concurrent AR controls are handled in prose.

Citations still to add before submission: Llama 3, LiveCodeBench, UltraFeedback, NuminaMath and CodeAlpaca are named without citations, matching the existing treatment of KiCad and NanoCoder. A primary source for SiLU is also still needed.

## EAGLE-3 token supervision replaces reconstruction for fine-tuned targets, 16 September 2026

Source is `../eagle_auf_results.md`, a completed run covering three fits and all 384 evaluation requests. This supersedes the fine-tuned EAGLE-3 rows previously taken from `../new_phase1.md`, which trained those maps by reconstruction toward base Qwen3-4B features.

The earlier gap is now closed. The Method EAGLE-3 subsection had been corrected to say reconstruction for both settings, because that was what had been run. It now states token supervision for a fine-tuned target and reconstruction for a replaced one, matching both the implementation and the DFlash pairing.

Measured against the unadapted EAGLE-3 drafter on the same adapted target and prompts:

| Workload | Unadapted | Token-supervised maps | Ratio | Acceptance | vs AR |
| --- | ---: | ---: | ---: | --- | ---: |
| GSM8K | 71.19 | 120.35 | 1.6906 | 3.3466 to 5.6951 | 3.8217 |
| NanoCoder | 81.77 | 119.30 | 1.4588 | 3.8519 to 5.6677 | 3.7517 |
| KiCad | 73.82 | 151.86 | 2.0570 | 3.7944 to 7.3632 | 4.9997 |

The same three maps trained by reconstruction reach 71.89, 75.34 and 78.70 tokens per second, so token supervision is 67.41, 58.35 and 92.96 percent faster. The manuscript presents this as an independent replication of the objective-to-setting pairing on a second drafter family, and notes that the reconstruction variant falls below the unadapted drafter on NanoCoder. The report itself warns that the two arms differ in objective and supervision workload and are not a controlled change of one loss term, which the manuscript states.

Architecture and recipe now recorded in Method: three square maps $W_i\in\mathbb R^{2560\times2560}$ at identity, 19,660,800 trainable parameters, taps at zero-based indices 2, 18 and 33, folded export $[F_1W_1\;F_2W_2\;F_3W_3]$. The objective is a recurrent adaptation of accept-until-fail, not DFlash's masked-block implementation: up to eight successive predictions per anchor, a detached prefix-correctness weight that includes the first wrong token, a saved token continuing a path only where it matches the greedy prediction, an in-head vocabulary mask so an out-of-head target token ends a path without contributing a loss term, and per-example normalization by that example's supervised-position count. Method states each of these. One epoch, 512 updates, 4,096 examples per LoRA, up to 32 anchors, AdamW at 1e-4, seed 42, two L40S.

Fitting cost in Table 9 updated from the reconstruction figures to 2.73 minutes and 0.091 GPU-hours for GSM8K and 19.16 minutes and 0.639 GPU-hours for KiCad. The cost paragraph no longer claims EAGLE-3 fits are cheapest because reconstruction skips the draft body, since the fine-tuned EAGLE-3 fits now backpropagate through the frozen drafter. It instead attributes the per-example saving to the reconstruction objective and notes that these token-supervised fits reused an existing feature cache and needed no new extraction.

These runs are single timing passes with no repeated-run intervals, and their returned-token counts differ from the controls, so throughput and acceptance are reported rather than summed latency.

## Citations added for named assets

Added and cited: Llama 3 (`dubey2024llama3`), LiveCodeBench (`jain2024livecodebench`), UltraFeedback (`cui2023ultrafeedback`), NuminaMath (`li2024numinamath`), CodeAlpaca (`chaudhary2023codealpaca`) and SiLU (`elfwing2018silu`). The unused GELU entry was removed because the measured nonlinear arm uses SiLU. All 31 remaining entries are cited and resolve with no LaTeX warnings.

These six entries were written from model knowledge rather than fetched, because this session had no verified network retrieval for them. Their titles, authors, years and identifiers must be checked against the primary records before submission, in the same way the earlier entries in this log were verified. KiCad and NanoCoder remain internal domain names with no external citation.

## Citation verification, 16 September 2026

The six entries added from model knowledge were checked against primary records. Three were correct, two were wrong and one had minor errors. All are now corrected and the flag on them is lifted.

| Key | Status | Correction |
| --- | --- | --- |
| `grattafiori2024llama3` | Was wrong | The arXiv v3 author list is led by Aaron Grattafiori, not Abhimanyu Dubey. Entry and key renamed, and the citation in Experiments updated. Verified at https://arxiv.org/abs/2407.21783 |
| `cui2024ultrafeedback` | Was wrong | The subtitle is "Boosting Language Models with Scaled AI Feedback", not "with High-quality Feedback". Three authors were missing, Bingxiang He, Ruobing Xie and Yankai Lin, and the order was wrong. The venue is ICML 2024, so the entry is now an inproceedings and the key year changed from 2023. Verified at https://arxiv.org/abs/2310.01377 |
| `li2024numinamath` | Minor errors | Aligned to the dataset's own BibTeX: "Shengyi Costa Huang" rather than "Shengyi Huang", publisher Numina, and the canonical URL https://huggingface.co/AI-MO/NuminaMath-CoT |
| `jain2024livecodebench` | Correct | Title, all ten authors in order, year and arXiv id confirmed at https://arxiv.org/abs/2403.07974 |
| `chaudhary2023codealpaca` | Correct | Matches the repository's own citation block at https://github.com/sahil280114/codealpaca |
| `elfwing2018silu` | Correct | Neural Networks volume 107, pages 3--11, 2018, confirmed. The arXiv preprint is 1702.03118 from 2017 |

The final build resolves all 31 bibliography entries with no undefined citations and no BibTeX warnings.

Unit convention: fitting time is now reported in minutes everywhere. The introduction previously said 159.8 seconds where Table 9 said 2.66 minutes. The introduction now says 2.66 minutes, matching the table exactly.

## Experiments rewritten in the reference-paper structure, 16 September 2026

Section 5 now follows the organization used by PARD-2 Section 4 and RepSpec Section 4: setup, main results, ablation studies, then generalization, serving and overhead. Setup uses their bold lead-ins (Models, Datasets and benchmarks, Metrics, Baselines, Implementation details), main results lead with two large speedup-and-acceptance tables, and every ablation closes with an explicit statement of what it shows.

The internal transfer labels T1 to T4 were removed from the manuscript. Transfers are now named by the target they accelerate and the model their drafter came from: Qwen3-8B from Qwen3-4B, Qwen3-4B from Qwen3-8B, Llama3.2-3B from Llama3.1-8B, and Llama3.1-8B from Qwen3-4B. The labels remain only in this log and in the reports.

Main tables now carry both speedup over autoregressive decoding and average acceptance length for every cell, following PARD-2 Table 1. The best entry in each row or pair is bold in every table. In the cross-model table that means the native drafter is bold on most cells, which is the honest reading: RelaySpec's claim there is recovery without target-specific drafter training, not superiority. Llama3.2-3B appears as a RelaySpec-only row because it has no published native drafter, which removes a row rather than leaving cells blank.

### Harmonic mean as the selection criterion

Harmonic mean is used only where an ablation must choose one configuration, namely the placement ablation and the objective ablation. It is not reported in the main results, the capacity ablation, the EAGLE-3 table or the cost table. Values are computed on speedup ratios.

| Selection | Candidates and harmonic means | Chosen |
| --- | --- | --- |
| Objective, fine-tuned target | Accept-until-fail 1.3762, decaying cross-entropy 1.3530 | Accept-until-fail |
| Objective, replaced target | MSE100 0.9214, MSE50 0.9186, decaying cross-entropy 0.6226, accept-until-fail 0.6042 | Reconstruction |
| Placement, fine-tuned target | Five maps 1.376, draft body 1.355, direct fusion 1.265, joint map 1.244, rank 128 0.999, rank 56 0.751, rank 28 0.565 | Five per-stream maps |

The fine-tuned objective case is the one that justifies the criterion in the manuscript. On the arithmetic mean the two token objectives are nearly tied, 1.4505 against 1.4449, because decaying cross-entropy's strong KiCad result offsets its weak NanoCoder result of 1.031. The harmonic mean separates them because it is pulled down by the smallest entry. Among the two token objectives the ordering reverses between settings, with decaying cross-entropy ahead for the replaced target at 0.6226 against 0.6042, which the manuscript states so that neither token objective is presented as dominant.

### Other corrections in this pass

Bolded cells written as `\textbf{$+37.6$}` do not render bold, because the emphasis does not cross into math mode. Three such cells were converted to `$\mathbf{+37.6}$`. The Method cross-reference to the capacity ablation was repointed from the removed `sec:generality` label to `sec:ablation-capacity`, and the capacity table is now referenced from its own prose. The build has no undefined references and every table and figure is referenced.

## Experiments consolidated into one main table, 16 September 2026

All single-request results now live in one table of 21 rows covering both settings, every target and every workload, with speedup over autoregressive decoding, acceptance length, the matching reference drafter and the ratio to it. Llama3.2-3B has no published native drafter, so its three reference cells carry the spanning text "none published" rather than being left blank, which keeps the no-empty-cell rule while showing that the row is a real measurement.

Targets are written $A \leftarrow B$, meaning target $A$ accelerated by a drafter built for $B$. The internal transfer labels T1 to T4 no longer appear anywhere in the manuscript.

Each reference drafter is now named and defined where it is introduced. The \emph{unadapted drafter} is the released checkpoint run against a fine-tuned target with no adaptation. The \emph{native drafter} is one trained for the target being accelerated. The four placement-ablation arms are defined in an explicit list before their table: single fused projection, draft-transformer LoRA, single joint map and rank-$r$ maps. The phrase "draft body" no longer appears without saying that it means a LoRA on the attention and MLP projections inside the draft transformer.

Figure 3 now covers both settings in three panels: fine-tuned scaling in examples, fine-tuned scaling in anchors, and transfer scaling in examples. The transfer panel is new and reads the MSE50 coverage table in `../phase2_results.md` for the 4{,}096, 8{,}192 and 16{,}384-example three-epoch fits. Recovery of the native drafter rises monotonically on all four workloads, from 97.6 to 99.9 percent on Math and 75.2 to 84.8 percent on Code, so the workloads that transfer worst are also the ones with the most headroom. `figures/draw_scaling.py` asserts both grids are complete and saves the plotted values to `figures/scaling_data.json`.

Whitespace was caused by float placement. All six tables and both figures now use `[t]`, which removed the sparse pages. Body pages now carry between 584 and 802 words each, where the previous layout had a 309-word page and large mid-page gaps.

Checks on the final build: no undefined references, no unreferenced tables or figures, no overfull boxes, no LaTeX or BibTeX warnings, no em dashes and no prose semicolons.

## Main table pinned ahead of the ablations, 16 September 2026

The 21-row main results table was floating past the ablation subsections and landing beside the placement-ablation table. Two changes fix the ordering. The table is now declared immediately after the opening sentence of the main-results subsection rather than after its two discussion paragraphs, and `placeins` is loaded so a `\FloatBarrier` at the end of that subsection prevents any main-results float from drifting into the ablations.

`placeins` controls float placement only. It does not touch geometry, fonts, margins or the ICLR style files, so the template remains unmodified in every respect the venue specifies.

Resulting order: the main table sits at the top of the page carrying the main-results subsection, the ablation subsections follow, and the placement-ablation table appears with them. Body pages carry between 584 and 802 words, so the barrier introduced no new whitespace.

## Introduction figures and section sync, 17 September 2026

Sections 1, 2, 3 and 6 were copied from `../yashas_relayspec_iclr2027/sections/`. The Introduction was then changed only to replace its architecture figure with the second autoregressive comparison, remove the obsolete architecture-figure reference, link both new figures in the prose, and update the GSM8K and KiCad percentages to match the current 128-prompt results. The Method retains the research partner's text apart from removal of its reference to the replaced architecture figure. Experiments was not copied or rewritten.

Figures 1 and 2 use the `Saved AR comparison` tables in `../new_phase1.md` and `../new_phase2.md`. Figure 1 uses the DFlash native and five-map AUF rows for GSM8K, KiCad and NanoCoder. Figure 2 uses the DFlash native and five-map MSE25 rows for Math, GSM8K, Code and Chat. Each pair shares the saved autoregressive mean throughput for the same target and dataset. The plotted metric is the ratio of arithmetic mean per-request throughput to that autoregressive mean. `figures/draw_speedups.py` checks the saved ratios against the corresponding throughput values and writes `figures/speedups_data.json`.

## Related-work folding comparison, 17 September 2026

RepSpec's ICLR 2026 paper adds redundant linear structure during draft-model training and merges it into the backbone for inference. This is directly relevant to RelaySpec's foldable training-time maps, but RepSpec changes the drafter's training architecture while RelaySpec trains maps before the existing fusion projection and freezes the draft transformer. Huang et al. (ACL 2025) and Chen et al. (NeurIPS 2025) study linear or affine mappings between model representations for steering vectors, probes, and features. They do not evaluate reuse of a speculative drafter, so the short Related Work section omits them. Folding RelaySpec maps eliminates a separate mapping operation. It does not establish equal total inference cost across source and target model widths.

## Method and DFlash background consolidation, 17 September 2026

The separate Preliminaries section was removed. The current Method first explains DFlash's 16-token block and five-stream fusion, then defines the mapped and folded computations with all feature and matrix widths. Its training subsection retains the identity-initialized AUF rule for LoRA-fine-tuned targets and the Xavier-initialized, paired-feature and post-fusion reconstruction rule for cross-model transfer. The short EAGLE-3 and inference subsection retains recurrent AUF supervision, reduced-vocabulary termination, unchanged verification, cache cleanup and the requirement for a separate cross-tokenizer proposal bridge. Section 4 still reports training cohort sizes, supervised coverage and optimization schedules. No experiment results were changed in this pass.

Evidence checked: `../phase1.pdf`, `../phase1_results.md`, and `../phase1_probe.md` for DFlash block drafting, AUF supervision and identity initialization; `../phase2.pdf`, `../phase2_results.md`, and `../phase2_probe.md` for paired features, relative layer-plus-context reconstruction, folding and tokenizer limits. The earlier `../yashas_relayspec_iclr2027/sections/method.tex` supplies the concise objective-level descriptions. The supplied DFlash and RepSpec PDFs were used for presentation structure, without importing their claims. The build places the Conclusion on page 15, down from page 16 before this consolidation, so the 9-page main-text limit still requires later cuts, mainly in Experiments.

## Section 5 source audit and compression, 17 September 2026

Sections 1, 2, 3, 4 and 6 were frozen for this pass. Section 5 was checked against `../phase1.pdf`, `../phase2.pdf`, their four result/probe Markdown reports, `../eagle_auf_results.md`, `../more_phase1.md`, `../more_phase2.md` and `../small_scaling.md`. The main table and cross-transfer serving rows come from Phase 1/2 result reports. Nonlinear-map rows come from `more_phase1.md` and `more_phase2.md`. EAGLE-3 LoRA AUF rows come from `eagle_auf_results.md`; its cross-transfer MSE rows come from `more_phase2.md`. Figure 3 combines `small_scaling.md` with the Phase 1 scaling export and Phase 2 MSE50 coverage table. Figure 4 comes from the Phase 1 serving export. The rank-56 pooled adapter is a separate serving experiment from Figure 4's five-map measurements.

Three manuscript claims failed this check. The former public-runtime table comparing a single-projection interface, PARD and SD$^2$ (193.68/105.63/19.92 tokens/s) occurs only in manuscript copies, not in the specified experiment reports, so it was removed from active Section 5 pending a raw measurement record. The claim that base-model-generated training responses matched adapted-target-generated responses at $1.001\times$ conflated a separate dense-versus-sampled NanoCoder result and was removed. The Figure 3 prose incorrectly compared KiCad's eight-anchor map result with the 32-anchor body-LoRA result; the actual eight-anchor gains are 59.65\% and 59.33\%. At 16 examples, all four cross-transfer workloads are below autoregressive throughput, so the main text now states that limitation.

No active Section 5 table reports a trained-map SVD truncation. The remaining rank-$r$ rows in the placement table are separately trained low-rank replacements of whole maps, as stated in the Phase 1 ablation export. SVD correction results remain in Analysis and its appendix. Figure 3 now plots throughput as a percentage of the appropriate unadapted or native drafter in all three panels. Its Phase 1 example axis says presentations because some larger budgets repeat training examples. The selected-map fit-time statement excludes feature capture and does not describe every exploratory fit, some of which exceeded half an hour.

## Abstract and title, 17 September 2026

The title is centered through its declaration in `main.tex`. The ICLR 2027 style file remains unchanged; its official title macro sets small caps but does not center a long title, which stretched the interword spaces under full justification. The abstract's DFlash LoRA range (6--85\% over unadapted) is rounded from the five selected rows in Section 5 Table~\ref{tab:main}, sourced from the Phase 1 reports. Its Qwen3-4B-to-Qwen3-8B retention values (100.2\% MATH, 99.4\% GSM8K versus native) are the Phase 2 selected MSE25 mean-throughput ratios, not latency ratios. Its EAGLE-3 range (46--106\% over unadapted) is rounded from `../eagle_auf_results.md`. The abstract names both LoRA activation and a change of model size or family as target changes. No limitation statement or cross-architecture absolute-throughput comparison is included.

## One-time Method anchor clarification, 17 September 2026

The Method now explains the DFlash anchor as the last committed target token and the first known token in a block. The following 15 masked positions are proposals; during training a saved response token becomes the clean anchor and up to 15 following saved tokens supply labels. The anchor has no loss. This follows DFlash's block-construction description in `../relayspec_ref/2602.06036v2.pdf`, the masking procedure in `../phase1_results.md` under the DFlash token objective, and the corresponding Phase 1/2 PDF and Markdown reports. Sections 1, 2, 3, and 6 remained unchanged in this exception.

## Experimental protocol moved to appendix, 17 September 2026

The compact Training and runtime paragraph was removed from Section 5. Appendix~\ref{app:experimental-protocol} now separates selected LoRA-fine-tuned and cross-transfer fitting, serial Transformers evaluation, EAGLE counter comparability, and concurrent vLLM serving. The selected fit sizes and schedules follow `../phase1.pdf`, `../phase2.pdf`, `../phase1_results.md`, and `../phase2_results.md`. Anchor and reconstruction details were cross-checked with `../phase1_probe.md` and `../phase2_probe.md`. The 384-request three-adapter serving mixture follows the Phase 1 serving record and Figure 4 caption. Tables in Section 5 link to the protocol; the frozen Sections 1--4 and 6 were not edited.

## Section 6 compression, 17 September 2026

With explicit user authorization, Section 6 now combines the interface-recovery and held-out-prediction summaries, removes repeated cross-transfer architecture setup, and leaves one sentence for the live midpoint result. Appendix `app:midpoint-full` explains the no-retraining intervention and its complete 12-row table. The LoRA correction-SVD summary is shortened; Appendix `app:low-rank-corrections` defines correction versus whole-map truncation, the retained-latency-reduction ratio, and links both complete rank tables. The GSM8K rank-128 latency-retention range is 95.1--99.6\% across two repeats; the NanoCoder rank-256 range is 57.8--89.3\%. These are behavioral/timing results, not proof that the learned maps lie on a low-rank manifold. Sources checked: `../phase1.pdf`, `../phase2.pdf`, `../phase1_results.md`, `../phase1_probe.md`, `../phase2_results.md`, and `../phase2_probe.md`.

## Speedup notation and table ranking, 17 September 2026

Every active Section 5 reference-relative throughput ratio and harmonic mean now carries a multiplication sign, including cells whose column heading already says `×ref`. This is a notation-only edit: the reported measurements were not changed. Where a table compares alternatives and already bolds the best result, it now underlines the second best. The placement table's KiCad winner is the five-map result (+89.5\%), while the joint-map result (+88.4\%) is second; its earlier bolding was incorrect. Section 6's three-recipe interface-recovery table also marks the best and second-best error and cosine results. The 2026-09-17 build has no undefined references or overfull boxes.

## Target terminology and Section 5 float order, 17 September 2026

With explicit user authorization, Method and Introduction now use `LoRA-fine-tuned target` and `cross-transfer` consistently. The old inference paragraph in Method Section 4.3 was removed. The only remaining use of `replaced` in the active manuscript describes swapping a five-map bank in an appendix control, not a target. The main-results table declaration was moved after its Section 5.2 discussion. This is a layout-only source reorder that retains the table, values, label, and reference. It allows Section 5.3 to begin on page 5 rather than leaving its lower area empty. The build has no unresolved references or overfull boxes.

## Abstract and contribution-block compression, 17 September 2026

The abstract now states that DFlash throughput improves by up to 85\% on some LoRA-fine-tuned targets and that the approach generalizes to EAGLE-3 drafters, retaining the measured 46--106\% range. The repeated Introduction contributions itemize was removed. The abstract and page flow build cleanly; Section 5.3 remains on page 5 and the Conclusion remains on page 11.

## Architecture and objective tables moved to appendix, 17 September 2026

With explicit user authorization, the placement and objective ablation tables were moved from Section 5 to Appendix A.1, retaining their labels, units, result values, and best/second-best formatting. Section 5 now records only the selected conclusions: five per-stream maps, AUF for LoRA-fine-tuned targets, and reconstruction MSE for cross-transfer. The nonlinear and supervision-budget subsections explicitly state that their comparisons use these selected architecture and objective choices. This reduces the main text by one page, with the Conclusion now starting on page 10. Sources remain the Phase 1 and Phase 2 reports logged for the original Section 5 audit.

## Nonlinear-map table moved to appendix, 17 September 2026

With explicit user authorization, the matched linear-versus-nonlinear comparison was moved to Appendix A.1. Section 5.3.3 retains its measured conclusion and links the complete table. No results or units changed. The build has no unresolved references or overfull boxes, and the Conclusion remains on page 10.

## Adaptation-cost fit times, 17 September 2026

With explicit user authorization, Section 5's adaptation-cost prose qualification was removed. Table 5 now reports fit time in seconds: DFlash LoRA-fine-tuned fits span 160--319 seconds across the three reported domains, DFlash cross-transfer is 1,494 seconds, EAGLE-3 LoRA-fine-tuned fits span 164--1,149 seconds, and EAGLE-3 cross-transfer is 788 seconds. These values come from `../more_phase1.md`, `../more_phase2.md`, and `../eagle_auf_results.md`; the ranges retain domain-specific variation. The table was tightened to avoid overflow and builds cleanly.

## Concurrent-serving detail and cross-transfer figure, 17 September 2026

With explicit user authorization, the unsupported rank-56 pooled-adapter statement was removed from active Section 5. Figure 5 now plots the complete Qwen3-8B $\leftarrow$ Qwen3-4B cross-transfer concurrency record at 1, 8, 16, and 32 clients for Math, GSM8K, Code, and Chat. It is generated by `figures/draw_transfer_serving.py` directly from the complete T1 table in `../phase2_results.md`. Appendix A.2 now contains the moved 16-client cross-transfer table and the corresponding 16-client shared LoRA-serving table. The latter is taken from the `peak_clients_trial0`, oracle-policy rows in `../phase1_results.md`. The active source builds with resolved references and no overfull boxes.

## Per-target fit times and scaling-time appendix, 17 September 2026

With explicit user authorization, the aggregate adaptation-cost table was removed. Tables 1--3 now report the exact optimizer-loop fit time for every selected target and drafter family. DFlash LoRA-fine-tuned times are 159.8, 319.3, 165.4, 243.6, and 241.2 seconds for Qwen3-4B GSM8K, KiCad, NanoCoder and Qwen3-8B GSM8K and CodeAlpaca respectively. These values come from the Phase 1 scaling export and its Qwen3-8B study. Selected DFlash cross-transfer times are 1,494.6 seconds (T1), 1,145.4 seconds (T2), 1,233.0 seconds (T4), and 1,234.8 seconds (T3), from `../phase2_results.md`. EAGLE-3 LoRA-fine-tuned times are 163.9, 184.2, and 1,149.3 seconds from `../eagle_auf_results.md`; its cross-transfer time is 787.9 seconds from `../more_phase2.md`.

Figure 6, generated by `figures/draw_scaling_fit_times.py`, gives the optimizer-loop fit times for every point in Figure 3. It reads the Phase 1 scaling table, Phase 2 MSE50 coverage table, and `../small_scaling.md`; its parsed data are saved in `figures/scaling_fit_times_data.json`. Appendix A.1 now records the selected map recipes with their learning rates and epoch counts, and notes repeated records when a source dataset has fewer than 4,096 examples. Section 5's placement paragraph was rewritten to state the compared parameterizations and their formulas. Repeated caption definitions of $\tau$ were removed while its definition remains in the Section 5 metrics paragraph.

## Multi-LoRA routing and pooled-map appendix, 17 September 2026

Section 5.5 now states that each of the three LoRA-fine-tuned target adapters has a matching map bank and that the deployment is a hard request-level mixture. It separates the explicit target-ID selection used in Figure 4 from the independently measured prompt-only router. Appendix `app:serving-details` gives the complete historical rank-56 AUF map-bank recipe, the TF--IDF/logistic-router recipe, all two-repeat explicit-ID and learned-router results, and the full pooled-map measurements. These entries are supported by Phase 1 studies 23, 24, 28, and 29 in `../phase1_results.md`, cross-checked against `../phase1.pdf`. The pooled-map and router measurements use an earlier rank-56 sampled-AUF setup and are reported separately from the current Figure 4 map-bank serving run.

## Client-side serving throughput and shared cross-transfer fits, 17 September 2026

Figure 5 now pairs server aggregate throughput with client-side per-request throughput. The latter is computed from the Phase 2 T1 output-token totals divided by the sum of the 128 recorded client-side request latencies, rather than from server wall time or a RelaySpec/native ratio. Figure 4 explicitly calls its already-recorded arithmetic mean `mean client-side per-request throughput`. Both figures retain server aggregate throughput in panel (a). Sources: `../phase1_results.md` serving export and the `Optimized vLLM concurrency`, `Optimized serving latency`, and `Optimized serving output and memory accounting` tables in `../phase2_results.md`.

The Phase 2 PDF and Markdown report one selected optimizer-loop fit per cross-transfer target--drafter pair, shared by its four workload evaluations. They do not report independently fitted maps or per-dataset fit times. Tables 1 and 3 therefore label these cells `Shared fit time (s)`, and the appendix makes the shared-fit scope explicit. This preserves the recorded 1,494.6 s (T1), 1,145.4 s (T2), 1,233.0 s (T4), and 1,234.8 s (T3) pair-level values without assigning them to individual datasets.

## Output-match wording audit, 17 September 2026

At the user's request, all manuscript discussion of output agreement, disagreement, or sequence equivalence was removed, including the Method's generic greedy-verification claim and the appendix's serving qualification. Section 5 retains the sole requested operational statement: for every single-request evaluation with a native DFlash control, the native DFlash output was verified to match the adapted output. Section 5 also now identifies BF16 Transformers as the single-request DFlash evaluation engine. This statement is supported by the selected single-request audit records in `../phase1_results.md` and `../phase2_results.md`.
