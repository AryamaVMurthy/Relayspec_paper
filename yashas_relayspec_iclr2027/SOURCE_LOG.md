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
- PARD and PARD-2 train the drafter model itself. Their results do not establish that a frozen DFlash transformer can directly use arbitrary new target features.
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

Contributions summarize the reported interface, drafter-model and fusion comparisons, supervision budgets, shared serving, and controlled map interventions. Detailed evidence and limitations still need to be drafted in the remaining manuscript sections.

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

The five-stream native fusion and the 4096-to-2560 map dimensions are checked against `../phase2_results.md`, section “T1 pruned sweep: method and interpretation”, and `../phase1_probe.md`, section “What exactly was adapted?”. The notation is uniform in Figure 1 and Section 2.2. The new target width is d_t, the input width expected by the reused fusion projection is d_r, and the drafter transformer width is d_d. The diagram uses column vectors: H_i has shape d_t by n, W_i has shape d_r by d_t, W_i H_i has shape d_r by n, and F_i has shape d_d by d_r. The concatenated original fusion F has shape d_d by 5d_r. The folded fusion G has shape d_d by 5d_t. All arrow segments in the generator are checked to be horizontal or vertical.

The Section 2 DFlash description was checked against Sections 4.1, 4.2, and A.3 of `../relayspec_ref/2602.06036v2.pdf`, plus the masked-block contracts in both phase reports and all four Markdown reports. The official `z-lab/Qwen3-4B-DFlash-b16` checkpoint configuration was also checked at revision `c2b880f72981615c03795e58c532feed186d64df`. Its block size is 16. The official `z-lab/dflash` reference generator was checked on 2026-09-16 and defaults to temperature zero. The deployed block therefore has one clean anchor and 15 masked proposal positions, and the paper reports greedy decoding. Five target streams are fused and normalized, then supplied as key and value context to every transformer layer of the drafter. Only target features before the anchor are visible to the block. The original target is absent from adapted inference.

The token vocabulary assumption applies to the displayed speculative-sampling equation. The Llama-to-Qwen transfer uses the separate tokenizer bridge documented in the T4 report. Greedy bridge measurements do not establish stochastic distribution preservation. Reference features used as reconstruction targets are denoted H_i^(r), distinct from new-target inputs H_i.

## Evidence-based rewrite after objective and probe review

Consulted both report PDFs, including the renamed `../phase2.pdf`, plus all four results/probe Markdown reports. The Section 6 audit used fresh text extractions under `../tmp/analysis_check/`.

- AUF motivation: Phase 1 PDF discussion of expected accepted-prefix length and `phase1_probe.md` section “What AUF taught these maps”. Later correct predictions can lower CE without extending the prefix. This is motivation, not a proof of optimal throughput.
- LoRA-target versus cross-model motivation: Phase 2 PDF opening and `phase2_probe.md` section 11. Identity initialization preserves the existing feature interface for a LoRA-fine-tuned target. Random rectangular maps need to learn inputs usable by the frozen drafter.
- Compression and direction: `phase1_probe.md` sections 5 and 9. For the tested GSM8K checkpoint, the identity-preserving rank-128 correction gives 5.928 average acceptance length versus 5.988 for the full maps. In two matched timing repeats, it retains 95.1% and 99.6% of the full maps' measured latency reduction. These variants still use the same dense folded inference projection. NanoCoder timing is less stable, so no rank is claimed to work across tasks.
- LoRA shift: `phase1_probe.md` section 10 and “Paired-feature geometry and optimization”. Aligned versus shuffled shift cosines do not establish token-specific reversal. This is not proof that no useful information is recovered.
- Cross-size intervention and refinement: `phase2_probe.md` sections 6, 7, and 11. Midpoint maps improve the tested Math and Code endpoints. Warm AUF improves GSM8K and weakens other tested workloads. The recipes differ in initialization, coverage, and optimizer settings. No universal loss ranking is claimed.
- Performance scope: `phase1_results.md` architecture/objective and pooled-serving comparisons, and `phase2_results.md` selected T1 standalone table. Keep mean per-request, pooled, and aggregate throughput separate.
- Added `lin2026eda`: https://arxiv.org/abs/2603.09527 , verified preprint metadata and shared/private adaptation description.
- Added `ramakrishnan2025omnidraft`: https://proceedings.neurips.cc/paper_files/paper/2025/file/3c2fe1417eed1c6ff9acf169617981ea-Paper-Conference.pdf , NeurIPS 2025. Cross-vocabulary cache and online adaptation.

The final Section 3 comparisons distinguish the parameter and deployment choices directly. EDA trains a lightweight private component in a shared/private drafter and regenerates target-specific data. Online Speculative Decoding continually updates drafters from serving queries. OmniDraft combines online caching, hybrid distillation, and vocabulary conversion. PARD and PARD-2 train parallel drafters for target-independent or dual-mode operation. SD$^2$ injects target-derived steering inside every drafter feed-forward layer and fully fine-tunes the drafter in its main method, while also reporting a frozen-drafter ablation. RelaySpec instead trains five maps at the existing DFlash feature interface, keeps the drafter transformer frozen, and folds the maps into the fusion projection for inference.

Root AGENTS.md now requires checking the supplied report PDFs and Markdown evidence when drafting technical claims, and following the reference papers' structure with original, simple-English prose.

## Method and experiments source audit, 16 September 2026

The Method and Experiments sections were drafted after checking `../phase1.pdf`, the renamed `../phase2.pdf`, `../phase1_results.md`, `../phase1_probe.md`, `../phase2_results.md`, and `../phase2_probe.md`. PDF text extractions used for review are under `../tmp/pdfs/`. Claims below are reported experimental results. They were reconciled across the reports but were not independently rerun while writing the manuscript.

| Manuscript claim | Evidence used | Scope retained in the manuscript |
| --- | --- | --- |
| Five-map interface, identity initialization, AUF support, frozen components, and folding | Phase 1 PDF Sections 2.1 and 2.3, `phase1_results.md` architecture/objective contract, and `phase1_probe.md` | Applies to same-width Qwen3 LoRA targets. AUF includes the first incorrect proposal. |
| Paired feature reconstruction and relative MSE | Phase 2 PDF Sections 3.1 and 3.2, `phase2_results.md` T1 method contract, and `phase2_probe.md` Section 2 | Applies to cross-model transfer. Source and target process the same saved prefix. The selected maps use Xavier initialization. |
| Selected LoRA results and comparison methods | Phase 1 PDF Table 1 and `phase1_results.md` rows for `maps_full`, `body_r128`, `fusion_delta`, and decaying CE | Main table uses 4,096 examples, 32 anchors, one epoch, and mean per-request TPS. It does not combine later sweep maxima with the selected checkpoint. |
| LoRA supervision sweep | Phase 1 PDF Section 4.2 and the complete 96-cell table in `phase1_results.md` | Reported maxima are marked as inspected sweep maxima rather than independent confirmation. Repeated examples in the larger GSM8K and KiCad cells remain a limitation. |
| T1 to T4 selected transfer results | Phase 2 PDF Table 1 and `phase2_results.md` selected standalone tables | T1, T2, and T4 use native-drafter ratios. T3 uses AR ratios because no native drafter exists. T4 includes tokenizer conversion. T3 uses block size 16 despite block size 10 pretraining. |
| Cross-model training budget and recipe comparison | Phase 2 PDF Table 2 and Sections 5 to 6, `phase2_results.md` size/coverage tables, and `phase2_probe.md` | MSE, AUF, and decay are described as complete recipes because learning rate and supervision coverage differ. No loss-only causal claim is made. |
| Output agreement | Opening qualification in `phase2_results.md` and `phase2_probe.md` | Native and adapted Transformers outputs match for T1, T2, and T4. vLLM outputs can differ, so its measurements are not called lossless. |
| Shared serving | Phase 1 PDF Section 4.3 and deployment tables in `phase1_results.md` | The historical rank-56 bank is kept separate from the newer five-map deployment. Mean request TPS and aggregate workload TPS are not interchanged. |

Dataset citations for MATH, GSM8K, LiveCodeBench, UltraFeedback, and NuminaMath follow the Phase 2 report bibliography. The vLLM citation follows the same report. The public-data provenance warning remains relevant: the saved 16,384-example transfer cohort is reproducible from its retained IDs and hashes, but the normalized cohort alone does not provide a complete from-scratch public dataset revision trail.

The Introduction now gives only a short overview of the two training cases. The detailed AUF and cross-model reconstruction procedures were moved to Section 4. The AUF support rule is checked against `phase1_probe.md` and `phase2_results.md`: it includes the correct prefix and first incorrect proposal. The cross-model description is checked against the paired-feature and loss definitions in `phase2_results.md` and `phase2_probe.md`: the original and new targets process the same saved text, and the normalized MSE covers each mapped feature and the fused context after RMSNorm.

## Analysis probe selection, 16 September 2026

Section 6 was shortened after reviewing how the supplied DFlash, PARD-2, RepSpec, and SD$^2$ papers introduce ablations and analysis. The revised section poses a concrete question before each intervention, gives the smallest comparison needed to answer it, and then states a scoped conclusion. It uses completed probes from both phases. It does not use layer removal, bank swaps, incomplete KiCad or Code rows, weight energy as a proxy for speed, or the one-step optimizer diagnostic in the main paper. It includes the strongest completed identity-preserving SVD correction result and the completed rectangular-identity probe.

| Analysis statement | Primary evidence | Qualification kept in the text |
| --- | --- | --- |
| Maps for LoRA-fine-tuned targets mainly benefit drafting by changing context direction | `phase1_probe.md` Sections 8 to 10 and the completed live five-map intervention table | Parallel and perpendicular variants use dynamic controls. The conclusion rests on acceptance length because NanoCoder timing retention is unstable. |
| A lower-rank correction preserves most of the GSM8K benefit | `phase1_probe.md` section “Five-W AUF: mechanistic interventions”, `phase1.pdf` Section 5, and repeated timing rows in the five-map mechanism report | SVD is applied to $W_i-I$ and the identity path is retained. Rank 128 gives acceptance length 5.928 versus 5.988 for the full maps and retains 95.1% and 99.6% of measured latency reduction in two repeats. The folded inference projection remains dense. |
| The maps do not simply reverse the target LoRA | `phase1_probe.md` Section 10 and paired-feature geometry table | Mean aligned cosines are compared with shuffled-position controls. This tests literal pre-RMSNorm cancellation, not every form of useful recovery. |
| MSE restores the interface expected by the source drafter | `phase2_probe.md` Sections 4 and 5 | Training-position reconstruction is expected from the objective. The held-out CE and correct-prefix measurements provide the separate prediction check and use only eight examples per workload. |
| Moving token-trained maps toward MSE improves live decoding | `phase2_probe.md` Section 6 and all four completed midpoint measurements | The intervention averages all five maps and their scales. It does not isolate one layer or feature property. Endpoints were timed in separate passes. |
| Rectangular-identity initialization does not close the acceptance gap | `phase2_probe.md` Section 10, Phase 2 PDF Tables 27 to 29, and `phase2_results.md` section “Completed T1 rectangular-identity retry” | MSE changes only initialization. AUF and decay CE change initialization and learning rate together, so their gains cannot be assigned to initialization alone. |
| AUF refinement after MSE is workload-dependent | `phase2_probe.md` Section 7 | The best GSM8K endpoint weakens Math, Code, and Chat. This does not prove overfitting. |

The resulting interpretation is deliberately scoped. Adaptation first needs to give the frozen drafter features it can use, then may redirect those features for a target or workload. Adaptation to a LoRA-fine-tuned target starts with a usable identity interface, while cross-model transfer starts with a larger interface mismatch. Warm AUF supplies evidence for later workload specialization, but not proof of overfitting. The evidence does not establish a universal ordering of objectives or model pairs.

Section 6.2 identifies the analyzed setting directly as a Qwen3-8B target with a pretrained Qwen3-4B drafter and does not use the internal label T1. The new-target features are 4,096-dimensional, the paired original-target features are 2,560-dimensional, and each recipe supplies five maps of shape $2560\times4096$. Table 1 contains only the Phase 1 direction intervention and the LoRA-shift cosines, radial energy, and RMSNorm error. The main correction-SVD endpoints, whole-context-scaling endpoints, and rank-1 and rank-2 entire-map SVD endpoints are stated in the surrounding paragraphs and reported fully in the appendix. Table 2 reports only the response-position interface measurements from `phase2_probe.md` Section 4. The held-out prediction results from Section 5 are summarized in the surrounding paragraph and reported in full in Appendix Tables 9--12. The midpoint intervention is likewise summarized in the paragraph and linked to Appendix Table 13. Table 3 uses MATH, GSM8K, Code, and Chat as columns. Its first three rows give MSE, AUF, and decay-CE throughput, and its next three rows give their acceptance lengths. Every cell shows the actual Xavier $\rightarrow$ rectangular-identity endpoints with a horizontal arrow and no delta. Native endpoints are omitted from this initialization-only comparison. The complete diagnostic variants, reconstruction measurements, and fitting costs remain in the appendix.

## Analysis appendix audit, 16 September 2026

Appendix A reports every result behind a Section 6 claim. The LoRA-fine-tuned-target appendix includes all native, full, parallel, perpendicular, correction-strength, and whole-context-scaling rows, plus the complete paired LoRA-shift geometry table. The Qwen3-4B-drafter-to-8B-target appendix includes all context and per-layer geometry rows, every held-out diagnostic variant on MATH, GSM8K, Code, and Chat, all 12 parent/midpoint/MSE live endpoints, and all 16 warm-refinement endpoints.

The appendix also includes completed secondary probes omitted from the main analysis: every single-map removal, every completed foreign-bank swap, all first-pass whole-map and correction-rank rows, and both repeated timing cohorts for the selected ranks. Incomplete Code removal cohorts are not presented as completed results. All numeric appendix tables state their metrics and units in the caption or column heading. Primary evidence is `../phase1_probe.md` Sections 3 to 10 and the final five-map mechanism report, plus `../phase2_probe.md` Sections 4 to 8 and 10. The appendix remains after the references under the existing `\appendix` command, so it does not enter the nine-page ICLR main-text limit.

The appendix layout was tightened without changing any values. Per-layer reconstruction results are arranged as aligned MSE, AUF, and decay-CE panels. The four held-out diagnostic tables share one ordered float block while retaining their separate table numbers and labels. Float barriers keep each subsection's tables before the next subsection, and appendix-only float settings top-align grouped tables instead of vertically distributing them across otherwise empty pages.
