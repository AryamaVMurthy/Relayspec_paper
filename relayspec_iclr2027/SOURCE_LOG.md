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
