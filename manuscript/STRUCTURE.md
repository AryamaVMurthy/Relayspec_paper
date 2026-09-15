# RelaySpec: manuscript structure

Status: recommended structure for section-by-section drafting, 15 September 2026. This is an outline, not a verified submission-ready paper.

## Source roles

- `../relay.pdf`: current manuscript; the title and existing prose are working material.
- `../phase1_results.md`, `../phase1_probe.md`, and `../phase1.pdf`: experimental reports for target specialization, controls, serving, and mechanisms.
- `../phase2_results.md`, `../phase2_probe.md`, and `../phase2 (1).pdf`: experimental reports for cross-model transfer and its controls.
- `../relayspec_ref/`: literature and historical reviewer feedback, not evidence for our own measurements.

Phase labels describe research history. Use scientific questions and experimental settings to organize the paper. A measurement appearing in an experimental report is not automatically verified for publication; check its configuration, provenance, and interpretation when drafting the relevant section.

## Central question

How much adaptation to a changed target can be localized in the feature interface while keeping a pretrained speculative draft transformer frozen?

The two settings are (1) adaptation to fine-tuned targets and (2) transfer across target models. Introduce them together, then use the same order in method, results, and analysis. Use the existing title as a working title until the contribution and evidence audit is complete.

## Recommended main text

Page budgets include figures and tables. They are editorial targets, not ICLR requirements. The total is 8.75 pages, leaving 0.25 page for layout adjustment within the nine-page initial limit.

| Part | Pages | Reader's question |
|---|---:|---|
| Title, abstract, and 1 Introduction | 1.25 | What problem is solved, why does it matter, and what is established? |
| 2 Related Work | 0.50 | How is the contribution different from the closest prior work? |
| 3 Background and Problem Formulation | 0.65 | What does the existing drafter consume, and what is allowed to change? |
| 4 RelaySpec | 1.85 | How does the interface train, export, and operate? |
| 5 Experiments | 2.80 | Does it work under fair, reproducible comparisons? |
| 6 Analysis and Limitations | 1.50 | Which choices matter, why, and where does it fail? |
| 7 Conclusion | 0.20 | What general lesson is justified by the evidence? |

### Abstract (unnumbered)

One paragraph: problem, interface intervention, two adaptation settings, representative quantitative evidence with explicit denominators, and the main boundary. Draft this last. Do not use the count of experimental cells as a substitute for the scientific finding.

### 1 Introduction

- Establish the deployment problem when a feature-conditioned drafter's target changes.
- Explain why reusing the draft body is worth investigating.
- Introduce the two settings and the shared interface idea in plain language.
- Briefly distinguish the proposal from target-independent drafting and draft-body training.
- End with three evidence-backed contributions: method, evaluation, and explanatory findings. Final wording depends on verification.

Avoid a long generic LLM introduction and repeated disclaimers about what is not novel. State each prior-work dependency at the relevant point.

### 2 Related Work

Use three compact thematic paragraphs rather than a long catalog:

1. Feature-conditioned and parallel speculative drafting: DFlash, EAGLE, PARD, PARD-2.
2. Draft adaptation and reuse: the closest specialization and cross-target adaptation work, including EDA and direct alignment as applicable after source verification.
3. Linear interfaces and structural re-parameterization: RepSpec and relevant representation-adaptation work.

Explain for each category what is adapted, whether target features remain required, and whether per-target fitting is needed. Move the expanded comparison table to the appendix unless it proves essential to understanding novelty.

### 3 Background and Problem Formulation

Define target, reference model, pretrained drafter, feature streams, target/draft widths, and verification. Specify frozen versus trainable parameters and offline versus online access. Formulate both adaptation settings.

Include only inherited DFlash details needed to understand the adaptation. Retain concise context visibility and conditioning semantics; put the full layer equations and architecture inventory in Appendix B.

### 4 RelaySpec

#### 4.1 Linear Feature Interface

Five maps, shapes, fusion, normalization, and initialization. Figure 1 shows training and exported inference, with frozen/trainable components visibly distinguished. Explain why this intervention is plausible without claiming universal linear compatibility.

#### 4.2 Adaptation to Fine-Tuned Targets

Identity initialization, target-generated supervision, the attributed prefix-based objective, support/stop-gradient convention, and gradient flow through the frozen body. Compare alternative objectives empirically; do not presume one wins.

#### 4.3 Cross-Model Transfer

Reference/target feature pairing, interface reconstruction, normalization and weighting, and source-model use during fitting. Explain the cross-tokenizer alignment contract; reference complete bridge specifications in Appendix B.

#### 4.4 Export and Inference

Folding identity and resulting shapes; runtime selection and cache behavior; preservation of the target verification rule. State the assumptions behind ideal greedy equivalence and distinguish them from observed numerical behavior. Explain exactly which operation folding eliminates.

### 5 Experiments

#### 5.1 Experimental Setup and Metrics

Model/checkpoint pairs, workloads, splits, data provenance, selection protocol, hardware, backend, precision, decoding/stopping settings, and baselines. Define acceptance convention, mean-request throughput, aggregate throughput, AR-relative speedup, and native-relative recovery. Identify unmatched comparisons explicitly. Full recipes go in Appendix A.

#### 5.2 Adaptation to Fine-Tuned Targets

Table 1: strongest matched interface/body/fusion and objective controls on the same workloads and budgets. Include useful 8B evidence compactly. Present wins and losses together. Draw from phase-one experimental reports, subject to verification.

#### 5.3 Cross-Model Transfer

Table 2: all four transfer directions and workloads, with absolute throughput and clearly defined comparisons. Preserve missing native controls as missing. Discuss cross-family degradation beside its results. Draw from phase-two reports, subject to verification.

#### 5.4 Shared Serving and Adaptation Cost

Figure 2: aggregate versus per-request throughput across concurrency. Keep different serving implementations separate. Compact cost table: fitting time, preparation time where available, training/storage/peak-memory quantities, and relevant exclusions. Report measured overhead separately from analytic counts. Do not introduce pending experiments as completed evidence.

### 6 Analysis and Limitations

#### 6.1 Interface and Objective Controls

Interpret the matched controls from Section 5 without duplicating their table. Distinguish effects of capacity, initialization, objective, and training budget. Put the complete architecture/objective matrix in Appendix C.

#### 6.2 Mechanistic Interventions

Figure 3: the most diagnostic live interventions for specialization and transfer. Explain what changing direction, preserving the identity path, or interpolating maps demonstrates. Identify confounds and counterexamples. Keep cross-model recipe comparisons distinct from objective-only ablations.

#### 6.3 Supervision and Cost Tradeoffs

Figure 4: compact data/anchor/token-coverage curves. Distinguish repeated examples from unique data, and post hoc maxima from the checkpoints used in headline evaluation. Full grids go in Appendix D.

#### 6.4 Limitations

Concise boundaries: workloads, model families, DFlash-specific evidence, concurrency, numerical agreement, seeds, reused evaluation cohorts, complete adaptation cost, and reproduction gaps that remain at submission. This subsection must describe the final state, not today's tentative gaps if they are later resolved.

### 7 Conclusion

One short paragraph answering the central question, with the main practical condition for useful reuse. No new numerical claims or promised results.

## After the main text

Unnumbered AI Use Statement (required), Ethics Statement (recommended), and Reproducibility Statement (recommended), then References and appendices. These statements are excluded from the main-text page count by the official guidance. Preserve anonymity in the review version. The reproducibility statement points to real artifacts and must not promise artifacts that are absent.

## Appendix structure

- A: Experimental Reproducibility — checkpoint/data pins, prompt manifests, splits, seeds, recipes, metric definitions, environments, evaluation/selection protocols, and measured numerical agreement.
- B: Architecture, Alignment, and Verification — full inherited layer details, attention masks, tokenizer bridge, cache transitions, folding algebra, assumptions and correctness argument.
- C: Complete Adaptation Controls — architecture/objective matrix, matched budgets, initialization controls, pooled versus separate adapters.
- D: Supervision and Cost Results — complete grids, coverage, uncertainty, preparation/fitting costs, analytic versus measured accounting.
- E: Serving Results — all configurations, per-request/aggregate metrics, missing baselines, dispatch and implementation variants.
- F: Additional Mechanistic Evidence — extended interventions, counterexamples, derivations and interpretation limits.
- G: Extended Related-Work Comparison — architectural distinctions; clearly label literature-reported performance as external measurements.

No appendix for unperformed experiments. Keep internal provenance IDs and research chronology in the working evidence map unless needed for reproducibility.

## How the current manuscript maps into this outline

| Current section | Proposed destination |
|---|---|
| 1 Introduction | 1 Introduction |
| 2 Change the interface... | 3 Background + 4.1/4.4; extended details in B |
| 3 Train for compatibility... | 4.2 and 4.3, specialization first |
| 4 Experimental design | 5.1 and A |
| 5 Results... | 5.2 and 5.3 |
| 6 Shared serving... | 5.4 and E |
| 7 Why the interface helps | 6.1/6.2 and F |
| 8 Supervision scaling... | 6.3; costs in 5.4; full grids in D |
| 9 Positioning, limits, impact | 2, 6.4, and Ethics Statement |
| 10 Conclusion | 7 Conclusion |

## Why this organization fits the reference papers

- DFlash: take the link between a bottleneck, a design choice, and a controlled experiment; use realistic serving evaluation. Its two-column layout is not an ICLR template.
- PARD-2: take the shared-method/two-settings organization and the separation of main results, component ablations, and serving tradeoffs. Its late related-work placement is a stylistic choice; early positioning helps RelaySpec because it builds on several close precedents.
- RepSpec: take explicit training/export distinction, folding algebra, matched controls, and separate training/inference cost accounting. Its camera-ready page length does not determine our initial limit.
- Openreviews.pdf: keep the decisive causal controls, comparison conditions, and core implementation semantics in the main text. The collection mixes ICLR and ICML reviews and includes PARD rather than PARD-2.

## Verification workflow

For each section: define its claim; locate exact report evidence; verify metric/configuration and citations; identify unresolved gaps; draft prose/equations; check consistency with tables and other sections; record review status.

Suggested order: structure -> problem formulation -> method -> experimental setup -> results -> analysis/limitations -> related work -> introduction -> conclusion -> abstract/title -> statements and final format audit.

All sections currently have structure only in `main.tex`. Existing content in `relay.pdf` is available for migration; it has not been certified through this workflow.

## Official format sources

- Author guidance: https://iclr.cc/Conferences/2027/AuthorGuidelines
- Official template: https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip

The accompanying starter uses the official unmodified style, US Letter, anonymous review mode, and author-year citations. No source `.tex` for the current `relay.pdf` was present when this starter was created.
