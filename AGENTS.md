# Writing rules for this workspace

Apply these rules when drafting, editing, reviewing, or explaining the RelaySpec paper and its supporting material.

## Source roles

- RelaySpec is the main paper being drafted for ICLR.
- Phase 1 and phase 2 are experimental reports of the research and experiments. They are evidence sources, not canonical papers.
- Verify the manuscript one section at a time against the experimental reports. Preserve the distinction between reported results and independently verified results.

## Required evidence checks

Before drafting, revising, or explaining technical claims in the paper, always consult the relevant material in both Phase 1 and Phase 2 and the supplied Markdown reports:

- `phase1.pdf` and `phase2.pdf`
- `phase1_results.md` and `phase1_probe.md`
- `phase2_results.md` and `phase2_probe.md`
- Any original experiment notes referenced by those reports when needed to resolve a claim.

Use the PDFs for the research narrative and the Markdown reports for procedures, measured results, controls, and qualifications. Do not derive motivation solely from model dimensions or a plausible explanation. Check the actual comparisons and probes. Distinguish motivation, observation, interpretation, and untested hypotheses. Preserve recipe differences, metric definitions, incomplete coverage, and negative results. Do not combine historical checkpoints or different experiments into one result. If sources disagree, identify the discrepancy and use the supported, explicitly scoped result. Record important claim-to-source links in the manuscript source log.

## Reference-paper style

Consult the relevant papers in `relayspec_ref/` when organizing and writing manuscript sections. Follow their concise research-paper structure: concrete problem, method intuition, brief procedure, supporting evidence, and specific contributions. Use original wording relevant to RelaySpec. Do not copy prose or import another paper's claims. The simple-English rules below continue to apply. Keep established concepts in Preliminaries, comparisons in Related Work, and detailed RelaySpec procedures in Method.

## Required writing style

Write in simple, direct academic English. Keep the meaning technically correct. Explain each point in normal language instead of compressing ideas into research buzzwords.

- Be straight to the point.
- Do not add filler, hype, vague claims, or unnecessary background.
- Do not stack technical terms together when a normal sentence can explain the idea.
- Use the correct technical term only when it is needed. Make its meaning clear from the sentence.
- Prefer concrete explanations of what is being done, measured, or compared.
- Avoid phrases such as "first-error localization", "transfer dynamics", "representation quality", and "specialization-generalization trade-off" unless they are necessary. Explain what they mean directly.
- Write "deterministic checks" instead of "symbolic verification" when that accurately describes the procedure.
- Do not use em dashes.
- Do not use semicolons in prose. Preserve them in code or mathematical notation when required for correctness.
- Keep sentences reasonably short.
- Do not use overly formal, inflated, or unnecessarily complicated wording.
- Do not repeat the same idea in different words.
- Do not introduce new claims to make the writing sound stronger.
- Preserve important experimental details, assumptions, limitations, equations, datasets, model names, and citations.
- Make every paragraph answer a clear question: what are we doing, why are we doing it, or how will we measure it?
- Write so that a technically capable undergraduate can understand the text on the first read without losing the research meaning.

The target style is a researcher clearly explaining the work to another researcher. Be concise, concrete, and natural. Do not try to make the work sound sophisticated.

## Numerical reporting

- Every table column containing numeric measurements must identify the metric and its unit in the column heading or table caption. Stating the unit once is sufficient. It does not need to be repeated in every cell.
- Use the full metric name. In particular, write "acceptance length", not "acceptance". Define $\tau$ as tokens per verification step wherever a table could otherwise be ambiguous.
- Label throughput in tokens/s. Write speedups with the multiplication sign, such as $1.2\times$, and percentages with the percent sign.
- Name dimensionless quantities explicitly, such as cosine similarity or an error ratio. Do not present unexplained raw numbers.

## Analysis appendix coverage

- Every experimental probe cited or interpreted in the main analysis must have its complete measured results in the appendix, including matching controls, all tested variants, negative results, metric definitions, units, and material limitations.
- Link each main-text probe claim to the corresponding appendix subsection or table.
- Additional completed probes may be reported in the appendix even when they are omitted from the main argument. Clearly distinguish complete results from partial, stopped, or confounded runs.

## Explain the procedure before naming it

For each section, explain the actual procedure before naming or summarizing it. Introduce a technical label afterward only if it is useful.

Examples:

- Prefer "check whether the model can identify where a proposed solution first becomes wrong" over "first-error localization".
- Prefer "We train a fresh copy of the student on only one task form and then test it on all five forms" over "We construct a directed cross-task transfer matrix". The term "transfer matrix" may be used afterward once the text explains what it represents.
- Prefer "Prior work shows that training on one task can help some other tasks much more than others" over "Prior work finds asymmetric transfer after task-specific fine-tuning".

These examples illustrate style. They do not establish procedures or findings for RelaySpec.

## Complete sentences and connected explanations

Use RepSpec, PARD, and DFlash as references for how to develop an explanation, not as sources of sentences to copy. The relevant patterns are:

- RepSpec Section 3.1 first explains the operation in words, introduces the components and notation, gives the equations, and then explains how training-time operations are combined for inference.
- PARD Sections 3 and 3.1 state what the method will do before introducing its prediction objective and inference procedure.
- DFlash Section 4.1 connects a design choice to the limitation it addresses, explains how it works, and points to evidence for its effect.

Apply these patterns with RelaySpec's own evidence and the following rules:

- Write complete sentences with a clear subject and verb. Do not compress prose into fragments, labels, or a list of implementation facts.
- Begin each paragraph by identifying the operation, question, or setting being discussed. Do not start with an unexplained symbol, loss name, initialization choice, or result.
- Connect each sentence to the preceding explanation. State the reason, consequence, or next step when the relationship would otherwise be unclear. Use transitions only when they express a real relationship.
- Introduce what a component does and why it is needed before giving its name, dimensions, or formula. Do not make readers infer its purpose from notation alone.
- Lead into an equation with a sentence explaining what it computes. Define every new symbol before or immediately after the equation, and explain the result when its role is not already clear.
- Name the object being changed or measured. Avoid vague openings such as "This improves performance" when the reader cannot tell what "this" or "performance" refers to.
- Distinguish the motivation for a design from evidence that it works. A smooth explanation must not turn an intuition into an established finding.
- Use connected sentences to describe a sequence of operations. Do not jump from initialization to export, or from training to evaluation, without explaining the change of stage.
- Keep sentences reasonably short, but retain the words needed to explain their meaning. Combine closely related short sentences when separating them makes the paragraph choppy.
- Compress by removing repeated claims, unnecessary background, and duplicate definitions. Do not remove subjects, logical connections, assumptions, or explanations merely to save lines.
- Read each paragraph as continuous prose before finishing. A reader should understand what is being done and why without reconstructing the argument from isolated statements.

For example, prefer "Before inference, we multiply each learned map into its fusion block. The drafter can then use the combined projection without a separate mapping operation." over "Maps folded into fusion. No mapping overhead." This illustrates sentence construction, not permission to add an unsupported efficiency claim.

## Citations and cross-references

Use citations and cross-references wherever needed to support claims and help readers find relevant material. Refer to the appendix, sections, figures, tables, and equations when appropriate, including from the main paper. Make each reference accurate and specific. Cite prior work where its ideas, methods, or findings are used.

## Check before finishing a section

Confirm that the text explains the actual procedure, comparison, or measurement. Remove repeated points and unnecessary jargon. Check that simpler wording preserves the technical meaning, evidence, assumptions, and limitations. Check that prose contains no em dashes or semicolons. Check that paragraph openings establish context, sentences connect logically, and compression has not made the explanation abrupt or incomplete.
