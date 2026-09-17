# RelaySpec paper for ICLR 2027

This LaTeX project uses the official anonymous review format. The abstract and main sections are drafted. Figures 1 and 2 compare single-request throughput with autoregressive decoding of each target. The abstract reports results from the experimental reports. Those experiments have not been independently rerun.

## Edit and compile

Open `main.tex` as the root document. Write each section in its corresponding file under `sections/`. Add verified bibliography entries to `references.bib`. Use `\citet{key}` for narrative citations and `\citep{key}` for parenthetical citations. Use labels and `\ref{label}` for section, figure, table, and appendix references.

Run from this folder:

```bash
python3 build.py
```

Requires Python 3, pdfLaTeX, and BibTeX. The compiled paper is `output/pdf/paper.pdf`. Build artifacts are in `build/`. The script runs BibTeX once citations exist. An empty bibliography produces no visible reference list until entries are cited.

For Overleaf, upload this folder and select `main.tex` as the main document.

## Format

The project uses the official ICLR 2027 style and bibliography files without modification. It uses US Letter, anonymous review mode, automatic line numbers, and the template's fonts, spacing, and margins. Do not add geometry settings, reduce font sizes, or change the style files. Leave `\iclrfinalcopy` disabled for initial submission.

Initial main text must fit within 9 pages. Rebuttal and camera-ready main text may use 10 pages. References and appendices are excluded. Appendices follow references. The abstract must be one paragraph.

AI disclosure is required in the paper and submission form. Ethics and reproducibility statements are recommended. These statements are excluded from the main-text limit. Keep the AI-use and ethics statements within one page each. Fill them with accurate content before submission.

The six main section headings are the working outline for this project. They are not prescribed by ICLR. DFlash background is included at the start of Method.

## Sources and final checks

Official sources checked on 15 September 2026:

- [Author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)
- [Official LaTeX package](https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip)
- [AI policy for authors](https://iclr.cc/Conferences/2027/AIPolicyForAuthors)

`template-source.json` records the download URL and hashes of the unmodified style files.

The template sets document formatting. It cannot ensure scientific correctness or satisfy submission-form obligations. Before submission, check anonymity throughout all artifacts, page count, citation accuracy, completed statements, authorship, OpenReview profiles, submission eligibility, reciprocal reviewing, dual-submission rules, and deadlines against the current official guidance. Follow the writing rules in the parent `AGENTS.md`.

## Figures

`figures/relayspec_lora_ar.pdf` is Figure 1 and compares unchanged DFlash with RelaySpec on three LoRA-fine-tuned Qwen3-4B targets. `figures/relayspec_transfer_ar.pdf` is Figure 2 and compares a native Qwen3-8B drafter with the reused Qwen3-4B drafter on four workloads. Every plotted speedup uses the matching autoregressive target as the `1.00×` baseline. Regenerate both with `python3 figures/draw_speedups.py`. The script reads the saved AR comparisons in `../new_phase1.md` and `../new_phase2.md` and records the plotted values in `figures/speedups_data.json`.

`figures/relayspec_overview.pdf` is the earlier architecture diagram. It is retained as an editable asset but is not included in the current paper.

## Agreed section structure

Abstract, Introduction, Related Work, Method, Experiments, Analysis, Conclusion. The abstract is unnumbered. Method introduces the DFlash computation where it is needed, then covers RelaySpec's maps, training procedures, and export. Experiments reports performance and controlled comparisons. Analysis explains map behavior and includes limitations. Required and recommended statements, references, and appendices follow the main text.
