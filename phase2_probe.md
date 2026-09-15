# Phase 2 probes: why does MSE transfer the drafter better?

<!-- AGREEMENT_NOTE -->
In the selected Transformers comparisons with native-drafter controls (T1, T2 and T4), native and adapted drafters matched all 128 output sequences per task. In vLLM, token sequences diverged, consistent with documented numerical and batching variability. The authors completed a manual audit of all divergent answers and found them semantically similar. Semantic similarity does not imply identical token counts. [vLLM documentation](https://docs.vllm.ai/en/stable/features/batch_invariance/).
<!-- /AGREEMENT_NOTE -->





## 1. The answer supported by the experiments

**For T1, the strongest explanation is that MSE restores the representation expected by an already-trained, frozen drafter.** Cold AUF and decay CE improve token prediction from a random interface, but leave that interface much farther from the source model's features. Their weakness is especially visible on Code and Chat. Moving their maps toward an MSE solution improves both offline predictions and live decoding.

This is evidence for **representation compatibility as an important mechanism**, not proof that MSE is universally the better loss. The recipes differ in learning rate and supervision coverage. GSM8K benefits from subsequent AUF training, and some offline rankings fail to predict live rankings. Those counterexamples are part of the result.

Scope: the completed **T1 cross-size probes** used in Phase 2, including training-feature geometry, all completed AUF interventions, held-out common-input diagnostics, four live midpoint interventions and the four warm-refinement endpoints examined by those diagnostics. This is not a restatement of the separate LoRA interpretability studies. The completed rectangular-identity retry below adds a direct initialization intervention and a combined initialization/LR comparison for the token objectives.

## 2. What is being mapped, and what the losses ask for

The target is Qwen3-8B. The reused drafter was trained for Qwen3-4B. At one token position, take five 8B vectors $h_i$, each with 4,096 coordinates. Map each to the 2,560 coordinates expected by the source drafter:

$$
\widehat y_i=W_i h_i,\qquad W_i\in\mathbb R^{2560\times4096},\qquad i=1,\ldots,5.
$$

Here $i$ names a layer slot, not a token. The original frozen fusion has five blocks $F_i$. The context passed into the frozen draft body is

$$
\widehat c=N\left(\sum_{i=1}^5F_iW_i h_i\right).
$$

$N$ is the original RMSNorm. The paired source model sees **the same saved token prefix**, producing vectors $y_i$ and reference context $c_s=N(\sum_i F_i y_i)$. The source model is used for training supervision, not run live in the mapped deployment.

MSE explicitly asks for agreement with that source interface. At each supervised position its loss is

$$
\ell_{\mathrm{MSE}}=\frac15\sum_{i=1}^5D(W_i h_i,y_i)+D(\widehat c,c_s),\qquad
D(u,v)=\frac{\|u-v\|_2^2}{\|v\|_2^2+10^{-6}}.
$$

The denominator makes this a relative reconstruction error: errors are judged against the reference vector's magnitude. Positions are averaged within each example, then examples receive equal weight. This is **not** vocabulary cross-entropy.

Token losses instead ask the frozen drafter to predict the saved target continuation. If $\ell_j=-\log q(x_j)$ is CE at draft position $j$, AUF supervises positions up to and **including** the first wrong prediction. Its support is

$$
s_j=\prod_{k<j}\mathbf1[\widehat x_k=x_k].
$$

Thus the first position always receives supervision; position two receives it only if position one was correct. Padding and invalid labels are excluded. Decay CE retains later valid positions with decaying positional weights. Neither objective explicitly asks that $W_i h_i$ resemble $y_i$. The probes below compare the actual trained recipes, not a newly standardized optimization experiment.

| Checkpoint | Map initialization | Training | Direct supervision |
| --- | --- | --- | --- |
| MSE | Same random Xavier maps | 8,192 examples, 3 epochs, LR 0.001 | Layer + context reconstruction on 50% of prompt and response positions |
| Cold AUF | Same random Xavier maps | 8,192 examples, 3 epochs, LR 0.0001 | Target-token CE on the accepted prefix plus first wrong position; 32 response anchors/example |
| Cold decay CE | Same random Xavier maps | 8,192 examples, 3 epochs, LR 0.0001 | Position-decayed target-token CE; 32 response anchors/example |
| Warm AUF | MSE checkpoint above | 3 additional epochs on 4,096 or 8,192 overlapping/unseen examples | AUF refinement |

The fusion, draft body and source embedding/head remain frozen. Five maps are folded into one effective fusion at inference. Architecture overhead therefore does not explain the large acceptance differences between these three five-map endpoints.

## 3. First check: is the MSE advantage real in decoding?

Each task has 128 prompts, Transformers single-request evaluation, greedy nonthinking generation and a 2,048-token cap. Rows are ordered by speed within each task. **Mean TPS** averages each request's tokens/second. **Pooled TPS** divides all output tokens by summed request time. **Latency reduction** is $100(1-T/T_{native})$; negative values mean slower than native. **Acceptance** is the pooled number of emitted tokens per verification step, including correction/bonus and end truncation. It is not answer accuracy.



| Task | Recipe | Mean TPS | Pooled TPS | × native | Latency reduction | Acceptance |
| --- | --- | --- | --- | --- | --- | --- |
| math | MSE | 219.48 | 216.72 | 0.9986× | +0.32% | 7.9776 |
| math | Decay CE | 194.69 | 187.71 | 0.8858× | -15.09% | 6.9474 |
| math | AUF | 194.22 | 186.76 | 0.8837× | -15.67% | 6.8923 |
| gsm | MSE | 173.55 | 170.79 | 0.9753× | -1.80% | 6.2568 |
| gsm | Decay CE | 163.47 | 158.50 | 0.9187× | -9.70% | 5.6926 |
| gsm | AUF | 162.05 | 157.39 | 0.9107× | -10.46% | 5.6773 |
| code | MSE | 126.18 | 124.01 | 0.8322× | -16.54% | 4.6912 |
| code | Decay CE | 61.56 | 64.80 | 0.4060× | -123.03% | 2.4489 |
| code | AUF | 58.37 | 61.90 | 0.3850× | -133.47% | 2.3456 |
| chat | MSE | 79.49 | 76.54 | 0.8882× | -14.51% | 2.8125 |
| chat | Decay CE | 51.31 | 49.11 | 0.5733× | -78.46% | 1.8115 |
| chat | AUF | 49.93 | 47.84 | 0.5578× | -83.19% | 1.7667 |



These are timing comparisons with the native **8B drafter**, whereas the representation reference in the loss is the **4B source model**. Do not confuse those two references.

## 4. Does MSE actually restore the expected features?

Yes, on the measured training positions. The geometry probe used 64 training records and 7,090 positions: 2,994 prompt and 4,096 response. It sampled up to 64 available positions per segment per record. Segment statistics pool positions; they do not give every record equal weight. This differs from the training loss's record weighting.

A feature error near zero means a mapped vector resembles its paired 4B vector. Cosine near one means their directions agree. Context error repeats the comparison **after frozen fusion and RMSNorm**. These geometry errors use a denominator floor $\max(\|v\|^2,10^{-6})$, rather than the training loss's additive epsilon.


| Objective | Prompt context error | Response context error |
| --- | --- | --- |
| mse | 0.216170 | 0.136147 |
| auf | 1.228544 | 0.940413 |
| decay7 | 1.189191 | 0.899731 |

| Objective | Layer | Segment | Feature error | Source cosine |
| --- | --- | --- | --- | --- |
| mse | 0 | prompt | 0.060359 | 0.975595 |
| mse | 0 | response | 0.033437 | 0.984793 |
| mse | 1 | prompt | 0.191632 | 0.899016 |
| mse | 1 | response | 0.068966 | 0.965129 |
| mse | 2 | prompt | 0.192591 | 0.901045 |
| mse | 2 | response | 0.083779 | 0.957394 |
| mse | 3 | prompt | 0.191474 | 0.902611 |
| mse | 3 | response | 0.110141 | 0.944060 |
| mse | 4 | prompt | 0.180501 | 0.909023 |
| mse | 4 | response | 0.090359 | 0.954783 |
| auf | 0 | prompt | 2.742802 | 0.195425 |
| auf | 0 | response | 2.794326 | 0.214868 |
| auf | 1 | prompt | 2.859036 | 0.112855 |
| auf | 1 | response | 2.540209 | 0.120631 |
| auf | 2 | prompt | 21.592222 | 0.141729 |
| auf | 2 | response | 12.066430 | 0.175011 |
| auf | 3 | prompt | 6.882195 | 0.144134 |
| auf | 3 | response | 4.716092 | 0.226237 |
| auf | 4 | prompt | 3.814032 | 0.190191 |
| auf | 4 | response | 3.408715 | 0.266906 |
| decay7 | 0 | prompt | 2.710681 | 0.201154 |
| decay7 | 0 | response | 2.753600 | 0.220411 |
| decay7 | 1 | prompt | 2.827380 | 0.119284 |
| decay7 | 1 | response | 2.522013 | 0.126873 |
| decay7 | 2 | prompt | 21.544376 | 0.145896 |
| decay7 | 2 | response | 12.034999 | 0.179880 |
| decay7 | 3 | prompt | 6.830830 | 0.151039 |
| decay7 | 3 | response | 4.675968 | 0.234286 |
| decay7 | 4 | prompt | 3.730459 | 0.201127 |
| decay7 | 4 | response | 3.298069 | 0.278744 |


MSE response-layer cosines are 0.944–0.985, compared with 0.121–0.267 for AUF. Its response context error is 0.136, versus 0.940 for AUF and 0.900 for decay. However, MSE was explicitly trained to minimize these errors. **The stronger question is whether this also helps on unseen problems and in actual decoding.**

## 5. Held-out test: same inputs, different maps

Eight prompt-length quantiles were selected per task, giving 32 examples outside the training manifest. Each uses the same saved native-target continuation across variants. Each example contributes eight identically seeded draft blocks: **64 blocks per task per variant**. This is a small diagnostic subset of the existing evaluation set, not a fresh final test set.

Read the columns as follows:

- **Full CE:** average negative log probability of the saved correct token over all valid block labels, irrespective of which loss trained the mapper. Lower is better.
- **AUF support:** percentage of valid labels that would receive AUF loss under these predictions. It includes the first wrong token; it is not a correctness percentage.
- **Correct prefix:** how many consecutive draft predictions are correct before the first error, averaged over blocks, without a bonus token. This is teacher-forced, offline measurement—not live acceptance length.
- **Prompt/response error:** normalized context reconstruction error. Positions are averaged within each record and then the eight records are averaged equally for each segment.

“Halfway” means $W_i^{half}=(W_i^{token}+W_i^{MSE})/2$, applied to all five maps with **no retraining**. “Warm seen/unseen” means AUF refinement from the MSE parent with overlapping/disjoint training examples, respectively.


### MATH

| Variant | Full CE ↓ | AUF support | Correct prefix ↑ | Prompt error ↓ | Response error ↓ |
| --- | --- | --- | --- | --- | --- |
| MSE | 1.6939 | 51.81% | 6.7656 | 0.1821 | 0.1236 |
| Decay halfway to MSE | 1.8043 | 49.79% | 6.4844 | 0.7590 | 0.4928 |
| Decay CE | 1.9534 | 47.13% | 6.0625 | 1.3852 | 0.9058 |
| Warm unseen 4k | 2.0634 | 52.45% | 6.9219 | 0.3910 | 0.3867 |
| Warm seen 4k | 2.0695 | 50.64% | 6.6094 | 0.3947 | 0.3827 |
| AUF halfway to MSE | 2.1282 | 48.51% | 6.2812 | 0.7768 | 0.5110 |
| Warm unseen 8k | 2.1507 | 54.47% | 7.2188 | 0.4347 | 0.4084 |
| Warm seen 8k | 2.1760 | 56.49% | 7.5625 | 0.4362 | 0.4088 |
| AUF | 2.6028 | 45.74% | 5.8438 | 1.4131 | 0.9333 |
| Random initial | 8.6261 | 6.81% | 0.0000 | 4.4041 | 4.9689 |



### GSM

| Variant | Full CE ↓ | AUF support | Correct prefix ↑ | Prompt error ↓ | Response error ↓ |
| --- | --- | --- | --- | --- | --- |
| Decay halfway to MSE | 1.5407 | 55.01% | 7.1875 | 0.7967 | 0.4933 |
| Decay CE | 1.6564 | 50.70% | 6.5312 | 1.4673 | 0.8850 |
| Warm seen 4k | 1.7356 | 50.91% | 6.5625 | 0.4074 | 0.3686 |
| Warm seen 8k | 1.7586 | 57.48% | 7.5781 | 0.4506 | 0.3911 |
| Warm unseen 8k | 1.7793 | 55.65% | 7.3125 | 0.4535 | 0.3911 |
| MSE | 1.7817 | 53.07% | 6.8750 | 0.2026 | 0.1146 |
| Warm unseen 4k | 1.8745 | 55.65% | 7.2812 | 0.4053 | 0.3692 |
| AUF halfway to MSE | 1.9572 | 55.76% | 7.3125 | 0.8212 | 0.5098 |
| AUF | 2.0566 | 54.14% | 7.0781 | 1.5231 | 0.9074 |
| Random initial | 9.6898 | 7.00% | 0.0156 | 4.0335 | 4.6817 |



### CODE

| Variant | Full CE ↓ | AUF support | Correct prefix ↑ | Prompt error ↓ | Response error ↓ |
| --- | --- | --- | --- | --- | --- |
| MSE | 2.6039 | 40.46% | 4.8750 | 0.3942 | 0.2864 |
| Warm unseen 4k | 3.4776 | 32.24% | 3.6719 | 0.5446 | 0.4510 |
| Warm unseen 8k | 3.5000 | 32.79% | 3.7344 | 0.5785 | 0.4877 |
| Warm seen 8k | 3.5864 | 32.02% | 3.6250 | 0.5754 | 0.4818 |
| Warm seen 4k | 3.6100 | 33.77% | 3.8906 | 0.5456 | 0.4543 |
| Decay halfway to MSE | 3.7045 | 25.33% | 2.6406 | 0.9133 | 0.6970 |
| AUF halfway to MSE | 4.1200 | 28.62% | 3.1250 | 0.9301 | 0.7159 |
| Decay CE | 4.6468 | 17.98% | 1.5625 | 1.5436 | 1.2464 |
| AUF | 5.5081 | 17.00% | 1.4219 | 1.5820 | 1.2868 |
| Random initial | 9.8854 | 7.24% | 0.0312 | 3.7889 | 4.6308 |



### CHAT

| Variant | Full CE ↓ | AUF support | Correct prefix ↑ | Prompt error ↓ | Response error ↓ |
| --- | --- | --- | --- | --- | --- |
| MSE | 3.8074 | 28.50% | 2.9062 | 0.3017 | 0.3255 |
| Warm unseen 8k | 4.1924 | 26.64% | 2.6406 | 0.4872 | 0.4895 |
| Warm seen 4k | 4.2015 | 26.99% | 2.6719 | 0.4507 | 0.4547 |
| Warm seen 8k | 4.2630 | 24.65% | 2.3594 | 0.4868 | 0.4873 |
| Warm unseen 4k | 4.2904 | 24.42% | 2.3281 | 0.4504 | 0.4519 |
| Decay halfway to MSE | 4.5984 | 21.03% | 1.8750 | 0.9437 | 0.7632 |
| AUF halfway to MSE | 4.9769 | 22.78% | 2.1094 | 0.9668 | 0.7841 |
| Decay CE | 5.6398 | 16.36% | 1.2188 | 1.7617 | 1.3749 |
| AUF | 6.2094 | 13.55% | 0.8438 | 1.8059 | 1.4210 |
| Random initial | 10.7415 | 7.48% | 0.0000 | 4.9159 | 4.0375 |



**What these tables establish:** Code MSE has CE 2.604 versus AUF 5.508, and correct prefix 4.875 versus 1.422. Chat has CE 3.807 versus 6.209 and prefix 2.906 versus 0.844. This is more than a favorable reconstruction metric: the fixed drafter also predicts the target continuation better on these held-out inputs.

**Where that explanation stops:** GSM8K decay has lower CE than MSE but a shorter correct prefix; warm seen-8k has a longer prefix than MSE despite worse reconstruction. Math warm seen-8k also has a longer offline prefix, yet its full live evaluation is worse than MSE. Reconstruction, CE and an eight-example prefix probe cannot each be treated as a universal speed predictor.

## 6. Intervention: move the token-trained maps toward MSE

The halfway maps change the representations before the **same frozen drafter**. The table includes both original endpoints for comparison. Latency reduction here is relative to the corresponding cold token-trained endpoint, not native; MSE is compared to that same endpoint in its row. These endpoints were timed in separate passes, so exact timing deltas are descriptive.



| Task | Compared with | Variant | Mean TPS | Pooled TPS | × native | × cold parent | Total seconds | Latency reduction / parent | Acceptance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| code | AUF | MSE | 126.18 | 124.01 | 0.8322× | 2.1618× | 569.96 | +50.08% | 4.6912 |
| code | AUF | AUF halfway to MSE | 81.64 | 85.41 | 0.5384× | 1.3986× | 827.50 | +27.53% | 3.2512 |
| code | AUF | AUF | 58.37 | 61.90 | 0.3850× | 1.0000× | 1141.85 | +0.00% | 2.3456 |
| code | Decay CE | MSE | 126.18 | 124.01 | 0.8322× | 2.0498× | 569.96 | +47.75% | 4.6912 |
| code | Decay CE | Decay CE halfway to MSE | 84.68 | 88.03 | 0.5585× | 1.3756× | 802.88 | +26.39% | 3.3368 |
| code | Decay CE | Decay CE | 61.56 | 64.80 | 0.4060× | 1.0000× | 1090.76 | +0.00% | 2.4489 |
| math | AUF | MSE | 219.48 | 216.72 | 0.9986× | 1.1300× | 406.20 | +13.83% | 7.9776 |
| math | AUF | AUF halfway to MSE | 206.81 | 201.87 | 0.9410× | 1.0648× | 436.09 | +7.48% | 7.5624 |
| math | AUF | AUF | 194.22 | 186.76 | 0.8837× | 1.0000× | 471.37 | +0.00% | 6.8923 |
| math | Decay CE | MSE | 219.48 | 216.72 | 0.9986× | 1.1273× | 406.20 | +13.39% | 7.9776 |
| math | Decay CE | Decay CE halfway to MSE | 208.80 | 203.73 | 0.9500× | 1.0725× | 432.11 | +7.86% | 7.5748 |
| math | Decay CE | Decay CE | 194.69 | 187.71 | 0.8858× | 1.0000× | 468.98 | +0.00% | 6.9474 |



The halfway intervention improves both cold recipes on both tasks, but does not reach MSE. This supports a useful direction from the token-trained solutions toward the MSE solution. It does **not** isolate which layer or geometric property caused the improvement: all five maps change together, including their scales. It is neither MSE+AUF training nor an ensemble of two drafters.

## 7. Warm AUF: does a good starting interface solve everything?

No. Starting from the 8k/MSE50 parent, all four three-epoch refinements improve GSM8K but reduce throughput on Math, Code and Chat. The table shows all 16 endpoints examined in the held-out diagnostic. Latency reduction is against native; fit time includes the MSE parent and AUF stage.



| Task | AUF data | × native | × MSE parent | Mean TPS | Latency reduction / native | Acceptance | Total fit min |
| --- | --- | --- | --- | --- | --- | --- | --- |
| chat | unseen 4k | 0.8548× | 0.9624× | 76.51 | -17.33% | 2.7495 | 32.39 |
| chat | seen 8k | 0.8446× | 0.9509× | 75.59 | -19.06% | 2.7038 | 45.32 |
| chat | unseen 8k | 0.8431× | 0.9492× | 75.46 | -19.32% | 2.7030 | 47.42 |
| chat | seen 4k | 0.8420× | 0.9480× | 75.36 | -18.58% | 2.7403 | 31.52 |
| code | unseen 4k | 0.7417× | 0.8913× | 112.46 | -26.52% | 4.3499 | 32.39 |
| code | seen 4k | 0.7273× | 0.8740× | 110.28 | -27.86% | 4.2769 | 31.52 |
| code | unseen 8k | 0.7051× | 0.8473× | 106.92 | -31.27% | 4.1818 | 47.42 |
| code | seen 8k | 0.7043× | 0.8463× | 106.78 | -32.01% | 4.1525 | 45.32 |
| gsm | unseen 8k | 1.0606× | 1.0874× | 188.72 | +5.53% | 6.6218 | 47.42 |
| gsm | seen 8k | 1.0431× | 1.0696× | 185.62 | +3.50% | 6.5891 | 45.32 |
| gsm | unseen 4k | 1.0413× | 1.0677× | 185.30 | +4.01% | 6.5340 | 32.39 |
| gsm | seen 4k | 1.0339× | 1.0601× | 183.97 | +3.40% | 6.4951 | 31.52 |
| math | seen 8k | 0.9927× | 0.9941× | 218.18 | -1.32% | 7.8662 | 45.32 |
| math | unseen 8k | 0.9911× | 0.9925× | 217.82 | -1.83% | 7.8410 | 47.42 |
| math | seen 4k | 0.9717× | 0.9731× | 213.57 | -3.20% | 7.8145 | 31.52 |
| math | unseen 4k | 0.9694× | 0.9707× | 213.05 | -3.42% | 7.7785 | 32.39 |



This rules out the simple explanation “AUF only loses because it started randomly” as a complete account. A good MSE initialization helps substantially relative to cold AUF, but AUF can still trade away cross-workload performance. That is consistent with specialization of token prediction at the expense of broadly useful feature compatibility. It is **not proof of overfitting**: we do not have the matched training/held-out curves or controlled objective experiment needed to establish that.

## 8. What parts of the learned AUF maps are necessary?

These are the older, completed **AUF-parent** interventions. They are not interventions on the MSE checkpoint. Let $W_{0,i}$ be the random initial map and $\Delta W_i=W_i-W_{0,i}$ the learned change.

- **restore0–restore4:** reset just that layer's map to its random initial value. The numbers identify slots 0–4.
- **delta1/2/32/128:** keep only that many singular directions of each $\Delta W_i$, then add the result to $W_{0,i}$. The original map itself is not truncated.
- **fusion_remove:** find the leading 32 right-singular directions of $F_i\Delta W_i$ in the 8B input space, and remove the update's action on those directions: $W'_{i}=W_{0,i}+\Delta W_i(I-Q_iQ_i^\top)$.
- **random_remove:** the same operation using 32 random orthonormal directions per layer. This controls for simply removing a small input subspace.

For the directional interventions, define the pre-RMS initial context $z_0=\sum_iF_iW_{0,i}h_i$ and learned change $\delta=\sum_iF_i\Delta W_i h_i$. Decompose

$$
\delta_{\parallel}=\frac{\delta^\top z_0}{\max(\|z_0\|^2,10^{-12})}z_0,\qquad
\delta_{\perp}=\delta-\delta_{\parallel}.
$$

**Radial** feeds $z_0+\delta_{\parallel}$ to RMSNorm. **Perpendicular** feeds $z_0+\delta_{\perp}$. **Hook control** reconstructs $z_0+\delta$ through the same intervention path. The reference direction is the **randomly initialized mapped context**, not a pretrained identity interface and not the native 8B context.

### Full Math intervention results

These 17 complete cohorts all agree with their specified matching control on 128/128 outputs. “Latency reduction” below is against that control; negative means the intervention took longer. The native TPS denominator is the existing Math native control, while the intervention-control denominator is node-matched. Timing passes are separate.


| Intervention | Node | Control | Mean TPS | × native (TPS) | Summed request seconds | Acceptance | TPS × control | Latency speedup × control | Latency reduction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unmodified | node08 | unmodified | 195.3339 | 0.8887 | 469.13 | 6.8923 | 1.0000 | 1.0000 | 0.00% |
| unmodified | node09 | unmodified | 194.7817 | 0.8862 | 470.59 | 6.8923 | 1.0000 | 1.0000 | 0.00% |
| random_remove | node09 | unmodified | 194.4424 | 0.8847 | 470.94 | 6.8735 | 0.9983 | 0.9993 | -0.07% |
| hook_control | node08 | hook_control | 193.6967 | 0.8813 | 472.79 | 6.8956 | 1.0000 | 1.0000 | 0.00% |
| hook_control | node09 | hook_control | 193.4648 | 0.8802 | 473.26 | 6.8956 | 1.0000 | 1.0000 | 0.00% |
| restore1 | node09 | unmodified | 189.0736 | 0.8603 | 482.30 | 6.7770 | 0.9707 | 0.9757 | -2.49% |
| restore0 | node08 | unmodified | 188.0331 | 0.8555 | 488.25 | 6.6363 | 0.9626 | 0.9608 | -4.07% |
| restore2 | node08 | unmodified | 180.0708 | 0.8193 | 511.39 | 6.3676 | 0.9219 | 0.9174 | -9.01% |
| perpendicular | node09 | hook_control | 155.3280 | 0.7067 | 601.45 | 5.4175 | 0.8029 | 0.7869 | -27.09% |
| delta128 | node09 | unmodified | 153.9099 | 0.7003 | 601.05 | 5.4182 | 0.7902 | 0.7829 | -27.72% |
| restore3 | node09 | unmodified | 100.1638 | 0.4557 | 905.34 | 3.5735 | 0.5142 | 0.5198 | -92.39% |
| restore4 | node08 | unmodified | 71.1581 | 0.3238 | 1386.93 | 2.3407 | 0.3643 | 0.3383 | -195.64% |
| delta32 | node08 | unmodified | 58.2957 | 0.2652 | 1542.19 | 2.1151 | 0.2984 | 0.3042 | -228.73% |
| fusion_remove | node08 | unmodified | 38.6362 | 0.1758 | 2288.12 | 1.4086 | 0.1978 | 0.2050 | -387.73% |
| delta2 | node09 | unmodified | 29.2849 | 0.1332 | 3067.04 | 1.0780 | 0.1503 | 0.1534 | -551.75% |
| delta1 | node08 | unmodified | 29.0863 | 0.1323 | 3112.75 | 1.0377 | 0.1489 | 0.1507 | -563.51% |
| radial | node08 | hook_control | 28.2457 | 0.1285 | 3185.37 | 1.0155 | 0.1458 | 0.1484 | -573.73% |


Resetting slots 3 or 4 severely damages acceptance; resetting earlier slots has smaller effects. Removing fusion-sensitive directions is disastrous, whereas removing random directions barely changes performance. These interventions establish that the learned interface is functionally used, not merely a harmless extra matrix.

Rank-two truncation nearly eliminates useful drafting. Even rank 128 retains only about 79% of parent mean TPS. This cross-size AUF update is not demonstrated to be compressible to the tiny ranks that worked for some LoRA updates.

Radial-only drafting also collapses. Perpendicular-only drafting is substantially better, but still loses about 20% of parent mean TPS. It would be incorrect to conclude that the parallel component is universally useless: this is a finite intervention relative to a random context, with RMSNorm after the sum. Positive pure rescaling is approximately removed by RMSNorm, but removing one component from a finite combined update need not preserve the normalized direction of the full result.

### Code intervention status

The fusion-sensitive removal stopped after 39 prompts and random removal after 22. Their partial timings are not performance results. The Math removal finding is therefore not a completed cross-domain replication.

## 9. Energy measurements: what they mean, and what they do not

For singular values $\sigma_j$, the energy retained by the first $k$ directions is

$$
E_k=\frac{\sum_{j=1}^{k}\sigma_j^2}{\sum_j\sigma_j^2}.
$$

The probe measures three matrices separately: the full map $W_i$, its learned change $\Delta W_i$, and the fusion-transformed change $F_i\Delta W_i$. It also measures the actual correction vectors $F_i\Delta W_i h_i$ across sampled tokens, separately for prompt and response. These activation vectors are **not centered**: a constant mean contributes energy. Their dominant directions are fitted separately for each segment and layer.

A low-rank cloud of observed vectors does not imply a low-rank map works on every input. In particular, a large prompt correction-energy percentage is not evidence that response-time acceptance is preserved. The live truncation results above directly demonstrate why energy and speed must remain separate measurements.


| Objective | Layer | Top-2 map-update energy | Top-2 fused-update energy | Top-2 prompt correction energy | Top-2 response correction energy |
| --- | --- | --- | --- | --- | --- |
| mse | 0 | 2.52% | 30.19% | 76.19% | 58.65% |
| mse | 1 | 0.81% | 5.17% | 99.96% | 55.79% |
| mse | 2 | 1.04% | 2.38% | 99.79% | 84.34% |
| mse | 3 | 0.73% | 2.42% | 98.96% | 58.04% |
| mse | 4 | 0.71% | 3.75% | 93.50% | 67.39% |
| auf | 0 | 11.62% | 48.45% | 42.54% | 43.73% |
| auf | 1 | 5.12% | 22.07% | 99.38% | 51.03% |
| auf | 2 | 4.46% | 9.23% | 97.91% | 51.68% |
| auf | 3 | 3.16% | 10.75% | 84.13% | 48.91% |
| auf | 4 | 3.63% | 16.51% | 91.39% | 84.28% |
| decay7 | 0 | 10.38% | 47.21% | 42.56% | 43.27% |
| decay7 | 1 | 4.51% | 20.11% | 99.20% | 52.15% |
| decay7 | 2 | 4.18% | 8.74% | 97.27% | 50.14% |
| decay7 | 3 | 3.09% | 10.83% | 83.44% | 47.88% |
| decay7 | 4 | 3.41% | 15.95% | 91.70% | 84.10% |


### Complete recorded spectra

Each cell lists percentages at **rank 1 / rank 2 / rank 32 / rank 128**, in that order. These include the less visually striking values, rather than selecting only the top-two result. They are descriptive measurements, not four additional validated deployment variants per row.



| Objective | Slot | Quantity | Segment | Energy % at ranks 1 / 2 / 32 / 128 |
| --- | --- | --- | --- | --- |
| auf | 0 | W_matrix_energy | not applicable | 0.60 / 1.17 / 7.99 / 18.46 |
| auf | 0 | activation_correction_energy | prompt | 27.20 / 42.54 / 97.22 / 99.90 |
| auf | 0 | activation_correction_energy | response | 24.78 / 43.73 / 97.01 / 99.89 |
| auf | 0 | delta_matrix_energy | not applicable | 6.07 / 11.62 / 65.11 / 91.55 |
| auf | 0 | fused_delta_matrix_energy | not applicable | 27.65 / 48.45 / 95.18 / 99.44 |
| auf | 1 | W_matrix_energy | not applicable | 0.23 / 0.45 / 4.79 / 15.42 |
| auf | 1 | activation_correction_energy | prompt | 98.88 / 99.38 / 99.87 / 99.99 |
| auf | 1 | activation_correction_energy | response | 44.43 / 51.03 / 88.57 / 98.78 |
| auf | 1 | delta_matrix_energy | not applicable | 2.61 / 5.12 / 39.31 / 74.99 |
| auf | 1 | fused_delta_matrix_energy | not applicable | 11.30 / 22.07 / 69.40 / 91.43 |
| auf | 2 | W_matrix_energy | not applicable | 0.20 / 0.40 / 4.72 / 15.38 |
| auf | 2 | activation_correction_energy | prompt | 96.96 / 97.91 / 99.55 / 99.96 |
| auf | 2 | activation_correction_energy | response | 45.33 / 51.68 / 86.62 / 98.92 |
| auf | 2 | delta_matrix_energy | not applicable | 2.24 / 4.46 / 39.08 / 78.04 |
| auf | 2 | fused_delta_matrix_energy | not applicable | 5.19 / 9.23 / 59.83 / 92.77 |
| auf | 3 | W_matrix_energy | not applicable | 0.16 / 0.31 / 4.13 / 14.67 |
| auf | 3 | activation_correction_energy | prompt | 78.41 / 84.13 / 94.49 / 99.01 |
| auf | 3 | activation_correction_energy | response | 44.36 / 48.91 / 80.41 / 96.90 |
| auf | 3 | delta_matrix_energy | not applicable | 1.69 / 3.16 / 29.35 / 68.51 |
| auf | 3 | fused_delta_matrix_energy | not applicable | 6.27 / 10.75 / 51.55 / 85.96 |
| auf | 4 | W_matrix_energy | not applicable | 0.21 / 0.36 / 4.04 / 14.49 |
| auf | 4 | activation_correction_energy | prompt | 75.94 / 91.39 / 97.03 / 99.22 |
| auf | 4 | activation_correction_energy | response | 82.31 / 84.28 / 94.56 / 98.73 |
| auf | 4 | delta_matrix_energy | not applicable | 2.38 / 3.63 / 23.66 / 57.51 |
| auf | 4 | fused_delta_matrix_energy | not applicable | 12.13 / 16.51 / 37.93 / 68.53 |
| decay7 | 0 | W_matrix_energy | not applicable | 0.55 / 1.08 / 7.82 / 18.41 |
| decay7 | 0 | activation_correction_energy | prompt | 27.52 / 42.56 / 96.80 / 99.87 |
| decay7 | 0 | activation_correction_energy | response | 24.71 / 43.27 / 96.58 / 99.87 |
| decay7 | 0 | delta_matrix_energy | not applicable | 5.38 / 10.38 / 61.85 / 90.42 |
| decay7 | 0 | fused_delta_matrix_energy | not applicable | 27.17 / 47.21 / 94.31 / 99.31 |
| decay7 | 1 | W_matrix_energy | not applicable | 0.23 / 0.44 / 4.75 / 15.41 |
| decay7 | 1 | activation_correction_energy | prompt | 98.65 / 99.20 / 99.83 / 99.98 |
| decay7 | 1 | activation_correction_energy | response | 45.51 / 52.15 / 87.90 / 98.60 |
| decay7 | 1 | delta_matrix_energy | not applicable | 2.33 / 4.51 / 36.47 / 72.57 |
| decay7 | 1 | fused_delta_matrix_energy | not applicable | 10.35 / 20.11 / 66.59 / 90.07 |
| decay7 | 2 | W_matrix_energy | not applicable | 0.21 / 0.40 / 4.75 / 15.46 |
| decay7 | 2 | activation_correction_energy | prompt | 96.22 / 97.27 / 99.42 / 99.94 |
| decay7 | 2 | activation_correction_energy | response | 43.81 / 50.14 / 85.73 / 98.74 |
| decay7 | 2 | delta_matrix_energy | not applicable | 2.15 / 4.18 / 37.31 / 76.74 |
| decay7 | 2 | fused_delta_matrix_energy | not applicable | 4.66 / 8.74 / 58.60 / 92.27 |
| decay7 | 3 | W_matrix_energy | not applicable | 0.16 / 0.32 / 4.15 / 14.71 |
| decay7 | 3 | activation_correction_energy | prompt | 78.20 / 83.44 / 94.03 / 98.88 |
| decay7 | 3 | activation_correction_energy | response | 43.29 / 47.88 / 79.25 / 96.57 |
| decay7 | 3 | delta_matrix_energy | not applicable | 1.70 / 3.09 / 28.76 / 67.87 |
| decay7 | 3 | fused_delta_matrix_energy | not applicable | 6.31 / 10.83 / 51.69 / 86.02 |
| decay7 | 4 | W_matrix_energy | not applicable | 0.22 / 0.36 / 4.03 / 14.46 |
| decay7 | 4 | activation_correction_energy | prompt | 76.39 / 91.70 / 96.98 / 99.16 |
| decay7 | 4 | activation_correction_energy | response | 82.12 / 84.10 / 94.19 / 98.55 |
| decay7 | 4 | delta_matrix_energy | not applicable | 2.27 / 3.41 / 21.76 / 54.77 |
| decay7 | 4 | fused_delta_matrix_energy | not applicable | 11.99 / 15.95 / 36.06 / 66.32 |
| mse | 0 | W_matrix_energy | not applicable | 3.60 / 5.68 / 12.97 / 26.35 |
| mse | 0 | activation_correction_energy | prompt | 50.63 / 76.19 / 92.43 / 97.83 |
| mse | 0 | activation_correction_energy | response | 52.06 / 58.65 / 85.34 / 95.59 |
| mse | 0 | delta_matrix_energy | not applicable | 1.59 / 2.52 / 7.16 / 18.24 |
| mse | 0 | fused_delta_matrix_energy | not applicable | 22.80 / 30.19 / 60.20 / 77.17 |
| mse | 1 | W_matrix_energy | not applicable | 1.59 / 2.54 / 9.15 / 22.82 |
| mse | 1 | activation_correction_energy | prompt | 99.94 / 99.96 / 99.98 / 99.99 |
| mse | 1 | activation_correction_energy | response | 52.61 / 55.79 / 75.67 / 89.65 |
| mse | 1 | delta_matrix_energy | not applicable | 0.48 / 0.81 / 4.76 / 15.52 |
| mse | 1 | fused_delta_matrix_energy | not applicable | 2.70 / 5.17 / 23.27 / 46.80 |
| mse | 2 | W_matrix_energy | not applicable | 2.39 / 3.83 / 10.73 / 22.91 |
| mse | 2 | activation_correction_energy | prompt | 99.68 / 99.79 / 99.94 / 99.98 |
| mse | 2 | activation_correction_energy | response | 69.37 / 84.34 / 91.60 / 96.67 |
| mse | 2 | delta_matrix_energy | not applicable | 0.63 / 1.04 / 5.05 / 15.99 |
| mse | 2 | fused_delta_matrix_energy | not applicable | 1.24 / 2.38 / 24.34 / 57.01 |
| mse | 3 | W_matrix_energy | not applicable | 1.63 / 2.46 / 9.05 / 21.15 |
| mse | 3 | activation_correction_energy | prompt | 98.43 / 98.96 / 99.52 / 99.78 |
| mse | 3 | activation_correction_energy | response | 48.60 / 58.04 / 76.42 / 89.34 |
| mse | 3 | delta_matrix_energy | not applicable | 0.46 / 0.73 / 4.78 / 16.00 |
| mse | 3 | fused_delta_matrix_energy | not applicable | 1.27 / 2.42 / 19.60 / 47.95 |
| mse | 4 | W_matrix_energy | not applicable | 2.39 / 3.60 / 9.94 / 21.23 |
| mse | 4 | activation_correction_energy | prompt | 88.05 / 93.50 / 96.99 / 98.29 |
| mse | 4 | activation_correction_energy | response | 64.72 / 67.39 / 80.66 / 89.34 |
| mse | 4 | delta_matrix_energy | not applicable | 0.46 / 0.71 / 4.47 / 15.06 |
| mse | 4 | fused_delta_matrix_energy | not applicable | 2.35 / 3.75 / 10.83 / 28.20 |



## 10. Does rectangular-identity initialization close the gap?

**Probe conclusion — complete.** Retain Xavier-initialized MSE as the selected recipe. The identity/LR retry improves AUF and decay CE, but neither overtakes MSE in average acceptance length on any of the four datasets. MSE itself loses measured TPS under identity initialization. GSM8K is the exception to MSE’s TPS lead among the new fits: AUF is faster on this 128-prompt cohort, while MSE still has the longest average acceptance. Treat this as a measured timing/acceptance disagreement, not proof of timing noise or of uniform MSE superiority in wall-clock speed. The improvement of AUF/decay cannot be attributed to initialization separately from their LR increase.

**Completed, 15 September 2026.** T1: Qwen3-8B target with the frozen Qwen3-4B DFlash drafter. All five maps start at the rectangular identity

$$
W_i^{(0)}=[I_{2560}\;0_{2560\times1536}],\qquad i=1,\ldots,5.
$$

This copies the first 2,560 target coordinates initially; it does not imply that corresponding coordinates have the same meaning across models. The frozen fusion, RMSNorm, embeddings, vocabulary head and draft body are retained. All three fits use the same 8,192 original training examples, three epochs, final checkpoints and LR $10^{-3}$. Training responses retain the original 4,096-token cap. MSE uses the original normalized layer-plus-context reconstruction loss on fixed 25% prompt/response positions. AUF and decay CE retain 32 response anchors and their original loss definitions; decay uses $\gamma=7$. Optimizer, global batch eight and scheduling follow the original loops.

Evaluation uses Transformers, 128 prompts per dataset and a 2,048-token output cap. Four independent one-GPU shards cover disjoint prompts; this is still single-request decoding, not concurrent serving. TPS is the arithmetic mean of per-request tokens/sec. Average acceptance length pools the recorded emitted block lengths across verification steps, including the target/bonus contribution. Native timing and token references are reused; no new baseline is used in these comparisons. Rows are ordered by speed versus native within each dataset.

| Dataset | Loss | TPS | Avg. acceptance length | Speed vs native |
| --- | --- | ---: | ---: | ---: |
| Math | MSE25 | 213.70 | 7.983 | 0.9723× |
| Math | AUF | 206.16 | 7.326 | 0.9380× |
| Math | Decay CE | 203.74 | 7.348 | 0.9270× |
| GSM8K | AUF | 175.35 | 6.166 | 0.9854× |
| GSM8K | Decay CE | 169.72 | 6.073 | 0.9538× |
| GSM8K | MSE25 | 166.70 | 6.217 | 0.9368× |
| Code | MSE25 | 119.49 | 4.569 | 0.7881× |
| Code | Decay CE | 76.02 | 3.042 | 0.5014× |
| Code | AUF | 73.10 | 2.931 | 0.4821× |
| Chat | MSE25 | 76.13 | 2.747 | 0.8506× |
| Chat | Decay CE | 61.11 | 2.183 | 0.6828× |
| Chat | AUF | 60.77 | 2.163 | 0.6790× |

### Old versus identity initialization

| Dataset | Loss | Old TPS | Identity TPS | TPS change | Old avg. length | Identity avg. length | Length change |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Math | MSE25 | 218.75 | 213.70 | -2.31% | 7.973 | 7.983 | +0.14% |
| Math | AUF | 194.22 | 206.16 | +6.15% | 6.892 | 7.326 | +6.29% |
| Math | Decay CE | 194.69 | 203.74 | +4.65% | 6.947 | 7.348 | +5.77% |
| GSM8K | AUF | 162.05 | 175.35 | +8.21% | 5.677 | 6.166 | +8.60% |
| GSM8K | Decay CE | 163.47 | 169.72 | +3.82% | 5.693 | 6.073 | +6.68% |
| GSM8K | MSE25 | 172.35 | 166.70 | -3.28% | 6.204 | 6.217 | +0.21% |
| Code | MSE25 | 125.98 | 119.49 | -5.15% | 4.690 | 4.569 | -2.58% |
| Code | Decay CE | 61.56 | 76.02 | +23.49% | 2.449 | 3.042 | +24.20% |
| Code | AUF | 58.37 | 73.10 | +25.24% | 2.346 | 2.931 | +24.94% |
| Chat | MSE25 | 78.31 | 76.13 | -2.78% | 2.802 | 2.747 | -1.94% |
| Chat | Decay CE | 51.31 | 61.11 | +19.09% | 1.812 | 2.183 | +20.52% |
| Chat | AUF | 49.93 | 60.77 | +21.73% | 1.767 | 2.163 | +22.41% |

**What this establishes.** Among these three new fits, MSE25 has the highest average acceptance length on **all four datasets**, and the highest measured TPS on Math, Code and Chat. AUF has the highest measured GSM8K TPS (175.35 versus 166.70 for MSE), despite slightly lower acceptance (6.166 versus 6.217). Acceptance length is not the only determinant of elapsed time; this single 128-prompt measurement does not establish that timing noise alone explains the reversal. There are no repeated timing trials here.

Relative to the old runs, the combined identity initialization and higher LR improve AUF/decay acceptance and TPS on every dataset, with approximately 19–25% TPS gains on Code/Chat. These changes still do not close the acceptance-length gap to MSE. **Their causal contributions cannot be separated:** old AUF/decay used Xavier initialization and LR $10^{-4}$, whereas new AUF/decay use identity and LR $10^{-3}$.

MSE retains LR $10^{-3}$, making its comparison an initializer-only intervention. Identity yields 2.31–5.15% lower measured TPS on all four datasets, essentially unchanged/slightly higher acceptance on Math/GSM (+0.14%/+0.21%), and lower acceptance on Code/Chat (−2.58%/−1.94%). Thus identity offers no demonstrated advantage for MSE; it does **not** uniformly worsen acceptance. This supports retaining the original Xavier MSE recipe, without claiming that these runs prove why MSE works better or that identity alone helped the token losses.

| Loss | Fit wall time | Fit GPU-hours (two GPUs) |
| --- | ---: | ---: |
| MSE25 | 11.11 min | 0.370 |
| AUF | 30.14 min | 1.005 |
| Decay CE | 29.54 min | 0.985 |

Fits: 34146–34148. Mapped evaluation shards: 34160–34171. Source and machine-readable comparison: `experiments/dflash_identity_20260915/`; final checkpoints remain in the original Phase 2 `runs/T1_identity_*_n8192_e3/` directories.

## 11. What explanation can we defend?

**Observed:** MSE restores the paired source features and normalized context far more closely. On held-out Code/Chat it also improves token prediction and correct-prefix length, and on all four full workloads it beats the tested cold token-loss endpoints. Moving token-loss maps halfway toward MSE improves live acceptance and throughput. AUF refinement from MSE helps GSM8K while weakening the other workloads.

**Interpretation:** changing model size creates a substantial interface mismatch for a frozen pretrained drafter. Direct feature supervision provides a strong way to repair it across many prompt and response positions. Token supervision must discover a useful interface indirectly through the frozen draft network; AUF additionally suppresses later-position losses when early predictions are wrong. The initial held-out AUF support is only about 7%, making this a plausible obstacle to learning a broad interface from scratch. That support is measured after the fact on fixed diagnostic blocks, not a recorded training-time gradient history.

**Why this can differ from LoRA:** same-model adaptation starts with a usable pretrained interface and often needs to adjust proposal behavior toward the adapted target. Cross-size transfer starts here with random rectangular maps and must first make a different representation usable. It is plausible that token objectives help the former more readily and reconstruction helps the latter. The experiments support this account but do not isolate target shift from architecture, data, initialization and optimization differences.

**What is not established:** that MSE alone caused the gain; that AUF cannot work at another learning rate or coverage; that reconstruction error uniquely determines acceptance; that low matrix rank explains MSE's advantage; that the Code/Chat decline is proven overfitting; or that any geometry percentage is a speedup-retention percentage. The halfway intervention changes many properties together. The sample has eight diagnostic examples per domain and no multi-seed uncertainty analysis.

The defensible claim is: **in this T1 recipe comparison, restoring the source drafter's feature interface is associated with—and interventions support its role in—better transfer than cold token-loss training. Subsequent token optimization is workload-dependent.**

## Sources and scope checks

- [Phase 2 results](phase2_results.md): original endpoint and intervention reports.
- [Selected geometry and intervention implementation](experiments/dflash_mse100_20260914/probes_selected.py), [feature CSV](experiments/dflash_mse100_20260914/results/geometry_features.csv), [energy CSV](experiments/dflash_mse100_20260914/results/geometry_energy.csv).
- [Held-out diagnostic results](experiments/dflash_joint_completion_20260914/results/diagnostics.json), [diagnostic measurement code](experiments/dflash_joint_completion_20260914/diagnose.py), [fixed selection and midpoint definition](experiments/dflash_joint_completion_20260914/prepare.py).
- [All four completed live midpoint measurements](experiments/dflash_joint_completion_20260914/intervention_measurements/): fetched from the completed node09 artifacts for this report; the older local summary listed only the first cohort.
- [Warm refinement results](experiments/dflash_mse_auf_20260914/results/results.json): only the four 8k/MSE50-parent, epoch-three endpoints used by the diagnostic are included here.

No new fitting or evaluation was launched to write this document. Incomplete Code removal probes are marked incomplete. Historical source-MSE runs excluded from the current study are not imported. These numerical-runtime limitations remain relevant to the paper's losslessness claims.
