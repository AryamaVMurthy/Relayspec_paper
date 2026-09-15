# Phase 2 — complete results

<!-- AGREEMENT_NOTE -->
In the selected Transformers comparisons with native-drafter controls (T1, T2 and T4), native and adapted drafters matched all 128 output sequences per task. In vLLM, token sequences diverged, consistent with documented numerical and batching variability. The authors completed a manual audit of all divergent answers and found them semantically similar. Semantic similarity does not imply identical token counts. [vLLM documentation](https://docs.vllm.ai/en/stable/features/batch_invariance/).
<!-- /AGREEMENT_NOTE -->





**Current selected recipe:** five maps, 16,384 examples, 25% prompt+response MSE, three epochs, LR0.001. Optimized concurrency is complete; the historical studies below retain their own recipes.

<!-- STANDALONE_AR_COMPLETE_BEGIN -->
## Completed standalone AR baselines — 15 September 2026

All 16 T1–T4/task baselines are complete: 128 prompts per task, Transformers, 2,048-token response cap. T2 resumed saved partial measurements; T4 reuses its existing completed run. No concurrent-serving AR results are included here.

Mean TPS is the arithmetic mean of per-request output tokens/second. Pooled TPS is total output tokens divided by summed request seconds. A dash denotes the unavailable T3 native drafter.

| Transfer | Task | Prompts | AR mean TPS | AR pooled TPS | Native mean TPS | Mapped mean TPS | Native / AR | Mapped / AR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1 | math | 128/128 | 37.8373 | 37.7338 | 219.79 | 220.26 | 5.8088 | 5.8212 |
| T1 | gsm | 128/128 | 37.7622 | 37.7580 | 177.94 | 176.87 | 4.7121 | 4.6838 |
| T1 | code | 128/128 | 37.4970 | 37.4598 | 151.63 | 127.61 | 4.0438 | 3.4032 |
| T1 | chat | 128/128 | 35.0880 | 35.1556 | 89.50 | 80.93 | 2.5507 | 2.3065 |
| T2 | math | 128/128 | 39.4153 | 39.4552 | 224.52 | 216.37 | 5.6963 | 5.4895 |
| T2 | gsm | 128/128 | 39.7241 | 39.7216 | 177.32 | 169.53 | 4.4638 | 4.2677 |
| T2 | code | 128/128 | 39.6034 | 39.6920 | 171.68 | 130.83 | 4.3350 | 3.3035 |
| T2 | chat | 128/128 | 40.4786 | 40.5417 | 95.34 | 81.16 | 2.3553 | 2.0050 |
| T3 | math | 128/128 | 61.8956 | 62.0106 | — | 145.43 | — | 2.3496 |
| T3 | gsm | 128/128 | 64.4510 | 64.4528 | — | 140.60 | — | 2.1815 |
| T3 | code | 128/128 | 63.0876 | 63.1301 | — | 118.57 | — | 1.8794 |
| T3 | chat | 128/128 | 64.7250 | 64.8830 | — | 104.43 | — | 1.6134 |
| T4 | math | 128/128 | 41.0048 | 40.8312 | 130.07 | 121.94 | 3.1721 | 2.9738 |
| T4 | gsm | 128/128 | 41.1094 | 41.0486 | 127.35 | 90.84 | 3.0978 | 2.2097 |
| T4 | code | 128/128 | 40.7880 | 40.7296 | 122.44 | 80.60 | 3.0019 | 1.9761 |
| T4 | chat | 128/128 | 40.9593 | 40.9169 | 114.42 | 60.52 | 2.7935 | 1.4776 |

12/28 concurrent-serving AR comparisons are complete; unmeasured cells remain blank in the paper. Raw completed T2 measurements and the T4 reuse audit are retained under `experiments/dflash_ar_completion_20260915/`; the paper includes the complete AR inventory.
<!-- STANDALONE_AR_COMPLETE_END -->

<!-- OPTIMIZED_SERVING_BEGIN -->
## Optimized vLLM concurrency — complete, 15 September 2026

**52/52 complete cells; 6,656 requests.** Same selected five-map 16,384-example, 25% prompt+response MSE, three epochs, LR0.001 checkpoints. T1: 1/8/16/32 clients; T2–T4: 16 clients; 128 prompts per cell, response cap2,048, one L40S per engine. BF16, FlashAttention, 15 draft proposals, compilation mode3, automatic scheduling, capacity32, batch budget32,768, memory fraction0.8. Prefix caching is reset before each cell. T1–T3 use V1; T4 uses its V2 tokenizer bridge. vLLM0.28.0+cu129 / PyTorch2.13.0+cu129 / Transformers5.16.1.

Aggregate TPS = returned tokens / workload wall time including drain. Acceptance = 1 + accepted draft tokens / verification steps. Every retained cell has zero observed JIT events. Native/mapped inputs agree; output sequences can differ, and their equality counts are reported. T3 has no native drafter. AR and mapped output sequences can differ; ratios remain descriptive throughput comparisons.

| Pair | Task | Clients | AR TPS | Native TPS | Mapped TPS | Mapped/native | Mapped/AR | Native acceptance | Mapped acceptance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1 | chat | 16 |  | 1091.1376 | 1065.9270 | 0.9769 |  | 3.3116 | 2.8976 |
| T1 | chat | 8 |  | 705.5131 | 682.4391 | 0.9673 |  | 3.3055 | 2.9081 |
| T1 | chat | 32 |  | 1349.4410 | 1273.8183 | 0.9440 |  | 3.3548 | 2.9171 |
| T1 | chat | 1 |  | 114.8256 | 106.6839 | 0.9291 |  | 3.3250 | 2.8511 |
| T1 | code | 8 |  | 1106.8665 | 1045.0456 | 0.9441 |  | 5.7157 | 4.8930 |
| T1 | code | 32 |  | 1940.5186 | 1826.4532 | 0.9412 |  | 5.6801 | 4.8433 |
| T1 | code | 1 |  | 192.4316 | 176.5162 | 0.9173 |  | 5.7188 | 4.8261 |
| T1 | code | 16 |  | 1748.1943 | 1548.4744 | 0.8858 |  | 5.9057 | 4.8933 |
| T1 | gsm | 32 |  | 2251.4677 | 2612.5624 | 1.1604 |  | 6.4576 | 6.2556 |
| T1 | gsm | 8 |  | 1301.7095 | 1448.2625 | 1.1126 |  | 6.4364 | 6.2291 |
| T1 | gsm | 1 |  | 216.5400 | 227.9476 | 1.0527 |  | 6.4544 | 6.2485 |
| T1 | gsm | 16 |  | 2207.4294 | 2262.0264 | 1.0247 |  | 6.4501 | 6.2538 |
| T1 | math | 32 |  | 2866.9640 | 3317.4305 | 1.1571 |  | 8.0196 | 7.8274 |
| T1 | math | 16 |  | 2524.3059 | 2716.2608 | 1.0760 |  | 8.1100 | 7.9303 |
| T1 | math | 1 |  | 276.0892 | 292.2282 | 1.0585 |  | 8.1021 | 7.9168 |
| T1 | math | 8 |  | 1752.3877 | 1840.9399 | 1.0505 |  | 8.1382 | 7.9425 |
| T2 | chat | 16 | 941.9935 | 1888.6876 | 1431.2145 | 0.7578 | 1.5193 | 3.3935 | 2.7878 |
| T2 | code | 16 | 825.2923 | 2840.1513 | 2002.7911 | 0.7052 | 2.4268 | 6.1631 | 4.5371 |
| T2 | gsm | 16 | 1021.5417 | 3160.9924 | 3078.8715 | 0.9740 | 3.0139 | 6.3663 | 6.0125 |
| T2 | math | 16 | 959.4204 | 4482.4186 | 3710.7575 | 0.8278 | 3.8677 | 8.1087 | 7.5337 |
| T3 | chat | 16 | 1072.8188 | — | 1899.9516 | — | 1.7710 | — | 2.9890 |
| T3 | code | 16 | 1213.6652 | — | 2077.6174 | — | 1.7119 | — | 3.5285 |
| T3 | gsm | 16 | 727.3025 | — | 2241.5894 | — | 3.0821 | — | 3.8534 |
| T3 | math | 16 | 1126.9266 | — | 2856.0855 | — | 2.5344 | — | 4.7128 |
| T4 | chat | 16 | 520.0872 | 1440.0379 | 524.3994 | 0.3642 | 1.0083 | 3.9860 | 2.0426 |
| T4 | code | 16 | 585.5837 | 1296.8281 | 636.9735 | 0.4912 | 1.0878 | 4.3870 | 2.9868 |
| T4 | gsm | 16 | 394.4365 | 1354.5007 | 853.1347 | 0.6299 | 2.1629 | 4.2114 | 3.4393 |
| T4 | math | 16 | 558.2185 | 1894.9986 | 1046.9444 | 0.5525 | 1.8755 | 5.0914 | 4.6810 |


### Optimized serving latency

Times are seconds. Request latency begins after client-slot acquisition; TTFT/TPOT are request means. P50/P95 use linear interpolation.

| Pair | Task | Clients | Arm | Mean latency | P50 | P95 | Mean TTFT | Mean TPOT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1 | math | 1 | native | 2.5022 | 1.9783 | 6.0672 | 0.0610 | 0.0036 |
| T1 | math | 1 | mapped | 2.3954 | 1.9568 | 6.0645 | 0.0570 | 0.0034 |
| T1 | math | 8 | native | 3.0475 | 2.4700 | 7.5546 | 0.1085 | 0.0044 |
| T1 | math | 8 | mapped | 3.0165 | 2.3837 | 7.1910 | 0.1013 | 0.0042 |
| T1 | math | 16 | native | 3.7486 | 3.0312 | 9.5862 | 0.1404 | 0.0054 |
| T1 | math | 16 | mapped | 3.6437 | 2.9381 | 8.8155 | 0.1326 | 0.0052 |
| T1 | math | 32 | native | 6.0780 | 4.7099 | 14.7883 | 0.2343 | 0.0085 |
| T1 | math | 32 | mapped | 5.7893 | 4.5672 | 13.9775 | 0.2225 | 0.0082 |
| T1 | gsm | 1 | native | 1.2699 | 1.2208 | 2.1387 | 0.0607 | 0.0044 |
| T1 | gsm | 1 | mapped | 1.2084 | 1.1554 | 1.8884 | 0.0565 | 0.0042 |
| T1 | gsm | 8 | native | 1.5823 | 1.4651 | 2.4610 | 0.1023 | 0.0053 |
| T1 | gsm | 8 | mapped | 1.4633 | 1.4081 | 2.3123 | 0.0947 | 0.0051 |
| T1 | gsm | 16 | native | 1.8758 | 1.8070 | 3.0163 | 0.1321 | 0.0064 |
| T1 | gsm | 16 | mapped | 1.8032 | 1.7229 | 2.8949 | 0.1277 | 0.0061 |
| T1 | gsm | 32 | native | 2.9147 | 2.8591 | 4.7650 | 0.2301 | 0.0100 |
| T1 | gsm | 32 | mapped | 2.8004 | 2.7941 | 4.5333 | 0.2119 | 0.0097 |
| T1 | code | 1 | native | 2.9031 | 2.7523 | 6.3634 | 0.0797 | 0.0050 |
| T1 | code | 1 | mapped | 3.1931 | 2.9895 | 6.6035 | 0.0756 | 0.0056 |
| T1 | code | 8 | native | 3.8611 | 3.6013 | 9.0026 | 0.1471 | 0.0064 |
| T1 | code | 8 | mapped | 4.2089 | 3.8346 | 9.4944 | 0.1429 | 0.0072 |
| T1 | code | 16 | native | 4.8024 | 4.6220 | 10.4947 | 0.2645 | 0.0082 |
| T1 | code | 16 | mapped | 5.3376 | 4.9221 | 10.7961 | 0.2185 | 0.0092 |
| T1 | code | 32 | native | 7.8022 | 7.3494 | 18.3365 | 0.5312 | 0.0133 |
| T1 | code | 32 | mapped | 8.9815 | 8.1295 | 22.2090 | 0.5017 | 0.0149 |
| T1 | chat | 1 | native | 6.3228 | 5.7432 | 15.2275 | 0.0661 | 0.0095 |
| T1 | chat | 1 | mapped | 6.6403 | 5.8821 | 15.5002 | 0.0618 | 0.0102 |
| T1 | chat | 8 | native | 7.7411 | 6.6453 | 18.6244 | 0.1116 | 0.0117 |
| T1 | chat | 8 | mapped | 8.1193 | 7.6911 | 18.6664 | 0.1043 | 0.0124 |
| T1 | chat | 16 | native | 9.0748 | 8.4432 | 21.6225 | 0.1382 | 0.0140 |
| T1 | chat | 16 | mapped | 9.8675 | 8.8479 | 25.0961 | 0.1335 | 0.0149 |
| T1 | chat | 32 | native | 14.1901 | 13.0860 | 33.5474 | 0.2756 | 0.0222 |
| T1 | chat | 32 | mapped | 15.3876 | 13.8915 | 35.3062 | 0.2438 | 0.0235 |
| T2 | math | 16 | native | 2.3477 | 1.9228 | 5.3224 | 0.0838 | 0.0033 |
| T2 | math | 16 | mapped | 2.8383 | 2.2386 | 6.8471 | 0.0932 | 0.0039 |
| T2 | gsm | 16 | native | 1.1209 | 1.0238 | 1.8106 | 0.0764 | 0.0038 |
| T2 | gsm | 16 | mapped | 1.2991 | 1.1988 | 2.1079 | 0.0854 | 0.0045 |
| T2 | code | 16 | native | 2.6756 | 2.3462 | 7.0383 | 0.1391 | 0.0050 |
| T2 | code | 16 | mapped | 3.5451 | 3.1625 | 8.4194 | 0.1494 | 0.0074 |
| T2 | chat | 16 | native | 5.5901 | 4.9862 | 13.0436 | 0.0855 | 0.0086 |
| T2 | chat | 16 | mapped | 7.2925 | 6.6014 | 16.4874 | 0.0933 | 0.0111 |
| T3 | math | 16 | mapped | 2.6552 | 1.6690 | 7.6166 | 0.0764 | 0.0064 |
| T3 | gsm | 16 | mapped | 1.1155 | 0.9944 | 1.8925 | 0.0721 | 0.0058 |
| T3 | code | 16 | mapped | 2.9680 | 2.8203 | 5.2391 | 0.1205 | 0.0080 |
| T3 | chat | 16 | mapped | 3.9137 | 3.5926 | 7.9535 | 0.0743 | 0.0085 |
| T4 | math | 16 | native | 5.3021 | 3.3372 | 13.3923 | 0.1359 | 0.0094 |
| T4 | math | 16 | mapped | 8.1771 | 5.0493 | 23.5010 | 0.1912 | 0.0146 |
| T4 | gsm | 16 | native | 1.8637 | 1.7225 | 3.1809 | 0.1305 | 0.0092 |
| T4 | gsm | 16 | mapped | 3.3865 | 3.0428 | 5.4807 | 0.1665 | 0.0163 |
| T4 | code | 16 | native | 4.2101 | 3.7615 | 8.1708 | 0.2374 | 0.0110 |
| T4 | code | 16 | mapped | 8.6798 | 7.9233 | 16.7502 | 0.2989 | 0.0239 |
| T4 | chat | 16 | native | 5.2347 | 4.8845 | 9.6485 | 0.1389 | 0.0111 |
| T4 | chat | 16 | mapped | 13.8815 | 13.1287 | 27.4767 | 0.1883 | 0.0288 |

### Optimized serving output and memory accounting

Memory is the post-cell allocated/reserved worker snapshot in GiB, not peak memory.

| Pair | Task | Clients | Arm | Output tokens | Wall seconds | Accepted draft tokens | Verification steps | Allocated GiB | Reserved GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1 | math | 1 | native | 88428 | 320.2878 | 77626 | 10930 | 30.5914 | 30.8281 |
| T1 | math | 1 | mapped | 89603 | 306.6199 | 78326 | 11324 | 30.9190 | 31.2285 |
| T1 | math | 8 | native | 87904 | 50.1624 | 77207 | 10816 | 30.5914 | 31.0391 |
| T1 | math | 8 | mapped | 91446 | 49.6735 | 79978 | 11520 | 30.9190 | 31.4395 |
| T1 | math | 16 | native | 88970 | 35.2453 | 78111 | 10986 | 30.5914 | 31.3867 |
| T1 | math | 16 | mapped | 90527 | 33.3278 | 79151 | 11421 | 30.9190 | 31.7871 |
| T1 | math | 32 | native | 91908 | 32.0576 | 80585 | 11480 | 30.5914 | 32.2930 |
| T1 | math | 32 | mapped | 91230 | 27.5002 | 79607 | 11660 | 30.9190 | 32.7090 |
| T1 | gsm | 1 | native | 35201 | 162.5612 | 29841 | 5471 | 30.5914 | 32.2930 |
| T1 | gsm | 1 | mapped | 35259 | 154.6803 | 29612 | 5642 | 30.9190 | 32.7090 |
| T1 | gsm | 8 | native | 35953 | 27.6198 | 30460 | 5603 | 30.5914 | 32.2930 |
| T1 | gsm | 8 | mapped | 34857 | 24.0682 | 29257 | 5595 | 30.9190 | 32.7090 |
| T1 | gsm | 16 | native | 35186 | 15.9398 | 29823 | 5472 | 30.5914 | 32.2930 |
| T1 | gsm | 16 | mapped | 35313 | 15.6112 | 29663 | 5646 | 30.9190 | 32.7090 |
| T1 | gsm | 32 | native | 34926 | 15.5125 | 29613 | 5426 | 30.5914 | 32.2930 |
| T1 | gsm | 32 | mapped | 35005 | 13.3987 | 29405 | 5595 | 30.9190 | 32.7090 |
| T1 | code | 1 | native | 71508 | 371.6022 | 59126 | 12530 | 30.5914 | 32.2930 |
| T1 | code | 1 | mapped | 72148 | 408.7329 | 57212 | 14953 | 30.9190 | 32.7090 |
| T1 | code | 8 | native | 73176 | 66.1110 | 60502 | 12830 | 30.5914 | 32.2930 |
| T1 | code | 8 | mapped | 74255 | 71.0543 | 59112 | 15184 | 30.9190 | 32.7090 |
| T1 | code | 16 | native | 72589 | 41.5223 | 60428 | 12318 | 30.5914 | 32.5840 |
| T1 | code | 16 | mapped | 73832 | 47.6805 | 58766 | 15094 | 30.9190 | 33.0430 |
| T1 | code | 32 | native | 70883 | 36.5279 | 58501 | 12500 | 30.5914 | 34.2051 |
| T1 | code | 32 | mapped | 76011 | 41.6167 | 60352 | 15703 | 30.9190 | 34.6641 |
| T1 | chat | 1 | native | 92932 | 809.3316 | 65049 | 27978 | 30.5914 | 34.2051 |
| T1 | chat | 1 | mapped | 90678 | 849.9690 | 58892 | 31815 | 30.9190 | 34.6641 |
| T1 | chat | 8 | native | 92311 | 130.8424 | 64468 | 27963 | 30.5914 | 34.2051 |
| T1 | chat | 8 | mapped | 91975 | 134.7739 | 60366 | 31636 | 30.9190 | 31.7383 |
| T1 | chat | 16 | native | 90406 | 82.8548 | 63183 | 27333 | 30.5914 | 31.4941 |
| T1 | chat | 16 | mapped | 92152 | 86.4524 | 60372 | 31815 | 30.9190 | 32.0117 |
| T1 | chat | 32 | native | 91738 | 67.9822 | 64470 | 27378 | 30.5914 | 32.4512 |
| T1 | chat | 32 | mapped | 92651 | 72.7349 | 60911 | 31772 | 30.9190 | 32.9180 |
| T2 | math | 16 | native | 90452 | 20.1793 | 79433 | 11174 | 31.2812 | 31.8184 |
| T2 | math | 16 | mapped | 91256 | 24.5923 | 79175 | 12118 | 31.8356 | 32.4531 |
| T2 | gsm | 16 | native | 35396 | 11.1977 | 29928 | 5577 | 31.2812 | 31.8184 |
| T2 | gsm | 16 | mapped | 34718 | 11.2762 | 28942 | 5774 | 31.8356 | 32.4531 |
| T2 | code | 16 | native | 66900 | 23.5551 | 56195 | 10884 | 31.2812 | 32.6191 |
| T2 | code | 16 | mapped | 61617 | 30.7656 | 48058 | 13587 | 31.8356 | 33.2539 |
| T2 | chat | 16 | native | 91027 | 48.1959 | 64301 | 26865 | 31.2812 | 32.6191 |
| T2 | chat | 16 | mapped | 88848 | 62.0787 | 56987 | 31875 | 31.8356 | 33.2539 |
| T3 | math | 16 | mapped | 68413 | 23.9534 | 53847 | 14503 | 32.3260 | 33.0000 |
| T3 | gsm | 16 | mapped | 23541 | 10.5019 | 17337 | 6076 | 32.3260 | 33.0000 |
| T3 | code | 16 | mapped | 51379 | 24.7298 | 36737 | 14529 | 32.3260 | 33.8320 |
| T3 | chat | 16 | mapped | 64619 | 34.0109 | 42924 | 21581 | 32.3260 | 33.8320 |
| T4 | math | 16 | native | 86317 | 45.5499 | 69361 | 16953 | 30.9004 | 31.8594 |
| T4 | math | 16 | mapped | 83062 | 79.3375 | 65312 | 17743 | 30.8913 | 31.9961 |
| T4 | gsm | 16 | native | 24508 | 18.0938 | 18591 | 5789 | 30.9004 | 31.8594 |
| T4 | gsm | 16 | mapped | 29169 | 34.1904 | 20612 | 8450 | 30.8913 | 31.9961 |
| T4 | code | 16 | native | 50067 | 38.6073 | 38571 | 11388 | 30.9004 | 32.8242 |
| T4 | code | 16 | mapped | 48159 | 75.6060 | 31958 | 16085 | 30.8913 | 32.9609 |
| T4 | chat | 16 | native | 65217 | 45.2884 | 48774 | 16334 | 30.9004 | 32.8242 |
| T4 | chat | 16 | mapped | 62812 | 119.7789 | 32000 | 30692 | 30.8913 | 32.9609 |

### Optimized acceptance survival

L counts accepted draft tokens without the verifier bonus. All15 position probabilities are preserved in `paper/phase2/data/optimized_serving.json`.

| Pair | Task | Clients | Arm | P(L≥1) | P(L≥4) | P(L≥8) | P(L≥12) | P(L≥15) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1 | math | 1 | native | 0.9126 | 0.6310 | 0.4211 | 0.2931 | 0.2158 |
| T1 | math | 1 | mapped | 0.9031 | 0.6173 | 0.4075 | 0.2866 | 0.2073 |
| T1 | math | 8 | native | 0.9117 | 0.6398 | 0.4226 | 0.2965 | 0.2179 |
| T1 | math | 8 | mapped | 0.9031 | 0.6195 | 0.4098 | 0.2876 | 0.2095 |
| T1 | math | 16 | native | 0.9128 | 0.6349 | 0.4225 | 0.2951 | 0.2131 |
| T1 | math | 16 | mapped | 0.8997 | 0.6163 | 0.4079 | 0.2866 | 0.2122 |
| T1 | math | 32 | native | 0.9090 | 0.6243 | 0.4146 | 0.2927 | 0.2136 |
| T1 | math | 32 | mapped | 0.8981 | 0.6081 | 0.4017 | 0.2774 | 0.2016 |
| T1 | gsm | 1 | native | 0.8830 | 0.5323 | 0.2925 | 0.1649 | 0.0972 |
| T1 | gsm | 1 | mapped | 0.8719 | 0.5085 | 0.2806 | 0.1546 | 0.0909 |
| T1 | gsm | 8 | native | 0.8863 | 0.5279 | 0.2931 | 0.1631 | 0.0966 |
| T1 | gsm | 8 | mapped | 0.8744 | 0.5128 | 0.2790 | 0.1496 | 0.0865 |
| T1 | gsm | 16 | native | 0.8843 | 0.5283 | 0.2962 | 0.1625 | 0.0963 |
| T1 | gsm | 16 | mapped | 0.8760 | 0.5108 | 0.2798 | 0.1553 | 0.0930 |
| T1 | gsm | 32 | native | 0.8844 | 0.5367 | 0.2945 | 0.1627 | 0.0971 |
| T1 | gsm | 32 | mapped | 0.8745 | 0.5147 | 0.2785 | 0.1528 | 0.0895 |
| T1 | code | 1 | native | 0.8678 | 0.4592 | 0.2263 | 0.1277 | 0.0872 |
| T1 | code | 1 | mapped | 0.8284 | 0.3912 | 0.1553 | 0.0746 | 0.0462 |
| T1 | code | 8 | native | 0.8692 | 0.4544 | 0.2267 | 0.1279 | 0.0873 |
| T1 | code | 8 | mapped | 0.8255 | 0.3996 | 0.1606 | 0.0782 | 0.0513 |
| T1 | code | 16 | native | 0.8686 | 0.4707 | 0.2419 | 0.1414 | 0.1013 |
| T1 | code | 16 | mapped | 0.8303 | 0.3923 | 0.1616 | 0.0819 | 0.0537 |
| T1 | code | 32 | native | 0.8683 | 0.4554 | 0.2249 | 0.1232 | 0.0776 |
| T1 | code | 32 | mapped | 0.8277 | 0.3901 | 0.1585 | 0.0782 | 0.0477 |
| T1 | chat | 1 | native | 0.7118 | 0.2144 | 0.0667 | 0.0268 | 0.0153 |
| T1 | chat | 1 | mapped | 0.6397 | 0.1580 | 0.0425 | 0.0183 | 0.0108 |
| T1 | chat | 8 | native | 0.7055 | 0.2105 | 0.0647 | 0.0275 | 0.0178 |
| T1 | chat | 8 | mapped | 0.6406 | 0.1636 | 0.0482 | 0.0206 | 0.0124 |
| T1 | chat | 16 | native | 0.7095 | 0.2102 | 0.0639 | 0.0279 | 0.0175 |
| T1 | chat | 16 | mapped | 0.6391 | 0.1616 | 0.0468 | 0.0215 | 0.0142 |
| T1 | chat | 32 | native | 0.7113 | 0.2139 | 0.0684 | 0.0298 | 0.0184 |
| T1 | chat | 32 | mapped | 0.6380 | 0.1632 | 0.0485 | 0.0226 | 0.0151 |
| T2 | math | 16 | native | 0.9127 | 0.6334 | 0.4279 | 0.2943 | 0.2107 |
| T2 | math | 16 | mapped | 0.8999 | 0.5890 | 0.3823 | 0.2535 | 0.1782 |
| T2 | gsm | 16 | native | 0.8815 | 0.5334 | 0.2887 | 0.1524 | 0.0909 |
| T2 | gsm | 16 | mapped | 0.8715 | 0.5005 | 0.2587 | 0.1335 | 0.0788 |
| T2 | code | 16 | native | 0.8717 | 0.4912 | 0.2599 | 0.1624 | 0.1163 |
| T2 | code | 16 | mapped | 0.8083 | 0.3557 | 0.1376 | 0.0657 | 0.0372 |
| T2 | chat | 16 | native | 0.7055 | 0.2138 | 0.0717 | 0.0369 | 0.0250 |
| T2 | chat | 16 | mapped | 0.6293 | 0.1508 | 0.0409 | 0.0155 | 0.0085 |
| T3 | math | 16 | mapped | 0.8097 | 0.4124 | 0.1851 | 0.0177 | 0.0084 |
| T3 | gsm | 16 | mapped | 0.8311 | 0.3282 | 0.0586 | 0.0020 | 0.0000 |
| T3 | code | 16 | mapped | 0.7548 | 0.2720 | 0.0713 | 0.0007 | 0.0000 |
| T3 | chat | 16 | mapped | 0.6701 | 0.1950 | 0.0503 | 0.0013 | 0.0001 |
| T4 | math | 16 | native | 0.8528 | 0.4561 | 0.2214 | 0.0188 | 0.0004 |
| T4 | math | 16 | mapped | 0.6900 | 0.3565 | 0.1896 | 0.1002 | 0.0417 |
| T4 | gsm | 16 | native | 0.8680 | 0.3835 | 0.0788 | 0.0033 | 0.0002 |
| T4 | gsm | 16 | mapped | 0.6567 | 0.2432 | 0.0746 | 0.0340 | 0.0109 |
| T4 | code | 16 | native | 0.8364 | 0.3928 | 0.1230 | 0.0053 | 0.0000 |
| T4 | code | 16 | mapped | 0.6732 | 0.1764 | 0.0415 | 0.0143 | 0.0042 |
| T4 | chat | 16 | native | 0.8044 | 0.3310 | 0.1002 | 0.0053 | 0.0001 |
| T4 | chat | 16 | mapped | 0.4881 | 0.0685 | 0.0132 | 0.0027 | 0.0007 |

<!-- OPTIMIZED_SERVING_END -->

## Historical study status — 14 September 2026

**Stopped.** The controller was terminated and allocations **33992 / 33993** were cancelled. The later MSE → AUF jobs **34021 / 34024 / 34026 / 34029** are also cancelled. No Phase 2 job remains running or queued for automatic resumption. Completed checkpoints and raw outputs are preserved. Partial cohorts below are not reported as 128-prompt results. The execution plan is in [phase2.md](phase2.md).

**Earlier selected five-map recipe (superseded):** MSE50, **8,192 examples, three epochs, LR $10^{-3}$**, final checkpoint. The matching completed 16k MSE50 endpoint is retained. This is the practical five-map selection, not a claim that this backbone beats every trained native drafter. Loss comparisons use different learning rates and are recipe comparisons.

## T1 MSE → AUF refinement: stopped 14 September 2026

**Stopped at the user's request.** Jobs 34021, 34024, 34026 and 34029 were cancelled; both evaluation launch controllers were terminated. No automatic continuation is scheduled. Raw outputs, fitted checkpoints and partial training state are preserved. The earlier 16-fit / 48-checkpoint / 192-cohort sweep is cancelled, not awaiting completion.

This is **8B target + transferred 4B-trained drafter**, not the LoRA study. Only the five interface maps are trained. Each run starts from its MSE epoch-3 checkpoint and a fresh AUF optimizer; the source fusion, RMSNorm, drafter body, embeddings and output head remain frozen. AUF uses the existing first-error-inclusive prefix CE, response anchors, LR $10^{-4}$, seed 42, 32 anchors/example and global batch 8 on two GPUs. Each fit follows one three-epoch schedule; epochs 1/2/3 are saved along that trajectory. No additional positional decay is applied.

All completed results below belong to the **unseen** arm; the seen arm has no completed evaluation. A stopped or unlaunched configuration is not evidence of its performance.

**35/192 complete evaluation cohorts.** Incomplete cohorts are excluded from performance comparisons. The seen arm uses the first 4k/8k unique records. Its 4k-parent/8k-AUF cells have 50% overlap; other seen cells have 100% overlap. Unseen cells have zero overlap. No repeated records.

Start from MSE epoch 3; continue one three-epoch AUF trajectory and evaluate epochs 1, 2 and 3. All use the same four 128-prompt workloads, greedy nonthinking Transformers decoding and a 2,048-token response cap. Each workload is reported separately. AUF epoch 1 here uses the first third of a three-epoch schedule; it is not an independently scheduled one-epoch fit.

Speedup is the ratio of arithmetic mean per-request TPS to the existing matched native control; acceptance is pooled emitted tokens per verification step, including the correction/bonus token and the actual EOS/cap truncation. The initial target-prefill token is outside these steps; the report checks that summed step lengths equal output tokens minus that initial token for every request. Latency reduction is 1 − (total request seconds / native total request seconds); it is distinct from the mean-TPS ratio. MSE fit time is the parent cost, AUF fit time is cumulative through the displayed checkpoint. Fitting GPU-hours include both stages but exclude feature extraction, model setup, evaluation and allocation idle time. Existing MSE parents are reused; their cost is included for recipe comparison, not charged again as new work. All performance rows are sorted by descending speedup within workload.

| Workload | MSE coverage / examples | AUF data / examples | AUF epoch | vs native | MSE parent vs native | vs MSE parent | TPS change | Latency reduction | Avg acceptance | TPS | MSE min | AUF min | MSE + AUF fit GPU-h |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chat | 50% / 8192 | unseen (0% overlap) / 4096 | 2 | 0.8440× | 0.8882× | 0.9503× | -15.60% | -18.25% | 2.7174 | 75.54 | 17.79 | 9.71 | 0.917 |
| chat | 50% / 4096 | unseen (0% overlap) / 4096 | 3 | 0.8321× | 0.8096× | 1.0277× | -16.79% | -21.32% | 2.6440 | 74.47 | 9.56 | 13.12 | 0.756 |
| chat | 50% / 4096 | unseen (0% overlap) / 8192 | 3 | 0.8266× | 0.8096× | 1.0210× | -17.34% | -21.56% | 2.6456 | 73.99 | 9.56 | 28.24 | 1.260 |
| chat | 50% / 8192 | unseen (0% overlap) / 4096 | 1 | 0.8230× | 0.8882× | 0.9266× | -17.70% | -20.95% | 2.6693 | 73.66 | 17.79 | 4.87 | 0.756 |
| chat | 50% / 4096 | unseen (0% overlap) / 4096 | 2 | 0.8177× | 0.8096× | 1.0100× | -18.23% | -23.22% | 2.6104 | 73.19 | 9.56 | 8.79 | 0.612 |
| chat | 50% / 4096 | unseen (0% overlap) / 8192 | 2 | 0.8151× | 0.8096× | 1.0068× | -18.49% | -23.18% | 2.6204 | 72.96 | 9.56 | 18.86 | 0.947 |
| chat | 50% / 4096 | unseen (0% overlap) / 4096 | 1 | 0.7971× | 0.8096× | 0.9846× | -20.29% | -26.00% | 2.5586 | 71.35 | 9.56 | 4.47 | 0.468 |
| chat | 100% / 4096 | unseen (0% overlap) / 4096 | 1 | 0.7961× | 0.8537× | 0.9325× | -20.39% | -26.15% | 2.5617 | 71.25 | 18.30 | 4.58 | 0.763 |
| chat | 50% / 4096 | unseen (0% overlap) / 8192 | 1 | 0.7831× | 0.8096× | 0.9672× | -21.69% | -27.49% | 2.5271 | 70.09 | 9.56 | 9.42 | 0.633 |
| code | 50% / 8192 | unseen (0% overlap) / 4096 | 3 | 0.7417× | 0.8322× | 0.8913× | -25.83% | -26.52% | 4.3499 | 112.46 | 17.79 | 14.60 | 1.080 |
| code | 50% / 8192 | unseen (0% overlap) / 4096 | 2 | 0.7144× | 0.8322× | 0.8585× | -28.56% | -29.98% | 4.2266 | 108.33 | 17.79 | 9.71 | 0.917 |
| code | 50% / 4096 | unseen (0% overlap) / 4096 | 3 | 0.6915× | 0.7522× | 0.9194× | -30.85% | -35.07% | 4.0563 | 104.86 | 9.56 | 13.12 | 0.756 |
| code | 50% / 8192 | unseen (0% overlap) / 4096 | 1 | 0.6913× | 0.8322× | 0.8307× | -30.87% | -35.15% | 4.0373 | 104.81 | 17.79 | 4.87 | 0.756 |
| code | 50% / 4096 | unseen (0% overlap) / 8192 | 3 | 0.6865× | 0.7522× | 0.9126× | -31.35% | -34.37% | 4.0875 | 104.09 | 9.56 | 28.24 | 1.260 |
| code | 50% / 4096 | unseen (0% overlap) / 8192 | 2 | 0.6806× | 0.7522× | 0.9048× | -31.94% | -36.08% | 4.0086 | 103.20 | 9.56 | 18.86 | 0.947 |
| code | 50% / 4096 | unseen (0% overlap) / 4096 | 2 | 0.6751× | 0.7522× | 0.8974× | -32.49% | -37.69% | 3.9776 | 102.35 | 9.56 | 8.79 | 0.612 |
| code | 50% / 4096 | unseen (0% overlap) / 4096 | 1 | 0.6406× | 0.7522× | 0.8516× | -35.94% | -45.21% | 3.7637 | 97.13 | 9.56 | 4.47 | 0.468 |
| code | 50% / 4096 | unseen (0% overlap) / 8192 | 1 | 0.6214× | 0.7522× | 0.8261× | -37.86% | -48.19% | 3.6992 | 94.22 | 9.56 | 9.42 | 0.633 |
| gsm | 50% / 4096 | unseen (0% overlap) / 8192 | 3 | 1.0445× | 0.9422× | 1.1086× | +4.45% | +4.18% | 6.6206 | 185.86 | 9.56 | 28.24 | 1.260 |
| gsm | 50% / 4096 | unseen (0% overlap) / 4096 | 3 | 1.0425× | 0.9422× | 1.1065× | +4.25% | +3.68% | 6.5104 | 185.51 | 9.56 | 13.12 | 0.756 |
| gsm | 50% / 4096 | unseen (0% overlap) / 8192 | 2 | 1.0378× | 0.9422× | 1.1015× | +3.78% | +3.50% | 6.5128 | 184.66 | 9.56 | 18.86 | 0.947 |
| gsm | 50% / 4096 | unseen (0% overlap) / 4096 | 2 | 1.0056× | 0.9422× | 1.0673× | +0.56% | +0.46% | 6.4061 | 178.94 | 9.56 | 8.79 | 0.612 |
| gsm | 50% / 8192 | unseen (0% overlap) / 4096 | 2 | 0.9769× | 0.9753× | 1.0017× | -2.31% | -1.74% | 6.3272 | 173.84 | 17.79 | 9.71 | 0.917 |
| gsm | 50% / 4096 | unseen (0% overlap) / 8192 | 1 | 0.9480× | 0.9422× | 1.0063× | -5.20% | -4.94% | 6.0639 | 168.70 | 9.56 | 9.42 | 0.633 |
| gsm | 50% / 8192 | unseen (0% overlap) / 4096 | 1 | 0.9474× | 0.9753× | 0.9714× | -5.26% | -5.28% | 5.9575 | 168.58 | 17.79 | 4.87 | 0.756 |
| gsm | 50% / 4096 | unseen (0% overlap) / 4096 | 1 | 0.9457× | 0.9422× | 1.0038× | -5.43% | -5.61% | 5.9792 | 168.29 | 9.56 | 4.47 | 0.468 |
| math | 50% / 4096 | unseen (0% overlap) / 8192 | 3 | 0.9961× | 0.9764× | 1.0201× | -0.39% | -0.79% | 7.8768 | 218.93 | 9.56 | 28.24 | 1.260 |
| math | 50% / 4096 | unseen (0% overlap) / 4096 | 3 | 0.9818× | 0.9764× | 1.0055× | -1.82% | -2.53% | 7.7689 | 215.78 | 9.56 | 13.12 | 0.756 |
| math | 50% / 8192 | unseen (0% overlap) / 4096 | 3 | 0.9694× | 0.9986× | 0.9707× | -3.06% | -3.42% | 7.7785 | 213.05 | 17.79 | 14.60 | 1.080 |
| math | 50% / 4096 | unseen (0% overlap) / 8192 | 2 | 0.9623× | 0.9764× | 0.9855× | -3.77% | -4.03% | 7.7204 | 211.50 | 9.56 | 18.86 | 0.947 |
| math | 50% / 8192 | unseen (0% overlap) / 4096 | 2 | 0.9604× | 0.9986× | 0.9617× | -3.96% | -4.66% | 7.5780 | 211.07 | 17.79 | 9.71 | 0.917 |
| math | 50% / 4096 | unseen (0% overlap) / 4096 | 2 | 0.9593× | 0.9764× | 0.9825× | -4.07% | -5.01% | 7.5572 | 210.85 | 9.56 | 8.79 | 0.612 |
| math | 50% / 4096 | unseen (0% overlap) / 8192 | 1 | 0.9158× | 0.9764× | 0.9379× | -8.42% | -9.72% | 7.2421 | 201.28 | 9.56 | 9.42 | 0.633 |
| math | 50% / 8192 | unseen (0% overlap) / 4096 | 1 | 0.8951× | 0.9986× | 0.8964× | -10.49% | -12.01% | 7.1195 | 196.74 | 17.79 | 4.87 | 0.756 |
| math | 50% / 4096 | unseen (0% overlap) / 4096 | 1 | 0.8922× | 0.9764× | 0.9138× | -10.78% | -12.15% | 7.1537 | 196.10 | 9.56 | 4.47 | 0.468 |


### Fit and checkpoint inventory at stop

Times below are AUF-only wall-clock minutes on two GPUs; MSE-parent fitting times appear in the performance table above. Epoch times are cumulative.

| Fit | Saved AUF epochs | AUF fit / elapsed min | Status |
| --- | --- | ---: | --- |
| `T1_mse100_n4096_e3_lr0.001_auf_unseen_n4096` | 1, 2, 3 | 13.44 | Three epochs complete |
| `T1_mse100_n4096_e3_lr0.001_auf_unseen_n8192` | 1, 2, 3 | 28.11 | Three epochs complete |
| `T1_mse100_n8192_e3_lr0.001_auf_unseen_n4096` | None | 0.60 | Stopped; last saved progress 60/1536 steps |
| `T1_mse50_n4096_e3_lr0.001_auf_unseen_n4096` | 1, 2, 3 | 13.16 | Three epochs complete |
| `T1_mse50_n4096_e3_lr0.001_auf_unseen_n8192` | 1, 2, 3 | 28.29 | Three epochs complete |
| `T1_mse50_n8192_e3_lr0.001_auf_unseen_n4096` | 1, 2, 3 | 14.64 | Three epochs complete |
| `T1_mse50_n8192_e3_lr0.001_auf_unseen_n8192` | 1, 2, 3 | 29.67 | Three epochs complete |

### Interrupted evaluations — excluded from performance results

| Checkpoint / workload | Saved prompts | Status |
| --- | ---: | --- |
| `T1_mse100_n4096_e3_lr0.001_auf_unseen_n4096_epoch1_code_part0` | 24/128 | Interrupted; no final speedup reported |
| `T1_mse50_n8192_e3_lr0.001_auf_unseen_n4096_epoch3_chat_part0` | 87/128 | Interrupted; no final speedup reported |
| `T1_mse50_n8192_e3_lr0.001_auf_unseen_n4096_epoch3_gsm_part0` | 127/128 | Interrupted; no final speedup reported |
| `T1_mse50_n8192_e3_lr0.001_auf_unseen_n8192_epoch1_chat_part0` | 20/128 | Interrupted; no final speedup reported |
| `T1_mse50_n8192_e3_lr0.001_auf_unseen_n8192_epoch1_code_part0` | 16/128 | Interrupted; no final speedup reported |

Machine-readable complete results: [CSV](experiments/dflash_mse_auf_20260914/results/results.csv). The same checkpoint is evaluated on multiple workloads; its fit cost is not incurred separately per task.

## Scope and completion

| Study | Completed at stop | Remaining status |
| --- | --- | --- |
| T1 pruned 4k/8k recipe comparison | 8 fits × 4 tasks = 32 evaluations | Complete |
| T1 MSE50 16k endpoint | 4 tasks | Complete; existing fit reused |
| LoRA MSE50 / MSE100 | 3 domains each | Complete |
| T1 selected P2 architecture / loss | 10 fits × Math/Code = 20 evaluations | Complete |
| Native body r128 AUF / decay | 2 fits × 4 tasks = 8 evaluations | Complete |
| EAGLE standalone | 4 tasks | Complete |
| AR / native / initializer controls | 8 complete AR/native cohorts | 4 initializer cohorts stopped or unlaunched |
| Math mechanism study | Geometry and all 17 inference cohorts | Complete |
| Code mechanism confirmation | Unmodified parent, 128 prompts | 2 removals interrupted |
| T1 production serving | Mapped at 4 and 8 clients, 512 requests each | Other 18 cells interrupted or unlaunched |
| T2 / T3 / T4 | Earlier completed work inventoried below | Paused; no new runs |

## Reading the numbers

Performance tables are grouped by target/workload and sorted by **native throughput ratio, highest first**. The cross-workload recipe table uses its explicitly labeled geometric mean. Rows without a measured native reference are placed last within their group. Training, geometry and status tables have no speedup to rank. Probe tables additionally show a derived native TPS ratio; their original causal-control comparisons are preserved.

TPS means the arithmetic mean of per-request output tokens/second unless a table explicitly says aggregate. Native ratios use the unchanged native drafter with the same target. Acceptance includes the correction/bonus token. Fit time excludes feature capture, setup and evaluation; repeated task rows share a fit. A negative gain or ratio below one is a measured regression, not missing data.

The small numerical checks do not explain every full-cohort disagreement. Earlier historical rows and EAGLE do not inherit the current DFlash agreement certificate.

## T1 pruned sweep: method and interpretation

This is the current T1 study, transferring the Qwen3-4B DFlash drafter to the frozen Qwen3-8B target. T2/T3/T4 are outside the active scope. Training and standalone evaluation use the pinned Transformers/SpecForge components; the production concurrency experiment uses vLLM and has a separate table. The user stopped remaining evaluations; only complete cohorts are reported.

### Models and the interface

The target supplies five hidden vectors $h_i\in\mathbb{R}^{4096}$. The source drafter expects five vectors of width 2,560. Both use selected zero-based target-layer IDs `[1, 9, 17, 25, 33]`. The original source fusion matrix is $F\in\mathbb{R}^{2560\times12800}$; write its five column blocks as $F_i\in\mathbb{R}^{2560\times2560}$.

The main architecture learns five bias-free maps $W_i\in\mathbb{R}^{2560\times4096}$:

$$
\widehat c=N\!\left(\sum_{i=1}^{5}F_iW_i h_i\right).
$$

$N$ is the source drafter's original RMSNorm, with its gain and epsilon frozen. The source drafter body, source token embeddings and source vocabulary head also remain frozen. The 8B target performs verification. The 4B transformer is needed for paired training features, not during mapped inference.

Each $W_i$ starts from independently drawn Xavier-uniform entries using the seeded generator (seed42). There is no learned or identity initialization: these maps are rectangular. For inference, export a single fusion matrix $[F_1W_1,\ldots,F_5W_5]$ where the blocks are concatenated **horizontally**. This is a $2560\times20480$ matrix. All mappings and the fusion are linear, so the extra matrix products can be folded before inference; the original RMSNorm remains afterward. Export uses BF16.

| Architecture | What is trained | Initialization | Parameters |
| --- | --- | --- | --- |
| Five maps | Five separate $W_i$; frozen $F$ | Xavier-uniform maps | 52,428,800 |
| Direct fusion | One $G\in\mathbb{R}^{2560\times20480}$ | $G_0=[F_1W_{0,1},\ldots,F_5W_{0,5}]$ | 52,428,800 |
| BA28 / BA56 / BA128 | Each full map is $W_i=B_iA_i$ | Rank-$r$ SVD approximation of $W_{0,i}$, splitting singular values equally between factors | 931,840 / 1,863,680 / 4,259,840 |
| Joint map | $J\in\mathbb{R}^{12800\times20480}$ before frozen $F$ | Block diagonal of the five initial maps | 262,144,000 |
| Native body-LoRA128 | LoRA on attention and MLP projections in the **8B-native drafter** | PEFT initialization; rank128, alpha256, dropout0 | 48,496,640 |

BA is a low-rank **replacement of each whole map**, not a low-rank residual added to a retained full map. Its rank truncation changes its starting function. Direct fusion keeps the initial folded function but does not constrain subsequent training to the five-map factorization. Joint mapping starts with separate branches but can learn cross-layer connections. The native-body control uses a different pretrained backbone, retains its native frozen fusion/embeddings/head, and must not be presented as an interface-only comparison within the 4B drafter.

### Data, coverage and training loop

All T1 fits reuse the audited saved 8B-generated cohort of16,384 unique sequences. The4k/8k/16k sizes are the first4,096/8,192/16,384 entries of that fixed order, not fresh generated cohorts. Training responses have a4,096-token generation cap; sequences can end earlier. Source and target Transformers perform causal teacher-forced passes over the **same saved prompt and response token IDs**. No independent 4B response is generated.

The authoritative cohort preserves each ID, token hash, source row ID and original shard hashes. The present normalized cohort does not itself name a public dataset revision; that source-dataset attribution must be reconciled from original generation metadata before claiming a complete from-scratch public-data reproduction. Saved-token reproduction is pinned independently by the cohort and token hashes.

MSE50 keeps $\lceil n_p/2\rceil$ prompt positions and $\lceil n_r/2\rceil$ response positions, sampled separately without replacement. Seeds depend on the saved sequence's token hash and segment. Sampling is fixed across epochs. MSE100 keeps every valid prompt and response position, including the final token. Padding is excluded in both. Full hidden states are processed causally before positions are selected; sparse persistence does not mean independently encoding isolated tokens.

Token-objective training uses the cached full target context and32 sampled valid block anchors per example. Each block contains one clean anchor followed by15 masked positions. Only response labels are supervised; preceding prompt hidden states remain context. Context attention cannot see target states at or after the anchor, and separate draft blocks cannot attend to one another. Within-block draft attention is bidirectional. Labels at offset $j$ are the saved target token at the anchor plus $j$, for $j=1,\ldots,15$.

All current T1 fits use three epochs, final checkpoint, two GPUs, two examples per GPU per microbatch, and two accumulation microbatches: eight examples per optimizer update. There are1,536 updates at4k,3,072 at8k and6,144 at16k. Each epoch uses a deterministic shuffle with seed42 plus epoch index. AdamW has zero weight decay; gradient norm is clipped at1. Learning rate warms up over5% of updates and follows cosine decay. MSE uses LR$10^{-3}$; AUF and decaying CE use LR$10^{-4}$. Therefore these are **recipe comparisons**, not losses isolated at one common learning rate.

Trainable parameters and MSE reconstruction calculations are FP32; cached features and exports are BF16. Token-objective draft computation uses BF16 autocast with the frozen vocabulary head. Reported fit time excludes feature collection, model setup, interface initialization and export; those costs must remain separate from end-to-end adaptation cost.

### Exact losses

For a predicted feature $u$ and reference $v$, define

$$
D(u,v)=\frac{\|u-v\|_2^2}{\|v\|_2^2+10^{-6}}.
$$

For saved example $b$ and selected token position $t$, $y_{bti}$ is the paired 4B feature, and $h_{bti}$ is its8B counterpart. The MSE loss is

$$
\ell_{bt}=\frac15\sum_{i=1}^{5}D(W_i h_{bti},y_{bti})+
D\!\left(N\!\left(\sum_i F_iW_i h_{bti}\right),N\!\left(\sum_i F_i y_{bti}\right)\right),
$$

$$
L_{\mathrm{MSE}}=\frac1B\sum_{b=1}^{B}\frac{1}{|S_b|}\sum_{t\in S_b}\ell_{bt}.
$$

Each example gets equal weight after averaging over its own selected positions. This is not a single token-weighted average over all sequences. MSE backpropagation ends at the interface/context reconstruction; it does not traverse the drafter transformer or vocabulary prediction.

For either token objective, let $q_{baj}(x_{b,a+j})$ be the drafter probability assigned to the saved token at offset $j$ of anchor $a$. Let $m_{baj}$ indicate a valid supervised label. The microbatch loss is

$$
L(w)=-\frac{\sum_{b,a}\sum_{j=1}^{15}m_{baj}w_{baj}\log q_{baj}(x_{b,a+j})}{\sum_{b,a}\sum_{j=1}^{15}m_{baj}w_{baj}}.
$$

For the upstream DFlash decaying CE objective,

$$
w_{baj}=\exp\!\left(-\frac{j-1}{7}\right).
$$

For AUF, let $r_{bak}$ be1 when the current argmax at valid offset $k$ matches its saved label (invalid positions do not terminate support). Then

$$
w_{baj}=\prod_{k=1}^{j-1}r_{bak}.
$$

The empty product for $j=1$ is1. Thus AUF includes every correct leading token **and the first wrong token**, dropping later positions. The support calculation is detached from gradients and recomputed from current predictions. This is the repository's AUF formula implementation on top of pinned SpecForge; it is not claimed to be author-released AUF code. Neither objective here combines AUF with a decaying tail.

Full-vocabulary CE is computed in two-block chunks. Numerators and denominators are accumulated across those chunks before microbatch reduction. Across two GPUs and two accumulation microbatches, gradients correspond to the average of four locally normalized losses:

$$
L_{\mathrm{step}}=\frac14\sum_{g=1}^{2}\sum_{u=1}^{2}L_{g,u}.
$$

This differs from one global token-weighted reduction when valid support counts vary. Fusion, BA, joint and native-body controls use the same AUF/decay contracts; only the trainable parameters and associated backbone change.

### Evaluation and scope of claims

Standalone evaluation uses128 fixed prompts each from MATH test, GSM8K test, pinned LiveCodeBench releases and UltraFeedback-derived single-turn instructions. These are not HumanEval or MT-Bench. Use the saved nonthinking prompt IDs, greedy decoding, natural EOS and a2,048-token response cap. No new checkpoint is selected from a development evaluation. The confirmation prompts have nevertheless been used for recipe selection in this sweep; their original name does not make the current comparison an untouched statistical confirmation.

Standalone TPS is the arithmetic mean of per-request output tokens divided by request latency. Pooled acceptance is total runtime-reported emitted tokens divided by verification steps, including correction/bonus tokens. Latency speedup is summed reference request time divided by summed candidate request time. These are different statistics. EAGLE uses its pinned Transformers tree implementation and its separately documented emitted/verification accounting.

BF16 AR and speculative execution can select different tokens around numerical ties. Full probe outputs are additionally compared against their matching unmodified control. No speed table should silently substitute a small check for a full-output comparison.

P5 serves the fixed512-request mixture at concurrency1/4/8/16/32 for AR, native DFlash, mapped DFlash and native vLLM EAGLE. It uses pinned BF16 vLLM FLASH_ATTN, upstream batch-invariant operations, compilation disabled and full CUDA graphs. Prefix caching is enabled and reset before measurement; warmup is excluded. This is an explicit runtime configuration, not the original compiled-mode timing setup. The actual uninstrumented server must reproduce reviewed small outputs before each full pass. Report aggregate throughput and matched-concurrency speedups separately from mean request TPS; report TTFT and latency quantiles. Memory snapshots are not a measured peak. Full-cohort agreement and absence of timed JIT are required before accepting comparative results.

### Reading the mechanisms

Geometry describes MSE50/AUF/decay8k checkpoints on64 training examples and7,090 sampled positions (2,994 prompt,4,096 response). Activation energy is uncentered and includes any constant correction. It is not variance explained after mean subtraction.

Live interventions use the AUF parent, not the selected MSE parent. Delta compression truncates each $W_i-W_{0,i}$; branch restoration replaces one trained map with its initializer. Radial/perpendicular splits are relative to the initial fused context before RMSNorm, with a matched hook control. Subspace removal eliminates either the leading32 right-singular directions of $F_i(W_i-W_{0,i})$ or32 random directions per branch. The latter is rank-matched, not energy-matched.

The selected Code confirmation compares unmodified, leading-direction removal and random removal on the same node. Its results must be measured, not inferred from Math. Negative results remain evidence: this AUF parent is slower than the native drafter, so a percentage of "native speedup retained" would be misleading. Report direct parent-relative speed, latency and acceptance instead.

The complete saved-output audit is reported separately. Completed DFlash variants so far match native DFlash on all128 prompts per cohort, but native DFlash and AR differ on a substantial subset. Small-check target-logit traces explain particular events only.

## T1 recipe comparison: 4k and 8k

Three epochs, frozen draft body, five maps; 128 prompts per workload. AUF and decaying CE use 32 block anchors at LR 1e-4. MSE uses the stated fraction of prompt plus response tokens at LR 1e-3. This compares training recipes, not losses at a common learning rate. Acceptance includes the correction/bonus token.

Each workload cell is **speedup versus native / pooled accepted length**. The geometric mean weights the four workloads equally; it is a comparison summary, not measured mixed-traffic throughput. Missing cells are not completed at stop, never treated as zero.

| Recipe | Examples | Fit min | Math | GSM8K | Code | Chat | Geomean × native |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MSE 100% | 8192 | 53.90 | 0.9979× / 7.999 | 0.9901× / 6.224 | 0.8283× / 4.697 | 0.8914× / 2.819 | 0.9241 |
| MSE 50% | 8192 | 17.79 | 0.9986× / 7.978 | 0.9753× / 6.257 | 0.8322× / 4.691 | 0.8882× / 2.812 | 0.9211 |
| MSE 100% | 4096 | 18.30 | 0.9971× / 7.971 | 0.9797× / 6.187 | 0.7785× / 4.434 | 0.8537× / 2.681 | 0.8977 |
| MSE 50% | 4096 | 9.56 | 0.9764× / 7.969 | 0.9422× / 6.165 | 0.7522× / 4.422 | 0.8096× / 2.662 | 0.8652 |
| DFlash decay CE | 8192 | 28.24 | 0.8858× / 6.947 | 0.9187× / 5.693 | 0.4060× / 2.449 | 0.5733× / 1.812 | 0.6597 |
| AUF | 8192 | 31.64 | 0.8837× / 6.892 | 0.9107× / 5.677 | 0.3850× / 2.346 | 0.5578× / 1.767 | 0.6447 |
| DFlash decay CE | 4096 | 13.02 | 0.7912× / 6.151 | 0.7585× / 4.750 | 0.3324× / 2.000 | 0.4823× / 1.576 | 0.5569 |
| AUF | 4096 | 13.29 | 0.7741× / 6.017 | 0.7415× / 4.580 | 0.3180× / 1.910 | 0.4687× / 1.532 | 0.5408 |

Selected practical five-map recipe: **MSE50, 8,192 examples, three epochs, LR 1e-3**, final checkpoint. MSE100 gives nearly identical acceptance at about three times the 8k fitting time. Reuse the already completed matching 16k MSE50 result; no new 16k fit is scheduled. The authorized architecture controls retain AUF/decaying CE at LR 1e-4, using the common 8k/three-epoch schedule. Selection on these benchmarks is exploratory, not an independent held-out confirmation.

## Complete-token MSE results

Only complete 128-prompt cohorts appear below. All comparisons use the same prompt IDs and established MSE50 controls. Mean request TPS and summed-latency ratios are different statistics; output differences can affect latency ratios. Representative checks alone do not prove full-cohort token equivalence. The native-relative comparisons have a separate full-output audit.

| Fit | Task | Mean TPS | × native (TPS) | × native (latency) | Observed AR TPS ratio | × MSE50 (TPS) | Acceptance | MSE50 acceptance | Fit min | GPU h |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1_mse100_n8192_e3_lr0.001 | math | 219.3158 | 0.9979 | 1.0039 | 5.7963 | 0.9993 | 7.9994 | 7.9776 | 53.90 | 1.7967 |
| T1_mse100_n4096_e3_lr0.001 | math | 219.1429 | 0.9971 | 0.9994 | 5.7917 | 1.0211 | 7.9711 | 7.9689 | 18.30 | 0.6101 |
| T1_mse100_n8192_e3_lr0.001 | gsm | 176.1834 | 0.9901 | 0.9961 | 4.6656 | 1.0152 | 6.2244 | 6.2568 | 53.90 | 1.7967 |
| T1_mse100_n4096_e3_lr0.001 | gsm | 174.3330 | 0.9797 | 0.9855 | 4.6166 | 1.0399 | 6.1869 | 6.1646 | 18.30 | 0.6101 |
| T1_mse100_n8192_e3_lr0.001 | code | 125.5835 | 0.8283 | 0.8562 | 3.3492 | 0.9953 | 4.6968 | 4.6912 | 53.90 | 1.7967 |
| T1_mse100_n4096_e3_lr0.001 | code | 118.0456 | 0.7785 | 0.8108 | 3.1481 | 1.0350 | 4.4341 | 4.4222 | 18.30 | 0.6101 |
| T1_mse100_n8192_e3_lr0.001 | chat | 79.7790 | 0.8914 | 0.8749 | 2.2737 | 1.0036 | 2.8186 | 2.8125 | 53.90 | 1.7967 |
| T1_mse100_n4096_e3_lr0.001 | chat | 76.4124 | 0.8537 | 0.8340 | 2.1777 | 1.0545 | 2.6811 | 2.6619 | 18.30 | 0.6101 |
| lora_math_mse100 | math | 124.7985 | 0.9817 | 0.9849 | 3.2155 | 0.9709 | 4.4696 | 4.4665 | 0.57 | 0.0190 |
| lora_kicad_mse100 | kicad | 196.8256 | 1.2050 | 1.2321 | 5.1228 | 1.0122 | 7.2064 | 7.2083 | 6.36 | 0.2120 |
| lora_nanocoder_mse100 | nanocoder | 148.1504 | 0.9863 | 0.9886 | 3.7980 | 1.0093 | 4.9742 | 4.9573 | 0.62 | 0.0208 |

T1 4k MSE100 was evaluated on node09; its MSE50/native controls came from node08. Both use L40S GPUs and the pinned benchmark, but these are separate timing passes, not a repeat-controlled timing comparison. Do not attribute small throughput differences solely to coverage. Acceptance counters and saved output comparisons are reported separately.

### Training

| Fit | Examples | Epochs | Updates | Minutes | GPU hours |
| --- | --- | --- | --- | --- | --- |
| T1_mse100_n4096_e3_lr0.001 | 4096 | 3 | 1536 | 18.30 | 0.6101 |
| T1_mse100_n8192_e3_lr0.001 | 8192 | 3 | 3072 | 53.90 | 1.7967 |
| lora_math_mse100 | 4096 | 1 | 512 | 0.57 | 0.0190 |
| lora_kicad_mse100 | 4096 | 1 | 512 | 6.36 | 0.2120 |
| lora_nanocoder_mse100 | 4096 | 1 | 512 | 0.62 | 0.0208 |

## Selected T1 architecture results

Five-W AUF, decay CE, MSE50 and MSE100 reuse the completed 8k/three-epoch fits. Native DFlash is the 1× reference. Rows are grouped by task and sorted by native TPS ratio, highest first. MSE uses LR 1e-3; AUF/decay uses LR 1e-4. These are recipe comparisons, not a matched-learning-rate loss comparison.

Only complete 128-prompt cohorts appear. All rows use the base Qwen3-8B target. Source-drafter interfaces and native-drafter body controls are different backbones. Ratios use the matching previously measured native/AR controls; separate timing passes and BF16 execution differences remain qualifications.

| Cell | Task | TPS | × native | AR TPS ratio | Latency speedup × native | Emitted / verification | Fit min |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1_selected_body128_auf_n8192_e3 | math | 230.8243 | 1.0502 | 6.1004 | 1.0504 | 8.6097 | 31.79 |
| T1_selected_body128_decay7_n8192_e3 | math | 226.1528 | 1.0290 | 5.9770 | 1.0273 | 8.4241 | 31.78 |
| Native DFlash (reference) | math | 219.7874 | 1.0000 | 5.8087 | 1.0000 | 8.2108 | no fit |
| Five-W — MSE50 | math | 219.4753 | 0.9986 | 5.8005 | 1.0032 | 7.9776 | 17.79 |
| Five-W — MSE100 | math | 219.3158 | 0.9979 | 5.7963 | 1.0039 | 7.9994 | 53.90 |
| T1_selected_joint_decay7_n8192_e3 | math | 195.2239 | 0.8882 | 5.1596 | 0.8746 | 6.9644 | 33.29 |
| Five-W — decay CE | math | 194.6915 | 0.8858 | 5.1455 | 0.8689 | 6.9474 | 28.24 |
| Five-W — AUF | math | 194.2206 | 0.8837 | 5.1330 | 0.8645 | 6.8923 | 31.64 |
| T1_selected_joint_auf_n8192_e3 | math | 193.3218 | 0.8796 | 5.1093 | 0.8673 | 6.9562 | 36.40 |
| T1_selected_ba128_decay7_n8192_e3 | math | 149.9202 | 0.6821 | 3.9622 | 0.6564 | 5.2374 | 24.06 |
| T1_selected_ba128_auf_n8192_e3 | math | 143.0149 | 0.6507 | 3.7797 | 0.6274 | 4.9728 | 29.49 |
| T1_selected_ba56_decay7_n8192_e3 | math | 119.0842 | 0.5418 | 3.1473 | 0.5211 | 4.2197 | 24.61 |
| T1_selected_ba56_auf_n8192_e3 | math | 109.9603 | 0.5003 | 2.9061 | 0.4765 | 3.7969 | 25.22 |
| T1_selected_ba28_decay7_n8192_e3 | math | 93.2439 | 0.4242 | 2.4643 | 0.4070 | 3.2391 | 28.12 |
| T1_selected_ba28_auf_n8192_e3 | math | 82.5736 | 0.3757 | 2.1823 | 0.3621 | 2.8760 | 29.76 |
| T1_selected_fusion_decay7_n8192_e3 | math | 80.2527 | 0.3651 | 2.1210 | 0.3530 | 2.8394 | 27.68 |
| T1_selected_fusion_auf_n8192_e3 | math | 80.1810 | 0.3648 | 2.1191 | 0.3537 | 2.8402 | 26.90 |
| T1_native_EAGLE3 | math | 72.1780 | 0.3284 | 1.9076 | 0.3421 | 3.6199 | external checkpoint |
| T1_selected_body128_auf_n8192_e3 | gsm | 194.8627 | 1.0951 | 5.1603 | 1.0958 | 7.0986 | 31.79 |
| T1_selected_body128_decay7_n8192_e3 | gsm | 187.6579 | 1.0546 | 4.9695 | 1.0519 | 6.8111 | 31.78 |
| Native DFlash (reference) | gsm | 177.9411 | 1.0000 | 4.7121 | 1.0000 | 6.4788 | no fit |
| Five-W — MSE100 | gsm | 176.1834 | 0.9901 | 4.6656 | 0.9961 | 6.2244 | 53.90 |
| Five-W — MSE50 | gsm | 173.5459 | 0.9753 | 4.5958 | 0.9823 | 6.2568 | 17.79 |
| Five-W — decay CE | gsm | 163.4700 | 0.9187 | 4.3289 | 0.9116 | 5.6926 | 28.24 |
| Five-W — AUF | gsm | 162.0495 | 0.9107 | 4.2913 | 0.9053 | 5.6773 | 31.64 |
| T1_native_EAGLE3 | gsm | 75.5792 | 0.4247 | 2.0014 | 0.4492 | 3.6926 | external checkpoint |
| T1_selected_body128_auf_n8192_e3 | code | 155.3287 | 1.0244 | 4.1424 | 1.0369 | 5.8636 | 31.79 |
| T1_selected_body128_decay7_n8192_e3 | code | 153.8642 | 1.0148 | 4.1034 | 1.0164 | 5.7508 | 31.78 |
| Native DFlash (reference) | code | 151.6250 | 1.0000 | 4.0437 | 1.0000 | 5.6441 | no fit |
| Five-W — MSE50 | code | 126.1823 | 0.8322 | 3.3651 | 0.8581 | 4.6912 | 17.79 |
| Five-W — MSE100 | code | 125.5835 | 0.8283 | 3.3492 | 0.8562 | 4.6968 | 53.90 |
| T1_selected_joint_decay7_n8192_e3 | code | 70.9838 | 0.4682 | 1.8931 | 0.5178 | 2.8271 | 33.29 |
| T1_selected_joint_auf_n8192_e3 | code | 68.2183 | 0.4499 | 1.8193 | 0.5004 | 2.7418 | 36.40 |
| T1_native_EAGLE3 | code | 63.8810 | 0.4213 | 1.7036 | 0.4656 | 3.3384 | external checkpoint |
| Five-W — decay CE | code | 61.5596 | 0.4060 | 1.6417 | 0.4484 | 2.4489 | 28.24 |
| Five-W — AUF | code | 58.3696 | 0.3850 | 1.5566 | 0.4283 | 2.3456 | 31.64 |
| T1_selected_ba128_decay7_n8192_e3 | code | 43.8303 | 0.2891 | 1.1689 | 0.3157 | 1.7185 | 24.06 |
| T1_selected_ba128_auf_n8192_e3 | code | 41.5391 | 0.2740 | 1.1078 | 0.2974 | 1.6302 | 29.49 |
| T1_selected_ba56_decay7_n8192_e3 | code | 38.7609 | 0.2556 | 1.0337 | 0.2747 | 1.4981 | 24.61 |
| T1_selected_ba56_auf_n8192_e3 | code | 37.3163 | 0.2461 | 0.9952 | 0.2623 | 1.4330 | 25.22 |
| T1_selected_ba28_decay7_n8192_e3 | code | 35.8736 | 0.2366 | 0.9567 | 0.2503 | 1.3653 | 28.12 |
| T1_selected_ba28_auf_n8192_e3 | code | 34.7612 | 0.2293 | 0.9270 | 0.2431 | 1.3335 | 29.76 |
| T1_selected_fusion_auf_n8192_e3 | code | 33.8244 | 0.2231 | 0.9021 | 0.2360 | 1.2953 | 26.90 |
| T1_selected_fusion_decay7_n8192_e3 | code | 33.6595 | 0.2220 | 0.8977 | 0.2353 | 1.2921 | 27.68 |
| T1_selected_body128_auf_n8192_e3 | chat | 91.9628 | 1.0275 | 2.6209 | 1.0202 | 3.4139 | 31.79 |
| T1_selected_body128_decay7_n8192_e3 | chat | 90.2557 | 1.0084 | 2.5723 | 1.0076 | 3.3477 | 31.78 |
| Native DFlash (reference) | chat | 89.5033 | 1.0000 | 2.5508 | 1.0000 | 3.3236 | no fit |
| Five-W — MSE100 | chat | 79.7790 | 0.8914 | 2.2737 | 0.8749 | 2.8186 | 53.90 |
| Five-W — MSE50 | chat | 79.4945 | 0.8882 | 2.2656 | 0.8733 | 2.8125 | 17.79 |
| T1_native_EAGLE3 | chat | 61.8191 | 0.6907 | 1.7618 | 0.7173 | 3.1783 | external checkpoint |
| Five-W — decay CE | chat | 51.3131 | 0.5733 | 1.4624 | 0.5603 | 1.8115 | 28.24 |
| Five-W — AUF | chat | 49.9253 | 0.5578 | 1.4229 | 0.5459 | 1.7667 | 31.64 |

EAGLE uses its pinned Transformers tree implementation. Its emitted/verification metric uses reported EOS/cap-trimmed tokens divided by upstream verification steps. Raw upstream acceptance is retained separately in JSON; all overshoot work remains timed. External pretraining cost is unknown. DFlash acceptance includes the correction/bonus token.

Fit costs below exclude feature extraction and inference. GPU-hours are summed device time for each fit; peak memory is the reported peak GPU allocation, not summed across devices. Setup and export are listed separately.

| Cell | Trainable parameters | Fit min | Fit GPU-hours | Setup s | Interface init s | Export s | Peak GiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1_selected_fusion_auf_n8192_e3 | 52,428,800 | 26.90 | 0.90 | 11.94 | 0.10 | 1.32 | 5.47 |
| T1_selected_ba28_auf_n8192_e3 | 931,840 | 29.76 | 0.99 | 7.24 | 1.72 | 1.01 | 4.52 |
| T1_selected_ba56_auf_n8192_e3 | 1,863,680 | 25.22 | 0.84 | 8.19 | 1.79 | 1.10 | 4.54 |
| T1_selected_ba128_auf_n8192_e3 | 4,259,840 | 29.49 | 0.98 | 10.54 | 1.77 | 1.03 | 4.59 |
| T1_selected_joint_auf_n8192_e3 | 262,144,000 | 36.40 | 1.21 | 9.59 | 0.03 | 2.05 | 9.80 |
| T1_selected_body128_auf_n8192_e3 | 48,496,640 | 31.79 | 1.06 | 8.27 | 0.00 | 1.83 | 8.58 |
| T1_selected_fusion_decay7_n8192_e3 | 52,428,800 | 27.68 | 0.92 | 8.51 | 0.09 | 1.19 | 5.47 |
| T1_selected_ba28_decay7_n8192_e3 | 931,840 | 28.12 | 0.94 | 7.05 | 1.83 | 0.90 | 4.52 |
| T1_selected_ba56_decay7_n8192_e3 | 1,863,680 | 24.61 | 0.82 | 7.00 | 1.79 | 0.92 | 4.53 |
| T1_selected_ba128_decay7_n8192_e3 | 4,259,840 | 24.06 | 0.80 | 7.52 | 1.80 | 0.87 | 4.59 |
| T1_selected_joint_decay7_n8192_e3 | 262,144,000 | 33.29 | 1.11 | 5.74 | 0.02 | 2.17 | 9.83 |
| T1_selected_body128_decay7_n8192_e3 | 48,496,640 | 31.78 | 1.06 | 10.84 | 0.00 | 1.84 | 8.58 |
| T1_sweep_auf_n8192_a32_e3 | 52,428,800 | 31.64 | 1.05 | 8.53 | 0.02 | 1.40 | 5.53 |
| T1_sweep_decay7_n8192_a32_e3 | 52,428,800 | 28.24 | 0.94 | 16.57 | 0.02 | 1.57 | 5.52 |
| T1_mse50_n8192_e3_lr0.001 | 52,428,800 | 17.79 | 0.59 | 6.58 | 0.02 | 1.39 | 3.89 |
| T1_mse100_n8192_e3_lr0.001 | 52,428,800 | 53.90 | 1.80 | 3.45 | 0.02 | 1.30 | 5.53 |

| Cell | Training complete | Completed eval tasks |
| --- | --- | --- |
| T1_selected_fusion_auf_n8192_e3 | True | math, code |
| T1_selected_ba28_auf_n8192_e3 | True | math, code |
| T1_selected_ba56_auf_n8192_e3 | True | math, code |
| T1_selected_ba128_auf_n8192_e3 | True | math, code |
| T1_selected_joint_auf_n8192_e3 | True | math, code |
| T1_selected_body128_auf_n8192_e3 | True | math, gsm, code, chat |
| T1_selected_fusion_decay7_n8192_e3 | True | math, code |
| T1_selected_ba28_decay7_n8192_e3 | True | math, code |
| T1_selected_ba56_decay7_n8192_e3 | True | math, code |
| T1_selected_ba128_decay7_n8192_e3 | True | math, code |
| T1_selected_joint_decay7_n8192_e3 | True | math, code |
| T1_selected_body128_decay7_n8192_e3 | True | math, gsm, code, chat |
| T1_sweep_auf_n8192_a32_e3 | True | math, gsm, code, chat |
| T1_sweep_decay7_n8192_a32_e3 | True | math, gsm, code, chat |
| T1_mse50_n8192_e3_lr0.001 | True | math, gsm, code, chat |
| T1_mse100_n8192_e3_lr0.001 | True | math, gsm, code, chat |

## Selected T1 mapper geometry

These probes describe three five-map checkpoints trained on 8,192 examples for three epochs: MSE50 (LR 1e-3), AUF and decay CE (both LR 1e-4). They are not a learning-rate-controlled loss-only comparison. They use the same initial maps and frozen fusion weights.

The probe uses 64 training records and 7,090 sampled positions: 2,994 prompt and 4,096 response positions. For each record, it samples up to 64 positions from each segment of the existing 50%-coverage cache. These are training-distribution measurements, not held-out generalization estimates.

Let $h_i$ be an 8B hidden vector, $y_i$ the paired 4B hidden vector, $W_i$ its trained map, and $W_{0,i}$ its initial map. Let $F_i$ be the block of the frozen fusion matrix that consumes mapped layer $i$. Layer numbers below identify the five mapper slots (0–4).

$$
\widehat y_i=W_i h_i,\qquad \Delta W_i=W_i-W_{0,i},\qquad c_i=F_i\Delta W_i h_i.
$$

The feature error is the token average of $\|\widehat y_i-y_i\|_2^2/\max(\|y_i\|_2^2,10^{-6})$. Cosine is also averaged per token. Each segment is pooled over its sampled positions; examples with more sampled positions contribute more observations.

For context error, concatenate all five mapped vectors, apply frozen fusion and its original RMSNorm, and compare with the corresponding fused/RMS-normalized 4B vectors using the same normalized squared-error formula.

| Objective | Prompt context error | Response context error |
| --- | --- | --- |
| mse | 0.216170 | 0.136147 |
| auf | 1.228544 | 0.940413 |
| decay7 | 1.189191 | 0.899731 |

Lower reconstruction error means closer agreement with the 4B representations. AUF/decay optimize token prediction, so their higher reconstruction error is not itself proof of a decoding defect. Decoding results are measured separately.

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

Energy retained in $k$ directions means the sum of the largest $k$ squared singular values divided by the sum of all squared singular values. Matrix energy describes weights. Activation energy instead describes the actual vectors $c_i$ on sampled inputs, after each layer’s fusion block.

The activation vectors are **not mean-centered**. Their energy includes any constant component; this is not a covariance-only or variance-explained statistic. Directions are fitted separately for each layer and segment. High activation energy at rank two does not say that rank-two truncation of the weight matrix preserves it.

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

The complete rank-1/2/32/128 values, including the full-map spectrum, are in `geometry_energy.csv`. Feature measurements are in `geometry_features.csv`.

These measurements alone establish neither a speedup nor a safe compression rank.

## Selected T1 probe measurements

Only complete 128-prompt Math cohorts appear. Ratios compare with the unmodified AUF parent on the same node; radial/perpendicular interventions instead use the matching hook control. Separate timing passes remain a limitation. Acceptance includes the correction/bonus token.

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

Latency speedup = summed control seconds / summed intervention seconds; below1 means slower. Latency reduction =100 × (1 − intervention seconds / control seconds); negative means increased latency. These are direct comparisons to the AUF parent, not percentages of native-drafter speedup retained. Matrix/activation energy alone does not establish decoding performance.

| Intervention | Node | Complete | Saved prompts |
| --- | --- | --- | --- |
| delta1 | node08 | True | 128 |
| delta2 | node09 | True | 128 |
| delta32 | node08 | True | 128 |
| delta128 | node09 | True | 128 |
| radial | node08 | True | 128 |
| perpendicular | node09 | True | 128 |
| fusion_remove | node08 | True | 128 |
| random_remove | node09 | True | 128 |
| restore0 | node08 | True | 128 |
| restore1 | node09 | True | 128 |
| restore2 | node08 | True | 128 |
| restore3 | node09 | True | 128 |
| restore4 | node08 | True | 128 |
| unmodified | node08 | True | 128 |
| hook_control | node08 | True | 128 |
| unmodified | node09 | True | 128 |
| hook_control | node09 | True | 128 |

## Selected T1 probe measurements

Only complete 128-prompt Code cohorts appear. Ratios compare with the unmodified AUF parent on the same node; radial/perpendicular interventions instead use the matching hook control. Separate timing passes remain a limitation. Acceptance includes the correction/bonus token.

| Intervention | Node | Control | Mean TPS | × native (TPS) | Summed request seconds | Acceptance | TPS × control | Latency speedup × control | Latency reduction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unmodified | node09 | unmodified | 58.5547 | 0.3862 | 1138.60 | 2.3456 | 1.0000 | 1.0000 | 0.00% |

Latency speedup = summed control seconds / summed intervention seconds; below1 means slower. Latency reduction =100 × (1 − intervention seconds / control seconds); negative means increased latency. These are direct comparisons to the AUF parent, not percentages of native-drafter speedup retained. Matrix/activation energy alone does not establish decoding performance.

| Intervention | Node | Complete | Saved prompts |
| --- | --- | --- | --- |
| unmodified | node09 | True | 128 |
| fusion_remove | node09 | False | 39 |
| random_remove | node09 | False | 22 |

## T1 graph-mode serving

Speedups use aggregate tokens/sec at matched concurrency. Mean per-request TPS is separate. Acceptance=1+accepted draft tokens/num_drafts. Memory is maximum of before/after snapshots, not a measured peak. Separate timing passes, one sample per cell; no confidence intervals.

BF16 vLLM FLASH_ATTN, upstream batch invariance, compilation disabled, full CUDA graphs. Fixed512-request mixture,128 per task; greedy,2,048-token cap. No instrumented diagnostic timings appear here.

| Method | Clients | Aggregate tok/s | × AR | × native | Mean request tok/s | Acceptance | Latency p50 / p95 (s) | TTFT p50 / p95 (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mapped | 4 | 482.5971 | not completed at stop | not completed at stop | 144.3129 | 4.4889 | 2.810 / 14.516 | 0.119 / 0.163 |
| mapped | 8 | 962.4771 | not completed at stop | not completed at stop | 144.6928 | 4.4889 | 2.758 / 14.525 | 0.118 / 0.167 |

| Method | Reference concurrency | Compared concurrency |
| --- | --- | --- |
| mapped | 4 | 8 |

Cross-concurrency checks compare the same method. They do not replace the matched AR checks.


2/20 full timing cohorts complete. Matched AR comparisons available: 0/20; matched native: 0/20. Observed pairwise disagreements: 0. Missing comparisons are not completed at stop, not verified.

| Method | Clients | Node | State |
| --- | --- | --- | --- |
| ar | 1 | node08 | not completed at stop |
| eagle3 | 1 | node09 | not completed at stop |
| mapped | 1 | node08 | not completed at stop |
| native | 1 | node09 | not completed at stop |
| ar | 4 | node09 | not completed at stop |
| eagle3 | 4 | node09 | not completed at stop |
| mapped | 4 | node08 | complete |
| native | 4 | node09 | not completed at stop |
| ar | 8 | node09 | not completed at stop |
| eagle3 | 8 | node09 | not completed at stop |
| mapped | 8 | node08 | complete |
| native | 8 | node09 | not completed at stop |
| ar | 16 | node09 | not completed at stop |
| eagle3 | 16 | node09 | not completed at stop |
| mapped | 16 | node08 | not completed at stop |
| native | 16 | node09 | not completed at stop |
| ar | 32 | node09 | not completed at stop |
| eagle3 | 32 | node09 | not completed at stop |
| mapped | 32 | node08 | not completed at stop |
| native | 32 | node09 | not completed at stop |

### LoRA MSE coverage results — completed 14 September

All fits use the same active LoRA target, 4,096 examples, one epoch, 512 updates, LR $10^{-4}$, and 32,768,000 trainable parameters. The 50% and 100% variants cover prompt **and** response positions. Evaluation is 128 prompts/domain with caps 2,048 / 8,192 / 2,048 for GSM8K / KiCad / NanoCoder. Acceptance includes the correction/bonus token. Fitting excludes feature extraction, setup, export and evaluation.

| Domain | Coverage | Mean tok/s | × native | Gain/native | Change/AUF | Observed × AR | Acceptance | Fit seconds | Fit GPU h |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| math | 50% | 128.54 | 1.0111 | +1.11% | -26.51% | 3.312 | 4.4665 | 24.00 | 0.0133 |
| math | 100% | 124.80 | 0.9817 | -1.83% | -28.65% | 3.215 | 4.4696 | 34.12 | 0.0190 |
| kicad | 100% | 196.83 | 1.2050 | +20.50% | -36.42% | 5.123 | 7.2064 | 381.61 | 0.2120 |
| kicad | 50% | 194.45 | 1.1905 | +19.05% | -37.19% | 5.061 | 7.2083 | 181.75 | 0.1010 |
| nanocoder | 100% | 148.15 | 0.9863 | -1.37% | -8.73% | 3.798 | 4.9742 | 37.45 | 0.0208 |
| nanocoder | 50% | 146.78 | 0.9772 | -2.28% | -9.57% | 3.763 | 4.9573 | 29.13 | 0.0162 |

Full coverage leaves acceptance almost unchanged relative to 50%. KiCad retains approximately 20% mean-throughput gain over native DFlash; GSM8K and NanoCoder remain near native performance. These are separate timing passes, so small differences do not establish a reliable coverage benefit. AR ratios are descriptive timings. Comparisons with AUF also differ in feature backend and supervision structure, and do not isolate the loss alone.

## All MSE50 endpoints, including the reused 16k fit

## MSE 50% coverage results

Only complete 128-prompt cohorts are included. Mean-TPS speedup and summed-latency speedup are distinct measurements. Different output lengths can affect latency ratios. No unmeasured result is inferred.

| Fit | Workload | Mean TPS | Aggregate TPS | × native (mean TPS) | Gain vs native | × native (latency) | Acceptance | Fit min | GPU h |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1_mse50_n16384_e3_lr0.001 | math | 219.54 | 216.86 | 0.9989 | -0.11% | 1.0038 | 7.989 | 37.21 | 1.2402 |
| T1_mse50_n8192_e3_lr0.001 | math | 219.48 | 216.72 | 0.9986 | -0.14% | 1.0032 | 7.978 | 17.79 | 0.5931 |
| T1_mse50_n4096_e3_lr0.001 | math | 214.61 | 212.34 | 0.9764 | -2.36% | 0.9829 | 7.969 | 9.56 | 0.3187 |
| T1_mse50_n4096_e1_lr0.0001 | math | 176.96 | 170.51 | 0.8051 | -19.49% | 0.7893 | 6.290 | 2.98 | 0.0994 |
| T1_mse50_n16384_e3_lr0.001 | gsm | 175.23 | 172.56 | 0.9848 | -1.52% | 0.9925 | 6.254 | 37.21 | 1.2402 |
| T1_mse50_n8192_e3_lr0.001 | gsm | 173.55 | 170.79 | 0.9753 | -2.47% | 0.9823 | 6.257 | 17.79 | 0.5931 |
| T1_mse50_n4096_e3_lr0.001 | gsm | 167.65 | 164.88 | 0.9422 | -5.78% | 0.9483 | 6.165 | 9.56 | 0.3187 |
| T1_mse50_n4096_e1_lr0.0001 | gsm | 111.75 | 109.45 | 0.6280 | -37.20% | 0.6295 | 3.946 | 2.98 | 0.0994 |
| T1_mse50_n16384_e3_lr0.001 | code | 128.54 | 125.89 | 0.8477 | -15.23% | 0.8711 | 4.752 | 37.21 | 1.2402 |
| T1_mse50_n8192_e3_lr0.001 | code | 126.18 | 124.01 | 0.8322 | -16.78% | 0.8581 | 4.691 | 17.79 | 0.5931 |
| T1_mse50_n4096_e3_lr0.001 | code | 114.05 | 114.02 | 0.7522 | -24.78% | 0.7890 | 4.422 | 9.56 | 0.3187 |
| T1_mse50_n4096_e1_lr0.0001 | code | 46.60 | 48.89 | 0.3074 | -69.26% | 0.3383 | 1.842 | 2.98 | 0.0994 |
| T1_mse50_n16384_e3_lr0.001 | chat | 81.03 | 78.60 | 0.9053 | -9.47% | 0.8968 | 2.888 | 37.21 | 1.2402 |
| T1_mse50_n8192_e3_lr0.001 | chat | 79.49 | 76.54 | 0.8882 | -11.18% | 0.8733 | 2.812 | 17.79 | 0.5931 |
| T1_mse50_n4096_e3_lr0.001 | chat | 72.46 | 69.81 | 0.8096 | -19.04% | 0.7966 | 2.662 | 9.56 | 0.3187 |
| T1_mse50_n4096_e1_lr0.0001 | chat | 40.51 | 39.82 | 0.4526 | -54.74% | 0.4544 | 1.467 | 2.98 | 0.0994 |
| lora_math_mse50 | math | 128.54 | 127.90 | 1.0111 | +1.11% | 1.0164 | 4.467 | 0.40 | 0.0133 |
| lora_kicad_mse50 | kicad | 194.45 | 200.06 | 1.1905 | +19.05% | 1.2172 | 7.208 | 3.03 | 0.1010 |
| lora_nanocoder_mse50 | nanocoder | 146.78 | 141.81 | 0.9772 | -2.28% | 0.9741 | 4.957 | 0.49 | 0.0162 |

### Completed training

Training cost is listed as soon as a final export exists; it does not imply evaluation is complete.

| Fit | Examples | Epochs | Updates | Training min | GPU hours |
| --- | --- | --- | --- | --- | --- |
| T1_mse50_n4096_e1_lr0.0001 | 4096 | 1 | 512 | 2.98 | 0.0994 |
| T1_mse50_n16384_e3_lr0.001 | 16384 | 3 | 6144 | 37.21 | 1.2402 |
| T1_mse50_n4096_e3_lr0.001 | 4096 | 3 | 1536 | 9.56 | 0.3187 |
| T1_mse50_n8192_e3_lr0.001 | 8192 | 3 | 3072 | 17.79 | 0.5931 |
| lora_math_mse50 | 4096 | 1 | 512 | 0.40 | 0.0133 |
| lora_kicad_mse50 | 4096 | 1 | 512 | 3.03 | 0.1010 |
| lora_nanocoder_mse50 | 4096 | 1 | 512 | 0.49 | 0.0162 |

### Shared feature extraction cost

One cache serves multiple fits; do not charge it once per fit. Times below are extraction and persistence wall time for one GPU, excluding fitting.

| Cache | Examples | Seconds | GPU hours |
| --- | --- | --- | --- |
| T2 | 4096 | 1032.78 | 0.2869 |
| T1 | 16384 | 3325.78 | 0.9238 |
| nanocoder | 4096 | 688.36 | 0.1912 |
| math | 4096 | 642.81 | 0.1786 |
| T3 | 4096 | 890.90 | 0.2475 |
| kicad | 4096 | 4048.77 | 1.1247 |

## All completed Phase 2 standalone observations (including earlier recipes)

Each row is a complete 128-prompt cohort. This inventory retains earlier one-epoch, continuation, T3/T4 and later selected runs instead of silently replacing them. Method identifiers distinguish those recipes; `_a32` denotes the token-objective anchor setting. The selected reports above give the current comparisons in a more compact form. Identical named cohorts copied between roots are deduplicated; distinct node repetitions are retained.

| Pair/task | Run or control | Node | Mean TPS | × native | Descriptive AR ratio | Acceptance | Fit min | Parameters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1/chat | `T1_selected_body128_auf_n8192_e3` | node08 | 91.9628 | 1.0275 | 2.6209 | 3.4139 | 31.7912 | 48496640 |
| T1/chat | `T1_selected_body128_decay7_n8192_e3` | node09 | 90.2557 | 1.0084 | 2.5723 | 3.3477 | 31.7757 | 48496640 |
| T1/chat | `native` | node08 | 89.5033 | 1.0000 | 2.5508 | 3.3236 | — | — |
| T1/chat | `T1_mse100_n4096_e3_lr0.001` | node09 | 76.4124 | 0.8537 | 2.1777 | 2.6811 | 18.3032 | 52428800 |
| T1/chat | `T1_priority_auf_n16384_e3_lr1e4` | historical | 56.3743 | 0.6299 | 1.6067 | 1.9829 | 59.0507 | 52428800 |
| T1/chat | `T1_sweep_decay7_n8192_a32_e3` | node09 | 51.3131 | 0.5733 | 1.4624 | 1.8115 | 28.2368 | 52428800 |
| T1/chat | `T1_sweep_auf_n8192_a32_e3` | historical | 49.9253 | 0.5578 | 1.4229 | 1.7667 | 31.6433 | 52428800 |
| T1/chat | `T1_sweep_decay7_n4096_a32_e3` | historical | 43.1691 | 0.4823 | 1.2303 | 1.5760 | 13.0235 | 52428800 |
| T1/chat | `T1_sweep_auf_n4096_a32_e3` | historical | 41.9533 | 0.4687 | 1.1957 | 1.5320 | 13.2874 | 52428800 |
| T1/chat | `T1_priority_auf_n4096_plus2_lr1e4` | historical | 41.2074 | 0.4604 | 1.1744 | 1.5191 | 8.9126 | 52428800 |
| T1/chat | `T1_fivew_auf_n4096_a32` | historical | 35.2203 | 0.3935 | 1.0038 | 1.2948 | 4.7556 | 52428800 |
| T1/chat | `ar` | node08 | 35.0880 | 0.3920 | 1.0000 | 1.0000 | — | — |
| T1/code | `T1_selected_body128_auf_n8192_e3` | node08 | 155.3287 | 1.0244 | 4.1424 | 5.8636 | 31.7912 | 48496640 |
| T1/code | `T1_selected_body128_decay7_n8192_e3` | node09 | 153.8642 | 1.0148 | 4.1034 | 5.7508 | 31.7757 | 48496640 |
| T1/code | `native` | node08 | 151.6250 | 1.0000 | 4.0437 | 5.6441 | — | — |
| T1/code | `T1_mse100_n8192_e3_lr0.001` | node08 | 125.5835 | 0.8283 | 3.3492 | 4.6968 | 53.9014 | 52428800 |
| T1/code | `T1_mse100_n4096_e3_lr0.001` | node09 | 118.0456 | 0.7785 | 3.1481 | 4.4341 | 18.3032 | 52428800 |
| T1/code | `T1_selected_joint_decay7_n8192_e3` | node08 | 70.9838 | 0.4682 | 1.8931 | 2.8271 | 33.2872 | 262144000 |
| T1/code | `T1_selected_joint_auf_n8192_e3` | node08 | 68.2183 | 0.4499 | 1.8193 | 2.7418 | 36.3991 | 262144000 |
| T1/code | `T1_priority_auf_n16384_e3_lr1e4` | historical | 67.6783 | 0.4464 | 1.8049 | 2.7210 | 59.0507 | 52428800 |
| T1/code | `T1_sweep_decay7_n8192_a32_e3` | node09 | 61.5596 | 0.4060 | 1.6417 | 2.4489 | 28.2368 | 52428800 |
| T1/code | `T1_sweep_auf_n8192_a32_e3 / selected8k_unmodified` | node09 | 58.5547 | 0.3862 | 1.5616 | 2.3456 | 31.6433 | 52428800 |
| T1/code | `T1_sweep_auf_n8192_a32_e3` | historical | 58.3696 | 0.3850 | 1.5566 | 2.3456 | 31.6433 | 52428800 |
| T1/code | `T1_sweep_decay7_n4096_a32_e3` | historical | 50.3969 | 0.3324 | 1.3440 | 1.9998 | 13.0235 | 52428800 |
| T1/code | `T1_priority_decay7_n4096_plus2_lr1e4` | historical | 49.1700 | 0.3243 | 1.3113 | 1.9418 | 8.7649 | 52428800 |
| T1/code | `T1_sweep_auf_n4096_a32_e3` | historical | 48.2183 | 0.3180 | 1.2859 | 1.9104 | 13.2874 | 52428800 |
| T1/code | `T1_priority_auf_n4096_plus2_lr1e4` | historical | 47.9888 | 0.3165 | 1.2798 | 1.9023 | 8.9126 | 52428800 |
| T1/code | `T1_selected_ba128_decay7_n8192_e3` | node08 | 43.8303 | 0.2891 | 1.1689 | 1.7185 | 24.0569 | 4259840 |
| T1/code | `T1_selected_ba128_auf_n8192_e3` | node08 | 41.5391 | 0.2740 | 1.1078 | 1.6302 | 29.4859 | 4259840 |
| T1/code | `T1_selected_ba56_decay7_n8192_e3` | node09 | 38.7609 | 0.2556 | 1.0337 | 1.4981 | 24.6055 | 1863680 |
| T1/code | `ar` | node08 | 37.4970 | 0.2473 | 1.0000 | 1.0000 | — | — |
| T1/code | `T1_selected_ba56_auf_n8192_e3` | node08 | 37.3163 | 0.2461 | 0.9952 | 1.4330 | 25.2175 | 1863680 |
| T1/code | `T1_fivew_auf_n4096_a32` | historical | 36.7751 | 0.2425 | 0.9807 | 1.4041 | 4.7556 | 52428800 |
| T1/code | `T1_selected_ba28_decay7_n8192_e3` | node09 | 35.8736 | 0.2366 | 0.9567 | 1.3653 | 28.1235 | 931840 |
| T1/code | `T1_selected_ba28_auf_n8192_e3` | node08 | 34.7612 | 0.2293 | 0.9270 | 1.3335 | 29.7632 | 931840 |
| T1/code | `T1_selected_fusion_auf_n8192_e3` | node09 | 33.8244 | 0.2231 | 0.9021 | 1.2953 | 26.8966 | 52428800 |
| T1/code | `T1_selected_fusion_decay7_n8192_e3` | node09 | 33.6595 | 0.2220 | 0.8977 | 1.2921 | 27.6807 | 52428800 |
| T1/gsm | `T1_selected_body128_auf_n8192_e3` | node08 | 194.8627 | 1.0951 | 5.1603 | 7.0986 | 31.7912 | 48496640 |
| T1/gsm | `T1_selected_body128_decay7_n8192_e3` | node09 | 187.6579 | 1.0546 | 4.9695 | 6.8111 | 31.7757 | 48496640 |
| T1/gsm | `native` | node08 | 177.9411 | 1.0000 | 4.7121 | 6.4788 | — | — |
| T1/gsm | `T1_mse100_n8192_e3_lr0.001` | node08 | 176.1834 | 0.9901 | 4.6656 | 6.2244 | 53.9014 | 52428800 |
| T1/gsm | `T1_priority_auf_n16384_e3_lr1e4` | historical | 175.1087 | 0.9841 | 4.6371 | 6.2470 | 59.0507 | 52428800 |
| T1/gsm | `T1_mse100_n4096_e3_lr0.001` | node09 | 174.3330 | 0.9797 | 4.6166 | 6.1869 | 18.3032 | 52428800 |
| T1/gsm | `T1_sweep_decay7_n8192_a32_e3` | node08 | 163.4700 | 0.9187 | 4.3289 | 5.6926 | 28.2368 | 52428800 |
| T1/gsm | `T1_sweep_auf_n8192_a32_e3` | historical | 162.0495 | 0.9107 | 4.2913 | 5.6773 | 31.6433 | 52428800 |
| T1/gsm | `T1_sweep_decay7_n4096_a32_e3` | historical | 134.9602 | 0.7585 | 3.5739 | 4.7496 | 13.0235 | 52428800 |
| T1/gsm | `T1_priority_decay7_n4096_plus2_lr1e4` | historical | 132.1416 | 0.7426 | 3.4993 | 4.6352 | 8.7649 | 52428800 |
| T1/gsm | `T1_sweep_auf_n4096_a32_e3` | historical | 131.9362 | 0.7415 | 3.4939 | 4.5797 | 13.2874 | 52428800 |
| T1/gsm | `T1_priority_auf_n4096_plus2_lr1e4` | historical | 125.7897 | 0.7069 | 3.3311 | 4.4761 | 8.9126 | 52428800 |
| T1/gsm | `T1_fivew_decay7_n4096_a32` | historical | 78.3291 | 0.4402 | 2.0743 | 2.7460 | 4.5598 | 52428800 |
| T1/gsm | `T1_fivew_auf_n4096_a32` | historical | 74.3406 | 0.4178 | 1.9686 | 2.6269 | 4.7556 | 52428800 |
| T1/gsm | `ar` | node08 | 37.7622 | 0.2122 | 1.0000 | 1.0000 | — | — |
| T1/math | `T1_selected_body128_auf_n8192_e3` | node08 | 230.8243 | 1.0502 | 6.1004 | 8.6097 | 31.7912 | 48496640 |
| T1/math | `T1_selected_body128_decay7_n8192_e3` | node09 | 226.1528 | 1.0290 | 5.9770 | 8.4241 | 31.7757 | 48496640 |
| T1/math | `native` | node08 | 219.7874 | 1.0000 | 5.8087 | 8.2108 | — | — |
| T1/math | `T1_mse100_n8192_e3_lr0.001` | node08 | 219.3158 | 0.9979 | 5.7963 | 7.9994 | 53.9014 | 52428800 |
| T1/math | `T1_mse100_n4096_e3_lr0.001` | node09 | 219.1429 | 0.9971 | 5.7917 | 7.9711 | 18.3032 | 52428800 |
| T1/math | `T1_sweep_mse_n4096_a24_e3` | historical | 209.5590 | 0.9535 | 5.5384 | 7.6579 | 6.1950 | 52428800 |
| T1/math | `T1_priority_auf_n16384_e3_lr1e4` | historical | 207.2180 | 0.9428 | 5.4766 | 7.4294 | 59.0507 | 52428800 |
| T1/math | `T1_sweep_mse_n4096_a8_e3` | historical | 206.1221 | 0.9378 | 5.4476 | 7.4395 | 1.0774 | 52428800 |
| T1/math | `T1_sweep_auf_n8192_a32_e3 / selected8k_unmodified` | node08 | 195.3339 | 0.8887 | 5.1625 | 6.8923 | 31.6433 | 52428800 |
| T1/math | `T1_selected_joint_decay7_n8192_e3` | node08 | 195.2239 | 0.8882 | 5.1596 | 6.9644 | 33.2872 | 262144000 |
| T1/math | `T1_sweep_auf_n8192_a32_e3 / selected8k_unmodified` | node09 | 194.7817 | 0.8862 | 5.1479 | 6.8923 | 31.6433 | 52428800 |
| T1/math | `T1_sweep_decay7_n8192_a32_e3` | node08 | 194.6915 | 0.8858 | 5.1455 | 6.9474 | 28.2368 | 52428800 |
| T1/math | `T1_sweep_auf_n8192_a32_e3 / selected8k_random_remove` | node09 | 194.4424 | 0.8847 | 5.1389 | 6.8735 | 31.6433 | 52428800 |
| T1/math | `T1_sweep_auf_n8192_a32_e3` | historical | 194.2206 | 0.8837 | 5.1330 | 6.8923 | 31.6433 | 52428800 |
| T1/math | `T1_sweep_auf_n8192_a32_e3 / selected8k_hook_control` | node08 | 193.6967 | 0.8813 | 5.1192 | 6.8956 | 31.6433 | 52428800 |
| T1/math | `T1_sweep_auf_n8192_a32_e3 / selected8k_hook_control` | node09 | 193.4648 | 0.8802 | 5.1131 | 6.8956 | 31.6433 | 52428800 |
| T1/math | `T1_selected_joint_auf_n8192_e3` | node08 | 193.3218 | 0.8796 | 5.1093 | 6.9562 | 36.3991 | 262144000 |
| T1/math | `T1_sweep_auf_n8192_a32_e3 / selected8k_restore1` | node09 | 189.0736 | 0.8603 | 4.9970 | 6.7770 | 31.6433 | 52428800 |
| T1/math | `T1_sweep_auf_n8192_a32_e3 / selected8k_restore0` | node08 | 188.0331 | 0.8555 | 4.9695 | 6.6363 | 31.6433 | 52428800 |
| T1/math | `T1_sweep_auf_n8192_a32_e3 / selected8k_restore2` | node08 | 180.0708 | 0.8193 | 4.7591 | 6.3676 | 31.6433 | 52428800 |
| T1/math | `T1_sweep_decay7_n4096_a32_e3` | historical | 173.8943 | 0.7912 | 4.5958 | 6.1506 | 13.0235 | 52428800 |
| T1/math | `T1_priority_decay7_n4096_plus2_lr1e4` | historical | 171.4343 | 0.7800 | 4.5308 | 6.0574 | 8.7649 | 52428800 |
| T1/math | `T1_sweep_auf_n4096_a32_e3` | historical | 170.1361 | 0.7741 | 4.4965 | 6.0168 | 13.2874 | 52428800 |
| T1/math | `T1_priority_auf_n4096_plus2_lr1e4` | historical | 169.2277 | 0.7700 | 4.4725 | 5.9179 | 8.9126 | 52428800 |
| T1/math | `T1_sweep_auf_n8192_a32_e3 / selected8k_perpendicular` | node09 | 155.3280 | 0.7067 | 4.1052 | 5.4175 | 31.6433 | 52428800 |
| T1/math | `T1_sweep_auf_n8192_a32_e3 / selected8k_delta128` | node09 | 153.9099 | 0.7003 | 4.0677 | 5.4182 | 31.6433 | 52428800 |
| T1/math | `T1_selected_ba128_decay7_n8192_e3` | node08 | 149.9202 | 0.6821 | 3.9622 | 5.2374 | 24.0569 | 4259840 |
| T1/math | `T1_selected_ba128_auf_n8192_e3` | node08 | 143.0149 | 0.6507 | 3.7797 | 4.9728 | 29.4859 | 4259840 |
| T1/math | `T1_sweep_auf_n4096_a8_e3` | historical | 127.2140 | 0.5788 | 3.3621 | 4.6098 | 10.3192 | 52428800 |
| T1/math | `T1_selected_ba56_decay7_n8192_e3` | node09 | 119.0842 | 0.5418 | 3.1473 | 4.2197 | 24.6055 | 1863680 |
| T1/math | `T1_selected_ba56_auf_n8192_e3` | node08 | 109.9603 | 0.5003 | 2.9061 | 3.7969 | 25.2175 | 1863680 |
| T1/math | `T1_fivew_decay7_n4096_a32` | historical | 107.7487 | 0.4902 | 2.8477 | 3.7605 | 4.5598 | 52428800 |
| T1/math | `T1_sweep_auf_n8192_a32_e3 / selected8k_restore3` | node09 | 100.1638 | 0.4557 | 2.6472 | 3.5735 | 31.6433 | 52428800 |
| T1/math | `T1_fivew_auf_n4096_a32` | historical | 95.3304 | 0.4337 | 2.5195 | 3.4532 | 4.7556 | 52428800 |
| T1/math | `T1_selected_ba28_decay7_n8192_e3` | node09 | 93.2439 | 0.4242 | 2.4643 | 3.2391 | 28.1235 | 931840 |
| T1/math | `T1_selected_ba28_auf_n8192_e3` | node08 | 82.5736 | 0.3757 | 2.1823 | 2.8760 | 29.7632 | 931840 |
| T1/math | `T1_selected_fusion_decay7_n8192_e3` | node09 | 80.2527 | 0.3651 | 2.1210 | 2.8394 | 27.6807 | 52428800 |
| T1/math | `T1_selected_fusion_auf_n8192_e3` | node09 | 80.1810 | 0.3648 | 2.1191 | 2.8402 | 26.8966 | 52428800 |
| T1/math | `T1_sweep_auf_n8192_a32_e3 / selected8k_restore4` | node08 | 71.1581 | 0.3238 | 1.8806 | 2.3407 | 31.6433 | 52428800 |
| T1/math | `T1_sweep_auf_n8192_a32_e3 / selected8k_delta32` | node08 | 58.2957 | 0.2652 | 1.5407 | 2.1151 | 31.6433 | 52428800 |
| T1/math | `T1_sweep_auf_n8192_a32_e3 / selected8k_fusion_remove` | node08 | 38.6362 | 0.1758 | 1.0211 | 1.4086 | 31.6433 | 52428800 |
| T1/math | `ar` | node08 | 37.8373 | 0.1722 | 1.0000 | 1.0000 | — | — |
| T1/math | `T1_sweep_auf_n8192_a32_e3 / selected8k_delta2` | node09 | 29.2849 | 0.1332 | 0.7740 | 1.0780 | 31.6433 | 52428800 |
| T1/math | `T1_sweep_auf_n8192_a32_e3 / selected8k_delta1` | node08 | 29.0863 | 0.1323 | 0.7687 | 1.0377 | 31.6433 | 52428800 |
| T1/math | `T1_sweep_auf_n8192_a32_e3 / selected8k_radial` | node08 | 28.2457 | 0.1285 | 0.7465 | 1.0155 | 31.6433 | 52428800 |
| T3/chat | `ar` | historical | 64.7250 | — | 1.0000 | 1.0000 | — | — |
| T3/chat | `T3_fivew_decay7_n4096_a32` | historical | 54.1219 | — | 0.8362 | 1.2650 | 4.9464 | 62914560 |
| T3/chat | `T3_fivew_auf_n4096_a32` | historical | 54.0551 | — | 0.8352 | 1.2580 | 5.1403 | 62914560 |
| T3/code | `T3_fivew_decay7_n4096_a32` | historical | 63.5709 | — | 1.0077 | 1.4733 | 4.9464 | 62914560 |
| T3/code | `ar` | historical | 63.0876 | — | 1.0000 | 1.0000 | — | — |
| T3/code | `T3_fivew_auf_n4096_a32` | historical | 59.2704 | — | 0.9395 | 1.4663 | 5.1403 | 62914560 |
| T3/code | `T3_ba28_auf_n4096_a32` | historical | 48.0659 | — | 0.7619 | 1.1120 | 4.4572 | 1003520 |
| T3/code | `T3_fusion_auf_n4096_a32` | historical | 47.8449 | — | 0.7584 | 1.0959 | 4.6463 | 62914560 |
| T3/code | `T3_ba28_decay7_n4096_a32` | historical | 47.2628 | — | 0.7492 | 1.0989 | 4.4416 | 1003520 |
| T3/code | `T3_fusion_decay7_n4096_a32` | historical | 46.5790 | — | 0.7383 | 1.0818 | 4.5926 | 62914560 |
| T3/gsm | `T3_fivew_decay7_n4096_a32` | historical | 97.8498 | — | 1.5182 | 2.3818 | 4.9464 | 62914560 |
| T3/gsm | `T3_fivew_auf_n4096_a32` | historical | 96.3963 | — | 1.4957 | 2.3962 | 5.1403 | 62914560 |
| T3/gsm | `ar` | historical | 64.4510 | — | 1.0000 | 1.0000 | — | — |
| T3/gsm | `initial` | historical | 41.3441 | — | 0.6415 | 1.0087 | — | — |
| T3/math | `T3_fivew_auf_n4096_a32` | historical | 120.2144 | — | 1.9422 | 2.9048 | 5.1403 | 62914560 |
| T3/math | `T3_fivew_decay7_n4096_a32` | historical | 115.6398 | — | 1.8683 | 2.8995 | 4.9464 | 62914560 |
| T3/math | `ar` | historical | 61.8956 | — | 1.0000 | 1.0000 | — | — |
| T3/math | `T3_ba28_auf_n4096_a32` | historical | 59.1095 | — | 0.9550 | 1.3343 | 4.4572 | 1003520 |
| T3/math | `T3_fusion_auf_n4096_a32` | historical | 57.3693 | — | 0.9269 | 1.3182 | 4.6463 | 62914560 |
| T3/math | `T3_ba28_decay7_n4096_a32` | historical | 56.7761 | — | 0.9173 | 1.2679 | 4.4416 | 1003520 |
| T3/math | `T3_fusion_decay7_n4096_a32` | historical | 55.1158 | — | 0.8905 | 1.2317 | 4.5926 | 62914560 |
| T3/math | `initial` | historical | 42.3190 | — | 0.6837 | 1.0084 | — | — |
| T4/chat | `ar` | historical | 40.9593 | — | 1.0000 | 1.0000 | — | — |
| T4/code | `ar` | historical | 40.7880 | — | 1.0000 | 1.0000 | — | — |
| T4/gsm | `ar` | historical | 41.1094 | — | 1.0000 | 1.0000 | — | — |
| T4/math | `ar` | historical | 41.0048 | — | 1.0000 | 1.0000 | — | — |

## Completed fits, including fits without completed evaluation

Fit identifiers preserve exact architecture/loss/size recipes. No inference result is inferred from a completed fit. Times for resumed fits follow each saved summary; do not sum repeated source/continuation training to estimate total allocation cost.

| Fit | Examples | Epochs | LR | Updates | Fit min | GPU h | Setup / initialization / export seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `T1_ba128_auf_n4096_a32` | 4096 | 1 | 0.0001 | 512 | 4.0568 | 0.1352 | 10.2284 / 1.7732 / 0.9332 |
| `T1_ba128_decay7_n4096_a32` | 4096 | 1 | 0.0001 | 512 | 4.0384 | 0.1346 | 8.5410 / 1.7643 / 1.1305 |
| `T1_ba28_auf_n4096_a32` | 4096 | 1 | 0.0001 | 512 | 4.3310 | 0.1444 | 9.9652 / 1.7694 / 0.9785 |
| `T1_ba28_decay7_n4096_a32` | 4096 | 1 | 0.0001 | 512 | 4.1859 | 0.1395 | 11.0379 / 1.7740 / 0.9926 |
| `T1_ba56_auf_n4096_a32` | 4096 | 1 | 0.0001 | 512 | 4.1549 | 0.1385 | 10.7209 / 1.7698 / 0.9764 |
| `T1_ba56_decay7_n4096_a32` | 4096 | 1 | 0.0001 | 512 | 4.0593 | 0.1353 | 11.2965 / 1.7753 / 1.0223 |
| `T1_body128_auf_n4096_a32` | 4096 | 1 | 0.0001 | 512 | 5.4524 | 0.1817 | 11.9674 / 0.0000 / 2.2291 |
| `T1_body128_decay7_n4096_a32` | 4096 | 1 | 0.0001 | 512 | 5.3792 | 0.1793 | 11.2803 / 0.0000 / 2.2115 |
| `T1_fivew_auf_n4096_a32` | 4096 | 1 | 0.0001 | 512 | 4.7556 | 0.1585 | 5.8439 / 0.0176 / 1.4577 |
| `T1_fivew_decay7_n4096_a32` | 4096 | 1 | 0.0001 | 512 | 4.5598 | 0.1520 | 9.5934 / 0.0190 / 1.2206 |
| `T1_fusion_auf_n4096_a32` | 4096 | 1 | 0.0001 | 512 | 4.8197 | 0.1607 | 10.1458 / 0.0948 / 1.0784 |
| `T1_fusion_decay7_n4096_a32` | 4096 | 1 | 0.0001 | 512 | 4.6981 | 0.1566 | 9.9911 / 0.0968 / 0.9068 |
| `T1_joint_auf_n4096_a32` | 4096 | 1 | 0.0001 | 512 | 5.5329 | 0.1844 | 9.2628 / 0.0250 / 2.6514 |
| `T1_joint_decay7_n4096_a32` | 4096 | 1 | 0.0001 | 512 | 5.5666 | 0.1856 | 7.6109 / 0.0229 / 2.4088 |
| `T1_mse100_n4096_e3_lr0.001` | 4096 | 3 | 0.001 | 1536 | 18.3032 | 0.6101 | 4.4681 / 0.0264 / 1.2480 |
| `T1_mse100_n8192_e3_lr0.001` | 8192 | 3 | 0.001 | 3072 | 53.9014 | 1.7967 | 3.4497 / 0.0223 / 1.3005 |
| `T1_priority_auf_n16384_e3_lr1e4` | 16384 | 3 | 0.0001 | 6144 | 59.0507 | 1.9684 | 10.4315 / 0.0208 / 1.5610 |
| `T1_priority_auf_n4096_plus2_lr1e4` | 4096 | 2 | 0.0001 | 1024 | 8.9126 | 0.2971 | 5.6374 / 0.0183 / 1.2830 |
| `T1_priority_decay7_n4096_plus2_lr1e4` | 4096 | 2 | 0.0001 | 1024 | 8.7649 | 0.2922 | 9.0154 / 0.0202 / 1.3091 |
| `T1_selected_ba128_auf_n8192_e3` | 8192 | 3 | 0.0001 | 3072 | 29.4859 | 0.9829 | 10.5382 / 1.7711 / 1.0282 |
| `T1_selected_ba128_decay7_n8192_e3` | 8192 | 3 | 0.0001 | 3072 | 24.0569 | 0.8019 | 7.5216 / 1.8048 / 0.8727 |
| `T1_selected_ba28_auf_n8192_e3` | 8192 | 3 | 0.0001 | 3072 | 29.7632 | 0.9921 | 7.2423 / 1.7154 / 1.0132 |
| `T1_selected_ba28_decay7_n8192_e3` | 8192 | 3 | 0.0001 | 3072 | 28.1235 | 0.9374 | 7.0466 / 1.8259 / 0.8988 |
| `T1_selected_ba56_auf_n8192_e3` | 8192 | 3 | 0.0001 | 3072 | 25.2175 | 0.8406 | 8.1886 / 1.7887 / 1.1010 |
| `T1_selected_ba56_decay7_n8192_e3` | 8192 | 3 | 0.0001 | 3072 | 24.6055 | 0.8202 | 6.9974 / 1.7860 / 0.9206 |
| `T1_selected_body128_auf_n8192_e3` | 8192 | 3 | 0.0001 | 3072 | 31.7912 | 1.0597 | 8.2654 / 0.0000 / 1.8327 |
| `T1_selected_body128_decay7_n8192_e3` | 8192 | 3 | 0.0001 | 3072 | 31.7757 | 1.0592 | 10.8394 / 0.0000 / 1.8358 |
| `T1_selected_fusion_auf_n8192_e3` | 8192 | 3 | 0.0001 | 3072 | 26.8966 | 0.8966 | 11.9389 / 0.0983 / 1.3188 |
| `T1_selected_fusion_decay7_n8192_e3` | 8192 | 3 | 0.0001 | 3072 | 27.6807 | 0.9227 | 8.5079 / 0.0939 / 1.1943 |
| `T1_selected_joint_auf_n8192_e3` | 8192 | 3 | 0.0001 | 3072 | 36.3991 | 1.2133 | 9.5867 / 0.0255 / 2.0476 |
| `T1_selected_joint_decay7_n8192_e3` | 8192 | 3 | 0.0001 | 3072 | 33.2872 | 1.1096 | 5.7356 / 0.0236 / 2.1655 |
| `T1_sweep_auf_n4096_a32_e3` | 4096 | 3 | 0.0001 | 1536 | 13.2874 | 0.4429 | 10.4930 / 0.0302 / 1.4725 |
| `T1_sweep_auf_n4096_a8_e3` | 4096 | 3 | 0.0001 | 1536 | 10.3192 | 0.3440 | 9.1982 / 0.0188 / 1.2071 |
| `T1_sweep_auf_n8192_a32_e3` | 8192 | 3 | 0.0001 | 3072 | 31.6433 | 1.0548 | 8.5263 / 0.0194 / 1.3958 |
| `T1_sweep_decay7_n4096_a32_e3` | 4096 | 3 | 0.0001 | 1536 | 13.0235 | 0.4341 | 8.7769 / 0.0246 / 1.3339 |
| `T1_sweep_decay7_n8192_a32_e3` | 8192 | 3 | 0.0001 | 3072 | 28.2368 | 0.9412 | 16.5740 / 0.0197 / 1.5714 |
| `T1_sweep_mse_n4096_a24_e3` | 4096 | 3 | 0.001 | 1536 | 6.1950 | 0.2065 | 3.3853 / 0.0194 / 1.1746 |
| `T1_sweep_mse_n4096_a8_e3` | 4096 | 3 | 0.001 | 1536 | 1.0774 | 0.0359 | 3.1741 / 0.0174 / 1.3830 |
| `T2_body128_auf_n4096_a32` | 4096 | 1 | 0.0001 | 512 | 3.7824 | 0.1261 | 9.0496 / 0.0000 / 1.2615 |
| `T2_body128_decay7_n4096_a32` | 4096 | 1 | 0.0001 | 512 | 3.7345 | 0.1245 | 10.2005 / 0.0000 / 1.4344 |
| `T2_fivew_auf_n4096_a32` | 4096 | 1 | 0.0001 | 512 | 5.5500 | 0.1850 | 7.0695 / 0.0179 / 1.5741 |
| `T2_fivew_decay7_n4096_a32` | 4096 | 1 | 0.0001 | 512 | 5.3331 | 0.1778 | 11.5667 / 0.0226 / 1.5821 |
| `T3_ba128_auf_n4096_a32` | 4096 | 1 | 0.0001 | 512 | 4.4946 | 0.1498 | 12.0737 / 3.8758 / 1.3747 |
| `T3_ba128_decay7_n4096_a32` | 4096 | 1 | 0.0001 | 512 | 4.4919 | 0.1497 | 19.2670 / 3.8591 / 1.4384 |
| `T3_ba28_auf_n4096_a32` | 4096 | 1 | 0.0001 | 512 | 4.4572 | 0.1486 | 13.2571 / 3.9048 / 1.3275 |
| `T3_ba28_decay7_n4096_a32` | 4096 | 1 | 0.0001 | 512 | 4.4416 | 0.1481 | 10.0451 / 3.8592 / 1.4272 |
| `T3_ba56_auf_n4096_a32` | 4096 | 1 | 0.0001 | 512 | 4.4592 | 0.1486 | 10.1981 / 3.8603 / 1.3467 |
| `T3_ba56_decay7_n4096_a32` | 4096 | 1 | 0.0001 | 512 | 4.4726 | 0.1491 | 13.5962 / 3.8715 / 1.4131 |
| `T3_fivew_auf_n4096_a32` | 4096 | 1 | 0.0001 | 512 | 5.1403 | 0.1713 | 9.0185 / 0.0197 / 2.3370 |
| `T3_fivew_decay7_n4096_a32` | 4096 | 1 | 0.0001 | 512 | 4.9464 | 0.1649 | 10.2386 / 0.0197 / 1.6812 |
| `T3_fusion_auf_n4096_a32` | 4096 | 1 | 0.0001 | 512 | 4.6463 | 0.1549 | 6.2923 / 0.0889 / 1.4870 |
| `T3_fusion_decay7_n4096_a32` | 4096 | 1 | 0.0001 | 512 | 4.5926 | 0.1531 | 8.7555 / 0.0986 / 1.3900 |
| `T3_joint_auf_n4096_a32` | 4096 | 1 | 0.0001 | 512 | 6.2167 | 0.2072 | 9.9182 / 0.0257 / 2.9022 |
| `T3_joint_decay7_n4096_a32` | 4096 | 1 | 0.0001 | 512 | 6.1806 | 0.2060 | 15.7252 / 0.0253 / 2.7877 |

## Interrupted or partial standalone outputs

Preserved for inspection only. Saved counts are progress, not completed benchmark results.

| Artifact | Node | Saved prompts | State |
| --- | --- | --- | --- |
| `T1_initial_gsm_part0.json` | node08 | 109 | stopped incomplete; excluded from final metrics |
| `T1_initial_math_part0.json` | node08 | 53 | stopped incomplete; excluded from final metrics |
| `T1_mse100_n8192_e3_lr0.001_chat_part0.json` | node08 | 43 | stopped incomplete; excluded from final metrics |
| `T1_mse100_n8192_e3_lr0.001_chat_part1.json` | node09 | 43 | stopped incomplete; excluded from final metrics |
| `T1_mse100_n8192_e3_lr0.001_chat_part2.json` | node09 | 42 | stopped incomplete; excluded from final metrics |
| `T1_priority_decay7_n4096_plus2_lr1e4_chat_part0.json` | historical | 16 | stopped incomplete; excluded from final metrics |
| `T1_sweep_auf_n4096_a8_e3_code_part0.json` | historical | 56 | stopped incomplete; excluded from final metrics |
| `T1_sweep_auf_n8192_a32_e3_selected8k_fusion_remove_code_part0.json` | node09 | 39 | stopped incomplete; excluded from final metrics |
| `T1_sweep_auf_n8192_a32_e3_selected8k_random_remove_code_part0.json` | node09 | 22 | stopped incomplete; excluded from final metrics |
| `T2_ar_code_part0.json` | historical | 1 | stopped incomplete; excluded from final metrics |
| `T2_ar_gsm_part0.json` | historical | 93 | stopped incomplete; excluded from final metrics |
| `T2_ar_math_part0.json` | historical | 45 | stopped incomplete; excluded from final metrics |
| `T3_ba56_auf_n4096_a32_code_part0.json` | historical | 108 | stopped incomplete; excluded from final metrics |
| `T3_ba56_auf_n4096_a32_math_part0.json` | historical | 116 | stopped incomplete; excluded from final metrics |
| `T3_ba56_decay7_n4096_a32_math_part0.json` | historical | 31 | stopped incomplete; excluded from final metrics |

## Plots and evidence

![Recipe scaling](experiments/dflash_mse100_20260914/results/figures/recipe_scaling.svg)

![Recipe fitting cost](experiments/dflash_mse100_20260914/results/figures/recipe_cost.svg)

![Math interventions](experiments/dflash_mse100_20260914/results/figures/math_interventions.svg)

[Acceptance distribution](experiments/dflash_mse100_20260914/results/selected_acceptance_distribution.csv) · [Serving acceptance distribution](experiments/dflash_mse100_20260914/results/serving_acceptance_distribution.csv) · [Stopped stage inventory](experiments/dflash_mse100_20260914/validation/user_stop.json) · [Execution plan](phase2.md)

No missing measurements have been extrapolated. Native-relative timing runs are single passes, not repeated-run confidence intervals. No restart is authorized by this report.


## Selected T1–T3 recipe: completed training and non-AR evaluation

**Training and the requested non-AR evaluation are complete for T1–T3.** Standalone AR timing is complete; see the completed AR table above. Concurrent AR remains pending. T3 has no native-drafter denominator and its Transformers outputs are not identical to saved AR; see the explicit audit below. Completion here means the specified runs and audits finished, not that every transfer achieved native parity.

All three selected fits train five Xavier-initialized linear maps using normalized layer-plus-context MSE,16,384 NuminaMath examples,25% fixed prompt/response coverage,three epochs,LR0.001,seed42 and global batch8 on two GPUs. Both teachers and the drafter outside the maps remain frozen; maps are folded into fusion for inference. Training response cap is4,096; evaluation cap is2,048. Sampling is fixed across epochs. T1 reuses its original saved cache; T2/T3 reuse4,096 compatible trajectories each and generate only the12,288 missing trajectories with vLLM. T3 preserves its historical prompt-template date and the unknown cap metadata for the reused records; see the collection contract.

Fit time is optimizer-loop wall time, not total collection/capture or scheduler allocation time. Final loss is the last training minibatch loss, not a held-out loss or epoch mean.

| Transfer | Target / source drafter | Fit min | Fit GPU-hours | Updates | Final minibatch loss | Standalone evaluation | vLLM non-AR evaluation |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| T1 | Qwen3-8B / Qwen3-4B | 24.91 | 0.8303 | 6,144 | 0.239890 | Complete: native + mapped,4 tasks | Complete: native + mapped,4 tasks × clients1/8/16/32 |
| T2 | Qwen3-4B / Qwen3-8B | 19.09 | 0.6363 | 6,144 | 0.270358 | Complete: native + mapped,4 tasks | Complete: native + mapped,4 tasks ×16 clients |
| T3 | Llama3.2-3B / Llama3.1-8B | 20.58 | 0.6861 | 6,144 | 0.421100 | Complete: mapped,4 tasks; qualified | Complete: mapped,4 tasks ×16 clients |

Every full task cell contains128 prompts. The44 completed serving method cells comprise32 T1,8 T2 and4 T3 cells (5,632 requests). Compatible historical T1 standalone controls are reused. Native-relative rates are not AR-speedup-retention percentages.

### Selected T1 standalone checkpoint

Single-request Transformers. Mean TPS averages request rates; pooled TPS divides total returned tokens by summed request times. Acceptance pools emitted block lengths. These rows reuse the selected16k/25% fit and saved native controls, not the identity-init retry.

| Dataset | Native mean TPS | Mapped mean TPS | Mapped/native | Native pooled TPS | Mapped pooled TPS | Native acceptance | Mapped acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| math | 219.79 | 220.26 | 1.0022× | 216.03 | 217.27 | 8.2108 | 7.9841 |
| gsm | 177.94 | 176.87 | 0.9940× | 173.86 | 173.75 | 6.4788 | 6.2416 |
| code | 151.63 | 127.61 | 0.8416× | 144.52 | 125.46 | 5.6441 | 4.7391 |
| chat | 89.50 | 80.93 | 0.9043× | 87.64 | 78.42 | 3.3236 | 2.8868 |



## Earlier serving profile — T1 selected 16k / 25% mapper

128 prompts per dataset, 2,048-token output cap, one L40S per engine; clients 1/8/16/32. Aggregate tokens/s is total returned tokens divided by workload wall time, including drain. Acceptance is `1 + accepted draft tokens / verification steps`. Native and mapped output IDs, input IDs, caps and finish reasons agree on every listed prompt; engine versions, GPU model and timing-validity checks match. Full AR validation and AR speedup remain deferred. These are native-relative measurements, not AR speedup-retention percentages.

| Dataset | Clients | Native tok/s | Mapped tok/s | Mapped / native | Native acceptance | Mapped acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| math | 1 | 184.17 | 189.84 | 1.0308× | 8.0856 | 7.9319 |
| math | 8 | 1489.24 | 1552.06 | 1.0422× | 8.0856 | 7.9319 |
| math | 16 | 2221.08 | 2325.92 | 1.0472× | 8.0856 | 7.9319 |
| math | 32 | 2533.49 | 2716.26 | 1.0721× | 8.0856 | 7.9319 |
| gsm | 1 | 144.40 | 148.03 | 1.0251× | 6.4725 | 6.2847 |
| gsm | 8 | 1143.84 | 1170.43 | 1.0232× | 6.4725 | 6.2847 |
| gsm | 16 | 1839.15 | 1904.40 | 1.0355× | 6.4725 | 6.2847 |
| gsm | 32 | 2220.14 | 2337.06 | 1.0527× | 6.4725 | 6.2847 |
| code | 1 | 131.26 | 117.00 | 0.8913× | 5.8005 | 4.8945 |
| code | 8 | 1037.92 | 924.55 | 0.8908× | 5.8005 | 4.8945 |
| code | 16 | 1559.34 | 1394.86 | 0.8945× | 5.8005 | 4.8945 |
| code | 32 | 1851.73 | 1653.06 | 0.8927× | 5.8005 | 4.8945 |
| chat | 1 | 76.17 | 69.90 | 0.9177× | 3.3027 | 2.8734 |
| chat | 8 | 651.22 | 601.48 | 0.9236× | 3.3027 | 2.8734 |
| chat | 16 | 1023.80 | 961.20 | 0.9389× | 3.3027 | 2.8734 |
| chat | 32 | 1149.97 | 1103.12 | 0.9593× | 3.3027 | 2.8734 |

## Completed T1 rectangular-identity retry

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

### Old versus new

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

| Loss | Fit wall time | Fit GPU-hours (two GPUs) |
| --- | ---: | ---: |
| MSE25 | 11.11 min | 0.370 |
| AUF | 30.14 min | 1.005 |
| Decay CE | 29.54 min | 0.985 |

**What this establishes.** Among these three new fits, MSE25 has the highest average acceptance length on **all four datasets**, and the highest measured TPS on Math, Code and Chat. AUF has the highest measured GSM8K TPS (175.35 versus 166.70 for MSE), despite slightly lower acceptance (6.166 versus 6.217). Acceptance length is not the only determinant of elapsed time; this single 128-prompt measurement does not establish that timing noise alone explains the reversal. There are no repeated timing trials here.

Relative to the old runs, the combined identity initialization and higher LR improve AUF/decay acceptance and TPS on every dataset, with approximately 19–25% TPS gains on Code/Chat. These changes still do not close the acceptance-length gap to MSE. **Their causal contributions cannot be separated:** old AUF/decay used Xavier initialization and LR $10^{-4}$, whereas new AUF/decay use identity and LR $10^{-3}$.

MSE retains LR $10^{-3}$, making its comparison an initializer-only intervention. Identity yields 2.31–5.15% lower measured TPS on all four datasets, essentially unchanged/slightly higher acceptance on Math/GSM (+0.14%/+0.21%), and lower acceptance on Code/Chat (−2.58%/−1.94%). Thus identity offers no demonstrated advantage for MSE; it does **not** uniformly worsen acceptance. This supports retaining the original Xavier MSE recipe, without claiming that these runs prove why MSE works better or that identity alone helped the token losses.

Fits: 34146–34148. Mapped evaluation shards: 34160–34171. Source and machine-readable comparison: `experiments/dflash_identity_20260915/`; final checkpoints remain in the original Phase 2 `runs/T1_identity_*_n8192_e3/` directories.

## Earlier serving profile — T2 selected 16k / 25% five-map MSE

Qwen3-4B target with the frozen Qwen3-8B drafter, compared with the target’s native drafter. Three epochs, Xavier initialization, LR0.001;16 concurrent clients,128 prompts per task,2,048-token output cap, one L40S per engine. Aggregate TPS includes workload drain; acceptance is1+accepted draft tokens/verification steps. All inputs, output tokens, caps and finish reasons match128/128 within each workload. Standalone AR timing is complete; concurrent AR remains pending. These are native-relative throughput ratios, not AR-speedup retention.

| Dataset | Native TPS | Mapped TPS | Mapped/native | Native acceptance | Mapped acceptance |
| --- | ---: | ---: | ---: | ---: | ---: |
| math | 3412.62 | 3049.92 | 0.8937× | 8.0578 | 7.6492 |
| gsm | 2619.51 | 2289.25 | 0.8739× | 6.3195 | 5.9828 |
| code | 2286.85 | 1741.66 | 0.7616× | 6.0305 | 4.6646 |
| chat | 1387.87 | 1095.09 | 0.7890× | 3.3130 | 2.7858 |

The larger transferred drafter does not recover native serving throughput in this direction. Transformers evaluation is still in progress and remains a separate timing comparison.

## T2 completed Transformers non-AR comparison

Same selected 16,384-example, 25%-coverage, three-epoch Xavier MSE checkpoint as the T2 serving table. Single-request Transformers, 128 prompts per dataset, 2,048-token cap. Standalone AR timing is complete; see the completed AR table above. Concurrent AR remains pending. Mean TPS averages per-request rates; pooled TPS divides total returned tokens by summed request time. Acceptance pools emitted block lengths, including the target contribution.

| Dataset | Native mean TPS | Mapped mean TPS | Mean TPS vs native | Native pooled TPS | Mapped pooled TPS | Native acceptance | Mapped acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| math | 224.52 | 216.37 | 0.9637× | 222.28 | 212.85 | 8.0911 | 7.6596 |
| gsm | 177.32 | 169.53 | 0.9561× | 173.61 | 165.37 | 6.3415 | 5.9874 |
| code | 171.68 | 130.83 | 0.7621× | 174.01 | 136.92 | 6.2619 | 4.8890 |
| chat | 95.34 | 81.16 | 0.8512× | 93.64 | 79.74 | 3.3608 | 2.8289 |


The full mapped run agrees exactly with saved AR outputs on242/512 prompts: Math68/128, GSM8K81/128, Code55/128 and Chat38/128. Audit34257 traced the first disagreement for every remaining prompt. All270 reproduce different target argmax tokens on the same accepted prefix under AR versus block verification. Complete saved T3 AR timings are included in the paper; concurrent AR remains deferred. Raw logits and per-case traces are persisted in the T3 validation artifacts.

## T3 descriptive non-AR throughput and acceptance

Llama3.2-3B target with the Llama3.1-8B drafter; selected five-map Xavier MSE recipe,16,384 examples,25% coverage,three epochs. Each cell uses128 prompts,2,048-token output cap and one L40S. These rates are not native-relative speedups: no native drafter denominator is established for T3. Complete saved Transformers AR timing is now included in the paper as a descriptive comparison; concurrent AR is deferred.

Transformers mean TPS averages request rates; pooled TPS divides total returned tokens by summed request times. vLLM aggregate TPS uses workload wall time including drain at16 clients. Their timing denominators differ. Transformers acceptance pools emitted block lengths; vLLM acceptance is1+accepted draft tokens/verification steps.

| Dataset | TF mean TPS | TF pooled TPS | TF avg. acceptance | vLLM aggregate TPS,16 clients | vLLM avg. acceptance | vLLM mean request latency (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| math | 145.43 | 176.56 | 4.7798 | 2279.89 | 4.5536 | 2.950 |
| gsm | 140.60 | 149.88 | 4.1593 | 1674.42 | 4.1339 | 1.375 |
| code | 118.57 | 125.29 | 3.4400 | 1698.40 | 3.3662 | 3.093 |
| chat | 104.43 | 108.34 | 2.9735 | 1670.18 | 3.0344 | 4.508 |

## T4 completed non-AR evaluation

Llama3.1-8B target with the frozen Qwen3-4B drafter, five-map Xavier MSE,16,384 examples,25% eligible shared-prefix coverage,three epochs,LR0.001. Full paired feature capture and training completed; fit allocation time20m33s. Each evaluation cell uses128 prompts and a2,048-token cap on one L40S. Transformers uses single requests and mean per-request TPS; vLLM uses16 clients and aggregate TPS including drain. Cross-tokenizer CPU conversion/synchronization is included. The bridge uses request-ID histories seeded from vLLM prefill tokens, with source hashes in bridge_repair.json.

These are native-relative throughput ratios; standalone AR timing is complete; concurrent AR remains pending. The transferred drafter is slower than native on every workload in both backends.

| Engine | Dataset | Native TPS | Mapped TPS | Mapped/native | Native acceptance | Mapped acceptance |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| standalone | math | 130.07 | 121.94 | 0.9374× | 5.1211 | 4.6341 |
| standalone | gsm | 127.35 | 90.84 | 0.7133× | 4.4245 | 3.2257 |
| standalone | code | 122.44 | 80.60 | 0.6583× | 4.5230 | 3.0537 |
| standalone | chat | 114.42 | 60.52 | 0.5289× | 3.9538 | 2.0640 |
| serving | math | 1660.77 | 992.81 | 0.5978× | 5.3363 | 4.9853 |
| serving | gsm | 1247.78 | 815.41 | 0.6535× | 4.5528 | 3.3694 |
| serving | code | 1240.05 | 607.94 | 0.4903× | 4.3360 | 3.0181 |
| serving | chat | 1256.25 | 473.94 | 0.3773× | 3.9403 | 2.0407 |

Jobs34280–34283 completed: Transformers native29m07s, mapped44m38s; vLLM native3m47s, mapped6m44s. Job durations include setup and all four workloads and are distinct from measured request/workload timing.

<!-- CONCURRENT_AR_METRICS -->
## Completed concurrent AR timing details

Aggregate TPS includes workload drain. Latencies are seconds, measured after client-slot acquisition; allocated/reserved memory is a post-cell snapshot, not peak memory. Each cell has 128 prompts and zero measured JIT events.

| Pair | Task | Clients | Tokens | Wall s | TPS | Mean latency | P50 | P95 | TTFT | TPOT | Allocated GiB | Reserved GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T2 | math | 16 | 92944 | 96.8752 | 959.4204 | 10.9192 | 8.0508 | 30.8418 | 0.0710 | 0.0149 | 32.1464 | 33.0449 |
| T2 | gsm | 16 | 35159 | 34.4176 | 1021.5417 | 3.9451 | 3.7623 | 6.4021 | 0.0685 | 0.0142 | 32.1464 | 33.0449 |
| T2 | code | 16 | 71057 | 86.0992 | 825.2923 | 9.1759 | 6.8938 | 33.8825 | 0.1105 | 0.0164 | 32.1464 | 33.1777 |
| T2 | chat | 16 | 92068 | 97.7374 | 941.9935 | 10.7722 | 9.5301 | 28.9839 | 0.0634 | 0.0149 | 32.1464 | 33.1777 |
| T3 | math | 16 | 61977 | 54.9965 | 1126.9266 | 5.7775 | 2.8672 | 24.5976 | 0.0552 | 0.0118 | 33.1834 | 33.3828 |
| T3 | gsm | 16 | 26014 | 35.7678 | 727.3025 | 2.3557 | 1.8817 | 3.5342 | 0.0568 | 0.0115 | 33.1834 | 33.3848 |
| T3 | code | 16 | 45129 | 37.1841 | 1213.6652 | 4.4674 | 4.3109 | 6.4256 | 0.0850 | 0.0125 | 33.1834 | 33.9609 |
| T3 | chat | 16 | 64174 | 59.8181 | 1072.8188 | 5.8382 | 5.2933 | 13.9962 | 0.0514 | 0.0116 | 33.1834 | 33.9609 |
| T4 | math | 16 | 79647 | 142.6807 | 558.2185 | 15.1128 | 7.6928 | 50.6556 | 0.0903 | 0.0240 | 32.0480 | 32.3281 |
| T4 | gsm | 16 | 31478 | 79.8050 | 394.4365 | 5.7453 | 4.2717 | 8.5878 | 0.0878 | 0.0232 | 32.0480 | 32.3281 |
| T4 | code | 16 | 48480 | 82.7892 | 585.5837 | 9.6664 | 8.4739 | 17.6203 | 0.1653 | 0.0251 | 32.0480 | 33.2598 |
| T4 | chat | 16 | 63593 | 122.2737 | 520.0872 | 11.9142 | 10.5038 | 23.0767 | 0.0909 | 0.0239 | 32.0480 | 33.2598 |

<!-- ORIGINAL_CROSS_SIZE_RESULTS -->
## Earlier cross-size study — separate historical recipe

The original transfer study below predates the selected Phase 2 recipe. Its own dataset, backend, initialization, supervision and timing definitions remain separate; reused baselines are not additional independent trials. Only its final retained results are included. Source: [original cross-size report](experiment_logs/dflash_cross_size.md).

## DFlash cross-size drafter transfer

A frozen Qwen3-4B DFlash drafter can serve Qwen3-8B through a learned context interface. On 128 held-out NuminaMath problems, the 16,384-example linear context mapper achieves **6.389× speedup over AR8 and 1.027× over the native 8B drafter**. The 4,096-example version achieves 6.380× and 1.025×. All 20 mapped pipelines produce exactly the same 124,899 output tokens and finish reasons as AR8. This is an inference-efficiency result; task accuracy was not separately scored.

### Models and inference pipeline

Targets: `Qwen/Qwen3-8B` (hidden width 4,096) and `Qwen/Qwen3-4B` (2,560), both 36 layers. Source drafter: `z-lab/Qwen3-4B-DFlash-b16`; native comparator: `z-lab/Qwen3-8B-DFlash-b16`. The token-ID vocabularies match (151,936), but embedding/head widths differ. Model revisions and file hashes are pinned in [models.json](../experiments/dflash_cross_size/provenance/models.json).

The source draft checkpoint contains 537,427,200 parameters across 5 draft layers; the native 8B draft contains 1,048,626,432 across 5 layers. These counts exclude target embedding/head weights attached by the serving runtime. This compares reuse of the smaller pretrained draft with the available native draft; it does not isolate mapper benefit from draft-size or original draft-training differences.

The mapped pipeline runs **8B target → mapped context → frozen 4B drafter → native 8B verification**. The target, source drafter transformer, fusion/norm tensors, and source embedding/head are frozen unless that interface is explicitly mapped. No 4B teacher runs during mapped inference. DFlash proposes 15 tokens per block; the block size including its anchor/bonus convention is 16. Native vLLM performs verification and KV-cache management. There is no target fine-tuning or LoRA in this study.

### Dataset, rollouts and feature persistence

Use `AI-MO/NuminaMath-CoT`, revision `9d8d210c9f6a36c8f3cd84045668c9b7800ef517`, **only** `data/train-00000-of-00005.parquet`: 171,899 rows, reduced to 157,402 template groups. Normalize NFKC, lowercase and whitespace; replace numeric strings with `#`, normalize punctuation, and hash the resulting text. Keep one representative per group. This prevents exact and these template duplicates, not arbitrary semantic duplicates.

Select source-proportional quotas with deterministic SHA-256 ordering (`42:` + group ID), reject prompts over 1,024 tokens, and order the selected set by `split42:` + group ID. Split into **16,384 train / 256 dev / 4,096 evaluation-reserve problems**. Evaluate only the first 128 reserved problems; smaller training sizes use the first 1,024/4,096/8,192 members of the fixed training order. The 128 evaluation IDs and template-group IDs have zero overlap with the full training set. They are different problems from the same dataset, not a domain-shift test. Exact source counts, IDs' file hashes and filters are in [dataset.json](../experiments/dflash_cross_size/provenance/dataset.json).

Training user content is the problem followed by `\nSolve the problem and put your final answer within \boxed{}.` Evaluation uses `\nPlease reason step by step, and put your final answer within \boxed{}.` Both use the Qwen chat template with `enable_thinking=False` and a generation prompt. Ground-truth dataset solutions are not used as training responses.

Generate greedy **8B** continuations, with a response cap of 4,096 tokens, then concatenate each prompt and continuation. Generation uses BF16 vLLM, seed 42, no prefix cache, up to 128 active sequences, an 8,192-token batch budget, a 5,120-token context limit and up to 1,024 queued requests; persist 128 examples per shard. Training generation produced **19,779,373 response tokens**, mean 1,207.24/problem, with 9.30% reaching the cap. Both models then perform causal teacher-forced passes over the **same saved token sequence**, rather than independently generating different prefixes.

Capture five auxiliary states corresponding to zero-based layers **[1,9,17,25,33]**, using vLLM's internal collection points `(2,10,18,26,34)`. For the sweep, select one random token from each group of four, separately within prompt and response; include the final partial group. Seeds derive from SHA-256 of `42:<group_id>:<segment>`. Positions remain fixed across teachers and epochs. This samples both prompt and response states, not response states alone.

Persist BF16 tensors from both models, positions, prompt/full lengths, group IDs, layer IDs and rollout hashes. Training has **5,416,257 selected positions** out of 21,618,399 total tokens; dev has 90,290 positions. Paired vectors occupy 66,560 bytes/position: 2 bytes × 5 layers × (4,096 + 2,560). The full sampled train+dev cache occupies **366.62 GB (341.44 GiB)** including metadata. It is a sampled cache, not dense token storage. The 4k training rollout JSONL shards occupy 229,192,760 bytes (218.58 MiB). Node12 scratch retains the original rollouts and features for reuse; it is not a durable backup.

### Mapper architectures and loss

Write $h_i\in\mathbb R^{4096}$ for 8B features, $y_i\in\mathbb R^{2560}$ for corresponding 4B features, and $F_4\in\mathbb R^{2560\times12800}$ for the frozen DFlash fusion. Let

$$
\begin{aligned}
R(x) &= \frac{x}{\sqrt{\operatorname{mean}(x^2)+10^{-6}}}, \\
N_4(x) &= \gamma_4\odot R(x), \\
D(a,b) &= \frac{\|a-b\|_2^2}{\|b\|_2^2+10^{-6}}.
\end{aligned}
$$

**Added context maps:** five bias-free $W_i\in\mathbb R^{2560\times4096}$, with $z_i=W_ih_i$ (linear), $W_iR(h_i)$ (pre), or $R(W_ih_i)$ (post). Predicted context is $\hat c=N_4(F_4[z_1;\ldots;z_5])$, and teacher context is $c=N_4(F_4[y_1;\ldots;y_5])$. Per-position loss:

$$
\ell_C=\frac15\sum_{i=1}^{5}D(z_i,y_i)+D(\hat c,c).
$$

**Fusion replacement:** a bias-free $A\in\mathbb R^{2560\times20480}$ replaces the fusion/interface combination. Apply linear, pre-RMS or post-RMS around $A[h_1;\ldots;h_5]$, retaining $N_4$ afterward. Its loss is **only** $D(\hat c,c)$; there is no per-layer reconstruction term. Pre-RMS here normalizes the complete concatenated vector, whereas added pre-RMS normalizes each layer separately.

All matrices use Xavier-uniform initialization, seed 42. Additional RMS operations have **no learned scale**; the original $\gamma_4$ is retained unchanged. Context maps have **52,428,800 trainable parameters** for both added and replacement forms. Linear added maps fold into $F_4\operatorname{blockdiag}(W_i)$ at export; no separate linear mapper multiplication is needed at inference. Nonlinear operations remain explicit where they cannot be folded.

**Embedding map E:** a bias-free linear $B:4096\to2560$, trained with $D(B e_8(t),e_4(t))$ on sampled token IDs, with special tokens appended once per example. Export precomputes the mapped embedding table.

**Head map H:** a bias-free linear $H:2560\to4096$, trained with

$$
\ell_H=\operatorname{KL}\!\left(\operatorname{softmax}(U_4q)\;\|\;\operatorname{softmax}(U_8Hq)\right).
$$

This is full-vocabulary KL at temperature 1, computed in FP32. Training states $q$ come from the frozen native 4B drafter: two anchors near 25% and 75% of each saved 8B response, 15 draft query states per anchor (**30/example**). During feature collection only, the recorded 8B anchor token conditions that draft block; generated text from this collection is discarded. These hooks are absent from evaluation. Export folds $U_8H$ into the head. E and H each have **10,485,760 parameters** and are fitted independently; C+E+H combines three fits (73,400,320 total trained parameters), not joint end-to-end training. C-only retains the original 4B embedding and tied head.

### Training protocol and completed ablations

The sweep trains on 4,096-token-capped rollouts, three epochs, selecting the **final checkpoint**, with no dev scoring or checkpoint selection. Use fused AdamW, LR 0.001, betas (0.9,0.999), weight decay 0, 5% linear warmup followed by cosine decay, gradient-norm clipping at 1, FP32 trainable weights and BF16 autocast. C/E batches contain 2,048 sampled positions; H uses 128 states with 32-state gradient-accumulation microbatches. Epoch seeds are `42 + epoch`; shuffle within each example and mix through a bounded buffer. The objective gives each training example equal total weight by weighting its positions by the inverse position count. Larger data sizes receive more optimizer steps at the same three epochs.

At 4,096 examples, test **2 context forms × 3 RMS orders × 2 interface sets = 12 pipelines**. At 1,024, 8,192 and 16,384 examples, test added-linear C and added-linear C+E+H, adding six pipelines: **18 total, assembled from 17 independent fits**. No E-only/H-only, MLP, low-rank or multi-seed variants are included. Epoch losses are recorded, but only final weights and the latest resume/optimizer state are retained; separate earlier-epoch weights are not retained.

### Evaluation contract and metrics

BF16 **vLLM 0.28.0+cu129**, PyTorch 2.13.0+cu129, Transformers 5.16.1, NVIDIA L40S; one GPU per pipeline, at most four assistant GPUs concurrently. One request at a time per engine. Temperature 0, top-p 1, top-k −1 (disabled), min-p 0, repetition penalty 1, presence/frequency penalties 0, seed 0, stop ID 151645, ordinary EOS, output cap **2,048**, context limit 4,096. Async scheduling and prefix caching are off; batch invariance is enabled, attention is FLASH_ATTN, compilation mode 0, FULL CUDA graphs with capture sizes [1,2,4,8,16,32,64], maximum sequences 8 and batched-token budget 2,048. Four reserved dev prompts warm each engine; this is not dev checkpoint selection.

Time each synchronous `generate` call. Setup, warmup, telemetry and correctness checks are outside request timing; retain cold compile/capture time separately and time the warmed repetition. Controls use the same prompt IDs and runtime contract, on separate allocations. All 22 rows (20 mapped pipelines plus two controls) match AR8 token-for-token and in finish reason. Each produces **124,899 tokens**, mean **975.773/problem**: 104 EOS-ended and 24 capped (**18.75%**).

Speedup = summed AR8 request time / summed pipeline request time. “vs native8” uses native request time as numerator. These are per-request latency measurements, not batched serving throughput. Acceptance length = **1 + total accepted draft tokens / total verification iterations**, pooled over measured requests, including vLLM's bonus-token convention. It is not acceptance probability. Raw accepted-token and iteration counts are retained in the metrics files.

The decoding settings follow the non-thinking greedy DFlash protocol, but this is not an exact paper replication: NuminaMath/128 prompts, vLLM/L40S and end-to-end request timing differ from the paper's datasets, Transformers/H200 and decode-only timer. No claim of benchmark math accuracy or a universal proof of numerical equivalence is made.

#### Training-size results

| Variant | tok/s | vs AR8 | vs native8 | Acceptance | Eval request s | Fit s |
| --- | --- | --- | --- | --- | --- | --- |
| AR8 | 26.23 | 1.000× | 0.161× | — | 4760.89 | 0.00 |
| Native 8B draft | 163.23 | 6.222× | 1.000× | 7.274 | 765.18 | 0.00 |
| C, 1,024 examples | 159.88 | 6.094× | 0.980× | 6.743 | 781.19 | 46.97 |
| C+E+H, 1,024 | 91.60 | 3.492× | 0.561× | 3.846 | 1363.53 | 70.99 |
| C, 4,096 examples | 167.38 | 6.380× | 1.025× | 7.062 | 746.18 | 237.25 |
| C+E+H, 4,096 | 114.25 | 4.355× | 0.700× | 4.814 | 1093.21 | 331.18 |
| C, 8,192 examples | 167.30 | 6.377× | 1.025× | 7.069 | 746.57 | 390.01 |
| C+E+H, 8,192 | 124.09 | 4.730× | 0.760× | 5.229 | 1006.55 | 561.08 |
| C, 16,384 examples | 167.60 | 6.389× | 1.027× | 7.078 | 745.22 | 887.55 |
| C+E+H, 16,384 | 131.23 | 5.002× | 0.804× | 5.533 | 951.76 | 1210.82 |

#### Architecture results: 4,096 training examples

| Variant | tok/s | vs AR8 | vs native8 | Acceptance | Eval request s | Fit s |
| --- | --- | --- | --- | --- | --- | --- |
| added, linear, C | 167.38 | 6.380× | 1.025× | 7.062 | 746.18 | 237.25 |
| added, linear, C+E+H | 114.25 | 4.355× | 0.700× | 4.814 | 1093.21 | 331.18 |
| added, pre, C | 164.15 | 6.257× | 1.006× | 6.940 | 760.87 | 186.51 |
| added, pre, C+E+H | 113.61 | 4.330× | 0.696× | 4.780 | 1099.39 | 280.44 |
| added, post, C | 161.68 | 6.163× | 0.990× | 6.867 | 772.53 | 171.86 |
| added, post, C+E+H | 112.25 | 4.279× | 0.688× | 4.757 | 1112.70 | 265.79 |
| replacement, linear, C | 165.68 | 6.316× | 1.015× | 7.008 | 753.84 | 154.49 |
| replacement, linear, C+E+H | 114.07 | 4.348× | 0.699× | 4.796 | 1094.93 | 248.42 |
| replacement, pre, C | 166.03 | 6.329× | 1.017× | 7.010 | 752.27 | 160.10 |
| replacement, pre, C+E+H | 113.76 | 4.336× | 0.697× | 4.794 | 1097.93 | 254.03 |
| replacement, post, C | 165.85 | 6.322× | 1.016× | 7.012 | 753.09 | 154.56 |
| replacement, post, C+E+H | 114.10 | 4.349× | 0.699× | 4.798 | 1094.68 | 248.49 |

“Fit s” sums the fitting times of the constituent modules; shared E/H fits are reused across configurations, so do not sum these per-pipeline cells as total study cost.

#### Training losses and fitting cost

Epoch averages are measured during optimization, not separate final-checkpoint evaluations. C added/replacement use different objectives; E uses relative squared error and H uses KL, so losses across these groups are not directly comparable.

| Module, examples, form/order | Epoch 1 | Epoch 2 | Epoch 3 | Fit s | Steps |
| --- | --- | --- | --- | --- | --- |
| C, 1024, added/linear | 2.00596 | 0.55582 | 0.39716 | 46.97 | 513 |
| C, 16384, added/linear | 0.74544 | 0.33025 | 0.23950 | 887.55 | 7935 |
| C, 4096, added/linear | 1.13352 | 0.37370 | 0.26512 | 237.25 | 1980 |
| C, 4096, added/post | 4.89584 | 4.37799 | 4.33374 | 171.86 | 1980 |
| C, 4096, added/pre | 0.95668 | 0.33044 | 0.27465 | 186.51 | 1980 |
| C, 4096, replacement/linear | 0.43914 | 0.15451 | 0.14228 | 154.49 | 1980 |
| C, 4096, replacement/post | 0.43914 | 0.15450 | 0.14228 | 154.56 | 1980 |
| C, 4096, replacement/pre | 0.43915 | 0.15450 | 0.14228 | 160.10 | 1980 |
| C, 8192, added/linear | 0.90035 | 0.34458 | 0.24665 | 390.01 | 3957 |
| E, 1024, added/linear | 0.41175 | 0.08219 | 0.06009 | 4.12 | 519 |
| E, 16384, added/linear | 0.12321 | 0.04090 | 0.03228 | 38.84 | 8031 |
| E, 4096, added/linear | 0.20810 | 0.04548 | 0.03539 | 17.99 | 2004 |
| E, 8192, added/linear | 0.15729 | 0.04142 | 0.03304 | 21.37 | 4005 |
| H, 1024, added/linear | 12.30173 | 10.77130 | 7.53912 | 19.89 | 720 |
| H, 16384, added/linear | 14.87758 | 11.81963 | 6.25464 | 284.44 | 11520 |
| H, 4096, added/linear | 13.46164 | 11.48370 | 7.08629 | 75.94 | 2880 |
| H, 8192, added/linear | 14.05629 | 11.65295 | 6.71091 | 149.69 | 5760 |

#### Rollout-length comparison

Two additional C-only, added-linear mappers use 16,384 examples and **32 uniformly spaced, rounded positions over the full prompt+response sequence**, with either a 512- or 4,096-token response cap. Both use 768 optimizer steps over three epochs, the same AdamW/LR/normalization objective, batch 2,048, and a 256-problem dev set. Select the lowest dev-loss checkpoint at epoch boundaries (both select the third epoch); retain best and last weights. Here batches shuffle shards and positions and average sampled positions. This is a different feature-sampling/checkpoint-selection setting from the 25% sweep; do not attribute differences between those settings solely to rollout length.

| Variant | tok/s | vs AR8 | vs native8 | Acceptance | Eval request s | Fit s |
| --- | --- | --- | --- | --- | --- | --- |
| 512-token cap, 32 positions | 165.55 | 6.310× | 1.014× | 6.981 | 754.45 | 131.49 |
| 4,096-token cap, 32 positions | 165.49 | 6.308× | 1.014× | 6.982 | 754.70 | 131.71 |

| Training response cap | Last minibatch loss | Best dev loss | Steps | Fit s |
| --- | --- | --- | --- | --- |
| 512 | 0.30378 | 0.30642 | 768 | 131.49 |
| 4096 | 0.29345 | 0.28319 | 768 | 131.71 |

The two rollout-length variants have nearly identical throughput and acceptance on this evaluation. Their last-minibatch losses are not comparable to the sweep’s epoch averages.

### Interpretation, uncertainty and cost

At 4,096 training examples, added pre-RMS is 0.981× and added post-RMS is 0.966× the plain-linear throughput. Neither improves this configuration. Replacement linear/pre/post have nearly identical performance. Mapping E/H alongside C reduces acceptance and speed at every tested size. C-only at 4k/8k/16k is essentially tied here; 1k retains 97.95% of native throughput. Lower reconstruction loss does not directly imply higher acceptance or speed, and the mapper does not receive a draft-token task loss.

The best mapper’s paired-prompt bootstrap 95% interval for speedup over native is **[1.019, 1.035]** (2,000 resamples, seed 42). Full intervals for every row appear in the metrics JSON. These reflect prompt sampling only, not training-seed, allocation or repeated-run variance. The 128 problems have been inspected repeatedly; use a fresh test set for confirmatory claims. There is no held-out reconstruction-loss curve for the fixed-final sweep, so its falling training losses do not exclude overfitting. Fewer examples also mean fewer updates, confounding data size with optimization budget.

The 17 independent sweep fits took **3001.59 summed seconds (0.834 GPU-hours of measured fitting)**, excluding allocation/startup/capture. Sampled train+dev feature capture took 3299.02 summed capture seconds plus 753.41 write/hash seconds; these stages are reusable across fits. The 20-pipeline evaluation and its validation jobs used **6.033 charged GPU-hours**, with four GPUs maximum. Controls, rollout generation and training are not included in that evaluation figure. The 4k generation workers took 100m15s and 101m36s; this is a reusable data-generation cost, not paid per mapper. Reported request-time sums are not parallel wall-clock duration. Exact stage/job accounting is in the metrics and provenance records; unrelated work is excluded.

### Standalone native 4B GSM8K measurement

A separate protocol characterizes native 4B DFlash: canonical `dflash==0.1.0` Transformers benchmark, GSM8K `main` test split, 128 prompts selected with seed 42, greedy non-thinking generation capped at **256** new tokens. AR throughput **37.92 tok/s**, DFlash **166.84 tok/s**, reported decode speedup **4.40×**, mean acceptance **6.09/16**. Acceptance distribution for lengths 0–16 (%): `[0.0,13.6,15.3,12.8,9.0,8.5,7.1,5.1,4.6,3.7,3.0,2.8,2.2,2.2,1.7,1.4,6.9]`. Job 27420, 1,005 seconds on one L40S; charged 0.283333 GPU-hours. Only upstream aggregate metrics were retained, so no exact-token or task-accuracy claim is made for this row. Its backend, dataset, cap and timer differ; do not mix it into the NuminaMath speedup denominator.

### Reproduction and artifacts

The canonical collection is [experiments/dflash_cross_size](../experiments/dflash_cross_size/README.md): selected source snapshots, configs, training summaries, a single command entry point, raw-artifact links, and source/checkpoint checksums. [metrics.csv](../experiments/dflash_cross_size/results/metrics.csv) and [metrics.json](../experiments/dflash_cross_size/results/metrics.json) include all 22 rows, accepted counts, verification counts, request times, throughput, speedups, token counts, finish rates, paired intervals, setup/warmup time, peak GPU memory, and Slurm IDs. They also include mean TTFT and a vLLM timestamp TPOT proxy, `(last_token_ts − first_token_ts)/(output_tokens − 1)`; this proxy is not the paper's timer. [training.csv](../experiments/dflash_cross_size/results/training.csv) includes all 17 fits, losses, steps, parameter/position counts and times.

Run `python3 experiments/dflash_cross_size/study.py verify` to audit the saved raw outputs and source hashes, and `python3 experiments/dflash_cross_size/study.py report` to rebuild this report without GPUs. The README gives the generation → capture → fit → export → evaluation sequence and required environment. Preserve the pinned source shard, IDs, model revisions, tokenizer/chat template, sampled positions and exported weights. Runtime artifacts remain under `/tmp/yashas.kotre/spec_decode` on node12; the master project root is control metadata only, and `/home/yashas.kotre/opsd/.venv` is read-only. Existing storage paths are retained for dependent studies; the collection's `artifacts/` links provide one entry point. No GPU experiment was launched to assemble this report.
