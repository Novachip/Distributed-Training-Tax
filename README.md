# Distributed-Training-Tax
Model for DiLoCo training tax for frontier AI runs, based on this paper: https://arxiv.org/html/2605.29359

The question: could a covert actor evade satellite-based detection by splitting a frontier training run across many small sites, and what would it cost them? Primary source: Rahman (2026), *Does Distributed Training Undermine Compute Governance?* (arXiv:2605.29359). An interactive model accompanies this note (`Distributed training tax model.html` in this folder); all figures below are its outputs at the stated settings.

## The model and its starting assumptions

The model prices a covert frontier run in three modes, distinguished by what each site looks like from above. **Mode A** — few large, datacentre-class nodes linked by dedicated fibre (the cheap way to distribute, and the visible one). **Mode B** — many sites small enough to pass as ordinary commercial buildings (~1–5 MW) on commercial internet. **Mode C** — Rahman's threat model: 16-H100-equivalent nodes below every proposed monitoring threshold, on consumer internet.

Fixed assumptions: a 90-day run at 30% MFU on H100-equivalents (2×10¹⁵ FLOP/s FP8 peak, 1.2 kW/chip all-in — Colossus 1's ratio, ~$50k installed per chip using Rahman's cost factors). The target is quality-adjusted, centralised-equivalent FLOP; for reference, Grok 4 ≈ 5×10²⁶, and 10²⁸ is roughly the Colossus 2 generation. The training tax is the nominal compute purchased divided by that target, i.e. 1/(η·χ), where η multiplies together the DiLoCo penalties — sync-interval staleness, gradient compression (150×, per Covenant-72B), replica divergence, pipeline activation quality, batch-size overrun and pipeline-bandwidth stalls — and χ is a Chinchilla-suboptimality penalty for over- or under-trained models. Chip count, cost and power all scale with the same factor.

The divergence term, which dominates, is calibrated so the model reproduces all six of Rahman's Table 1 configurations (inferred coefficient within ~5%; total η within ~1.3–1.6× across his Tables 1 and 7). For each target the solver finds the cheapest configuration, jointly optimising pipeline depth, sync interval, batch overrun factor and model size.

Two disciplines were added that Rahman's simulator omits, and they drive the results. First, **batch discipline**: fleet-wide tokens per optimiser step are capped at the critical batch size (default 15M tokens at 100B parameters, scaling with √N as a proxy for the data-scaling found by Zhang et al. 2024 and Merrill et al. 2025), with overruns paying the McCandlish steps–data trade-off (2/(1+f) — diminishing, not zero, returns). Second, **pipeline bandwidth**: compressed activations must actually fit through each site's link; if not, throughput degrades proportionally. Both the critical batch size and the activation compression ratio are adjustable, because they are the genuinely contested assumptions.

## Headline findings

**1. The decisive variable is a piece of undemonstrated ML science, not anything about datacentres.** Whether covert distribution works at frontier scale turns almost entirely on whether effective batch sizes can be pushed orders of magnitude beyond anything demonstrated. Rahman's configurations implicitly require it — his GPT-5-scale setup implies ~4 *billion*-token effective batches, ~100× beyond demonstrated critical batch sizes, while the DiLoCo experiments his efficiency factors are calibrated on all stayed within the normal range (≤16 replicas). Every other disagreement — node size, pipeline depth, memory richness — is second-order by comparison.

**2. Under demonstrated batch behaviour, hidden distribution doesn't get taxed — it hits a wall.** With the critical-batch default, Mode B *plateaus*: no amount of hardware reaches beyond ~2×10²⁶ at 1 Gbps per site, or ~10²⁷ at 10 Gbps. Mode C is unviable at any spend, at any target from 10²⁶ up. The mechanism is compounding: adding hardware adds replicas and sync burden faster than useful compute. A true frontier run (≥5×10²⁶ today) simply cannot be hidden in small buildings on 90 days under these assumptions — at any price.

**3. The distribution that works is the visible kind, and it gets more visible as runs grow.** At 10²⁷, the cheapest viable configuration is ~20 sites of ~24 MW each (~400k chips, ~$20B) at a 1.9× tax — every site a recognisable AI datacentre, detectable by exactly the cooling-signature methods in the main post. At 10²⁸, viability requires ~30 sites of ~150 MW on 100 Gbps fibre (1.7×), or ~9 sites of ~290 MW on 400 Gbps at ~1× — which is no longer evasion at all, just ordinary multi-campus training of the kind frontier labs already do openly.

**4. The tax is a cliff, not a slope.** At fixed infrastructure (25 MW nodes, 10 Gbps), Mode A costs 1.9× at 10²⁷, 7.6× at 2×10²⁷, and nothing works at 3×10²⁷. Because the divergence penalty compounds exponentially in replicas × sync interval, "pay a bit more tax" is not generally an option — configurations are either roughly affordable or unreachable. This is why summarising distributed training with a single tax multiplier misleads.

**5. Grant Rahman his batch assumption and the satellite argument genuinely fails.** With billion-token batches allowed, Mode B trains 10²⁷ at a 1.01× tax across 145 anonymous 1.8 MW buildings (~216k chips, ~$11B), and still manages 10²⁸ at 2.2× across ~3,000 sites. Mode C reaches 10²⁶ at ~1.9× using GH200-class memory-rich nodes (~$2B, ~2,600 sites) — though even here, 10²⁷ stays out of reach. In that world, detection must fall back on procurement tracking, chip registries and whistleblowers, exactly as Rahman argues. Notably, his memory-rich hardware choices only matter *jointly* with the batch assumption: with realistic batches, upgrading Mode C's nodes from 1.28 TB to 2.3 TB changes nothing.

| Target | Mode A (central) | Mode B (central) | Mode B (Rahman batches) | Mode C (Rahman batches) |
|---|---|---|---|---|
| 10²⁶ | 1.03× — 2 sites | 3.8× (1G) / 1.2× (10G) | ~1× | 21× (H100) / 1.9× (GH200) |
| 10²⁷ | 1.9× — 20 × 24 MW | unreachable | 1.01× — 145 × 1.8 MW | unreachable |
| 10²⁸ | 1.7× — 30 × 150 MW, 100G fibre | unreachable | 2.2× — ~3,000 sites | unreachable |

## What this means for the footnote

The original claim — distribution imposes a prohibitive tax and the clusters would anyway be detectable — survives in a sharper, conditional form. Under batch-size behaviour anyone has actually demonstrated, both halves are right simultaneously: the hideable configurations plateau below the frontier regardless of spend, and the viable configurations are dozens of satellite-recognisable AI datacentres (whose minimum size grows with the frontier). The claim's exposure is a single assumption: if effective batches can scale ~100× beyond current evidence, the tax largely evaporates and the detection burden shifts from imagery to procurement — a covert actor would need ~216k smuggled H100-equivalents even then. The honest formulation is therefore not "distribution is prohibitive" but "distribution at frontier scale requires betting billions, covertly, on an unproven extrapolation of ML science that the public literature has not validated past a few percent of the required scale — and racing rivals who don't have to".

## Caveats

Everything beyond ~10²⁶ extrapolates a simulator that itself extrapolates experiments at ≤16B parameters and ≤16 replicas (largest real DiLoCo-family run: Covenant-72B, 4.8×10²³ FLOP). The critical-batch scaling law above ~60M tokens is untested in both directions. Chinchilla-optimal data volumes at 10²⁷⁺ exceed plausibly available unique tokens — an unmodelled further penalty on all modes. Algorithmic progress (compression went 16×→150× in one research cycle) pushes taxes down over time, so these figures should be date-stamped mid-2026. Longer runs than 90 days shrink chip counts proportionally but conflict with racing incentives (the unregulated longest-useful-run bound is ~4.5 months, per Sevilla et al. 2022). Detection labels in the model use a heuristic 5 MW per-site threshold, not a physical detection model.

## Sources

- Rahman, R. (2026). *Does Distributed Training Undermine Compute Governance?* arXiv:2605.29359 — Tables 1 and 7, Appendices A–C, E and G; simulator documentation (github.com/robirahman/miri-decentralized-training-report).
- Epoch AI: Grok 4 training resources; Colossus 1 directory entry (280k H100e, 340 MW → ~1.2 kW/H100e); frontier data centres hub.
- Sevilla et al. (2022), *The Longest Training Run*; Sevilla & Troynikov (2025), *Could Decentralized Training Solve AI's Power Problem?*
- Lidin et al. (2026), Covenant-72B (arXiv:2603.08163) — largest demonstrated DiLoCo-family run.
- McCandlish et al. (2018), *An Empirical Model of Large-Batch Training*; Zhang et al. (2024), arXiv:2410.21676; Merrill et al. (2025), arXiv:2505.23971 — critical batch size.
- Hoffmann et al. (2022) and Besiroglu et al. (2024) — Chinchilla scaling, χ calibration.
