# DAI UAV Agent v4.3: Counterfactual Candidate Verification

## Goal

V4.2 showed that the frozen candidate pool has recovery headroom, but its one-pass multicandidate verifier was not reliable. Self-reported confidence concentrated around 0.90/0.95, small labels did not resolve small targets, and the final gate reused the same visual score. V4.3 changes inference only. It does not retrain, use `question_e`, or expose GT to the verifier.

## Method

1. Reuse the frozen v4.1 Target and Zoom candidates; do not regenerate the final box.
2. Verify concrete candidate boxes, including Zoom members inside a hypothesis cluster, instead of cluster representatives only.
3. Route only when the parent has not accepted the initial result, the alternative has cross-view or Zoom support, and its non-visual score is not lower than the initial score.
4. Render one comparison sheet: the original global image is on top and two context-preserving candidate crops are below.
5. Compare H0 with Hk twice. The second call swaps A/B labels, colors, crop positions, and crop scale.
6. Accept a pairwise result only when both calls map to the same candidate. Never use model-reported numeric confidence.
7. Replace the initial box only when exactly one alternative wins both counterbalanced calls and the independently recomputed gate still passes.

The independent score is

\[
S_{ind}=0.40S_{token}+0.20S_{target}+0.15S_{relation}+0.15S_{global}+0.10S_{shape}.
\]

It excludes all counterfactual-verifier outputs. The fixed formal configuration requires:

- original parent decision is not `accept`;
- Zoom or cross-view support;
- `S_ind(Hk) >= S_ind(H0)`;
- `IoU(Hk,H0) <= 0.85`;
- alternative deduplication at IoU 0.92;
- at most one alternative and exactly two counterbalanced calls.

The original v4.1 budget is at most three specialized perception calls. V4.3 adds at most two calls, so the specialized-call ceiling is five before human escalation.

## Protocol guards

- Single image and single physical viewpoint only.
- Candidate crops are transformed observations, not real multiview data.
- `question_e_used=false`.
- `gt_visible=false`; GT is joined only after final-box commitment.
- `self_reported_confidence_used=false`.
- `final_bbox_regenerated=false`; the final box must be selected from the frozen pool.

This v4.3 run is a development experiment motivated by the v4.2 test failure. For a final unbiased paper table, freeze all gates on an independent calibration split from DVGBench train before one final test evaluation.
