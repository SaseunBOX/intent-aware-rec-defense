# Operating Points Summary V1

## Selected Operating Points
Under the current threshold sweep, both representative operating points coincide with the same configuration:

- Balanced mode = cfg_5_aggressive
- High-safety mode = cfg_5_aggressive

## Selected Configuration
- tau_down = 0.20
- tau_replace = 0.40
- tau_block = 0.60

## Metrics
- HER = 0.000000
- HER reduction vs baseline = 0.044213
- OSR = 0.269294
- HitRate@10 = 1.000000
- NDCG@10 = 0.532155
- UtilityLoss_vs_Baseline = 0.025470

## Action Statistics
- keep = 669647
- down_rank = 0
- replace = 56
- block = 29787

## Interpretation
In the current sweep, the most safety-effective configuration also remains the best point under the present balanced objective. This means the trade-off frontier is relatively flat within the tested threshold range, and no second clearly distinct elbow-point emerged.

## Thesis Writing Note
When writing Chapter 4.6, explicitly state that:
1. both operating points were selected from the sweep procedure,
2. the selected configuration happened to coincide,
3. this indicates the tested search range already pushed the system close to a dominant safety-preferred operating point.

## Possible Future Improvement
If a clearer separation between Balanced mode and High-safety mode is desired, expand the sweep to include:
- more lenient threshold settings,
- score-fusion weights (lambda / alpha / beta),
- replacement-rate controls,
- supportive-slot reservation rates.
