# Chapter 4 and Chapter 5 Writing Summary V3

## Intent Model Upgrade Effect
The upgraded session-level intent model improved downstream ranking utility without changing harmful-exposure suppression.

### Baseline Full Pipeline with old intent model
- HER = 0.000048
- NDCG@10 = 0.532155
- UtilityLoss_vs_Baseline = 0.025470

### New intent_model_v2_light with best-NDCG configuration
- HER = 0.000048
- NDCG@10 = 0.540076
- UtilityLoss_vs_Baseline = 0.010965

### New intent_model_v2_light with balanced configuration
- HER = 0.000048
- NDCG@10 = 0.535596
- UtilityLoss_vs_Baseline = 0.019169

## Interpretation
The upgraded intent model increased NDCG@10 while HER remained unchanged. This suggests that better intent estimation mainly improved ranking calibration among retained candidates, while harmful suppression continued to be dominated by the hard threshold policy.

## Thesis Writing Note
This result is useful because it shows that improving the intent layer is not only a standalone classification improvement. It also yields measurable downstream benefit in recommendation quality, even when the intervention structure is unchanged.
