# Chapter 4 and Chapter 5 Writing Summary V2

## 1. Main Findings
- The injected baseline produced HER = 0.044213 with NDCG@10 = 0.546064.
- The thesis-style Full Pipeline V2 reduced HER to 0.000048 while preserving NDCG@10 = 0.532155.
- The corresponding utility loss relative to the injected baseline was 0.025470.
- This indicates that the multi-stage defense sharply reduced harmful exposure while preserving a high level of recommendation utility.

## 2. Ablation Interpretation
- Risk-only reduced HER more strongly than Intent-only, confirming that content-risk modeling is the most direct mechanism for suppressing harmful-promotional exposure.
- Intent-only changed OSR more clearly than Risk-only, suggesting that user-intent awareness mainly improves calibration of intervention and helps control unnecessary suppression.
- No_Replacement_V1 performed almost identically to Full_Pipeline_V2 on the current metrics, which suggests that replacement contributed less to the headline metrics under the current threshold range and small safe replacement pool.
- Global_Threshold_V1 slightly worsened HER relative to Full_Pipeline_V2 while keeping OSR and NDCG nearly unchanged, suggesting that a coarse global threshold can partially mimic the full policy but loses some precision in harmful exposure control.

## 3. Operating Points
- Balanced mode and High-safety mode both selected cfg_5_aggressive in the current sweep.
- This indicates that the current sweep frontier is relatively flat and already concentrated near a strongly safety-preferred region.

## 4. Model Quality
- intent_model_v1 overall accuracy = 0.647296
- risk_model_v1 overall accuracy = 0.995675

## 5. Chapter 5 Discussion Points
- The strongest safety gains emerged only after combining probabilistic intent estimation, probabilistic risk estimation, score fusion, and hard intervention rules.
- The full pipeline outperformed single-stage variants in overall safety-utility balance.
- The current study remains an offline simulation, so the results should be interpreted as controlled experimental evidence rather than direct online deployment performance.
- Future work should enlarge the safe replacement pool, improve calibration of the intent model, and expand the sweep space so that Balanced and High-safety operating points separate more clearly.
